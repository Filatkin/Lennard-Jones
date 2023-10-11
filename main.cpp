#include <iostream>
#include <fstream>
#include <array>
#include <random>
#include "Vector.h"

// getting data file path on the particular device
const std::string FILE_PATH = __FILE__;
const int amountOfSymbolsBeforeRootDirectory = 8;
const std::string DIR_PATH = FILE_PATH.substr(0, FILE_PATH.size() - amountOfSymbolsBeforeRootDirectory);

//number of particles, integration steps, density
const unsigned int N = 1000, numOfSteps = 10000;
// maximum number of particles along each coordinate
const unsigned int n = std::ceil(std::cbrt(N));
// integration step
const double dt = 1e-4;
// mass of particles, lennard-jones parameters, reduced density (rho_reduced = rho_real / sigma^3), reduced temperature (tem_reduced = tem_real / epsilon)
const double m = 1, sigma = 1, epsilon = 1, rho = 0.4, T_thermostat = 1;
// size of a box
const double step = 1 / std::cbrt(rho), L = n * step;
//cut range
const double rc = L / 2;
//potential energy and force at cut range
const double rc2 = rc * rc, rc6 = rc2 * rc2 * rc2;
const double Uc = 4 / rc6 * (1 / rc6 - 1), Fc = 24 * (2 / rc6 - 1) / rc6 / rc;

// initialization
std::array<Vector, N> initCoord() {
    std::array<Vector, N> coord;
    unsigned int a = 0;
    for (unsigned int i = 1; i <= n; i++)
        for (unsigned int j = 1; j <= n; j++)
            for (unsigned int k = 1; k <= n; k++) {
                if (a < N) {
                    coord[a] = Vector(i * step, j * step, k * step);
                    a++;
                } else {
                    i = n + 1;
                    j = i;
                    k = i;
                };
            }
    return coord;
}

std::array<Vector, N> initVel() {
    std::array<Vector, N> vel;
    // randomization
    std::random_device rd;
    //std::mt19937 gen(0);
    std::mt19937 gen(rd());
    double v_axis = std::sqrt(12 * T_thermostat);
    std::uniform_real_distribution<> dist_vel(-0.5, 0.5);
    // random uniform velocity distribution
    for (unsigned int i = 0; i < N; i++) {
        vel[i] = Vector(dist_vel(gen), dist_vel(gen), dist_vel(gen)) * v_axis;
    }
    return vel;
}

Vector floorVector(const Vector &r) {
    return Vector(std::floor(r.x), std::floor(r.y), std::floor(r.z));
}

//periodic boundary conditions
Vector periodicDist(const Vector &dr) {
    return dr - L * floorVector(dr / L + 0.5);
}

Vector periodicPosition(const Vector &r) {
    return r - L * floorVector(r / L);
}

// total interaction force and potential energy calculation
std::pair<std::array<Vector, N>, double> totalInteraction(const std::array<Vector, N> &r, std::vector<double> &dist) {
    double k = 0;
    // total force initialization
    std::array<Vector, N> F;
    for (unsigned int i = 0; i < N; i++) {
        F[i] = Vector(0, 0, 0);
    }
    // total potential initialization
    double U = 0;
    // calculation of total force and total potential energy
    for (unsigned int i = 0; i < N - 1; i++)
        for (unsigned int j = i + 1; j < N; j++) {
            Vector dr = periodicDist(r[i] - r[j]);
            const double r2 = dr * dr, r1 = std::sqrt(r2);
            dist[k] += r1;
            if (r1 <= rc) {
                const double r6 = r2 * r2 * r2, r8 = r6 * r2;
                Vector Force = dr * (24 / r8 * (2 / r6 - 1) - Fc / r1);
                F[i] = F[i] + Force;
                F[j] = F[j] - Force;
                U += 4 / r6 * (1 / r6 - 1) - Uc + Fc * (r1 - rc);
                //U += 4 / r6 * (1 / r6 - 1);
            }
            k++;
        }
    return std::make_pair(F, U);
}

// collecting and calculating data for velocity auto-correlation function
std::pair<std::vector<double>, std::vector<double>>
VelocityAutoCorrelationFunction(const std::vector<std::array<Vector, N>> &V) {
    std::vector<double> vacf(numOfSteps / 2, 0), norm = vacf;
    // fix t_delay = i * dt
    for (unsigned int i = 0; i < numOfSteps / 2; i++)
        // fix t0 = j * dt
        for (unsigned int j = 0; j <= numOfSteps / 2; j++)
            // sum over particles
            for (unsigned int k = 0; k < N; k++) {
                // vacf = <v(t0 + t_delay) * v(t0)>_t0,N
                vacf[i] += V[j][k] * V[i + j][k];
                norm[i] += V[j][k] * V[j][k];
            }
    return std::make_pair(vacf, norm);
}

int main() {
//  initialization;
    std::array<Vector, N> r = initCoord(), v = initVel(), v_mid = v, r0 = r, r_true = r0;
    std::vector<std::array<Vector, N>> v_all;
    v_all.reserve(numOfSteps + 1);
//  opening a file to write data, analyzed further in python
    std::ofstream outfile(DIR_PATH + "data.csv");
    outfile << N << "," << dt << "," << numOfSteps << "," << sigma << "," << epsilon << "," << m << "," << L
            << "," << rho << "," << T_thermostat << std::endl;
//  data containers for further usage
    std::vector<double> dist_all(N * (N - 1) / 2, 0);
    std::pair<std::array<Vector, N>, double> Total_old = totalInteraction(r, dist_all);
    std::array<Vector, N> F_old = Total_old.first;
//  saving initial values
    Vector P_old(0, 0, 0);
    double K_old = 0;
    for (unsigned int i = 0; i < N; i++) {
        P_old = P_old + v[i];
        double v2 = v[i] * v[i];
        outfile << std::sqrt(v2) << ",";
        K_old += v2;
    }
    v_all.emplace_back(v);
    double U_old = Total_old.second;
    K_old *= 0.5;
    outfile << std::sqrt(P_old * P_old) << "," << K_old << "," << U_old << "," << K_old + U_old << "," << 0
            << std::endl;
//  velocity verlet integration of motion equations
    for (unsigned int k = 0; k < numOfSteps; k++) {
//      kinetic energy initialization
        double K = 0, MSD = 0;
//      total momentum introduction
        Vector P(0, 0, 0);
//      one step of integration
        for (unsigned int i = 0; i < N; i++) {
            v_mid[i] = v[i] + F_old[i] * (0.5 * dt);
            r[i] = periodicPosition(r[i] + v_mid[i] * dt);
            r_true[i] = r_true[i] + v_mid[i] * dt;
            MSD += (r_true[i] - r0[i]) * (r_true[i] - r0[i]);
        }
//      new force calculation
        std::pair<std::array<Vector, N>, double> Total = totalInteraction(r, dist_all);
        std::array<Vector, N> F = Total.first;
//      velocity calculation
        for (unsigned int i = 0; i < N; i++) {
            v[i] = v_mid[i] + F[i] * (0.5 * dt);
//          write velocity in file
            double v2 = v[i] * v[i];
            outfile << std::sqrt(v2) << ",";
//          momentum calculation
            P = P + v[i];
//          kinetic energy calculation
            K += v2;
        }
        F_old = F;
        v_all.emplace_back(v);

        K *= 0.5;
//        velocity rescaling for NVE ensemble
//        double T = 2 * K / (N - 1) / 3;
//        std::cout << T << std::endl;
//        for (unsigned int i = 0; i < N; i++) {
//             v[i] = v[i] * std::sqrt(T_thermostat / T);
//        }
        double U = Total.second;
        // write in file
        outfile << std::sqrt(P * P) << "," << K << "," << U << "," << K + U << "," << MSD / N << std::endl;
    }
//  VACF calculation
    std::pair<std::vector<double>, std::vector<double>> vacf_full = VelocityAutoCorrelationFunction(v_all);
    std::vector<double> vacf = vacf_full.first, norm = vacf_full.second;

//  write in file data for VACF and PCF
    for (unsigned int i = 0; i < numOfSteps / 2; i++) {
        outfile << vacf[i] << "," << norm[i] << "," << std::endl;
    }
//  averaging data for pair correlation function
    for (unsigned int i = 0; i < N * (N - 1) / 2; i++) {
        outfile << dist_all[i] / (numOfSteps + 1) << ",";
    }
    outfile.close();
    return 0;
}
