#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include "Vector.h"

// getting data file path on the particular device
const std::string FILE_PATH = __FILE__;
const int amountOfSymbolsBeforeRootDirectory = 8;
const std::string DIR_PATH = FILE_PATH.substr(0, FILE_PATH.size() - amountOfSymbolsBeforeRootDirectory);

//number of particles, integration steps, density
const unsigned int N = 216, numOfSteps = 1000;
// maximum number of particles along each coordinate
const unsigned int n = std::ceil(std::cbrt(N));
// integration step
const double dt = 0.005;
// mass of particles, lennard-jones parameters, reduced density (rho_reduced = rho_real / sigma^3), reduced temperature (tem_reduced = tem_real / epsilon)
const double m = 1, sigma = 1, epsilon = 1, rho = 0.8, T_thermostat = 1.25;
// turning NVT-ensemble
const bool enable_NVT = true;
// size of a box
const double step = 1 / std::cbrt(rho), L = n * step;
//cut range
const double rc = L / 2;
//potential energy and force at cut range
const double rc2 = rc * rc, rc6 = rc2 * rc2 * rc2;
const double Uc = 4 / rc6 * (1 / rc6 - 1), Fc = 24 * (2 / rc6 - 1) / rc6 / rc;

// initialization
std::vector<Vector> initCoord() {
    std::vector<Vector> coord;
    coord.reserve(N);
    unsigned int a = 0;
    for (unsigned int i = 0; i < n; i++)
        for (unsigned int j = 0; j < n; j++)
            for (unsigned int k = 0; k < n; k++) {
                if (a < N) {
                    coord.emplace_back((i + 0.5) * step, (j + 0.5) * step, (k + 0.5) * step);
                    a++;
                } else {
                    i = n;
                    j = i;
                    k = i;
                }
            }
    return coord;
}

std::vector<Vector> initVel() {
    std::vector<Vector> vel;
    vel.reserve(N);
    double K = 0;
    // randomization
    std::random_device rd;
    //std::mt19937 gen(0);
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist_vel(-0.5, 0.5);
    // random uniform velocity distribution
    for (unsigned int i = 0; i < N; i++) {
        vel.emplace_back(dist_vel(gen), dist_vel(gen), dist_vel(gen));
        K += vel[i] * vel[i];
    }
    K *= 0.5;
    double T_current = K / (N - 1) * 2 / 3;
    // setting desired temperature
    double factor = std::sqrt(T_thermostat / T_current);
    for (unsigned int i = 0; i < N; i++) {
        vel[i] = vel[i] * factor;
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
std::pair<std::vector<Vector>, double> totalInteraction(const std::vector<Vector> &r, std::vector<double> &dist) {
    double k = 0;
    // total force initialization
    std::vector<Vector> F(N, Vector(0, 0, 0));
    // total potential initialization
    double U = 0;
    // calculation of total force and total potential energy
    for (unsigned int i = 0; i < N - 1; i++)
        for (unsigned int j = i + 1; j < N; j++) {
            Vector dr = periodicDist(r[i] - r[j]);
            const double r2 = dr * dr, r1 = std::sqrt(r2);
            dist.emplace_back(r1);
            k++;
            if (r1 <= rc) {
                const double r6 = r2 * r2 * r2, r8 = r6 * r2;
                Vector Force = dr * (24 / r8 * (2 / r6 - 1) - Fc / r1);
                F[i] = F[i] + Force;
                F[j] = F[j] - Force;
                U += 4 / r6 * (1 / r6 - 1) - Uc + Fc * (r1 - rc);
                //U += 4 / r6 * (1 / r6 - 1);
            }
        }
    return std::make_pair(F, U);
}

// collecting and calculating data for velocity auto-correlation function
std::pair<std::vector<double>, std::vector<double>>
VelocityAutoCorrelationFunction(const std::vector<std::vector<Vector>> &V) {
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
    std::vector<Vector> r = initCoord(), v = initVel(), v_mid = v, r0 = r, r_true = r0;
    std::vector<std::vector<Vector>> v_all;
    v_all.reserve(numOfSteps + 1);
//  opening a file to write data, analyzed further in python
    std::ofstream out_param_file(DIR_PATH + "parameters.csv");
    out_param_file << N << "," << dt << "," << numOfSteps << "," << sigma << "," << epsilon << "," << m << std::endl;
    out_param_file.close();
//  data containers for further usage
    std::vector<double> dist_all;
    std::pair<std::vector<Vector>, double> Total_old = totalInteraction(r, dist_all);
    std::vector<Vector> F_old = Total_old.first;
//  saving initial values
    std::ofstream outfile(DIR_PATH + "data.csv");
    outfile << rho << "," << T_thermostat << "," << L << std::endl;
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
//      first half-step of integration
        for (unsigned int i = 0; i < N; i++) {
            v_mid[i] = v[i] + F_old[i] * (0.5 * dt);
            r[i] = periodicPosition(r[i] + v_mid[i] * dt);
            r_true[i] = r_true[i] + v_mid[i] * dt;
            MSD += (r_true[i] - r0[i]) * (r_true[i] - r0[i]);
        }
//      new force calculation
        std::pair<std::vector<Vector>, double> Total = totalInteraction(r, dist_all);
        std::vector<Vector> F = Total.first;
//      second half-step of integration
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
        double T = 2 * K / (N - 1) / 3;
        double factor = std::sqrt(T_thermostat / T);
        // std::cout << T << std::endl;
        if (enable_NVT) {
            for (unsigned int i = 0; i < N; i++) {
                v[i] = v[i] * factor;
            }
        }
        double U = Total.second;
        // write in file
        outfile << std::sqrt(P * P) << "," << K << "," << U << "," << K + U << "," << MSD / N << std::endl;
    }
//  VACF calculation
    std::pair<std::vector<double>, std::vector<double>> vacf_full = VelocityAutoCorrelationFunction(v_all);
    std::vector<double> vacf = vacf_full.first, norm = vacf_full.second;

//  write in file data for VACF
    for (unsigned int i = 0; i < numOfSteps / 2; i++) {
        outfile << vacf[i] << "," << norm[i] << "," << std::endl;
    }
//  writing data for pair correlation function
    for (unsigned int i = 0; i < N * (N - 1) * (numOfSteps + 1) / 2; i++) {
        outfile << dist_all[i] << ",";
    }
    outfile.close();
    return 0;
}
