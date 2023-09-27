#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include "Vector.h"

const std::string FILE_PATH = __FILE__;
const int amountOfSymbolsBeforeRootDirectory = 8;
const std::string DIR_PATH = FILE_PATH.substr(0, FILE_PATH.size() - amountOfSymbolsBeforeRootDirectory);

//number of particles, integration steps
const unsigned int N = 512, numOfSteps = 2000;
// maximum number of particles along each coordinate
const unsigned int n = int(std::ceil(std::cbrt(N)));
// integration step value
const double dt = 0.001;
// units
// (e.g. for Argon molecules let sigma = 3.4 angstrom, mass = 68e-27 kg, energy at 300K = kT = 4.14e-21 joule)
// const double sigma_unit = 3.4, m_unit = 68e-27, E_unit = 4.14e-21;
// standard
const double sigma_unit = 1, m_unit = 1, E_unit = 1;
// lennard-jones parameters, mass of particles
// for argon with such units epsilon = 0.4 kT = 0.4 energy unit
//const double sigma = 1, epsilon = 0.4, m = 1;
// standard
const double sigma = 1, epsilon = 1.5, m = 1;
// size of a box
const double L = n * sigma;

// initialization
std::vector<Vector> initCoord() {
    std::vector<Vector> coord;
    coord.reserve(N);
    unsigned int a = 0;
    for (unsigned int i = 0; i < n; i++)
        for (unsigned int j = 0; j < n; j++)
            for (unsigned int k = 0; k < n; k++) {
                if (a < N)
                    coord.emplace_back(i * sigma, j * sigma, k * sigma);
                a++;
            }
    return coord;
}

std::vector<Vector> initVel() {
    std::vector<Vector> vel;
    vel.reserve(N);
    std::random_device rd;
    //std::mt19937 gen(0);
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist_vel(-L / 10, L / 10);
    // random uniform velocity distribution
    for (unsigned int i = 0; i < N; i++) {
        vel.emplace_back(dist_vel(gen), dist_vel(gen), dist_vel(gen));
    }
    return vel;
}

Vector floorVector(const Vector &r) {
    return Vector(std::floor(r.x), std::floor(r.y), std::floor(r.z));
}

//periodic boundary conditions
Vector periodicDist(const Vector &dr) {
    return dr - L * floorVector(dr * (1. / L) + 0.5);
}

Vector periodicPosition(const Vector &r) {
    return r - L * floorVector(r * (1. / L));
}

// pair interaction energy and force calculation
Vector pairInteractionForce(const Vector &rad) {
    const double r2 = rad * rad;
    const double r6 = r2 * r2 * r2;
    const double r8 = r6 * r2;
    const double sigma3 = sigma * sigma * sigma;
    const double sigma6 = sigma3 * sigma3;
    return rad * (24 * epsilon / r8 * sigma6 * (2 * sigma6 / r6 - 1.));
}

double pairInteractionPotential(const Vector &rad) {
    const double r2 = rad * rad;
    const double r6 = r2 * r2 * r2;
    const double sigma3 = sigma * sigma * sigma;
    const double sigma6 = sigma3 * sigma3;
    return 4 * epsilon * sigma6 / r6 * (sigma6 / r6 - 1.);
}

// total interaction force and potential energy calculation
std::pair<std::vector<Vector>, double> totalInteraction(const std::vector<Vector> &r) {
    // total force introduction
    std::vector<Vector> F(N, Vector(0, 0, 0));
    // total potential introduction
    double U = 0;
    // calculation of force and potential energy
    for (unsigned int i = 0; i < N - 1; i++)
        for (unsigned int j = i + 1; j < N; j++) {
            Vector dr = periodicDist(r[i] - r[j]);
            Vector Force = pairInteractionForce(dr);
            F[i] = F[i] + Force;
            F[j] = F[j] - Force;
            U += pairInteractionPotential(dr);
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
            // sum over ensemble
            for (unsigned int k = 0; k < N; k++) {
                // vacf = <v(t0 + t_delay) * v(t0)>_t0,N
                vacf[i] += V[j][k] * V[i + j][k];
                norm[i] += V[j][k] * V[j][k];
            }
    return std::make_pair(vacf, norm);
}

// collecting distances for pair correlation function
std::vector<double> distanceCalculation(const std::vector<Vector> &r) {
    std::vector<double> dist;
    dist.reserve(N * (N - 1) / 2);
    for (unsigned int i = 0; i < N - 1; i++)
        for (unsigned int j = i + 1; j < N; j++) {
            Vector dr = periodicDist(r[i] - r[j]);
            dist.emplace_back(std::sqrt(dr * dr));
        }
    return dist;
}

int main() {
    // initialization;
    std::vector<Vector> r = initCoord(), v = initVel();
    // save initial coordinates and r_true to calculate mean squared displacement (MSD), v_mid to use in integration
    std::vector<Vector> r0 = r, r_true = r, v_mid(N, Vector(0, 0, 0));

    // opening a file to write data, analyzed further in python
    std::ofstream outfile(DIR_PATH + "data.csv");
    outfile << N << "," << dt << "," << numOfSteps << "," << L << "," << sigma << "," << sigma_unit << "," << m_unit
            << "," << E_unit << std::endl;

    // data containers for further usage
    std::vector<Vector> F_old = totalInteraction(r).first;
    std::vector<std::vector<Vector>> v_all;
    std::vector<std::vector<double>> dist_all;
    v_all.reserve(numOfSteps);
    dist_all.reserve(numOfSteps);
    v_all.emplace_back(v);
    dist_all.emplace_back(distanceCalculation(r));
    // velocity verlet integration of motion equations;
    for (unsigned int k = 0; k < numOfSteps; k++) {
        // kinetic energy, mean squared displacement introduction
        double T = 0, MSD = 0;
        // total momentum introduction
        Vector P(0, 0, 0);
        // one step of integration
        for (unsigned int i = 0; i < N; i++) {
            v_mid[i] = v[i] + F_old[i] * (0.5 * dt);
            r[i] = periodicPosition(r[i] + v_mid[i] * dt);
            r_true[i] = r_true[i] + v_mid[i] * dt;
            // MSD calculation
            MSD += (r_true[i] - r0[i]) * (r_true[i] - r0[i]);
        }
        // new force calculation
        std::pair<std::vector<Vector>, double> Total = totalInteraction(r);
        std::vector<Vector> F = Total.first;
        // velocity calculation
        for (unsigned int i = 0; i < N; i++) {
            v[i] = v[i] + (F_old[i] + F[i]) * (dt / m / 2);
            //v[i] = v_mid[i] + F[i] * (0.5 * dt);
            // write velocity in file
            double v2 = v[i] * v[i];
            outfile << std::sqrt(v2) << ",";
            // momentum calculation
            P = P + v[i];
            // kinetic energy calculation
            T += v2;
        }
        F_old = F;

        v_all.emplace_back(v);
        dist_all.emplace_back(distanceCalculation(r));

        T = 0.5 * m * T;
        double U = Total.second;
        // write in file
        outfile << m * std::sqrt(P * P) << "," << m * T << "," << U << "," << T + U << "," << MSD / N << std::endl;
    }
    // VACF calculation
    std::pair<std::vector<double>, std::vector<double>> vacf_full = VelocityAutoCorrelationFunction(v_all);
    std::vector<double> vacf = vacf_full.first, norm = vacf_full.second, dist(N * (N - 1), 0);

    // averaging data for pair correlation function
    for (unsigned int i = 0; i < N * (N - 1) / 2; i++)
        for (unsigned int j = 0; j < numOfSteps; j++)
            dist[i] += dist_all[j][i];
    // write in file data for VACF and PCF
    for (unsigned int i = 0; i < numOfSteps / 2; i++) {
        outfile << vacf[i] << "," << norm[i] << "," << std::endl;
    }
    for (unsigned int i = 0; i < N * (N - 1) / 2; i++) {
        outfile << dist[i] / numOfSteps << ",";
    }
    outfile.close();
}
