#include <iostream>
#include <fstream>
#include <random>
#include <array>
#include "Vector.h"

// getting data file path on the particular device
const std::string FILE_PATH = __FILE__;
const int amountOfSymbolsBeforeRootDirectory = 8;
const std::string DIR_PATH = FILE_PATH.substr(0, FILE_PATH.size() - amountOfSymbolsBeforeRootDirectory);

//number of particles, integration steps, time between writing data for radial distribution function, size of the window and interval for sliding window algorithm, cut parameter to save rdf only after equilibration
const unsigned int N = 216, numOfSteps = 10000, stepsBetweenWriting = numOfSteps / 100, windowSize =
        numOfSteps / 5, interval = windowSize / 2, cutSteps = numOfSteps / 100;
// maximum number of particles along each coordinate
const unsigned int n = std::ceil(std::cbrt(N));
// integration step
const double dt = 0.005;
// mass of particles, lennard-jones parameters, reduced density (rho_reduced = rho_real / sigma^3), reduced temperature (tem_reduced = tem_real / epsilon)
const double m = 1, sigma = 1, epsilon = 1, rho = 0.8, T_thermostat = 1.25;
// usage of NVT-ensemble
const bool enable_NVT = false;
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
std::pair<std::vector<Vector>, double>
totalInteraction(const std::vector<Vector> &r, std::vector<double> &dist, const bool &writeDist) {
    // total force initialization
    std::vector<Vector> F(N, Vector(0, 0, 0));
    // total potential initialization
    double U = 0;
    // calculation of total force and total potential energy
    for (unsigned int i = 0; i < N - 1; i++)
        for (unsigned int j = i + 1; j < N; j++) {
            Vector dr = periodicDist(r[i] - r[j]);
            const double r2 = dr * dr, r1 = std::sqrt(r2);
            if (writeDist)
                dist.emplace_back(r1);
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

// mean squared displacement calculation using sliding window algorithm
std::vector<double> MeanSquaredDisplacement(const std::vector<std::vector<Vector>> &r) {
    std::vector<double> MSD(windowSize, 0);
    // fix t_delay = i * dt
    for (unsigned int i = 0; i < windowSize; i++)
        // fix t0 = j * dt
        for (unsigned int j = 0; j < numOfSteps - interval; j += interval)
            // sum over particles
            for (unsigned int k = 0; k < N; k++) {
                // vacf = <v(t0 + t_delay) * v(t0)>_t0,N
                MSD[i] += (r[j][k] - r[j + i][k]) * (r[j][k] - r[j + i][k]);
            }
    return MSD;
}

// velocity auto-correlation calculation using sliding window algorithm
std::pair<std::vector<double>, std::vector<double>>
VelocityAutoCorrelationFunction(const std::vector<std::vector<Vector>> &V) {
    std::vector<double> vacf(windowSize, 0), norm = vacf;
    // fix t_delay = i * dt
    for (unsigned int i = 0; i < windowSize; i++)
        // fix t0 = j * dt
        for (unsigned int j = 0; j < numOfSteps - interval; j += interval)
            // sum over particles
            for (unsigned int k = 0; k < N; k++) {
                // vacf = <v(t0 + t_delay) * v(t0)>_t0,N
                vacf[i] += V[j][k] * V[j + i][k];
                norm[i] += V[j][k] * V[j][k];
            }
    return std::make_pair(vacf, norm);
}

int main() {
//  initialization;
    std::vector<Vector> r = initCoord(), v = initVel(), vMid = v, rTrue = r;

//  data containers for further usage
    std::vector<double> dist_all;
    dist_all.reserve((int(numOfSteps / stepsBetweenWriting) + 1) * N * (N - 1) / 2);
    std::vector<std::vector<Vector>> v_all, r_all;
    v_all.reserve(numOfSteps + 1);
    r_all.reserve(numOfSteps + 1);
    Vector P(0, 0, 0);
    double Kin = 0;
    std::array<double, numOfSteps + 1> allMomentum = {}, allKineticEnergy = {}, allPotentialEnergy = {};

//  saving initial values
    std::pair<std::vector<Vector>, double> TotalForceAndPotential = totalInteraction(r, dist_all, false);

    for (unsigned int i = 0; i < N; i++) {
        P = P + v[i];
        Kin += v[i] * v[i];
    }
    Kin *= 0.5;
    v_all.emplace_back(v);
    r_all.emplace_back(rTrue);
    allMomentum[0] = std::sqrt(P * P);
    allKineticEnergy[0] = Kin;
    allPotentialEnergy[0] = TotalForceAndPotential.second;

//  velocity verlet (leap-frog) integration of motion equations
    std::vector<Vector> F_old = TotalForceAndPotential.first;
    bool writeDist;
    for (unsigned int k = 1; k <= numOfSteps; k++) {
        // kinetic energy initialization
        Kin = 0;
        // total momentum introduction
        P = Vector(0, 0, 0);
        // first half-step of integration
        for (unsigned int i = 0; i < N; i++) {
            vMid[i] = v[i] + F_old[i] * (0.5 * dt);
            r[i] = periodicPosition(r[i] + vMid[i] * dt);
            rTrue[i] = rTrue[i] + vMid[i] * dt;
        }
//      new force calculation
        if (k % stepsBetweenWriting == 0 && k > cutSteps)
            writeDist = true;
        else
            writeDist = false;
        TotalForceAndPotential = totalInteraction(r, dist_all, writeDist);
        std::vector<Vector> F = TotalForceAndPotential.first;
//      second half-step of integration
        for (unsigned int i = 0; i < N; i++) {
            v[i] = vMid[i] + F[i] * (0.5 * dt);
//          momentum calculation
            P = P + v[i];
//          kinetic energy calculation
            Kin += v[i] * v[i];
        }
        Kin *= 0.5;
        F_old = F;

        // velocity rescaling for NVE ensemble
        if (enable_NVT) {
            double T = 2 * Kin / (N - 1) / 3;
            double factor = std::sqrt(T_thermostat / T);
            for (unsigned int i = 0; i < N; i++) {
                v[i] = v[i] * factor;
            }
        }

        // saving
        v_all.emplace_back(v);
        r_all.emplace_back(rTrue);
        allMomentum[k] = std::sqrt(P * P);
        allKineticEnergy[k] = Kin;
        allPotentialEnergy[k] = TotalForceAndPotential.second;
    }
//  MSD calculation
    std::vector<double> MSD = MeanSquaredDisplacement(r_all);
//  VACF calculation
    std::pair<std::vector<double>, std::vector<double>> vacf_full = VelocityAutoCorrelationFunction(v_all);
    std::vector<double> vacf = vacf_full.first, norm = vacf_full.second;

//  writing in file
    std::ofstream outfile(DIR_PATH + "data.txt");
    outfile << N << "," << dt << "," << numOfSteps << "," << m << "," << sigma << "," << epsilon << "," << L << ","
            << rho << "," << T_thermostat << "," << interval << "," << stepsBetweenWriting << ","  << cutSteps << "," << std::endl;
    for (double &el: allMomentum)
        outfile << el << ",";
    outfile << std::endl;
    for (double &el: allKineticEnergy)
        outfile << el << ",";
    outfile << std::endl;
    for (double &el: allPotentialEnergy)
        outfile << el << ",";
    outfile << std::endl;
    for (std::vector<Vector> &vel: v_all)
        for (Vector &el: vel)
            outfile << el.x << "," << el.y << "," << el.z << ",";
    outfile << std::endl;
    for (double &el: MSD)
        outfile << el << ",";
    outfile << std::endl;
    for (double &el: vacf)
        outfile << el << ",";
    outfile << std::endl;
    for (double &el: norm)
        outfile << el << ",";
    outfile << std::endl;
    for (double &dist: dist_all) {
        outfile << dist << ",";
    }
    outfile.close();
    return 0;
}
