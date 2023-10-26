import csv
import matplotlib.pyplot as plt
import numpy as np

# getting data from file
with open("parameters.csv", 'r') as file:
    data = list(csv.reader(file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC))
N = int(data[0][0])
dt = float(data[0][1])
numOfSteps = int(data[0][2]) + 1
sigma = float(data[0][3])
epsilon = float(data[0][4])
m = float(data[0][5])
t_unit = sigma * (m / epsilon) ** 0.5


def diffusion_einstein(time, mean_squared_displacement):
    # cut ballistic part
    msd_cut = mean_squared_displacement[int(len(mean_squared_displacement) / 2):]
    # diffusivity out of msd using least squares method
    t_msd = time[int(len(t) / 2):]
    t_av = sum(t_msd) / len(t_msd)
    msd_av = sum(msd_cut) / len(msd_cut)
    return sum((t_msd - t_av) * (msd_cut - msd_av)) / sum((t_msd - t_av) ** 2) / 6


def diffusion_vacf(velocity_autocorrelation_function):
    return sum(velocity_autocorrelation_function) * dt / 3 * 2 / N / numOfSteps


def mean_free_path_kinetic_theory(velocities, diffusion):
    return 3 * numOfSteps * N * diffusion / sum(velocities)


t = np.arange(0, numOfSteps * dt, dt) * t_unit
dt *= t_unit

plt.rcParams['font.size'] = '20'

fig1 = plt.figure(1, figsize=(16, 8))
axis1 = fig1.subplots()
axis1.set_title("Mean squared displacement")
axis1.set_xlabel("Time")
axis1.set_ylabel("msd")

zeros = [0 for i in range(int(numOfSteps / 2))]
fig2 = plt.figure(2, figsize=(16, 8))
axis2 = fig2.subplots()
axis2.set_title("Normalized velocity auto-correlation function")
axis2.set_xlabel("Time")
axis2.set_ylabel("vacf")

fig3 = plt.figure(3, figsize=(16, 8))
axis3 = fig3.subplots()
dr = 0.01
x = np.arange(0, N ** (1 / 3) / 2, dr)
ones = np.ones(len(x))
axis3.set_title("Radial distribution function")
axis3.set_xlabel("r / sigma")
axis3.set_ylabel("g(r / sigma)")

d_msd = np.array([])
d_vacf = np.array([])
mfp_kin = np.array([])
er_d = np.array([])
mfp = np.array([])
er_mfp = np.array([])
file_names = ["data_gas2.csv", "data_liquid2.csv", "data_solid2.csv"]
for file in file_names:
    with open(file, 'r') as data_file:
        data = list(csv.reader(data_file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC))
    rho = float(data[0][0])
    T_thermostat = float(data[0][1])
    L = float(data[0][2])
    data.pop(0)
    v = np.array([data[i][j] for j in range(N) for i in range(numOfSteps)]) * sigma / t_unit
    msd = np.array([data[i][N + 4] for i in range(numOfSteps)]) * sigma ** 2
    vacf = np.array([data[i][0] for i in range(numOfSteps, int(numOfSteps / 2) + numOfSteps)]) * (
                sigma / t_unit) ** 2
    norm = np.array([data[i][1] for i in range(numOfSteps, int(numOfSteps / 2) + numOfSteps)]) * (
                sigma / t_unit) ** 2
    dist = np.array([data[int(numOfSteps / 2) + numOfSteps][i] for i in range(int(N * (N - 1) / 2) * numOfSteps)])

    # calculating
    # diffusivity out msd using einstein relation
    d1 = diffusion_einstein(t, msd)
    # diffusivity out of vacf using trapezoidal rule
    d2 = diffusion_vacf(vacf)
    d_msd = np.append(d_msd, d1)
    d_vacf = np.append(d_vacf, d2)
    # error in diffusivity
    er_d = np.append(er_d, abs(d2 - d1) * 100 / min(d1, d2))
    # mean free path from kinetic theory
    mfp1 = mean_free_path_kinetic_theory(v, d1)
    mfp2 = L ** 3 / N / 2 ** 0.5 / sigma ** 2 / np.pi
    mfp_kin = np.append(mfp_kin, mfp1)
    mfp = np.append(mfp, mfp2)
    er_mfp = np.append(er_mfp, abs(mfp1 - mfp2) * 100 / min(mfp2, mfp1))

    # plotting
    axis1.plot(t, msd)
    axis2.plot(t[0: int(numOfSteps / 2)], vacf / norm)
    counts, bins = np.histogram(dist, x)
    norm_counts = counts / rho / N / 4 / np.pi / x[1:] ** 2 / dr * 2 / numOfSteps
    axis3.plot(x[1:], norm_counts)
result = np.array([d_msd, d_vacf, er_d, mfp_kin, mfp, er_mfp])
print("Results correspond to gas, liquid, solid, respectively          ")
print("Diffusion coefficient out of mean squared displacement:         ", *[f"{a:.3f}" for a in d_msd])
print("Diffusion coefficient out of velocity auto-correlation function:", *[f"{a:.3f}" for a in d_vacf])
print("Difference between two methods (diffusion), %:                  ", *[f"{a:.3f}" for a in er_d])
print("Mean free path out of diffusion coefficient in kinetic theory:  ", *[f"{a:.3f}" for a in mfp_kin])
print("Mean free path out of gas kinetic theory:                       ", *[f"{a:.3f}" for a in mfp])
print("Difference between two methods (mean free path), %:             ", *[f"{a:.3f}" for a in er_mfp])
axis2.plot(t[0: int(numOfSteps / 2)], zeros)
axis3.plot(x, ones)
axis1.legend(["gas", "liquid", "solid"])
axis2.legend(["gas", "liquid", "solid", "reference"])
axis3.legend(["gas", "liquid", "solid", "reference"])
plt.show()
