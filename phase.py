import csv
import matplotlib.pyplot as plt
import numpy as np


def diffusion_einstein(time, mean_squared_displacement, cut):
    # cut ballistic part
    msd_cut = mean_squared_displacement[cut:]
    # diffusivity out of msd using least squares method
    t_msd = time[cut:]
    t_av = sum(t_msd) / len(t_msd)
    msd_av = sum(msd_cut) / len(msd_cut)
    return sum((t_msd - t_av) * (msd_cut - msd_av)) / sum((t_msd - t_av) ** 2) / 6


def diffusion_vacf(velocity_autocorrelation_function, number_of_particles, averSteps):
    return sum(velocity_autocorrelation_function) * dt / 3 / averSteps / number_of_particles


def mean_free_path_kinetic_theory(velocities, diffusion):
    return 3 * len(velocities) * diffusion / sum(velocities)


def mean_free_path_diffusion_regime(mean_squared_displacement, v_squared, time):
    tol = 1
    i = 0
    while abs(mean_squared_displacement[i] - v_squared * time[i] ** 2) < tol:
        i += 1
    return (mean_squared_displacement[i]) ** 0.5


plt.rcParams['font.size'] = '20'

fig1 = plt.figure(1, figsize=(16, 8))
axis1 = fig1.subplots()
axis1.set_title("Mean squared displacement")
axis1.set_xlabel("log(time)")
axis1.set_ylabel("log(msd)")

fig2 = plt.figure(2, figsize=(16, 8))
axis2 = fig2.subplots()
axis2.set_title("Normalized velocity auto-correlation function")
axis2.set_xlabel("Lag time")
axis2.set_ylabel("vacf")

fig3 = plt.figure(3, figsize=(16, 8))
axis3 = fig3.subplots()
axis3.set_title("Radial distribution function")
axis3.set_xlabel("r / sigma")
axis3.set_ylabel("g(r / sigma)")

d_msd = np.array([])
d_vacf = np.array([])
er_d = np.array([])
mfp_kin = np.array([])
mfp = np.array([])
mfp_ballistics = np.array([])
er_mfp = np.array([])
file_names = ["data_gas.txt", "data_liquid.txt", "data_solid.txt"]
for data_file in file_names:
    with open(data_file, 'r') as file:
        data = np.array(list(map(float, file.readline().split(',')[:-1])))
        for __ in range(3):
            next(file)
        v_axis = np.array(list(map(float, file.readline().split(',')[:-1])))
        msd = np.array(list(map(float, file.readline().split(',')[:-1])))
        vacf = np.array(list(map(float, file.readline().split(',')[:-1])))
        norm = np.array(list(map(float, file.readline().split(',')[:-1])))
        dist = np.array(list(map(float, file.readline().split(',')[:-1])))
    v = np.array(
        [((v_axis[i] * v_axis[i]) + (v_axis[i + 1] * v_axis[i + 1]) + (v_axis[i + 2] * v_axis[i + 2])) ** 0.5 for i in
         range(int(len(v_axis) / 3))])
    N = int(data[0])
    dt = float(data[1])
    numOfSteps = int(data[2]) + 1
    m = float(data[3])
    sigma = float(data[4])
    epsilon = float(data[5])
    L = float(data[6])
    rho = float(data[7])
    T_thermostat = float(data[8])
    interval = int(data[9])
    stepsBetweenWriting = int(data[10])
    cutSteps = int(data[11])

    v0_squared_av = (sum(v[:N]) / N)
    t_unit = sigma * (m / epsilon) ** 0.5
    t = np.arange(0, len(msd) * dt, dt) * t_unit

    # number of steps to average time in VACF and MSD
    averagingSteps = numOfSteps / interval - 1

    t *= t_unit
    msd *= sigma ** 2 / N / averagingSteps
    vacf *= (sigma / t_unit) ** 2
    norm *= (sigma / t_unit) ** 2
    v *= sigma / t_unit

    # calculating
    # diffusivity out msd using einstein relation
    d1 = diffusion_einstein(t, msd, cutSteps)
    # diffusivity out of vacf using trapezoidal rule
    d2 = diffusion_vacf(vacf, N, averagingSteps)
    d_msd = np.append(d_msd, d1)
    d_vacf = np.append(d_vacf, d2)
    # error in diffusivity
    er_d = np.append(er_d, abs(d2 - d1) * 100 / min(d1, d2))
    # mean free path from kinetic theory
    mfp1 = mean_free_path_kinetic_theory(v, d1)
    mfp2 = L ** 3 / N / 2 ** 0.5 / sigma ** 2 / np.pi
    mfp3 = mean_free_path_diffusion_regime(msd, 3 / 2 * T_thermostat, t)
    mfp_kin = np.append(mfp_kin, mfp1)
    mfp = np.append(mfp, mfp2)
    mfp_ballistics = np.append(mfp_ballistics, mfp3)
    er_mfp = np.append(er_mfp, abs(mfp1 - mfp2) * 100 / min(mfp2, mfp1))

    # plotting
    # axis1.plot(t, msd)
    axis1.plot(np.log(t[1:]), np.log(msd[1:]))
    axis2.plot(t[:len(vacf)], vacf / norm)
    dr = 0.01
    x = np.arange(0, N ** (1 / 3) / 2, dr)
    counts, bins = np.histogram(dist, x)
    norm_counts = counts / rho / N / 4 / np.pi / x[1:] ** 2 / dr * 2 / ((numOfSteps - cutSteps) / stepsBetweenWriting)
    axis3.plot(x[1:], norm_counts)
axis2.axhline(y=0, linestyle='-', color='red')
axis3.axhline(y=1, linestyle='-', color='red')
result = np.array([d_msd, d_vacf, er_d, mfp_kin, mfp, er_mfp])
print("Results correspond to gas, liquid, solid, respectively          ")
print("Diffusion coefficient out of mean squared displacement:         ", *[f"{a:.3f}" for a in d_msd])
print("Diffusion coefficient out of velocity auto-correlation function:", *[f"{a:.3f}" for a in d_vacf])
print("Difference between two methods (diffusion), %:                  ", *[f"{a:.3f}" for a in er_d])
print("Mean free path out of diffusion coefficient in kinetic theory:  ", *[f"{a:.3f}" for a in mfp_kin])
print("Mean free path out of gas kinetic theory:                       ", *[f"{a:.3f}" for a in mfp])
print("Mean free path out of ballistic regime:                         ", *[f"{a:.3f}" for a in mfp_ballistics])
print("Difference between two methods (mean free path), %:             ", *[f"{a:.3f}" for a in er_mfp])
axis1.legend(["gas", "liquid", "solid"])
axis2.legend(["gas", "liquid", "solid", "reference"])
axis3.legend(["gas", "liquid", "solid", "reference"])
plt.show()
