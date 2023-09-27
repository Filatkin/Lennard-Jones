import csv
import matplotlib.pyplot as plt
import numpy as np

# getting data from file
file = open("data.csv", "r")
data = list(csv.reader(file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC))
file.close()
# parameters of calculation
N = int(data[0][0])
dt = float(data[0][1])
numOfSteps = int(data[0][2])
L = float(data[0][3])
sigma = float(data[0][4])
sigma_unit = float(data[0][5])
m_unit = float(data[0][6])
E_unit = float(data[0][7])
t_unit = sigma_unit * (m_unit / E_unit) ** 0.5
dt = dt * t_unit
data.pop(0)

# distributing data from file
v = [float(data[i][j]) * sigma_unit / t_unit for j in range(N) for i in range(numOfSteps)]
p = [float(data[i][N]) * sigma_unit / t_unit * m_unit for i in range(numOfSteps)]
T = [float(data[i][N + 1]) * E_unit for i in range(numOfSteps)]
U = [float(data[i][N + 2]) * E_unit for i in range(numOfSteps)]
E = [float(data[i][N + 3]) * E_unit for i in range(numOfSteps)]
MSD = [float(data[i][N + 4]) * sigma_unit ** 2 for i in range(numOfSteps)]
VACF = [float(data[i][0]) * (sigma_unit / t_unit) ** 2 for i in range(numOfSteps, int(numOfSteps / 2 + numOfSteps))]
norm = [float(data[i][1]) * (sigma_unit / t_unit) ** 2 for i in range(numOfSteps, int(numOfSteps / 2 + numOfSteps))]
dist = [float(data[int(numOfSteps / 2 + numOfSteps)][i]) for i in range(int(N * (N - 1) / 2))]

# calculating values
t = [dt * i for i in range(numOfSteps)]

# diffusivity out of MSD using least squares method
t_av = sum(t) / numOfSteps
MSD_av = sum(MSD) / numOfSteps
D1 = sum([(t[i] - t_av) * (MSD[i] - MSD_av) for i in range(numOfSteps)]) / sum(
    [(t[i] - t_av) ** 2 for i in range(numOfSteps)]) / 6
print("Diffusion coefficient out of mean squared displacement:", D1, "length units * length units / time units")

# diffusivity out of VACF using trapezoidal rule
D2 = sum(VACF) * dt / 3 * 2 / N / numOfSteps
print("Diffusion coefficient out of velocity auto-correlation function:", D2, "length units * length units / time units")

# error in diffusivity
erD = abs(D2 - D1) * 100 / min(D1, D2)
print("Difference between two methods:", erD, "%")

# mean free path from boltzmann equation with BGK approximation
MFP = 3 / 2 * numOfSteps * N * D2 / sum(v)
print("Mean free path:", MFP, "length units")

# plotting
figure1, axis1 = plt.subplots(1, 3)

axis1[0].plot(t, p)
axis1[0].set_title("Momentum conservation test")
axis1[0].set_xlabel("t, time units")
axis1[0].set_ylabel("Total momentum, mass units * length units / time units")

axis1[1].plot(t, T, t, U, t, E)
axis1[1].set_title("Energy conservation test")
axis1[1].set_xlabel("t, time units")
axis1[1].set_ylabel("E, energy units")
axis1[1].legend(["Kinetic energy", "Potential energy", "Total energy"])

axis1[2].hist(v, bins=20, density=True)
axis1[2].set_title("Maxwell distribution test")
axis1[2].set_xlabel("Velocity, length units / time units")
axis1[2].set_ylabel("Density")

figure2, axis2 = plt.subplots(1, 3)
axis2[0].plot(t, MSD)
axis2[0].set_title("Mean squared displacement")
axis2[0].set_xlabel("t, time units")
axis2[0].set_ylabel("MSD, length units * length units")

zeros = [0 for i in range(int(numOfSteps / 2))]
axis2[1].plot(t[0: int(numOfSteps / 2)], [VACF[i] / norm[i] for i in range(int(numOfSteps / 2))],
              t[0: int(numOfSteps / 2)], zeros)
axis2[1].set_title("Normalized velocity auto-correlation function")
axis2[1].set_xlabel("t, time units")
axis2[1].set_ylabel("VACF")

dr = 0.01
x = np.arange(0, L / 2 / sigma, dr)
counts, bins = np.histogram(dist, x)
norm_counts = counts * (L / sigma) ** 3 / N ** 2 / 4 / np.pi / x[1:] ** 2 / dr * 2
ones = [1 for i in range(len(x))]
axis2[2].plot(x[1:], norm_counts, x, ones)
axis2[2].set_title("Radial distribution function")
axis2[2].set_xlabel("r / sigma")
axis2[2].set_ylabel("g(r / sigma)")
plt.show()
