import csv
import matplotlib.pyplot as plt
import numpy as np

# getting data from file
with open("data.csv", 'r') as file:
    data = list(csv.reader(file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC))
# parameters of calculation
N = int(data[0][0])
dt = float(data[0][1])
numOfSteps = int(data[0][2]) + 1
sigma = float(data[0][3])
epsilon = float(data[0][4])
m = float(data[0][5])
L = float(data[0][6])
rho = float(data[0][7])
T_thermostat = float(data[0][8])
data.pop(0)

# distributing data from file
t_unit = sigma * (m / epsilon) ** 0.5
v = np.array([data[i][j] for j in range(N) for i in range(numOfSteps)]) * sigma / t_unit
p = np.array([data[i][N] for i in range(numOfSteps)]) * sigma * m / t_unit
K = np.array([data[i][N + 1] for i in range(numOfSteps)]) * epsilon
U = np.array([data[i][N + 2] for i in range(numOfSteps)]) * epsilon
E = np.array([data[i][N + 3] for i in range(numOfSteps)]) * epsilon
MSD = np.array([data[i][N + 4] for i in range(numOfSteps)]) * sigma ** 2
VACF = np.array([data[i][0] for i in range(numOfSteps, int(numOfSteps / 2 + numOfSteps))]) * (sigma / t_unit) ** 2
norm = np.array([data[i][1] for i in range(numOfSteps, int(numOfSteps / 2 + numOfSteps))]) * (sigma / t_unit) ** 2
dist = np.array([data[int(numOfSteps / 2 + numOfSteps)][i] for i in range(int(N * (N - 1) / 2))])

t = np.arange(0, numOfSteps * dt, dt) * t_unit
dt *= t_unit

# calculating values

# mean free path out of MSD
# cut ballistic part
MSD_cut = MSD[int(len(MSD) / 2):]
#MFP = (MSD[int(len(MSD) / 2)]) ** 0.5
#print("Mean free path from mean squared displacement", MFP)
# diffusivity out of MSD using least squares method
t_MSD = t[int(len(t) / 2):]
t_av = sum(t_MSD) / len(t_MSD)
MSD_av = sum(MSD_cut) / len(MSD_cut)
D1 = sum((t_MSD - t_av) * (MSD_cut - MSD_av)) / sum((t_MSD - t_av) ** 2) / 6
print("Diffusion coefficient out of mean squared displacement:", D1)

# diffusivity out of VACF using trapezoidal rule
D2 = sum(VACF) * dt / 3 * 2 / N / numOfSteps
print("Diffusion coefficient out of velocity auto-correlation function:", D2)

# error in diffusivity
erD = abs(D2 - D1) * 100 / min(D1, D2)
print("Difference between two methods:", erD, "%")

# mean free path from boltzmann equation with BGK approximation
MFP = 3 / 2 * numOfSteps * N * D2 / sum(v)
print("Mean free path:", MFP)

# plotting
figure1, axis1 = plt.subplots(1, 3)

axis1[0].plot(t, p)
axis1[0].set_title("Momentum conservation test")
axis1[0].set_xlabel("Time")
axis1[0].set_ylabel("Total momentum")

E0 = [E[0] for i in range(numOfSteps)]
axis1[1].plot(t, E)
axis1[1].set_title("Energy conservation test")
axis1[1].set_xlabel("Time")
axis1[1].set_ylabel("Total energy")
# axis1[1].set_ylim([-1170, -1169])

counts, bins = np.histogram(v)
counts = counts / sum(counts) / bins[1:] ** 2
axis1[2].plot(bins[1:], (-np.log(counts)) ** 0.5)
axis1[2].set_title("Maxwell distribution test")
axis1[2].set_xlabel("Velocity")
axis1[2].set_ylabel("Linearized Distribution")

figure2, axis2 = plt.subplots(1, 2)
axis2[0].plot(t, MSD)
axis2[0].set_title("Mean squared displacement")
axis2[0].set_xlabel("Time")
axis2[0].set_ylabel("MSD")

zeros = [0 for i in range(int(numOfSteps / 2))]
axis2[1].plot(t[0: int(numOfSteps / 2)], VACF / norm, t[0: int(numOfSteps / 2)], zeros)
axis2[1].set_title("Normalized velocity auto-correlation function")
axis2[1].set_xlabel("Time")
axis2[1].set_ylabel("VACF")


figure3, axis3 = plt.subplots(1, 1)
dr = 0.001
x = np.arange(0, L / 2 / sigma, dr)
counts, bins = np.histogram(dist, x)
norm_counts = counts * L ** 3 / N ** 2 / 4 / np.pi / x[1:] ** 2 / dr * 2
ones = [1 for i in range(len(x))]
axis3.plot(x[1:], norm_counts, x, ones)
axis3.set_title("Radial distribution function")
axis3.set_xlabel("r / sigma")
axis3.set_ylabel("g(r / sigma)")
plt.show()
