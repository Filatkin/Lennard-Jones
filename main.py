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
with open("data.csv", 'r') as file:
    data = list(csv.reader(file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC))
rho = float(data[0][0])
T_thermostat = float(data[0][1])
L = float(data[0][2])
data.pop(0)
# distributing data from file
v = np.array([data[i][j] for j in range(N) for i in range(numOfSteps)]) * sigma / t_unit
p = np.array([data[i][N] for i in range(numOfSteps)]) * sigma * m / t_unit
K = np.array([data[i][N + 1] for i in range(numOfSteps)]) * epsilon
U = np.array([data[i][N + 2] for i in range(numOfSteps)]) * epsilon
E = np.array([data[i][N + 3] for i in range(numOfSteps)]) * epsilon

t = np.arange(0, numOfSteps * dt, dt) * t_unit
dt *= t_unit

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

counts, bins = np.histogram(v, 30)
counts = counts / sum(counts)
x = bins[1:] ** 2
y = -np.log(counts) + 2 * np.log(bins[1:])
axis1[2].plot(x, y, '.')
x_av = sum(x) / len(x)
y_av = sum(y) / len(y)
a = sum((x - x_av) * (y - y_av)) / sum((x - x_av) ** 2)
b = y_av - a * x_av
approx = np.array(a * x + b)
axis1[2].plot(x, approx)
axis1[2].set_title("Maxwell distribution test")
axis1[2].set_xlabel("v^2")
axis1[2].set_ylabel("Linearized Distribution")
plt.show()
