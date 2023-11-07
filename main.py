import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = '10'
with open("data.txt", 'r') as file:
    data = np.array(list(map(float, file.readline().split(',')[:-1])))
    P = np.array(list(map(float, file.readline().split(',')[:-1])))
    K = np.array(list(map(float, file.readline().split(',')[:-1])))
    U = np.array(list(map(float, file.readline().split(',')[:-1])))
    v = np.array(list(map(float, file.readline().split(',')[:-1])))
N = int(data[0])
dt = float(data[1])
numOfSteps = int(data[2]) + 1
m = float(data[3])
sigma = float(data[4])
epsilon = float(data[5])
L = float(data[6])
rho = float(data[7])
T_thermostat = float(data[8])
t_unit = sigma * (m / epsilon) ** 0.5

# cutting to equilibration
cut = int(numOfSteps / 20)
#cut = 0
# P = P[cut:] * sigma * m / t_unit
# K = K[cut:] * epsilon
# U = U[cut:] * epsilon
v = v[3 * cut:] * sigma / t_unit
E = K + U
t = np.arange(0, numOfSteps * dt, dt) * t_unit
# plotting
figure1, axis1 = plt.subplots(1, 3)

axis1[0].plot(t, P)
#axis1[0].axvline(x=numOfSteps * dt / 20, linestyle='-', color='red')
axis1[0].set_title("Momentum conservation test")
axis1[0].set_xlabel("Time")
axis1[0].set_ylabel("Total momentum")
axis1[0].grid()

E0 = [E[0] for i in range(len(E))]
axis1[1].plot(t, E)
axis1[1].axvline(x=numOfSteps * dt / 20, linestyle='-', color='red')
axis1[1].set_title("Energy conservation test")
axis1[1].set_xlabel("Time")
axis1[1].set_ylabel("Total energy")
axis1[1].grid()
# axis1[1].set_ylim([-1170, -1169])

counts, bins = np.histogram(v ** 2, bins=10)
counts = counts / sum(counts)
x = bins[1:]
# y = -np.log(counts) + 2 * np.log(bins[1:])
y = -np.log(counts)
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
axis1[2].grid()
plt.show()
