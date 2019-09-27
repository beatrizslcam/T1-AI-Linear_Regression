import numpy as np
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas


def update(theta, alpha, X, E):
    temp = alpha * (np.dot(X.T, E))
    theta = theta - temp
    return theta


def loss(m, E):
    temp = (np.dot(E.T, E))
    J = np.divide(temp, (2*m))
    return J


def straight_line(X, theta):
    H = np.dot(X, theta)
    return H


# reading the archieves
X = np.loadtxt('input/inputs_x.txt', delimiter=',')
xorigin = X[:]

Y = np.loadtxt('input/outputs_y.txt')
yorigin = Y[:]

# resizing the X to (50,1)
X = np.expand_dims(X, axis=1)
Y = np.expand_dims(Y, axis=1)

# adding a column with 1
X = np.insert(X, 0, 1, axis=1)


m = X.shape[0]

theta = np.array([0, 0]).reshape(2, 1)

alpha = 0.001
it = 1000

e = np.zeros(it)
for i in range(it):
    H = straight_line(X, theta)
    E = H - Y
    J = loss(m, E)
    e[i] = J
    theta = update(theta, alpha, X, E)

plt.figure()
plt.title("Loss X Updates")
plt.xlabel('J')
plt.ylabel('iteracoes')
plt.plot(range(it), e)

plt.figure()

plt.scatter(xorigin, yorigin, marker='*', c='yellow')
plt.plot(xorigin, H)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
