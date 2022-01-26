# This is my second attempt at programming the attitude with quaternions

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint

# satellite parameters
m = 1500    # kg
jx = 1440    # kg*m^2
jy = 2500
jz = 3850

r_c = (6378.135 + 650) * 1e3    # m
n = np.sqrt(3.986004e14/r_c**3)

q1 = np.array([-.5, -.5, .5, .5])
q2 = np.array([0, 0, 0, 1])

def qtoA(q1, q2, q3, q4):
    # define Attitude matrix components
    A11 = q1**2 - q2**2 - q3**2 + q4**2
    A21 = 2*(q2*q1 - q3*q4)
    A31 = 2*(q3*q1 + q2*q4)

    A12 = 2*(q1*q2 + q3*q4)
    A22 = -q1**2 + q2**2 - q3**2 + q4**2
    A32 = 2*(q3*q2 - q1*q4)

    A13 = 2*(q1*q3 -q2*q4)
    A23 = 2*(q2*q3 + q1*q4)
    A33 = -q1**2 - q2**2 + q3**2 + q4**2

    A_qtoA = np.array([[A11, A12, A13],
                       [A21, A22, A23],
                       [A31, A32, A33]])
    return A_qtoA


mu = 3.986004e5


def num_int(y, t):
    q1, q2, q3, q4, w1, w2, w3 = y

    w = np.array([[w1],
                   [w2],
                   [w3]])    # angular velocity vector  b wrt o
    h_w = np.array([[0],
                    [1000],
                    [0]])

    # compute the attitude matrix
    A11 = q1 ** 2 - q2 ** 2 - q3 ** 2 + q4 ** 2
    A21 = 2 * (q2 * q1 - q3 * q4)
    A31 = 2 * (q3 * q1 + q2 * q4)

    A12 = 2 * (q1 * q2 + q3 * q4)
    A22 = -q1 ** 2 + q2 ** 2 - q3 ** 2 + q4 ** 2
    A32 = 2 * (q3 * q2 - q1 * q4)

    A13 = 2 * (q1 * q3 - q2 * q4)
    A23 = 2 * (q2 * q3 + q1 * q4)
    A33 = -q1 ** 2 - q2 ** 2 + q3 ** 2 + q4 ** 2

    A_bo = np.array([[A11, A12, A13],     # this is the attitude matrix
                       [A21, A22, A23],
                       [A31, A32, A33]])


    j = np.array([[jx, 0, 0],    # MOI matrix
                 [0, jy, 0],
                 [0, 0, jz]])

    w_skew_sym = np.array([[0, -w3, w2],    # skew symetric matrix
                           [w3, 0, -w1],
                           [-w2, w1, 0]])

    A_bo_dot = np.matmul(A_bo, w_skew_sym)    # attitude matrix evolution equation

    # compute radius in body frame
    r_co = np.array([[0],
                    [0],
                    [-r_c]])    # radius in the LVLH frame
    r_cb = np.matmul(A_bo, r_co)    # radius in Body frame

    # compute W_bI
    w_OI = np.array([[0],
                      [n],
                      [0]])
    w_bI = w + np.matmul(A_bo, w_OI)

    wdot = np.matmul(np.linalg.inv(j), (3*mu/r_c**3 * np.cross(r_cb, np.matmul(j, r_cb), axis=0)) - np.cross(w_bI, (np.matmul(j, w_bI) + h_w), axis=0) - np.matmul(A_bo_dot, w_OI))
    # compute qdot
    epsilon_q = np.array([[q4, -q3, q2],
                          [q3, q4, -q1],
                          [-q2, q1, q4],
                          [-q1, -q2, -q3]])
    qdot = .5 * np.matmul(epsilon_q, w)

    return [qdot[0][0], qdot[1][0], qdot[2][0], qdot[3][0], wdot[0][0], wdot[1][0], wdot[2][0]]



w_initial = np.array([[.01],
                      [.1],
                      [.01]])
y0 = [0, 0, 0, 1, .01, .1, .01]
t0 = np.linspace(0, 1000, 1000)
sol = odeint(num_int, y0, t0)
plt.plot(t0, sol[:, 4], label="w1")
plt.plot(t0, sol[:, 5], label="w2")
plt.plot(t0, sol[:, 6], label="w3")
plt.xlabel("Time (s)")
plt.ylabel("w (rad/s)")
plt.title("Numerical Integration second attempt: Angular rates")


plt.grid()
plt.legend()
plt.show()

plt.plot(t0, sol[:, 0], label="q1")
plt.plot(t0, sol[:, 1], label="q2")
plt.plot(t0, sol[:, 2], label="q3")
plt.plot(t0, sol[:, 3], label="q4")
plt.xlabel("Time (s)")
plt.ylabel("Quaternions")
plt.title("Numerical integration second attempt: Attitude")
plt.grid()
plt.legend()
plt.show()