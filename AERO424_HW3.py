# THIS was my first attempt at getting the numerical simulation up and running
#theres a couple problems with this. One is I used Euler angles and the small angle approximation
# that throws off the stability of the numerical integration and the small angles
# are not valid after a certain period diverging from small angles



import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint

# satellite parameters
m = 1500    # kg
jx = 1440    # kg*m^2
jy = 2500
jz = 3850

r_c = (6378.135 + 650)    # km
h_w = 1000   # kgm^2/s
q1 = -.5
q2 = -.5
q3 = .5
q4 = .5

n = np.sqrt(3.986004e5/r_c**3)


def num_integrate(y, t, Jx, Jy, Jz, n, h):
    omega1, omega2, omega3, phi, theta, psi = y

    dydt = [1/jx * (n*(4*n*(Jz-Jy) + h)*phi + (n*(Jx-Jy+Jz) + h)*omega3),
            1/jy * (3*n**2*(Jz-Jx)*theta),
            1/jz * (n*(n*(Jx-Jy) + h)*psi - (n*(Jx-Jy+Jz) + h)*omega1),
            omega1,
            omega2,
            omega3]
    return dydt


# phi = np.arctan2(2*(q4*q1 + q2*q3), 1 -2*(q1**2 + q2**2))
# theta = np.arcsin(2*(q4*q2 - q3*q1))
# psi = np.arctan2(2*(q4*q3 + q1*q2), 1-2*(q2**2 + q3**2))
# print(180/np.pi*phi)
# print(180/np.pi*theta)
# print(180/np.pi*psi)

y0 = [.01, .1, .01, -np.pi/2, 0, np.pi/2]
period = 2*np.pi / np.sqrt(2.896004e5) * r_c**1.5
#t = np.linspace(0, int(3*period), int(3*period))
t = np.linspace(0, 1000, 1000)

sol = odeint(num_integrate, y0, t, args=(jx, jy, jz, n, h_w))

phi = sol[:, 3]
theta = sol[:, 4]
psi = sol[:, 5]

for i in range(len(phi)):
    # bound upper range
    while phi[i] > np.pi:
        phi[i] = phi[i] - 2*np.pi
    while theta[i] > np.pi:
        theta[i] = theta[i] - 2*np.pi
    while psi[i] > np.pi:
        psi[i] = psi[i] + 2*np.pi
    # Bound lower range
    while phi[i] < -np.pi:
        phi[i] = phi[i] + 2 * np.pi
    while theta[i] < -np.pi:
        theta[i] = theta[i] + 2*np.pi
    while psi[i] < -np.pi:
        psi[i] = psi[i] + 2 * np.pi

plt.plot(t, sol[:, 0], label="phidot")
plt.plot(t, sol[:, 1], label="thetadot")
plt.plot(t, sol[:, 2], label="psidot")
plt.title("Numerical integration First attempt: Angular rates")
plt.xlabel("Time (s)")
plt.ylabel(" w (rad/s)")
plt.legend()
plt.grid()
plt.show()





plt.plot(t, sol[:, 3], label="phi")
plt.plot(t, sol[:, 4], label="theta")
plt.plot(t, sol[:, 5], label="psi")
plt.title("Numerical Integration: Attitude Angles")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.legend()
plt.grid()
plt.show()