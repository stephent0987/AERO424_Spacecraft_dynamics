import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint

########################
# PROBLEM 1
########################

# Satellite Parameters
m = 1000    # kg
J_t = 1000    # kg*m^2
J_s = 700     # kg*m^2

RPY0 = np.array([[0], [0], [0]])   # initial 3-1-3 Euler angles in Radians
w_b0 = np.array([[.1], [.1], [1]])    # initial angular velocity in body frame radians

# Analytical solution for time evolution of angular velocity

w3_t = w_b0[2][0]     # angular velocity about the spin axis is a constant


def w_1(t):     # angular velocity in the transverse axes as a function of time
    w_1_calculation = .1*np.cos(.3*t) + .1*np.sin(.3*t)

    return w_1_calculation


def w_2(t):     # angular velocity in the transverse axes as a function of time
    w_2_calculation = .1*np.cos(.3*t) - .1*np.sin(.3*t)

    return w_2_calculation


time = np.linspace(0, 100, 1000)    # time from 0 to 100 seconds
w3_t = np.full(len(time), w3_t)
w_1_vector = np.zeros(len(time))
w_2_vector = np.zeros(len(time))
for i in range(len(time)):
    w_1_vector[i] = w_1(time[i])
for i in range(len(time)):
    w_2_vector[i] = w_2(time[i])

plt.plot(time, w3_t, label="Spin axis angular velocity (w3)")
plt.plot(time, w_1_vector, label="w1")
plt.plot(time, w_2_vector, label="w2")
plt.grid()
plt.legend()
plt.title("Exact solution: Angular velocity vs. time")
plt.ylabel("w (rad/s)")
plt.xlabel("Time (s)")
plt.show()

# compute the attitude angles
norm_H = np.sqrt((J_t*w_b0[0][0])**2 + (J_t*w_b0[1][0])**2 + (J_s*w_b0[2][0])**2)
theta_const = np.arccos(J_s*w_b0[2][0] / norm_H)
theta_t = np.full(len(time), theta_const)

w_p = .3    # hand calculation in rad/s
psi_t = np.zeros(len(time))
n = 0
for i in range(len(time)):
    psi_t[i] = w_p * time[i] - n*2*np.pi


    if psi_t[i] > np.pi:    # bound the domain from (-180, 180) degrees
        n += 1
    elif psi_t[i] < -np.pi:
        n -= 1

w_l = norm_H/J_t    # inertial nutation rate
phi_t = np.zeros(len(time))
n = 0
for i in range(len(time)):
    phi_t[i] = w_l * time[i] - n*2*np.pi
    if phi_t[i] > np.pi:    # bound the domain from (-180, 180) degrees
        n += 1
    elif phi_t[i] < -np.pi:
        n -= 1

theta_t = 180/np.pi * theta_t
phi_t = 180/np.pi * phi_t
psi_t = 180/np.pi * psi_t

# plot attitude time evolution
plt.plot(time, theta_t, label="theta(t)")
plt.plot(time, psi_t, label="psi(t)")
plt.plot(time, phi_t, label="phi(t)")
plt.legend()
plt.ylabel("Attitude Angle (rad)")
plt.xlabel("Time (s)")
plt.title("Exact Time Evolution of 3-1-3 Attitude Angles")
plt.grid()
plt.show()



def example(y, t,w_p):
    omega1, omega2, omega3 = y

    dydt = [w_p*omega2, -w_p*omega1, 0]
    return dydt


# initial condition
y0 = [.1, .1, 1]

t = np.linspace(0, 100, 10000)
y = odeint(example, y0, t, args=(w_p,))

plt.plot(t, y[:, 0], "b", label="Omega1(t)")
plt.plot(t, y[:, 1], "g", label="Omega2(t)")
plt.plot(t, y[:, 2], "r", label="Omega3(t)")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (rad/s)")
plt.title("Numerical solution to the angular velocity evolution")
plt.legend()
plt.grid()
plt.show()

# So obviously looking at the exact solution and the mumerical solution, I can't tell the difference
# because Scipy's ode solver is so good.

###########################################
# Problem 2A    "SAVE AGGIECOMMS!!!"
###########################################

# The following functions are recycled from HW1

def calculate_JD(Y, M, D, h, m, s):    # calculate the julian date for any time
    JD = 1721013.5 + 367*Y - int(7/4 * (Y + int((M+9)/12))) + int(275*M/9) + D + (60*h + m + s/60)/1440
    return JD


def calculate_theta_GMST(T0, h, m, s):    # calculates angle between ECI and ECEF (UNITS ARE SECONDS)
    theta = 24110.54841 + 8640184.812866 * T0 + .093104 * T0 ** 2 - 6.2e-6*T0**3 + 1.002737909350795*(3600*h + 60*m + s)
    return theta


def DCM_EI(Y, M, D, h, m, s):    # Calculates the DCM from ECI to ECEF

    # calculate the rotation angle of the earth wrt the ECI frame
    T0 = (calculate_JD(Y, M, D, 0, 0, 0) - 2451545) / 36525
    theta_s = calculate_theta_GMST(T0, h, m, s)
    multiples = int(theta_s/86400)
    theta = np.pi/180 * (theta_s - multiples*86400) / 240    # in rads
    print(theta)
    # Assemble DCM
    A_EI = np.array([[np.cos(theta), np.sin(theta), 0],
                    [-np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    return A_EI


def Atoq(A):    # Computes a quaternion from the attitude matrix
    trace_A = np.trace(A)

    if A[0][0] > A[1][1] and A[0][0] > A[2][2] and A[0][0] > trace_A:
        q_version1 = np.array([[1 + 2*A[0][0] - np.trace(A)],
                               [A[0][1] + A[1][0]],
                                [A[0][2] + A[2][0]],
                               [A[1][2] - A[2][1]]])
        q_verion1 = q_version1 / np.linalg.norm(q_version1)
        return q_version1

    elif A[1][1] > A[0][0] and A[1][1] > A[2][2] and A[1][1] > trace_A:
        q_version2 = np.array([[A[1][0] + A[0][1]],
                               [1 + 2*A[1][1] -np.trace(A)],
                               [A[1][2] + A[2][1]],
                               [A[2][0] - A[0][2]]])
        q_version2 = q_version2 / np.linalg.norm(q_version2)
        return q_version2

    elif A[2][2] > A[0][0] and A[2][2] > A[1][1] and A[2][2] > trace_A:
        q_version3 = np.array([[A[2][0] + A[0][2]],
                               [A[2][1] + A[1][2]],
                               [1 + 2*A[2][2] - np.trace(A)],
                               [A[0][1] - A[1][0]]])
        q_version3 = q_version3 / np.linalg.norm(q_version3)
        return q_version3

    else:
        q_version4 = np.array([[A[1][2] - A[2][1]],
                               [A[2][0] - A[0][2]],
                               [A[0][1] - A[1][0]],
                               [1 + np.trace(A)]])
        q_version4 = q_version4 / np.linalg.norm(q_version4)
        return q_version4


def qtoA(q1, q2, q3, q4):    # computes an attitude matrix given a quaternion
    # define Attitude matrix components
    A11 = q1**2 - q2**2 - q3**2 + q4**2
    A21 = 2*(q2*q1 - q3*q4)
    A31 = 2*(q3*q1 + q2*q4)

    A12 = 2*(q1*q2 + q3*q4)
    A22 = -q1**2 + q2**2 - q3**2 + q4**2
    A32 = 2*(q3*q2 - q1*q4)

    A13 = 2*(q1*q3 - q2*q4)
    A23 = 2*(q2*q3 + q1*q4)
    A33 = -q1**2 - q2**2 + q3**2 + q4**2

    A_qtoA = np.array([[A11, A12, A13],
                       [A21, A22, A23],
                       [A31, A32, A33]])
    return A_qtoA

# End of recycled functions

A_EI = DCM_EI(2035, 10, 1, 13, 37, 15)    # DCM from ECI to ECEF

long = -np.pi/180*96
A_TE = np.array([[np.cos(long), np.sin(long), 0],    # DCM from ECEF to Topgraphic
                 [-np.sin(long), np.cos(long), 0],
                 [0, 0, 1]])

A_LT = np.array([[0, 1, 0],
                 [0, 0, -1],
                 [-1, 0, 0]])

A_LI = np.matmul(np.matmul(A_LT, A_TE), A_EI)
print("\nProblem 2A\n")
print("The attitude matrix is:")
print(A_LI)

q_initial = Atoq(A_LI)
print("\nThe quaternion is:")
print(q_initial)

#############################
# PROBLEM 2B
#############################
print("\nProblem 2B\n")


w = np.array([[.01],
              [.029927060278441],
              [0]])

# numerically solve the odes


def q_kinematics(y, t, w1, w2):
    q1, q2, q3, q4 = y
    dydt = [.5 * (w1*q4 - w2*q3),
            .5 * (w1*q3 + w2*q4),
            -.5 * (w1*q2 - w2*q1),
            -.5 * (w1*q1 + w2*q2)]

    return dydt


w1 = w[0][0]
w2 = w[1][0]

# initial condition
y0 = [q_initial[0][0],
      q_initial[1][0],
      q_initial[2][0],
      q_initial[3][0]]
t = np.linspace(0, 2500, 2500)

solution = odeint(q_kinematics, y0, t, args=(w1, w2))
q_norm = np.zeros(len(solution[:, 0]))
for i in range(len(solution[:, 0])):
    q_norm[i] = np.linalg.norm(solution[i, :])


contact_time = np.array([])
# If we want the distance between the two quaternions to be 1% the expected attitude 1-<q1,a2> <= .01
New_contact = True    # this just makes sure that we only get on contact point when the dot product condition is met
num_contacts = 0
for i in range(len(solution[:, 0])):
    dot = 1 - np.matmul(np.transpose(q_initial), solution[i, :])**2
    if dot <= .01 and New_contact == True:
        contact_time = np.append(contact_time, [t[i]], axis=0)
        New_contact = False
        num_contacts += 1
    if dot > .01:
        New_contact = True

zeros_contact_time = np.zeros(len(contact_time))
# graph solution
plt.plot(t, solution[:, 0], label="q1(t)")
plt.plot(t, solution[:, 1], label="q2(t)")
plt.plot(t, solution[:, 2], label="q3(t)")
plt.plot(t, solution[:, 3], label="q4(t)")
plt.plot(contact_time, zeros_contact_time, "bo", label="Contacts")
plt.title("Quaternion evolution after the suddenlink boop")
plt.xlabel("Time (s)")
plt.ylabel("Quaternion element values")
plt.legend()
plt.xlabel("time (s)")
plt.grid()
print("The time after 10 more contacts have been established is {} minutes.".format(contact_time[11] / 60))
plt.show()




