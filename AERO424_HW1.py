import numpy as np

# PROBLEM 2


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

    # Assemble DCM
    A_EI = np.array([[np.cos(theta), np.sin(theta), 0],
                    [-np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    return A_EI


# Test vector validation of ECI to ECEF function
r_I = np.array([[7000.5],
                [8000],
                [-.5]])

r_E = np.matmul(DCM_EI(2021, 9, 17, 12, 0, 0), r_I)
# print("Problem 2A test vector is:")
# print(r_E)

########################
# problem 2B
########################


# from the handwork we calculate inclination
i = np.arccos(1.99096871e-7*4625.27*6000**2 / (-3*np.pi*1.08262668e-3*6378**2))


i_ben = np.pi/180*97.8    # two values for the satellite's orbital inclination
# I definitely was not able to calculate inclination correctly...


def DCM_OI():    # computes DCM from ECI frame to LVLH frame
    v_norm = 8.1506    # in km/s

    # form the satellite radius and velocity vector in the ECI frame

    R = np.array([[600],
                  [0],
                  [0]])
    V = np.array([[0],
                  [v_norm*np.cos(i_ben)],
                  [v_norm*np.sin(i_ben)]])

    # form LVLH frame in terms of the inertial vectors
    Z_LVLH = -R/np.linalg.norm(R)
    Y_LVLH = -(np.cross(R, V, axis=0)) / np.linalg.norm(np.cross(R, V, axis=0))
    X_LVLH = np.cross(Y_LVLH, Z_LVLH, axis=0)

    A_OI = np.column_stack((X_LVLH, Y_LVLH, Z_LVLH))
    return A_OI


# Test vector from ECI to LVLH

r_O = np.matmul(DCM_OI(),r_I)
#print("\nProblem 2B test vector from ECI to LVLH")
#print(r_O)

#######################
# PROBLEM 3A
#######################

# from quaternion to attitude matrix
#print("\nProblem 3A")
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


# test with quaternion
DCM_3A = qtoA(0, 0.2588, 0, 0.9659)

##print("\nAttitude matrix from the quaternion:")
#print(DCM_3A)


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
                               [1+ np.trace(A)]])
        q_version4 = q_version4 / np.linalg.norm(q_version4)
        return q_version4


DCM = np.array([[0.8660, 0, -0.5],
                [0, 1, 0],
                [0.5, 0, 0.866]])
#print("\nThe Quaternion computed from Attitude matrix:")
#print(Atoq(DCM))

###########################
# Problem 3B
###########################
#print("\nProblem 3B")


def q_multiply(q1, q2):    # multiplies two quaternions (or a vector pretending to be a quaternion)
    psi_q = np.array([[q1[3][0], q1[2][0], -q1[1][0], q1[0][0]],
                      [-q1[2][0], q1[3][0], q1[0][0], q1[1][0]],
                      [q1[1][0], -q1[0][0], q1[3][0], q1[2][0]],
                      [-q1[0][0], -q1[1][0], -q1[2][0], q1[3][0]]])
    q3 = np.matmul(psi_q, q2)
    return q3


def vector_q_transform(V, q):    # transforms a vector with a given quaternion
    V_4D = np.array([[V[0][0]],
                     [V[1][0]],
                     [V[2][0]],
                     [0]])
    qStar = np.array([[-q[0][0]],
                      [-q[1][0]],
                      [-q[2][0]],
                      [q[3][0]]])

    intermidiate_step = q_multiply(q, V_4D)
    V_Transformed = q_multiply(intermidiate_step, qStar)
    return V_Transformed

V = np.array([[20],
              [21],
              [12]])

q = np.array([[.09],
              [.18],
              [.27],
              [.939]])
q_star = ([[-.09],
           [-.18],
           [-.27],
           [.939]])
#print("\nPassive transformation: ")
##print(vector_q_transform(V, q))
#print("\nActive transformation:")
#print(vector_q_transform(V, q_star))


#############################
# problem 4
#############################
#print("\nProblem 4")

# Part A
A_EI = DCM_EI(2021, 9, 17, 19, 21, 0)
#print("\nDCM from ECI to ECEF is at 9/17/2021 19:21 GMT:")
#print(A_EI)

# part B

# HRBB Coordinates
lat = np.pi/180 * 30.618998
long = np.pi/180 * -96.338802

A_prime_ECEF = np.array([[np.cos(long), np.sin(long), 0],
                      [-np.sin(long), np.cos(long), 0],
                      [0, 0, 1]])

A_T_prime = np.array([[np.cos(lat), 0, -np.sin(lat)],
                         [0, 1, 0],
                         [np.sin(lat), 0, np.cos(lat)]])
A_T_ECEF = np.matmul(A_T_prime, A_prime_ECEF)
#print("\nThe DCM from ECEF to the Topographic frame is:")
#print(A_T_ECEF)

# part C
# From hand calculations...
A_BT = np.array([[0, 1, 0],
                 [0, 0, 1],
                 [1, 0, 0]])    # We need this attitude to point satellite antenna towards ground

# part D
A_BE = np.matmul(A_BT, A_T_ECEF)

A_BI = np.matmul(A_BE, A_EI)
#print("\nThe DCM from ECI to Target Body")
#print(A_BI)

# part E
A_BcI = np.array([[-.3739, -.6594, .6523],
                  [.2949, .5823, .7576],
                  [-.8793, .4756, -.0233]])
A = np.matmul(A_BI, np.transpose(A_BcI))
#print("\nDCM from current attitude to target ")
#print(A)


#############################
# Problem 5
#############################
#print("\nProblem 5")

x_nadir = np.array([[.356],
                    [.934],
                    [.001]])

x_sun = np.array([[-.945],
                  [.277],
                  [.172]])
x_nadir_norm = x_nadir / np.linalg.norm(x_nadir)
x_sun_norm = x_sun / np.linalg.norm(x_sun)

# compute a normalized vector halfway between sun and nadir vector
x_halfway = (x_sun_norm + x_nadir_norm) / np.linalg.norm((x_sun_norm + x_nadir_norm))

# compute quaternion components using definition of a quaternion
q1_3 = np.cross(x_sun_norm, x_halfway, axis=0)
q4 = np.matmul(np.transpose(x_sun_norm), x_halfway)


q = np.array([[q1_3[0][0]],
              [q1_3[1][0]],
             [q1_3[2][0]],
             [q4[0][0]]])

#print("\nQuaternion that transforms from the nadir vector to the sun vector: ")
#print(q)
x_sun_test = vector_q_transform(x_nadir, q)


# part B, going to use the function created previously to compute this sun vector
print("\n Part B")
astronomer_Whoopsie_quaternion = np.array([[.043],
                           [-.328],
                           [.764],
                           [.554]])
# create a conjugate quaternion because my first attempt isn't right
astronomer_Whoopsie_quaternion_star = np.array([[-.043],
                                           [.328],
                                           [-.764],
                                           [.554]])
x_moon = x_sun    # the sun vector was actually the moon vector
partb_sun_vector = vector_q_transform(x_moon, astronomer_Whoopsie_quaternion_star)

#print("\nSun vector for part B after the astronomer's whoopsie:")
#print(partb_sun_vector)

# Part C
# compute the quaternion you could've used originally, just done by multplying together
# the two intermediate quaternions

q_direct = q_multiply(astronomer_Whoopsie_quaternion_star, q)  # the direct quaternion from earth to sun
#print("\n The direct quaternion between from nadir to sun vector:")
#print(q_direct)

x_sun_test_direct = vector_q_transform(x_nadir, q_direct)

#print("\n The sun vector computed with the direct quaternion:")
#print(x_sun_test_direct)





