import numpy as np
import AERO424_HW1

# part A
# from handwork, the spacecraft attitude wrt G frame is..
print("\nPART A")
A_BG = np.array([[1/np.sqrt(2), 1/np.sqrt(2), 0],
                 [-1/np.sqrt(2), 1/np.sqrt(2), 0],
                 [0, 0, 1]])
print("\nFrom handwork, the spacecraft attitude wrt G frame is: ")
print(A_BG)

#####################
# PART B
####################
print("\n\nPART B\n\n")

# from handwork, the sun and earth vector in the spacecraft body frame:

e_G = np.array([[-1], [0], [0]])    # earth vector in G frame
s_G = np.array([[0], [1], [0]])    # sun vector in G frame

e_B = np.matmul(A_BG, e_G)    # earth vector in body frame
s_B = np.matmul(A_BG, s_G)    # sun vector in body frame

print("Earth vector in the spacecraft body frame:")
print(e_B)
print("\nSun vector in spacecraft body frame:")
print(s_B)

#################
# PART C
################
print("\n\nPart C\n\n")


def TRIAD(b1, b2, r1, r2):     # b for body frame. r for inertial frame
    # compute the body triad
    w1 = b1
    w2 = np.cross(b1, b2, axis=0) / np.linalg.norm(np.cross(b1, b2, axis=0))
    w3 = np.cross(w1, w2, axis=0)
    W = np.column_stack((w1, w2, w3))
    print("The orthonormal triad with the body vectors is:")
    print(W, "\n")
    # compute the G frame triad
    v1 = r1
    v2 = np.cross(r1, r2, axis=0) / np.linalg.norm(np.cross(r1, r2, axis=0))
    v3 = np.cross(v1, v2, axis=0)
    V = np.column_stack((v1, v2, v3))
    print("The orthonormal triad with the G frame is:")
    print(V, "\n")

    # compute attitude with the triads
    print("\n\nPART D\n\n")
    A_triad = np.matmul(W, np.transpose(V))
    return A_triad


A_BG_Triad = TRIAD(e_B, s_B, e_G, s_G)

print("The attitude matrix according to TRIAD is:")
print(A_BG_Triad)
print("\nSince we are getting the same thing as part A, TRIAD is validated.")

#################
print("\n\nPART E\n\n")
################


def Adjoint(M):    # computes adjoint of matrix M, 3x3 only
    a11 = M[1][1]*M[2][2] - M[1][2]*M[2][1]
    a21 = -1 * (M[0][1]*M[2][2] - M[0][2]*M[2][1])
    a31 = M[0][1] * M[1][2] - M[0][2] * M[1][1]

    a12 = -1 * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
    a22 = M[0][0] * M[2][2] - M[0][2] * M[2][0]
    a32 = -1 * (M[0][0] * M[1][2] - M[0][2] * M[1][0])    # all of these are the hardcoded co-factors

    a13 = M[1][0] * M[2][1] - M[1][1] * M[2][0]
    a23 = -1 * (M[0][0] * M[2][1] - M[0][1] * M[2][0])
    a33 = M[0][0] * M[1][1] - M[0][1] * M[1][0]

    AdjA = np.array([[a11, a21, a31],
                     [a12, a22, a32],
                     [a13, a23, a33]])
    return AdjA    # validated function


def QUEST(a, b1, b2, r1, r2):    # QUEST algorithm constrained to two inputs, a is the weight vector

    # essential quest parameters
    B = a[0]*np.matmul(b1, np.transpose(r1)) + a[1]*np.matmul(b2, np.transpose(r2))
    sigma = np.trace(B)
    S = B + np.transpose(B)
    Z = a[0]*np.cross(b1, r1, axis=0) + a[1]*np.cross(b2, r2, axis=0)
    kappa = np.trace(Adjoint(S))
    delta = np.linalg.det(S)

    # from Shuster and Oh eqn (72) the max eigenvalue where there's only two observations is below:
    cos_theta_part1 = np.matmul(np.transpose(b1), b2) * np.matmul(np.transpose(r1), r2)
    cos_theta_part2 = np.linalg.norm(np.cross(b1, b2, axis=0)) * np.linalg.norm(np.cross(r1, r2, axis=0))
    cos_theta = cos_theta_part1 + cos_theta_part2    # Shuster and Oh eqn. 73
    # this is a bit hand wavey but this Shuster and Oh research paper is assuring me that in the special case where
    # QUEST only has two observed inputs, the max eigenvalue has this exact closed form solution and newton raphson is
    # unnecessary.

    lambda_max = np.sqrt(a[0]**2 + a[1]**2 + 2*a[0]*a[1]*cos_theta)    # Shuster and Oh eqn. 72

    # compute optimal quaternion
    alpha = lambda_max**2 - sigma**2 + kappa
    beta = lambda_max - sigma
    gamma = (lambda_max + sigma)*alpha - delta    # Shuster and Oh eqn. 66

    X_vector_step1 = (alpha*np.identity(3) + beta*S + np.matmul(S, S))  # Shuster and Oh eqn.68
    X_vector = np.matmul(X_vector_step1, Z)

    q1_3 = 1 / np.sqrt(gamma**2 + np.linalg.norm(X_vector)**2) * X_vector   # Shuster and Oh eqn. 69
    q4 = 1 / np.sqrt(gamma**2 + np.linalg.norm(X_vector)**2) * gamma

    # most of this math was pulled from "Three-Axis Attitude Determination from Vector Observations"
    # from M.D Shuster and S.D Oh,  AIAA 81-4003

    # I just realized the QUEST lecture notes finally got uploaded in Canvas. That's a big oof, but at least
    # the research paper worked.

    return q1_3, q4


weights = np.array([1, 1])
Q1_3, Q4 = QUEST(weights, e_B, s_B, e_G, s_G)

q1 = Q1_3[0][0]
q2 = Q1_3[1][0]
q3 = Q1_3[2][0]
q4 = float(Q4)
q = np.array([[q1], [q2], [q3], [q4]])

print("Quest gives us this as the quaternion:")
print(q, "\n")
print("Which gives us the DCM:")
print(AERO424_HW1.qtoA(q1, q2, q3, q4), "\n")
print("THANKFULLY it is the same attitude matrix as part A :)")


# moving on to q-davenport's method

def K_B_Summation(b, r, a):    # first step of the q davenport method, just generalized so we can easily sum vectors
    b_cross = np.array([[0, b[2][0], -b[1][0], b[0][0]],
                        [-b[2][0], 0, b[0][0], b[1][0]],
                        [b[1][0], -b[0][0], 0, b[2][0]],
                        [-b[0][0], -b[1][0], -b[2][0], 0]])

    r_dot = np.array([[0, -r[2][0], r[1][0], r[0][0]],
                      [r[2][0], 0, -r[0][0], r[1][0]],
                      [-r[1][0], r[0][0], 0, r[2][0]],
                      [-r[0][0], -r[1][0], -r[2][0], 0]])

    K_i = a * np.matmul(np.transpose(b_cross), r_dot)
    return K_i


def Q_Davenport_Method(a, b1, b2, r1, r2) -> np.ndarray:    # method is constrained to two measurements
    K_B = K_B_Summation(b1, r1, a[0]) + K_B_Summation(b2, r2, a[1])    # K matrix
    eigenvalues, eigenvectors = np.linalg.eig(K_B)

    # find largest eigenvalue
    largest_element = -1
    for i in range(len(eigenvalues)):
        largeness_tracker = 0    # needs to equal length of eigenvalue list for it to be the correct eigenvalue
        for j in range(len(eigenvalues)):
            if eigenvalues[i] >= eigenvalues[j]:
                largeness_tracker += 1
        if largeness_tracker == len(eigenvalues):
            largest_element = i

    q = eigenvectors[:, largest_element]
    return q


q = Q_Davenport_Method(weights, e_B, s_B, e_G, s_G)
print("\nQ Davenport's Method gives us this as the quaternion:")
print(q, "\n")

q1, q2, q3, q4 = q[0], q[1], q[2], q[3]
A_Davenport = AERO424_HW1.qtoA(q1, q2, q3, q4)
print("This gives us the attitude matrix:")
print(A_Davenport, "\n")
print("This is thankfully the same thing as part A")

#############
# PART F
#############
print("\n\nPART F\n\n")

# now we do everything again, just adding noise.
e_G_noise = np.array([[-1.03], [.05], [.07]])
s_G_noise = np.array([[.1], [1.02], [-.04]])

e_B_noise = np.matmul(A_BG, e_G_noise)
s_B_noise = np.matmul(A_BG, s_G_noise)

A_noise_TRIAD = TRIAD(e_B_noise, s_B_noise, e_G_noise, s_G_noise)
print("The noisy TRIAD attitude matrix is:")
print(A_noise_TRIAD, "\n")

q1_3_N, q4_N = QUEST(weights, e_B_noise, s_B_noise, e_G_noise, s_G_noise)
q1 = Q1_3[0][0]
q2 = Q1_3[1][0]
q3 = Q1_3[2][0]
q4 = float(Q4)
q = np.array([[q1], [q2], [q3], [q4]])
print("The noisy QUEST quaternion is:")
print(q, "\n")
A_noise_Quest = AERO424_HW1.qtoA(q1, q2, q3, q4)
print("The noisy QUEST Attitude matrix is:")
print(A_noise_TRIAD, "\n")

Davenport_q_noise = Q_Davenport_Method(weights, e_B_noise, s_B_noise, e_G_noise, s_G_noise)
print("The noisy Davenport quaternion is:")
print(Davenport_q_noise, "\n")

q1, q2, q3, q4 = Davenport_q_noise[0], Davenport_q_noise[1], Davenport_q_noise[2], Davenport_q_noise[3]
A_Davenport_noise = AERO424_HW1.qtoA(q1, q2, q3, q4)
print("The noisy Davenport Attitude matrix is: ")
print(A_Davenport_noise)

print("\nSo comparing this output with the original attitude matrices without noise. We see a all methods")
print("other than Davenport produce an equally offset matrix from the exact solution. This is probably because")
print("The davenport method used numpy's eigenvalue function to directly compute the eigenvalue and thus is more")
print("accurate.")
