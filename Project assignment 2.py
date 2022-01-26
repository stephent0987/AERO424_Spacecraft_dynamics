import numpy as np
from AERO424_HW1 import DCM_EI as hw1    # import function calls from my first homework
from AERO424_HW1 import calculate_JD as jd
# will a
# given satellite parameters
m = 11100    # kg
Ix = 77217   # kg*m^2
Iy = Ix
Iz = 25000    # Kg*m^2
Frontal_area = 23.45    # m^2
Solar_panel_A = 37.16    # m^2
Residual_Mag_Dipole_M = 20    # Am^2
h = 600000    # m
C_d = 2.5
moment_arm = 2    # m
rho = 1.454e-13    # kg/m^3
Reflectance = .7



r = (6378.135 + h) * 1e3
def Euler_to_DCM(phi, theta, psi):    # computes the DCM given 3-2-1 Euler angles

    # 3 = phi  2 = theta  1 = psi, even though this definition makes me wanna cry

    A11 = np.cos(theta) * np.cos(phi)
    A21 = -np.cos(psi)*np.sin(phi) + np.sin(psi)*np.sin(theta)*np.cos(phi)
    A31 = np.sin(psi)*np.sin(phi) + np.cos(psi)*np.sin(theta)*np.cos(phi)

    A12 = np.cos(theta)*np.sin(phi)
    A22 = np.cos(psi)*np.cos(phi) + np.sin(psi)*np.sin(theta)*np.sin(phi)
    A32 = -np.sin(psi)*np.cos(phi) + np.cos(psi)*np.sin(theta)*np.sin(phi)

    A13 = -np.sin(theta)
    A23 = np.sin(psi)*np.cos(theta)
    A33 = np.cos(psi)*np.cos(theta)

    A = np.array([[A11, A12, A13],
                  [A21, A22, A23],
                  [A31, A32, A33]])
    return A


# gravity gradient torque.
# this should be written to where it does not care what frame the position vector is in
# general formula: 3*mu/R^5 * Rc cross J*Rc


def Lg(r1, r2, r3, J1, J2, J3, phi, theta, psi, *body):   # input meters and radians for things radii vector in not body
    mu = 3.986004e14    # m^3 / s^2
    r = np.sqrt(r1**2 + r2**2 + r3**2)
    r_vector = np.array([[r1], [r2], [r3]])
    J_b = np.array([[J1, 0, 0],
                  [0, J2, 0],
                  [0, 0, J3]])
    A_bo = Euler_to_DCM(phi, theta, psi)
    if body != True or body != False:
        r_vector = np.matmul(A_bo, r_vector)
    elif body == False:
        r_vector = np.matmul(A_bo, r_vector)

    L_g = 3*mu / r**5 * np.cross(r_vector, np.matmul(J_b, r_vector), axis=0)
    return L_g

# This is to validate that I am getting the same thing as my estimate from assigment 1
# I used smad's equation to calculate worst case which is equivalent to psi = 45 degrees
print("\n\n")
psi = np.pi/180 * 0
theta = np.pi/180 * 0
phi = np.pi/180 * 0
print("Function 1 Part A results:\n")
print("You would expect the gravity gradient to be zero any time a principle axis lines up with radial vector")
print("Currently body frame and LVLH frame are aligned and I get gravity gradient torques of:")
print(Lg(0, 0, r, Ix, Iy, Iz, phi, theta, psi))



psi = np.pi/180 * 45
theta = np.pi/180 * 0
phi = np.pi/180 * 0


print("Function 1 part b results:\n")
print("Gravity gradient torque worst case:")
print(Lg(0, 0, r, Ix, Iy, Iz, phi, theta, psi))
print("And that is close to the same thing as assignment 1")
print("\n\n")


###########################
# Function 2
###########################
print("######################")
print("FUNCTION2")
print("######################\n")
def Lm(xi, yi, zi,Vxi, Vyi, Vzi, mx, my, mz, phi, theta, psi):

    Spacecraft_dipole = np.array([[mx], [my], [mz]])


    # I am assuming that this position vector will be in ECI
    A_EI = hw1(2021, 11, 1, 17, 44, 0)    # inputting the time as of writting this line
    r_I = np.array([[xi], [yi], [zi]])
    r_E = np.matmul(A_EI, r_I)    # compute position vector in ECEF
    norm_RE = np.linalg.norm(r_E)


    # dipole model
    m = .77e22    # Am^2
    theta_m = np.pi/180 * 169.7
    alpha_m = np.pi/180 * 108.2

    m_vector = 6371.2e3**3 * np.array([[-1669.05e-9],
                                  [5077.05e-9],
                                  [-29554.63e-9]])

    # B as a function of position vector in the ECEF frame
    B_r = (3*np.matmul(np.transpose(m_vector), r_E) * r_E - norm_RE**2 * m_vector) / norm_RE**5
    # now compute the B vector from ECEF to body. This requires velocity...
    # compute A_OI
    v_I = np.array([[Vxi], [Vyi], [Vzi]])
    Z_LVLH = -r_I / np.linalg.norm(r_I)
    Y_LVLH = -(np.cross(r_I, v_I, axis=0)) / np.linalg.norm(np.cross(r_I, v_I, axis=0))
    X_LVLH = np.cross(Y_LVLH, Z_LVLH, axis=0)

    A_OI = np.column_stack((X_LVLH, Y_LVLH, Z_LVLH))    # DCM from inertial to LVLH
    A_bo = Euler_to_DCM(phi, theta, psi)    # DCM from LVLH to body

    A_BE = np.matmul(A_bo, np.matmul(A_OI, np.transpose(A_EI)))    # tranformation from ECEF to body

    B_r_Bodyframe = np.matmul(A_BE, B_r)

    L_m = np.cross(Spacecraft_dipole, B_r_Bodyframe, axis=0)
    return L_m

# these values came from a keplerian elements to intertial are and v vector program i made for AERO423
# I did my best to get the correct altitude and colatitude position
R_I = np.array([[-3961.21710246e3],
                [-1847.14587049e3],
                [5340.84402173e3]])
V_I = np.array([[2.77561211e3],
                [-7.09120092e3],
                [0.58814415e3]])
print("Magnetic torque at the specified worst case values")
print(Lm(R_I[0][0], R_I[1][0], R_I[2][0], V_I[0][0], V_I[1][0], V_I[2][0], 0, 20, 0, 0, .7171, 0))

R_I = np.array([[0],
                [6378.135e3 + h],
                [0]])
V_I = np.array([[np.sqrt(3.986004e14/(6378.135e3 + h))],    # setting to equatorial circular orbit to debug better
                [0],
                [0]])

print("Magnetic torque at an equatorial orbit")
print(Lm(R_I[0][0], R_I[1][0], R_I[2][0], V_I[0][0], V_I[1][0], V_I[2][0], 0, 20, 0, 0, .7171, 0))
print("The equatorial one should be smaller in magnitude")
print("These values are pretty close to assignment 1")

print("################")
print("FUNCTION 3")
print("################")

print("\n\n")

# compute aerodynamic torque
# I plan to store data of the outward normal unit vectors by a matrix that is (N by 3) with
# N faces, and store surface area in numpy array
R_I = np.array([[0],
                [6378.135e3 + h],
                [0]])
V_I = np.array([[np.sqrt(3.986004e14/(6378.135e3 + h))],    # setting to equatorial circular orbit to debug better
                [0],
                [0]])

n_test = np.array([[1, 0, 0],
                   [-1, 0, 0],
                   [0, 1, 0],
                   [0, -1, 0],
                   [0, 0, 1],
                   [0, 0, -1]])    # models a simple cubesat

A_test = np.array([7, 7, 7, 7, 7, 7])    # 6U surface area configuration, (but if its in m^2 and not U's lol this is just a test)

r_test = np.array([[.5, 0, 0],    # position vectors of the
                   [-.5, 0, 0],
                   [0, 1, 0],
                   [0, -1, 0],
                   [0, 0, 1.5],
                   [0, 0, -1.5]])    # models a 6U cubsat, but if it was meters instead of U

def density_model(h):
   h = h/1000

   if 86 <= h <= 91:    # this is just a test to use with a reference standard atmosphere
       A = 0
       B = -3.22622e-6
       C = 9.111460e-4
       D = -.2609971
       E = 5.944694

   elif 300 < h <= 500:
       A = 1.1405646e-10
       B = -2.130756e-7
       C = 1.570762e-4
       D = -.07029296
       E = -12.89844

   elif 500 < h <= 750:
       A = 8.105631e-12
       B = 2.358417e-9
       C = -2.635110e-6
       D = -.01562608
       E = -20.02246

   elif 750 < h <= 1000:
       A = -3.701195e-12
       B = -8.608611e-9
       C = 5.118829e-5
       D = -.06600998
       E = -6.137674
   else:
       raise ValueError("You have inputted a value that is beyond the bounds of this model")

   rho = np.exp(A*h**4 + B*h**3 + C*h**2 + D*h + E)
   return rho





def La(r_I, v_I, n,Cp_positions, A, Cd, phi, theta, psi):    # computes total aerodynamic torque on system
    w_e = 2*np.pi / (3600*23 + 60*56 + 4)    # angular velocity of earth
    w_e_vector = np.array([[0], [0], [w_e]])

    v_rel_I = v_I + np.cross(w_e_vector, r_I, axis=0)   # relative velocity in inertial frame

    # compute attitude matrices
    A_BO = Euler_to_DCM(phi, theta, psi)

    Z_LVLH = -r_I / np.linalg.norm(r_I)
    Y_LVLH = -(np.cross(r_I, v_I, axis=0)) / np.linalg.norm(np.cross(r_I, v_I, axis=0))
    X_LVLH = np.cross(Y_LVLH, Z_LVLH, axis=0)
    A_OI = np.column_stack((X_LVLH, Y_LVLH, Z_LVLH))

    A_BI = np.matmul(A_BO, A_OI)

    # compute relative velocity in body frame
    v_rel_b = np.matmul(A_BI, v_rel_I)
    norm_v_rel_b = np.linalg.norm(v_rel_b)

    cos_theta_aero = np.zeros(len(n))    # compute all relative angles of faces
    for k in range(len(n)):
        n_k = np.array([[n[k][0]], [n[k][1]], [n[k][2]]])
        cos_theta_aero[k] = np.matmul(np.transpose(n_k), v_rel_b) / np.linalg.norm(v_rel_b)

    # compute density
    h = np.linalg.norm(r_I) - 6378.135e3    # in m
    Rho = density_model(h)

    L = np.array([[0], [0], [0]])
    for k in range(len(n)):
       F_i = -.5*Rho*Cd*norm_v_rel_b*v_rel_b*A[k] * max(cos_theta_aero[k], 0)
       position_i = np.transpose(np.array([Cp_positions[k]]))
       L_i = np.cross(position_i, F_i, axis=0)
       L = L + L_i

    return L



print("Aerodynamic torque at 600km setting yaw to 45 degrees")

print(La(R_I, V_I, n_test, r_test, A_test, C_d, np.pi/4, 0, 0))

R_I = np.array([[0],
                [6378.135e3 + 90000],
                [0]])
V_I = np.array([[np.sqrt(3.986004e14/(6378.135e3 + 90000))],
                [0],
                [0]])
print("Aerodynamic torque at 90km")
print(La(R_I, V_I, n_test, r_test, A_test, C_d, np.pi/4, 0, 0))

R_I = np.array([[0],
                [6378.135e3 + 999e3],
                [0]])
V_I = np.array([[np.sqrt(3.986004e14/(6378.135e3 + 999e3))],
                [0],
                [0]])
print("Aerodynamic torque at 999km")
print(La(R_I, V_I, n_test, r_test, A_test, C_d, np.pi/4, 0, 0))
print("this makes sense and is only an order of magnitude smaller than assignment 1")

################################
# Function 4
################################
print("########################")
print("Function 4")
print("##########################")

def L_SRP(R_I, V_I, n, Cp_positions, A, C_Diff, C_spec, phi, theta, psi):
    T_UT1 = (jd(2021, 11, 2, 20, 0, 0) - 2451545) / 36525   # computed at the time that i am writing this function

    M_sun = np.pi/180 * 357.5277233 + 35999.05034 * T_UT1   # radians
    # compute sun vector in AU
    norm_r_es = 1.000140612 - .016708617*np.cos(M_sun) - .000139589*np.cos(2*M_sun)  # radius from sun to earth in AU
    R_I_AU = R_I / 1.495978707e11    # satellite position vector in AU

    epsilon = (23.439291 - .0130042 * T_UT1) * np.pi/180    # radians
    phi_s = (280.46 + 36000.771*T_UT1)   # degrees
    phi_ecliptic = (phi_s + 1.914666471*np.sin(M_sun) + .019994643*np.sin(2*M_sun)) * np.pi/180 # radians

    r_es_hat = np.array([[np.cos(phi_ecliptic)],
                         [np.cos(epsilon)*np.sin(phi_ecliptic)],
                         [np.sin(epsilon)*np.sin(phi_ecliptic)]])
    r_es = norm_r_es*r_es_hat

    r_sat_sun = r_es - R_I_AU    # in AU

    c = 299792458
    Rad_pressure = 1361 / (c * np.linalg.norm(r_sat_sun)**2)

    # compute the sun unit vector in the body frame
    # compute attitude matrices
    A_BO = Euler_to_DCM(phi, theta, psi)

    Z_LVLH = -R_I / np.linalg.norm(R_I)
    Y_LVLH = -(np.cross(R_I, V_I, axis=0)) / np.linalg.norm(np.cross(R_I, V_I, axis=0))
    X_LVLH = np.cross(Y_LVLH, Z_LVLH, axis=0)
    A_OI = np.column_stack((X_LVLH, Y_LVLH, Z_LVLH))

    A_BI = np.matmul(A_BO, A_OI)
    s_hat = np.matmul(A_BI, (r_sat_sun/np.linalg.norm(r_sat_sun)))
    print(s_hat)

    Lsrp = np.array([[0], [0], [0]])
    for k in range(len(n)):
        n_k = np.array([[n[k][0]], [n[k][1]], [n[k][2]]])
        cos_theta_SRP_i = np.matmul(np.transpose(n_k), s_hat)

        Frad = Rad_pressure * A[k] * (2*(C_Diff/3 + C_spec*cos_theta_SRP_i) * n_k  + (1-C_spec)*s_hat) * max(cos_theta_SRP_i, 0)
        test = A[k] * (2*(C_Diff/3 + C_spec*cos_theta_SRP_i))

        position_i = np.transpose(np.array([Cp_positions[k]]))
        Lsrp = Lsrp + np.cross(position_i, Frad, axis=0)

    return Lsrp

print("zeros for the reflectivity")
print(L_SRP(R_I, V_I, n_test, r_test, A_test, 0, 0,np.pi/4 , 0, 0))
print("Ones for the reflectivitiy")
print(L_SRP(R_I, V_I, n_test, r_test, A_test, 1 , 1 ,np.pi/4 , 0, 0))
print("Obviously theres some fishy stuff happening here")

