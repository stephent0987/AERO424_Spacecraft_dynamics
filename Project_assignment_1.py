import numpy as np

# Question 2 code for helping determine Nominal mode requirements

# Attitude determination accuracy. Convert the mission level requirement of 10 arcseconds into degrees
min_attitude_knowledge = 10 / 3600
print("\nThe subsystem shall have determine attitude to {} degrees or better during nominal operation."
      .format(round(min_attitude_knowledge, 4)))

min_attitude_control = .05 / 3600
print("The subsystem shall control the attitude to within {} degrees or better during nominal operation."
      .format(round(min_attitude_control, 8)))

# given satellite parameters
m = 11100    # kg
Ix = 77217   # kg*m^2
Iy = Ix
Iz = 25000    # Kg*m^2
Frontal_area = 23.45    # m^2
Solar_panel_A = 37.16    # m^2
Residual_Mag_Dipole_M = 20    # Am^2
h = 600    # km
C_d = 2.5
moment_arm = 2    # m
rho = 1.454e-13    # kg/m^3
Reflectance = .7
gg_worst_angle = 45*np.pi/180

# calculate worst case gravity gradient torque

mu = 3.986012e14    # m^3/s2    earth gravitational constant according to Fundamentals of Astrodynamics
R = (h + 6378.1) * 1e3     # adding the radius of the earth

T_g_max = 3*mu / (2*R**3) * abs(Iz - Iy) * np.sin(2*gg_worst_angle)
print("\nWorst case gravity torque:", T_g_max)

# calculate worst case Solar pressure torque
# the worst case scenario is when your incidence angle is 0 making cos(phi) = 1
T_s_max = 1367/3e8 * Solar_panel_A * (1+Reflectance) * 2
print("\nThe worst case Solar pressure torque:", T_s_max)

# calculate worst case aerodynamic drag torque
v = np.sqrt(mu / R)
T_a_max = .5 * rho * Frontal_area * C_d * v**2 * moment_arm
print("\nThe worst case aerodynamic torque: ", T_a_max)

# compute the maximum magnetic dipole torque. Approximation is assumed for the SSO orbit.
M = 7.96e15    # Tesla * m^3   magnetic dipole of the earth
B = 2*M / R**3    # approximate worst case magnetic field strength for polar orbits

T_m_max = Residual_Mag_Dipole_M * B
print("\nThe worst case magnetic torque: ", T_m_max)

# sum disturbance torques. My rationale to do this is even though it is unlikely to have all disturbance torque to act
# on the satellite in the same direction at the same time, it'll provide a comfortable safety margin for minimum storage
# capacity of momentum

T_sum_max = T_s_max + T_m_max + T_a_max + T_g_max
print("\nThe sum of all worst case disturbance torques: ", T_sum_max)

# use the cyclic disturbance torque equation from class slides
period = 2*np.pi / np.sqrt(mu) * R**(3/2)

h = T_sum_max * period / (4*np.sqrt(2))    # required angular momentum

# add a 20% margin and round to look nice in the requirement
print("\nThe subsystem shall have a minimum momentum storage of {} N*m*s".format(round(1.2 * h, 1)))

# calculate the needed torque authority by adding disturbance torques to the necessary slew torque. For the slew
# we want to get 180 degree rotation in 5 minutes. Using the SMAD half-angle equation:

T_minimum_slew = 4 * Ix * np.pi / (5*60)**2 + T_sum_max


# add 20% margin and round to look nice in the requirement
print("\nThe minimum torque authority required is:", round(1.2 * T_minimum_slew, 1))





