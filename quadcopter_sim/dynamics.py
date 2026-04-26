import numpy as np #imports a library used of maths

# Physical parameters
MASS = 0.5          # kg - typical 450-size quadcopter
GRAVITY = 9.81      # m/s²
# Drag coefficients (start simple, tune later)
AX, AY, AZ = 0.1, 0.1, 0.1

#implement code, i.e. formula

def translational_dynamics(phi, theta, psi, velocity, thrust, mass=MASS):
    """ 
    Compute linear accelerations in the inertial frame. 
    phi = roll angle(rad)
    theta = pitch angle(rad)
    psi = yaw angle(rad)
    velocity = [vs, vy, vz] in intertial frame (m/s)
    thrust = total thurst from all 4 motors (N)
    """

    vx, vy, vz = velocity

    # Shorthand trig - matches the paper notation 
    Cphi, Sphi = np.cos(phi), np.sin(phi)
    Ctheta, Stheta = np.cos(theta), np.sin(theta)
    Cpsi, Spsi = np.cos(psi), np.sin(psi)

    # Gravity Term
    gravity_vec = np.array([0,0,GRAVITY])

    # Thrust Rotation (body frame > intertial frame)
    thrust_vec = (thrust/mass)*np.array([
        Cpsi*Stheta*Cpsi+Spsi*Sphi,
        Spsi*Stheta*Cphi-Cpsi*Sphi,
        Ctheta*Cphi
    ])

    # Drag Term
    drag_vec = (1/mass)*np.array([
        AX*vx,
        AY*vy, 
        AZ*vz
    ])

    #Equation 21: acceleration = -g + thrust_rotated - drag
    
    acceleration = -gravity_vec + thrust_vec - drag_vec
    return acceleration

# Rotational parameters

IXX = 0.0196 # MOI roll axis (kg*m²)
IYY = 0.0196 # MOI pitch axis (kg*m²)
IZZ = 0.0264 # MOI yaw axis (kg*m²)

def rotational_dynamics(phi, theta, omega, torques):
    """
    Compute angular accelerations in the body frame. 
    phi = roll angle
    theta = pitch angle
    omega = [p,q,r] angular velocities in body fram (rad/s)
    torques = [tau_phi, tau_theta, tau_psi] torques about each axis (N*m)
    """

    p,q,r = omega
    tau_phi, tau_theta, tau_psi = torques
    p_dot = ((IYY - IZZ) * q * r + tau_phi)   / IXX
    q_dot = ((IZZ - IXX) * p * r + tau_theta) / IYY
    r_dot = ((IXX - IYY) * p * q + tau_psi)   / IZZ

    return np.array([p_dot, q_dot, r_dot])

def euler_kinematics(phi, theta, omega):
    """
    Convert body frame angular velocities [p, q, r] 
    to Euler angle rates [phi_dot, theta_dot, psi_dot].
    This is the bridge between body frame and world frame rotation.
    """
    p, q, r = omega #omega is the total angular velocity vector

    Cphi, Sphi     = np.cos(phi),   np.sin(phi)
    Ctheta, Stheta = np.cos(theta), np.sin(theta)
    Ttheta         = np.tan(theta)

    phi_dot   = p + (q * Sphi + r * Cphi) * Ttheta
    theta_dot = q * Cphi - r * Sphi
    psi_dot   = (q * Sphi + r * Cphi) / Ctheta

    return np.array([phi_dot, theta_dot, psi_dot])






    # Angualar accelerations - Newton-Euler Equations
