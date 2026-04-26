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

