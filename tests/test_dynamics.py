import numpy as np
from quadcopter_sim.dynamics import translational_dynamics, rotational_dynamics, euler_kinematics, MASS, GRAVITY

def test_hover_condition():
    """
    At hover: level attitude, zero velocity, thrust = mg. All accelerations should be zero.
    """
    hover_thrust = MASS*GRAVITY     # exactly counteracts gravity
    phi, theta, psi = 0,0,0         # perfectly level
    velocity = [0,0,0]              # stationary

    acc = translational_dynamics(phi, theta, psi, velocity, hover_thrust)

    # should be [0,0,0] - hovering means no acceleration
    np.testing.assert_allclose(acc, [0,0,0], atol=1e-6)

def test_gravity_only():
    """
    With zero thrust, should just fall under gravity/
    """

    acc = translational_dynamics(0,0,0, [0,0,0], thrust = 0)
    #only z acceleration, equal to -g (falling)
    assert abs(acc[0]<1e-6) #no x acceleration
    assert abs(acc[1]<1e-6) #no y acceleration

    np.testing.assert_allclose(acc[2], -GRAVITY, atol=1e-6)

    
    from quadcopter_sim.dynamics import rotational_dynamics, euler_kinematics

def test_no_torque_no_rotation():
    """
    With no torques and no current rotation, angular acceleration is zero.
    """
    omega = [0, 0, 0] # angular velocity vector
    torques = [0, 0, 0] # no torques applied

    alpha = rotational_dynamics(0, 0, omega, torques) # calls rotational_dynamics function assigning (roll = 0, pitch = 0, current angular velocities, current motor torques)
    np.testing.assert_allclose(alpha, [0, 0, 0], atol=1e-6) # testing: assert_allclose - assert that two things are 
    # close to equal, alpha should be [0,0,0]. atol = tolerance

def test_euler_kinematics_level():
    """
    When level (phi=theta=0), body rates map directly to Euler rates.
    p -> phi_dot, q -> theta_dot, r -> psi_dot
    """
    phi, theta = 0, 0
    omega = [0.1, 0.2, 0.3]   # some arbitrary angular rates
    rates = euler_kinematics(phi, theta, omega)

    # At zero attitude, p=phi_dot, q=theta_dot, r=psi_dot exactly
    np.testing.assert_allclose(rates, omega, atol=1e-6)

