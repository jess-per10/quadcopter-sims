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

from quadcopter_sim.dynamics import motor_mixing, ARM_LENGTH, THRUST_COEFF, DRAG_COEFF

def test_symmetric_hover():
    """
    Equal motor speeds should produced zero roll, ptch, yaw torques. 
    Only thrust should be non-zero. 
    """
    w = 100.0 #rad/s - same for all motors
    thrust, tau_phi, tau_theta, tau_psi = motor_mixing([w,w,w,w])

    #Thrust should be 4*kT*w²
    expected_thrust = 4 * THRUST_COEFF * w**2
    np.testing.assert_allclose(thrust, expected_thrust, atol=1e-6)

    #All torques should be zero - symmetric hover
    np.testing.assert_allclose(tau_phi,   0, atol=1e-6)
    np.testing.assert_allclose(tau_theta, 0, atol=1e-6)
    np.testing.assert_allclose(tau_psi,   0, atol=1e-6)

def test_roll_torque_direction():
    """
    Speeding up teh left mtoros (3 and 2) should produce a positive roll torque
    """
    w_base = 100.0
    w_high = 110.0 # left motors faster 

    _, tau_phi, _, _ = motor_mixing([w_base, w_high, w_high, w_base])    

    # Left motors faster = positive roll torque
    assert tau_phi > 0

def test_zero_motors():
    """
    All motors off means zero thrust and zero torques.
    """
    thrust, tau_phi, tau_theta, tau_psi = motor_mixing([0, 0, 0, 0])
    np.testing.assert_allclose(thrust,    0, atol=1e-10)
    np.testing.assert_allclose(tau_phi,   0, atol=1e-10)
    np.testing.assert_allclose(tau_theta, 0, atol=1e-10)
    np.testing.assert_allclose(tau_psi,   0, atol=1e-10)

from quadcopter_sim.simulation import run_simulation, state_derivative
from quadcopter_sim.dynamics import MASS, GRAVITY, THRUST_COEFF

def test_state_derivative_length():
    """
    State derivative must return 12 values
    """

    state = np.zeros(12)
    motor_speeds = [100,100,100,100]
    deriv = state_derivative(0,state,motor_speeds)
    assert len(deriv) == 12

def test_free_fall():
    """
    With zero motor speeds, quadcopter should fall under gravity. 
    After 1 second, z velocity should be approximately -9.81m/s.
    """
    
    initial_state = np.zeros(12) # start at rest at origin
    result = run_simulation(initial_state, [0,0,0,0], t_end = 1.0)
    
    #Final z velocity should be close to -g
    final_vz = result.y[8,-1]
    np.testing.assert_allclose(final_vz, -GRAVITY, rtol=0.1, atol=0.1)