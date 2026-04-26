import numpy as np
from quadcopter_sim.dynamics import translational_dynamics, MASS, GRAVITY

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

    
    