import numpy as np
import pytest
from quadcopter_sim.controller import PIDController

def test_proportional_only():
    """
    With ki = 0 and kd = 0, output should be kp * error.
    """

    pid = PIDController(kp=2.0, ki=0.0, kd=0.0)
    output = pid.update(setpoint=10.0, measurement=7.0, dt=0.1)

    #error = 10-7 = 3, output should be 2*3=6

    assert abs(output-6) < 1e-6


def test_zero_error():
    """
    When at setpoint, output should be zero (with no accumulated integral)
    """
    pid = PIDController(kp=1.0, ki=0.0, kd=0.0)
    output = pid.update(setpoint=5.0, measurement=5.0, dt=0.1)

    assert abs(output) < 1e-6

def test_integral_accumulation():
    """
    with ki>0, repeated calls with same error should increase output. 
    """

    pid = PIDController(kp=0.0, ki=1.0, kd=0.0)
    output1 = pid.update(setpoint=1.0, measurement=0.0, dt=0.1)
    output2 = pid.update(setpoint=1.0, measurement=0.0, dt=0.1)

    # Second call should have larger output due to accumullated integral
    assert output2 > output1

def test_ouput_limit():
    """
    Output should be clamped to output_limit
    """
    pid = PIDController(kp=100.0, ki=0.0, kd=0.0, output_limit=5.0)
    output = pid.update(setpoint=100.0, measurement=0.0, dt=0.1)
    # Without limit, output would be 10*1 = 10, but should be clamped to 5
    assert abs(output - 5.0) < 1e-6

def test_reset(): 
    """
    After reset, controller should behave as if freshly create.
    """
    pid = PIDController(kp=1.0, ki=1.0, kd=0.0)
    pid.update(setpoint=1.0, measurement=0.0, dt=0.1) # some error to accumulate integral
    pid.update(setpoint=1.0, measurement=0.0, dt=0.1) # more error to accumulate integral
    pid.reset()
    output_after_reset = pid.update(setpoint=1.0, measurement=0.0, dt=0.1)

    # Fresh controller with same inputs
    fresh_pid = PIDController(kp=1.0, ki=1.0, kd=0.0)
    output_fresh = fresh_pid.update(setpoint=1.0, measurement=0.0, dt=0.1)

    # After reset, output should be just kp*error with no integral
    assert abs(output_after_reset - output_fresh) < 1e-6