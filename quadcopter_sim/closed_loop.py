import numpy as np
from .controller import PIDController
from .dynamics import (
    motor_mixing, 
    translational_dynamics,
    rotational_dynamics,
    euler_kinematics,
    MASS, GRAVITY, THRUST_COEFF 
)

# Hover motor speed - same formula as before
HOVER_W = np.sqrt((MASS * GRAVITY) / (4 * THRUST_COEFF))

def make_attitude_controllers():
    """
    Create PID controllers for roll, pitch and yaw.
    Returns a dict of three PIDController instances. 
    These gains ae starting points and may need tuning for good performance - expect to tune.
    """
    return {
        'roll':  PIDController(kp=8.5, ki=0.0, kd=6.5, output_limit=0.5),
        'pitch': PIDController(kp=2.0, ki=0.0, kd=2.0, output_limit=0.5),
        'yaw':   PIDController(kp=2.0, ki=0.0, kd=0.3, output_limit=0.3),    
    }

def attitude_control_step(state, setpoints, controllers, dt):
    """
    One control step - read current attitude, compute PID outputs, 
    return motor speeds.

    state - full 12 element state vector
    setpoints - dict with keys 'roll', 'pitch', 'yaw' and 'attitude'
    controllers - dict of PIDController instances
    dt - timestep in seconds

    returns motor speeds (array of 4)
    """

    # Unpack current attitude from state
    phi, theta, psi = state[3], state[4], state[5]

    # Run PID for each attitude axis
    roll_cmd = controllers['roll'].update(
        setpoint = setpoints['roll'], measurement = phi, dt=dt)
    
    pitch_cmd = controllers['pitch'].update(
        setpoint=setpoints['pitch'], measurement=theta, dt=dt)

    yaw_cmd   = controllers['yaw'].update(
        setpoint=setpoints['yaw'], measurement=psi, dt=dt)
    
    # Convert PID outputs to motor speed adjustments
    # Each command adjusts motors differentially around hover speed
    w1 = HOVER_W - roll_cmd + pitch_cmd + yaw_cmd # front right (CCW)
    w2 = HOVER_W + roll_cmd - pitch_cmd + yaw_cmd # back left (CCW)
    w3 = HOVER_W + roll_cmd + pitch_cmd - yaw_cmd # front left (CW)
    w4 = HOVER_W - roll_cmd - pitch_cmd - yaw_cmd # back right (CW)

    # Clamp motor speeds to physical limits
    motor_speeds = np.clip([w1, w2, w3, w4], 0, HOVER_W * 2)

    return motor_speeds

def run_controlled_simulation(intial_state, setpoints, t_end = 5.0, dt=0.01):
    """
    Run simulation with attitude PID controller active. 
    Unlike run_simulation, this steps manually so the controller can update at each timestep. 

    Return arrays of time and state history. 
    """
    controllers = make_attitude_controllers()

    n_steps = int(t_end/dt)
    time_history = np.zeros(n_steps)
    state_history = np.zeros((12, n_steps))

    state = intial_state.copy()
    t = 0.0

    for i in range(n_steps):
        #store current state
        time_history[i] = t
        state_history[:, i] = state

        # Get motor commands from controller 
        motor_speeds = attitude_control_step(state, setpoints, controllers, dt)

        # Get thrust and torques from motor speeds
        thrust, tau_phi, tau_theta, tau_psi = motor_mixing(motor_speeds)

        # Compute derivates
        vel = state[6:9]
        omega = state[9:12]
        phi, theta, psi = state[3], state[4], state[5]

        acc = translational_dynamics(phi, theta, psi, vel, thrust)
        alpha = rotational_dynamics(phi, theta, omega, [tau_phi, tau_theta, tau_psi])
        euler_rates = euler_kinematics(phi, theta, omega)

        # euler integration - update state
        state[0:3] += state[6:9] * dt # position
        state[3:6] += euler_rates * dt # attitude
        state[6:9] += acc * dt # velocity
        state[9:12] += alpha * dt # angular velocity

        t += dt

    return time_history, state_history