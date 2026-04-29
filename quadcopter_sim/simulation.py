import numpy as np
from scipy.integrate import solve_ivp
from .dynamics import (
    translational_dynamics,
    rotational_dynamics,
    euler_kinematics,
    motor_mixing,
)

def state_derivative(t, state, motor_speeds):
    """
    Compute the derivative of the full state vector. 
    This is the function solve_ivp calls at every time step
    state = [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]

    Returns d(state)/dt - rate of change of each state variable
    """

    x,y,z = state[0:3] 
    phi, theta, psi = state[3:6] #doesn't include the first index
    vx, vy, vz = state[6:9]
    p, q, r = state[9:12] 
    
    # Step 1: motor speeds to thrust and torques
    thrust, tau_phi, tau_theta, tau_psi = motor_mixing(motor_speeds)

    # Step 2: thrust to linear acceleration
    acceleration = translational_dynamics(phi, theta, psi, [vx, vy, vz], thrust)

    # Step 3: torques to angular acceleration
    alpha = rotational_dynamics(phi, theta, [p, q, r], [tau_phi, tau_theta, tau_psi])

    # Step 4: body rates to euler angle rates
    euler_rates = euler_kinematics(phi, theta, [p, q, r])

    # Pack derivates in same order as state
    d_state = np.array([
        vx, vy, vz,          # position derivatives are velocities
        euler_rates[0],      # phi_dot
        euler_rates[1],      # theta_dot
        euler_rates[2],      # psi_dot
        acceleration[0],     # vx_dot
        acceleration[1],     # vy_dot
        acceleration[2],     # vz_dot
        alpha[0],           # p_dot
        alpha[1],           # q_dot
        alpha[2]            # r_dot
    ])

    return d_state

def run_simulation(initial_state, motor_speeds, t_start=0, t_end=5, dt=0.01):
    """
    Run sim from t_start to t_end
    
    initial_state  - 12-element state vector at t=0
    motor_speeds   - [w1, w2, w3, w4] constant motor speeds (rad/s)
    dt             - time step for output (seconds)
    """
    t_span = (t_start, t_end)
    t_eval = np.arange(t_start, t_end, dt)

    result = solve_ivp(
        fun=lambda t, s: state_derivative(t, s, motor_speeds),
        t_span=t_span,
        y0=initial_state,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6, atol=1e-6
    )

    return result 
