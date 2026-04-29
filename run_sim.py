import numpy as np
from quadcopter_sim.simulation import run_simulation
from quadcopter_sim.dynamics import MASS, GRAVITY, THRUST_COEFF

"""
Personal scratchpad for trying things out and running the simulation. 
"""

# Hover motor speed - solve kT*4*w^2 = mg for w, w - motor speed, assumed to be the same because of hover
hover_w = np.sqrt((MASS*GRAVITY)/(4*THRUST_COEFF))
print(f"Hover motor speed: {hover_w:.1f} rad/s")

# Intial state - start at 1 metre altitude, everything else zero
initial_state = np.zeros(12)
initial_state[2] = 1.0 # z = 1m (x,y,z) 2 corresponds to 3rd position in index

# Run for 3 seconds at hover thrust
result = run_simulation(initial_state, [hover_w]*4, t_end = 3) #initial_state, motor_speeds, t_start=0, t_end=5, dt=0.01 - if not mentaioned function will used default values  

# Print position at start and end 
print(f"Start position: x={result.y[0,0]:.3f}, y={result.y[1,0]:.3f}, z={result.y[2,0]:.3f}") #x,y,z for the first time step
print(f"End position:   x={result.y[0,-1]:.3f}, y={result.y[1,-1]:.3f}, z={result.y[2,-1]:.3f}") #x,y,z for the last timestep
print(f"Z drift:        {result.y[2,-1] - result.y[2,0]:.4f} metres") # subtract to see how much quadcopter drifted 
print(f"Number of timesteps: {result.y.shape[1]}")
print(f"Shape of result.y: {result.y.shape}")

# f means formatted string, {} means evaluate as python code and insert into text, :.3f, means format as a float with 3 decimal places
# y is the state vector solved by solve_ivp. 
# note result.y[0,0] means the first state variable and the first time step, result.y[0,-1] means the first state variable and the last time step.
