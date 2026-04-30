import numpy as np
import matplotlib.pyplot as plt
from quadcopter_sim.simulation import run_simulation
from quadcopter_sim.dynamics import MASS, GRAVITY, THRUST_COEFF
# import tools required for the sim

# ── Run two scenarios ──────────────────────────────────────────────────────

# Scenario 1: Hover at 1 metre
hover_w = np.sqrt((MASS*GRAVITY)/(4*THRUST_COEFF)) # motor speed for hover
initial_state = np.zeros(12) # all other state variables = 0
initial_state[2] = 1.0 # start at 1 metre

hover_result = run_simulation(initial_state, [hover_w]*4, t_end = 3)
#initial_state, motor_speeds, t_start=0, t_end=5, dt=0.01

# Scenario 2: free fall (motors off)
fall_result = run_simulation(initial_state, [0]*4, t_end = 3) # motor speed = 0

# ── Plot 1: Position over time ─────────────────────────────────────────────
fig, axes = plt.subplots(3,1,figsize = (10,8)) #creates one figure with 3 subplots stacked vertically (3 rows, 1 column)
fig.suptitle('Quadcopter Position - Hover Scenario', fontsize =14) #title

labels = ['X position (m)', 'Y position (m)', 'Z position (m)'] # each plot title
for i, (ax, label) in enumerate(zip(axes, labels)): #enumerate give the index i
    ax.plot(hover_result.t, hover_result.y[i], linewidth=2)
    ax.set_ylabel(label)
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=hover_result.y[i, 0], color='r', #refernce line 
               linestyle='--', alpha=0.5, label='Initial')
    ax.legend()

plt.tight_layout() #ensures subplots dont overlap 
plt.savefig('plots/position_hover.png', dpi=150)
plt.show()

# ── Plot 2: Attitude over time ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10,8))
fig.suptitle('Quadcopter Attitude - Hover Scenario', fontsize=14)

attitude_labels = ['Roll φ (rad)', 'Pitch θ (rad)', 'Yaw ψ (rad)']
for i, (ax, label) in enumerate(zip(axes, attitude_labels)):
    ax.plot(hover_result.t, hover_result.y[i+3], linewidth=2, color='orange') #now gives pitch, yaw and roll
    ax.set_ylabel(label)
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/attitude_hover.png', dpi=150)
plt.show()

# ── Plot 3: Free fall ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2,1,figsize = (10,6))
fig.suptitle('Quadcopter Free Fall - Motors Off', fontsize =14)

axes[0].plot(fall_result.t, fall_result.y[2], linewidth=2, color='red')
axes[0].set_ylabel('Z position (m)')
axes[0].set_xlabel('Time (s)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(fall_result.t, fall_result.y[8], linewidth=2, color='darkred') # z velocity over time should be a straight line sloping down to -9.81 m/s
axes[1].set_ylabel('Z velocity (m/s)')
axes[1].set_xlabel('Time (s)')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=-GRAVITY, color='gray'
                , linestyle='--', label=f'expected: {-GRAVITY} m/s²') # reference line for gravity 
axes[1].legend()

plt.tight_layout()
plt.savefig('plots/free_fall.png', dpi=150)
plt.show()

print("Plots saved to 'plots/' folder")

# ── Plot 4: 3D trajectory ──────────────────────────────────────────────────

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(hover_result.y[0], #x
         hover_result.y[1], #y
         hover_result.y[2], #z
         linewidth=2, color='purple', label='hover')

ax.plot(fall_result.y[0],
         fall_result.y[1],
         fall_result.y[2],
         linewidth=2, color='red', label='free fall')

# Mark start point 
ax.scatter([0], [0], [1], color = 'green', s=100, label='start', zorder=5)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Trajectory')
ax.legend()

plt.savefig('plots/3d_trajectory.png', dpi=150)
plt.show()
