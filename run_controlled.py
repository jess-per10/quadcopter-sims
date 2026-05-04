import numpy as np
import matplotlib.pyplot as plt
from quadcopter_sim.closed_loop import run_controlled_simulation

# Start with a 15 degree roll disturbance - controller should fix it

initial_state = np.zeros(12)
initial_state[2] = 1.0 # 1 m altitude
initial_state[3] = np.radians(15) # 15 degree roll

# Target: level hover
setpoints = {'roll':0.0, 'pitch':0.0, 'yaw': 0.0, 'attitude': 1.0}

t, states = run_controlled_simulation(initial_state, setpoints, t_end=5.0)

# ------------------- Plot attitude recover -------------------
fig, axes = plt.subplots(3,1,figsize=(10,8))
fig.suptitle('Attitude Control — Recovery from 15° Roll Disturbance', fontsize=13)

labels = ['Roll φ (deg)', 'Pitch θ (deg)', 'Yaw ψ (deg)']
for i, (ax, label) in enumerate(zip(axes, labels)):
    ax.plot(t, np.degrees(states[i+3]), linewidth=2)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='setpoint')
    ax.set_ylabel(label)
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig('plots/attitude_control.png', dpi = 150)
plt.show()

# -------------------Plot altitude-------------------
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(t, states[2], linewidth=2, color='purple')
ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='setpoint: 1m')
ax.set_ylabel('Altitude (m)')
ax.set_xlabel('Time (s)')
ax.set_title('Altitude During Attitude Recovery')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('plots/altitude_control.png', dpi=150)
plt.show()

# Print some stats
print(f"Initial roll: {np.degrees(initial_state[3]):.1f} deg")
print(f"Final roll:   {np.degrees(states[3,-1]):.3f} deg")
print(f"Settling time: check the plot - when does roll cross 0?")