import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Define rotation matrix functions (angles in degrees)
def rotation_matrix_x(theta):
    theta_rad = np.radians(theta)
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta_rad), -np.sin(theta_rad)],
        [0, np.sin(theta_rad), np.cos(theta_rad)]
    ])

def rotation_matrix_y(theta):
    theta_rad = np.radians(theta)
    return np.array([
        [np.cos(theta_rad), 0, np.sin(theta_rad)],
        [0, 1, 0],
        [-np.sin(theta_rad), 0, np.cos(theta_rad)]
    ])

def rotation_matrix_z(theta):
    theta_rad = np.radians(theta)
    return np.array([
        [np.cos(theta_rad), -np.sin(theta_rad), 0],
        [np.sin(theta_rad), np.cos(theta_rad), 0],
        [0, 0, 1]
    ])

# Initial position
initial_position = np.array([2, 3, 4])
rot_x = 30
rot_y = 45
rot_z = 60

# Compute individual rotation matrices
R_x = rotation_matrix_x(rot_x)
R_y = rotation_matrix_y(rot_y)
R_z = rotation_matrix_z(rot_z)

# Composite rotation (applied in order: X -> Y -> Z)
R_composite = R_z @ R_y @ R_x

# Final position
final_position = R_composite @ initial_position

print("Rotation matrix about X (30°):\n", R_x)
print("\nRotation matrix about Y (45°):\n", R_y)
print("\nRotation matrix about Z (60°):\n", R_z)
print("\nComposite rotation matrix (Z*Y*X):\n", R_composite)
print("\nInitial position:", initial_position)
print("Final position:", final_position)

# Animation setup
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Robotic Arm Rotation: 30° X → 45° Y → 60° Z')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([-2, 5])
ax.set_ylim([-2, 5])
ax.set_zlim([-2, 5])
ax.grid(True)

# Coordinate system and labels
ax.quiver(0, 0, 0, 4, 0, 0, color='r', alpha=0.3, label='X')
ax.quiver(0, 0, 0, 0, 4, 0, color='g', alpha=0.3, label='Y')
ax.quiver(0, 0, 0, 0, 0, 4, color='b', alpha=0.3, label='Z')

# Arm representation (line from origin to end-effector)
arm_line, = ax.plot([0, initial_position[0]], [0, initial_position[1]], [0, initial_position[2]], 
                   'o-', color='purple', linewidth=3, markersize=6, label='Arm')
end_effector = ax.scatter([], [], [], s=100, c='gold', marker='*', label='End-Effector')

# Trajectory tracking
trajectory_line, = ax.plot([], [], [], 'm--', linewidth=1.5, label='Trajectory')
trajectory_x, trajectory_y, trajectory_z = [], [], []

# Rotation progress text
progress_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

def init_animation():
    arm_line.set_data([0, initial_position[0]], [0, initial_position[1]])
    arm_line.set_3d_properties([0, initial_position[2]])
    end_effector._offsets3d = ([initial_position[0]], [initial_position[1]], [initial_position[2]])
    trajectory_line.set_data([], [])
    trajectory_line.set_3d_properties([])
    trajectory_x.append(initial_position[0])
    trajectory_y.append(initial_position[1])
    trajectory_z.append(initial_position[2])
    progress_text.set_text('Initial Position')
    return arm_line, end_effector, trajectory_line, progress_text

def update_animation(frame):
    # Interpolate rotation angles (0 to 100%)
    t = frame / 100
    current_rot_x = t * rot_x if t <= 1/3 else rot_x
    current_rot_y = max(0, min(rot_y, (t - 1/3) * 3 * rot_y)) if t > 1/3 else 0
    current_rot_z = max(0, min(rot_z, (t - 2/3) * 3 * rot_z)) if t > 2/3 else 0
    
    # Compute current rotation matrix
    R_current = rotation_matrix_z(current_rot_z) @ rotation_matrix_y(current_rot_y) @ rotation_matrix_x(current_rot_x)
    current_position = R_current @ initial_position
    
    # Update arm line
    arm_line.set_data([0, current_position[0]], [0, current_position[1]])
    arm_line.set_3d_properties([0, current_position[2]])
    
    # Update end effector position
    end_effector._offsets3d = ([current_position[0]], [current_position[1]], [current_position[2]])
    
    # Update trajectory
    trajectory_x.append(current_position[0])
    trajectory_y.append(current_position[1])
    trajectory_z.append(current_position[2])
    trajectory_line.set_data(trajectory_x, trajectory_y)
    trajectory_line.set_3d_properties(trajectory_z)
    
    # Update status text
    if t <= 1/3:
        progress_text.set_text(f'Rotating X: {current_rot_x:.1f}°')
    elif t <= 2/3:
        progress_text.set_text(f'Rotating Y: {current_rot_y:.1f}°')
    else:
        progress_text.set_text(f'Rotating Z: {current_rot_z:.1f}°')
    
    return arm_line, end_effector, trajectory_line, progress_text

# Create animation
ani = FuncAnimation(fig, update_animation, frames=100, init_func=init_animation,
                    interval=50, blit=True, repeat_delay=2000)

# Add legend and adjust view
ax.legend(loc='upper right')
ax.view_init(elev=20, azim=15)

plt.tight_layout()
plt.show()