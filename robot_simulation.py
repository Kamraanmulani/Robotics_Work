import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
import matplotlib.colors as mcolors

# Robot parameters
L1 = 1.0  # Length of link 1 (meters)
L2 = 0.8  # Length of link 2 (meters)
WORKSPACE_LIMITS = [0, 2, -1, 1]  # x_min, x_max, y_min, y_max

# Pick and place positions
PICK_POS = (0.4, 0.6)
PLACE_POS = (1.6, -0.6)

# Calculate inverse kinematics (2-link planar robot)
def inverse_kinematics(x, y):
    """Compute joint angles for given end-effector position (elbow-down solution)"""
    c2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    s2 = np.sqrt(1 - c2**2)
    theta2 = np.arctan2(s2, c2)
    
    k1 = L1 + L2 * c2
    k2 = L2 * s2
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    
    return theta1, theta2

# Forward kinematics
def forward_kinematics(theta1, theta2):
    """Compute joint positions from angles"""
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)
    return (x1, y1), (x2, y2)

# Generate trajectory (linear interpolation in joint space)
def generate_trajectory(start_pos, end_pos, steps=50):
    """Generate smooth trajectory between two points"""
    theta1_start, theta2_start = inverse_kinematics(*start_pos)
    theta1_end, theta2_end = inverse_kinematics(*end_pos)
    
    trajectory = []
    for t in np.linspace(0, 1, steps):
        theta1 = theta1_start + t * (theta1_end - theta1_start)
        theta2 = theta2_start + t * (theta2_end - theta2_start)
        _, end_effector = forward_kinematics(theta1, theta2)
        trajectory.append((theta1, theta2, end_effector))
    return trajectory

# ==========================================================================
# 1. Pick-and-Place Animation
# ==========================================================================
def create_animation():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(WORKSPACE_LIMITS[0], WORKSPACE_LIMITS[1])
    ax.set_ylim(WORKSPACE_LIMITS[2], WORKSPACE_LIMITS[3])
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title('Industrial Robot: Pick-and-Place Operation', fontsize=14)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    
    # Create work environment elements
    work_table = Rectangle((0.2, -0.1), 1.6, 0.1, color='gray', alpha=0.5)
    conveyor = Rectangle((1.7, -0.8), 0.3, 0.6, color='lightblue', alpha=0.7)
    ax.add_patch(work_table)
    ax.add_patch(conveyor)
    
    # Add labels
    ax.text(0.5, 0.7, "PICK AREA", fontsize=10, ha='center')
    ax.text(1.8, -0.9, "PLACE AREA", fontsize=10, ha='center')
    
    # Initialize robot components
    link1, = ax.plot([], [], 'o-', lw=4, color='dodgerblue', markersize=8)
    link2, = ax.plot([], [], 'o-', lw=4, color='royalblue', markersize=8)
    gripper = Circle((0, 0), radius=0.05, color='red')
    ax.add_patch(gripper)
    
    # Create object to be moved
    obj = Circle(PICK_POS, radius=0.04, color='gold')
    ax.add_patch(obj)
    
    # Generate trajectories
    pickup_traj = generate_trajectory((0, 0), PICK_POS)
    move_traj = generate_trajectory(PICK_POS, PLACE_POS)
    return_traj = generate_trajectory(PLACE_POS, (0, 0))
    full_trajectory = pickup_traj + move_traj + return_traj
    
    # Animation update function
    def update(frame):
        theta1, theta2, (ee_x, ee_y) = full_trajectory[frame]
        
        # Update robot position
        (x1, y1), (x2, y2) = forward_kinematics(theta1, theta2)
        link1.set_data([0, x1], [0, y1])
        link2.set_data([x1, x2], [y1, y2])
        gripper.center = (x2, y2)
        
        # Move object when gripper is at pick position
        if frame >= len(pickup_traj) and frame < len(pickup_traj) + len(move_traj):
            obj.center = (x2, y2)
            
        # Reset object after placement
        if frame == len(pickup_traj) + len(move_traj) - 1:
            obj.center = PLACE_POS
            
        return link1, link2, gripper, obj
    
    ani = FuncAnimation(fig, update, frames=len(full_trajectory), 
                        interval=30, blit=True)
    plt.tight_layout()
    return ani

# ==========================================================================
# 2. Workspace Visualization
# ==========================================================================
def visualize_workspace():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(WORKSPACE_LIMITS[0], WORKSPACE_LIMITS[1])
    ax.set_ylim(WORKSPACE_LIMITS[2], WORKSPACE_LIMITS[3])
    ax.set_aspect('equal')
    ax.set_title('Robot Workspace Reachability Map', fontsize=14)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.grid(True)
    
    # Generate workspace points
    points = []
    for theta1 in np.linspace(0, 2*np.pi, 100):
        for theta2 in np.linspace(-np.pi, np.pi, 50):
            _, (x, y) = forward_kinematics(theta1, theta2)
            if (WORKSPACE_LIMITS[0] <= x <= WORKSPACE_LIMITS[1] and 
                WORKSPACE_LIMITS[2] <= y <= WORKSPACE_LIMITS[3]):
                points.append((x, y))
    
    # Convert to array and plot with density-based coloring
    points = np.array(points)
    ax.hexbin(points[:, 0], points[:, 1], gridsize=30, cmap='viridis', 
              extent=WORKSPACE_LIMITS, alpha=0.85)
    
    # Add colorbar
    norm = mcolors.Normalize(vmin=0, vmax=50)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.9)
    cbar.set_label('Position Density', fontsize=10)
    
    # Mark key areas
    ax.plot(PICK_POS[0], PICK_POS[1], 'ro', markersize=8, label='Pick Position')
    ax.plot(PLACE_POS[0], PLACE_POS[1], 'go', markersize=8, label='Place Position')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig

# ==========================================================================
# Main Execution
# ==========================================================================
if __name__ == "__main__":
    # Create and display animation
    ani = create_animation()
    
    # Create and display workspace visualization
    workspace_fig = visualize_workspace()
    
    plt.show()