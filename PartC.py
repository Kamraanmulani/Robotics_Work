import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle, Polygon
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
from matplotlib import animation
import random
from IPython.display import HTML

# ==========================================================================
# 1. 4-DOF Robotic Arm Model (Revolute and Prismatic Joints)
# ==========================================================================
class FourDOFRobot:
    def __init__(self):
        # DH parameters: [a, alpha, d, theta]
        # Joint types: R = Revolute, P = Prismatic
        self.joints = [
            {'type': 'R', 'limits': [-np.pi/2, np.pi/2]},  # Base rotation
            {'type': 'R', 'limits': [-np.pi/3, np.pi/3]},  # Shoulder
            {'type': 'P', 'limits': [0, 0.5]},            # Elbow extension
            {'type': 'R', 'limits': [-np.pi/2, np.pi/2]}   # Wrist rotation
        ]
        self.dh_params = np.array([
            [0.0, 0, 0.5, 0],      # Link 1
            [0.4, -np.pi/2, 0, 0], # Link 2
            [0.0, np.pi/2, 0, 0],  # Link 3 (prismatic)
            [0.0, 0, 0.2, 0]       # Link 4
        ])
        self.home_position = [0, 0, 0.2, 0]
        self.current_angles = np.array(self.home_position)
        self.gripper_open = True
        
    def dh_matrix(self, a, alpha, d, theta):
        """Create a Denavit-Hartenberg transformation matrix"""
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        return np.array([
            [ct, -st*ca, st*sa, a*ct],
            [st, ct*ca, -ct*sa, a*st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])
    
    def forward_kinematics(self, joint_angles):
        """Compute the position of each joint and end effector"""
        T = np.eye(4)
        joint_positions = [T[:3, 3]]
        
        # Apply DH transformations for each joint
        for i in range(len(joint_angles)):
            a, alpha, d, theta = self.dh_params[i]
            joint_type = self.joints[i]['type']
            
            # Adjust parameters based on joint type
            if joint_type == 'R':
                theta += joint_angles[i]
            elif joint_type == 'P':
                d += joint_angles[i]
                
            T_i = self.dh_matrix(a, alpha, d, theta)
            T = T @ T_i
            joint_positions.append(T[:3, 3])
            
        return np.array(joint_positions)
    
    def inverse_kinematics(self, target_pos):
        """Simple inverse kinematics for target position (geometric approach)"""
        x, y, z = target_pos
        
        # Base rotation (joint0)
        theta0 = np.arctan2(y, x)
        
        # Adjust for arm reachability
        r = np.sqrt(x**2 + y**2)
        r = max(0.1, min(r, 0.7))  # Keep within reach
        
        # Shoulder angle (joint1)
        z_adjusted = z - 0.5  # Account for base height
        D = np.sqrt(r**2 + z_adjusted**2)
        theta1 = np.arctan2(z_adjusted, r)
        
        # Prismatic extension (joint2)
        extension = max(0, min(D - 0.4, 0.5))  # Min/max extension
        
        # Wrist rotation (joint3) - keep level
        theta3 = -theta1
        
        return np.array([theta0, theta1, extension, theta3])
    
    def set_joint_angles(self, angles):
        """Set joint angles with limit checking"""
        for i in range(len(angles)):
            low, high = self.joints[i]['limits']
            self.current_angles[i] = max(low, min(high, angles[i]))
    
    def move_to_position(self, target_pos):
        """Move end effector to target position"""
        angles = self.inverse_kinematics(target_pos)
        self.set_joint_angles(angles)
        return self.forward_kinematics(angles)[-1]

# ==========================================================================
# 2. Object Detection and Sorting System
# ==========================================================================
class ObjectDetector:
    def __init__(self):
        # Define color ranges for detection (HSV format)
        self.color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'green': ([40, 40, 40], [80, 255, 255]),
            'blue': ([100, 100, 100], [130, 255, 255]),
            'yellow': ([20, 100, 100], [40, 255, 255])
        }
        self.size_thresholds = {
            'small': 25,
            'medium': 40,
            'large': 60
        }
        
    def detect_objects(self, frame):
        """Detect colored objects in a frame and return their properties"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        
        objects = []
        
        # Detect objects for each color
        for color_name, (lower, upper) in self.color_ranges.items():
            # Create mask for color range
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Filter small contours
                if cv2.contourArea(contour) < 500:
                    continue
                
                # Get bounding circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                
                # Determine size category
                if radius < self.size_thresholds['small']:
                    size = 'small'
                elif radius < self.size_thresholds['medium']:
                    size = 'medium'
                else:
                    size = 'large'
                
                objects.append({
                    'position': (x, y),
                    'radius': radius,
                    'color': color_name,
                    'size': size,
                    'contour': contour
                })
        
        return objects

# ==========================================================================
# 3. Simulation Environment
# ==========================================================================
class RoboticSortingSimulation:
    def __init__(self):
        # Create robot and detector
        self.robot = FourDOFRobot()
        self.detector = ObjectDetector()
        
        # Simulation parameters
        self.conveyor_speed = 0.5
        self.objects = []
        self.sorted_count = {'red': 0, 'green': 0, 'blue': 0, 'yellow': 0}
        self.picked_object = None
        self.animation_speed = 50  # ms per frame
        
        # Create figure and axes
        self.fig = plt.figure(figsize=(14, 10))
        self.ax1 = self.fig.add_subplot(221, projection='3d')  # Robot view
        self.ax2 = self.fig.add_subplot(222)  # Camera view
        self.ax3 = self.fig.add_subplot(223)  # Sorting bins
        self.ax4 = self.fig.add_subplot(224)  # Statistics
        
        # Setup axes
        self.setup_axes()
        
        # Add slider for speed control
        ax_slider = plt.axes([0.25, 0.05, 0.65, 0.03])
        self.speed_slider = Slider(ax_slider, 'Speed', 10, 200, valinit=50)
        self.speed_slider.on_changed(self.update_speed)
        
        # Initialization
        self.camera_frame = self.generate_camera_frame()
        self.home_pos = self.robot.forward_kinematics(self.robot.home_position)
        
    def setup_axes(self):
        """Configure plot axes"""
        # Robot view (3D)
        self.ax1.set_title('4-DOF Robotic Arm')
        self.ax1.set_xlim([-1, 1])
        self.ax1.set_ylim([-1, 1])
        self.ax1.set_zlim([0, 1.2])
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        self.ax1.view_init(elev=30, azim=45)
        
        # Camera view
        self.ax2.set_title('Conveyor Belt Camera View')
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])
        
        # Sorting bins
        self.ax3.set_title('Sorting Bins')
        self.ax3.set_xlim([0, 4])
        self.ax3.set_ylim([0, 2])
        self.ax3.set_aspect('equal')
        self.ax3.set_xticks([])
        self.ax3.set_yticks([])
        
        # Draw bins
        bin_colors = ['red', 'green', 'blue', 'yellow']
        for i, color in enumerate(bin_colors):
            bin_rect = Rectangle((i+0.1, 0.1), 0.8, 0.8, color=color, alpha=0.3)
            self.ax3.add_patch(bin_rect)
            self.ax3.text(i+0.5, 0.5, f"{color.capitalize()}\n0", 
                         ha='center', va='center', fontsize=10)
        
        # Statistics
        self.ax4.set_title('Sorting Statistics')
        self.ax4.set_xticks([])
        self.ax4.set_yticks([])
        self.ax4.text(0.5, 0.9, "Objects Sorted: 0", 
                     ha='center', fontsize=12, fontweight='bold')
        
    def generate_camera_frame(self, width=640, height=480):
        """Generate synthetic camera frame with objects on conveyor"""
        # Create blank image
        frame = np.ones((height, width, 3), dtype=np.uint8) * 220
        
        # Draw conveyor belt
        cv2.rectangle(frame, (0, 200), (width, 280), (150, 150, 150), -1)
        for i in range(0, width, 30):
            cv2.line(frame, (i, 200), (i, 280), (100, 100, 100), 2)
        
        # Draw existing objects
        for obj in self.objects:
            x, y = obj['position']
            # Convert simulation position to camera coordinates
            cam_x = int(x * 100 + width/2)
            cam_y = int(240 - y * 100)
            radius = int(obj['radius'])
            color = self.color_to_bgr(obj['color'])
            
            # Draw object
            cv2.circle(frame, (cam_x, cam_y), radius, color, -1)
            cv2.circle(frame, (cam_x, cam_y), radius, (0, 0, 0), 2)
            
            # Draw object info
            cv2.putText(frame, f"{obj['size']}", (cam_x-20, cam_y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame
    
    def color_to_bgr(self, color_name):
        """Convert color name to BGR tuple"""
        colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255)
        }
        return colors.get(color_name, (0, 0, 0))
    
    def add_random_object(self):
        """Add a new random object to the conveyor"""
        color = random.choice(['red', 'green', 'blue', 'yellow'])
        size = random.choice(['small', 'medium', 'large'])
        radius = {
            'small': random.randint(15, 25),
            'medium': random.randint(30, 40),
            'large': random.randint(45, 55)
        }[size]
        
        self.objects.append({
            'position': (-1.0, random.uniform(-0.3, 0.3)),
            'radius': radius,
            'color': color,
            'size': size,
            'detected': False
        })
    
    def update_speed(self, val):
        """Update animation speed from slider"""
        self.animation_speed = val
    
    def update_conveyor(self):
        """Update conveyor belt movement"""
        # Move existing objects
        for obj in self.objects:
            x, y = obj['position']
            obj['position'] = (x + self.conveyor_speed * 0.02, y)
        
        # Remove objects that have passed the conveyor
        self.objects = [obj for obj in self.objects if obj['position'][0] < 1.5]
        
        # Randomly add new objects
        if random.random() < 0.05 and len(self.objects) < 5:
            self.add_random_object()
    
    def update_robot(self):
        """Update robot state and perform sorting actions"""
        # Detect objects in camera view
        detected_objects = self.detector.detect_objects(self.camera_frame)
        
        # Find closest undetected object
        target_object = None
        for obj in detected_objects:
            # Convert camera coordinates to simulation coordinates
            x, y = obj['position']
            sim_x = (x - 320) / 100
            sim_y = (240 - y) / 100
            
            # Only consider objects in picking range
            if 0.2 < sim_x < 0.8 and -0.3 < sim_y < 0.3:
                obj['sim_position'] = (sim_x, sim_y)
                obj['camera_position'] = (x, y)
                target_object = obj
                break
        
        # If we have a target and robot is idle
        if target_object and not self.picked_object:
            # Move to pick position
            sim_x, sim_y = target_object['sim_position']
            target_pos = (sim_x, sim_y, 0.02)  # Slightly above conveyor
            self.robot.move_to_position(target_pos)
            self.robot.gripper_open = False
            self.picked_object = target_object
            
            # Mark as picked in object list
            for obj in self.objects:
                dist = np.sqrt((obj['position'][0] - sim_x)**2 + 
                              (obj['position'][1] - sim_y)**2)
                if dist < 0.1:
                    obj['detected'] = True
                    break
        
        # If robot has picked an object
        elif self.picked_object:
            # Move to bin position
            color = self.picked_object['color']
            bin_positions = {
                'red': (0.5, 0.8, 0.1),
                'green': (1.5, 0.8, 0.1),
                'blue': (2.5, 0.8, 0.1),
                'yellow': (3.5, 0.8, 0.1)
            }
            bin_x, bin_y, bin_z = bin_positions[color]
            
            # Move above bin
            self.robot.move_to_position((bin_x, bin_y, bin_z + 0.2))
            
            # Move down to bin
            self.robot.move_to_position((bin_x, bin_y, bin_z))
            
            # Release object
            self.robot.gripper_open = True
            self.sorted_count[color] += 1
            self.picked_object = None
            
            # Move back to home position
            self.robot.set_joint_angles(self.robot.home_position)
    
    def update(self, frame):
        """Update simulation state for animation frame"""
        # Update conveyor
        self.update_conveyor()
        
        # Update camera frame
        self.camera_frame = self.generate_camera_frame()
        
        # Update robot and sorting logic
        self.update_robot()
        
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # Reapply axis settings
        self.setup_axes()
        
        # ==================================================================
        # Draw robot in 3D view
        # ==================================================================
        joint_positions = self.robot.forward_kinematics(self.robot.current_angles)
        
        # Draw links
        for i in range(len(joint_positions) - 1):
            x = [joint_positions[i][0], joint_positions[i+1][0]]
            y = [joint_positions[i][1], joint_positions[i+1][1]]
            z = [joint_positions[i][2], joint_positions[i+1][2]]
            self.ax1.plot(x, y, z, 'o-', lw=3, markersize=8, 
                         color=plt.cm.viridis(i/4))
        
        # Draw gripper
        gripper_pos = joint_positions[-1]
        gripper_size = 0.05
        if self.robot.gripper_open:
            # Draw open gripper
            self.ax1.plot([gripper_pos[0]], [gripper_pos[1]], [gripper_pos[2]], 
                         's', markersize=10, color='black')
        else:
            # Draw closed gripper with picked object
            if self.picked_object:
                color = self.picked_object['color']
                self.ax1.plot([gripper_pos[0]], [gripper_pos[1]], [gripper_pos[2]], 
                             'o', markersize=12, color=color)
        
        # Draw conveyor in 3D view
        conveyor_x = np.array([-1, 1, 1, -1])
        conveyor_y = np.array([-0.4, -0.4, 0.4, 0.4])
        conveyor_z = np.array([0, 0, 0, 0])
        self.ax1.plot_trisurf(conveyor_x, conveyor_y, conveyor_z, color='gray', alpha=0.3)
        
        # Draw objects on conveyor
        for obj in self.objects:
            x, y = obj['position']
            self.ax1.plot([x], [y], [0.03], 'o', markersize=obj['radius']/5, 
                         color=obj['color'])
        
        # ==================================================================
        # Draw camera view
        # ==================================================================
        self.ax2.imshow(cv2.cvtColor(self.camera_frame, cv2.COLOR_BGR2RGB))
        
        # Draw detection information
        detected_objects = self.detector.detect_objects(self.camera_frame)
        for obj in detected_objects:
            x, y = obj['position']
            cv2.circle(self.camera_frame, (int(x), int(y)), 5, (255, 0, 0), 2)
            cv2.putText(self.camera_frame, 
                       f"{obj['color']} {obj['size']}", 
                       (int(x) - 30, int(y) - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # ==================================================================
        # Draw sorting bins
        # ==================================================================
        bin_colors = ['red', 'green', 'blue', 'yellow']
        for i, color in enumerate(bin_colors):
            bin_rect = Rectangle((i+0.1, 0.1), 0.8, 0.8, color=color, alpha=0.3)
            self.ax3.add_patch(bin_rect)
            count = self.sorted_count[color]
            self.ax3.text(i+0.5, 0.5, f"{color.capitalize()}\n{count}", 
                         ha='center', va='center', fontsize=10)
        
        # Draw objects in bins (for visualization)
        for i, color in enumerate(bin_colors):
            count = min(5, self.sorted_count[color])  # Max 5 visible objects
            for j in range(count):
                self.ax3.plot(i+0.3 + j*0.1, 0.3, 'o', markersize=8, 
                             color=color, alpha=0.7)
        
        # ==================================================================
        # Draw statistics
        # ==================================================================
        total_sorted = sum(self.sorted_count.values())
        self.ax4.text(0.5, 0.8, f"Objects Sorted: {total_sorted}", 
                     ha='center', fontsize=14, fontweight='bold')
        
        # Draw pie chart of sorted objects
        if total_sorted > 0:
            sizes = [self.sorted_count[c] for c in bin_colors]
            self.ax4.pie(sizes, labels=bin_colors, autopct='%1.1f%%', 
                        colors=bin_colors, startangle=90)
        
        return self.ax1, self.ax2, self.ax3, self.ax4
    
    def run_simulation(self):
        """Run the simulation animation"""
        # Add initial objects
        for _ in range(3):
            self.add_random_object()
        
        # Create animation
        ani = FuncAnimation(self.fig, self.update, frames=100, 
                           interval=self.animation_speed, blit=False)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.show()
        
        # To save the animation (uncomment if needed)
        # ani.save('robotic_sorting_simulation.mp4', writer='ffmpeg', fps=30, dpi=150)
        
        return ani

# ==========================================================================
# Run the simulation
# ==========================================================================
if __name__ == "__main__":
    sim = RoboticSortingSimulation()
    sim.run_simulation()