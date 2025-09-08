#!/usr/bin/env python3
"""
ArUco Marker Landing Controller for UAV
========================================
This node provides autonomous landing capabilities on ArUco markers for UAVs.
It detects ArUco markers during flight, stores their world positions, and enables
the drone to return to any previously detected marker for inspection or landing.

Key Features:
- Continuous ArUco marker detection from camera feed
- Conversion of pixel coordinates to world coordinates using drone pose
- Permanent storage of detected marker positions
- Multi-phase landing sequence with safety checks
- Manual or automatic flight modes

Author: [Your Name]
Date: 2024
Version: 1.2
"""

import rospy
import actionlib
from actionlib_msgs.msg import GoalStatus
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped, Point, TransformStamped
from std_msgs.msg import Int32, Bool
from cv_bridge import CvBridge, CvBridgeError
from spar_msgs.msg import FlightMotionAction, FlightMotionGoal
from visualization_msgs.msg import Marker, MarkerArray


class ArUcoLandingController:
    """
    Main controller class for ArUco-based landing system.
    
    This class handles:
    - ArUco marker detection from camera images
    - Coordinate transformation from pixel to world frame
    - Storage and management of detected markers
    - Flight control for approaching and landing on markers
    - Visualization for RViz
    """
    
    def __init__(self):
        """Initialize the ArUco landing controller with all necessary parameters and connections."""
        rospy.init_node('aruco_landing_controller', anonymous=True)
        
        # ========== CONFIGURATION PARAMETERS ==========
        # These parameters can be adjusted in the launch file
        
        # Flight behavior parameters
        self.auto_fly_on_select = rospy.get_param("~auto_fly_on_select", False)  # Auto fly when marker selected
        self.fly_altitude = rospy.get_param("~fly_altitude", 2.0)  # Default altitude when flying to marker (m)
        
        # Camera offset from drone center (meters)
        # Adjust if camera is not mounted at drone's center
        self.camera_frame_offset_x = rospy.get_param("~camera_offset_x", 0.0)
        self.camera_frame_offset_y = rospy.get_param("~camera_offset_y", 0.0)
        
        # Landing sequence parameters
        self.landing_height = rospy.get_param("~landing_height", 0.15)  # Final hover height before touchdown (m)
        self.touchdown_enabled = rospy.get_param("~touchdown_enabled", False)  # Enable actual ground touchdown
        self.approach_height = rospy.get_param("~approach_height", 0.5)  # Initial approach altitude (m)
        
        # Flight velocity parameters
        self.vel_linear = rospy.get_param("~vel_linear", 0.2)  # Horizontal velocity (m/s)
        self.vel_yaw = rospy.get_param("~vel_yaw", 0.2)  # Yaw rate (rad/s)
        
        # Accuracy thresholds for waypoint convergence
        self.accuracy_pos = rospy.get_param("~acc_pos", 0.1)  # Position accuracy (m)
        self.accuracy_yaw = rospy.get_param("~acc_yaw", 0.1)  # Yaw accuracy (rad)
        
        # Camera parameters for OakD (IMX378 Fixed-Focus variant)
        # These are used for pixel-to-world coordinate transformation
        self.camera_fov_h = rospy.get_param("~camera_fov_h", 69.4)  # Horizontal FOV (degrees)
        self.camera_fov_v = rospy.get_param("~camera_fov_v", 42.5)  # Vertical FOV (degrees)
        self.image_width = rospy.get_param("~image_width", 640)  # Image width (pixels)
        self.image_height = rospy.get_param("~image_height", 480)  # Image height (pixels)
        
        # ========== ARUCO DETECTION SETUP ==========
        # Using 4x4 ArUco dictionary with IDs 0-99
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        # ========== STATE MANAGEMENT ==========
        # Core state variables for tracking markers and flight status
        self.current_pose = None  # Current drone pose from motion capture
        self.detected_markers = {}  # Currently visible markers {id: (pixel_x, pixel_y, timestamp)}
        self.stored_marker_positions = {}  # All ever-detected markers {id: (world_x, world_y)}
        self.selected_marker_id = None  # Currently selected marker for operations
        self.selected_marker_world_pos = None  # World position of selected marker
        self.landing_active = False  # Flag indicating if landing sequence is active
        self.marker_timeout = rospy.Duration(2.0)  # How long to keep "currently visible" status
        self.last_status_line = ""  # For single-line console updates
        
        # CV Bridge
        self.br = CvBridge()
        
        # TF2 for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Action client for SPAR
        action_ns = rospy.get_param("~action_topic", 'spar/flight')
        self.spar_client = actionlib.SimpleActionClient(action_ns, FlightMotionAction)
        rospy.loginfo("Waiting for SPAR action server...")
        self.spar_client.wait_for_server()
        rospy.loginfo("SPAR action server connected!")
        
        # Publishers
        self.marker_viz_pub = rospy.Publisher('/aruco_landing/markers', MarkerArray, queue_size=10)
        self.processed_image_pub = rospy.Publisher('/aruco_landing/image/compressed', CompressedImage, queue_size=10)
        # Also publish to the original ArUco topic for compatibility
        self.aruco_pub = rospy.Publisher('/processed_aruco/image/compressed', CompressedImage, queue_size=10)
        
        # Subscribers
        self.frame_sub = rospy.Subscriber('/depthai_node/image/compressed', CompressedImage, self.img_callback)
        self.pose_sub = rospy.Subscriber('/uavasr/pose', PoseStamped, self.pose_callback)
        self.select_marker_sub = rospy.Subscriber('/aruco_landing/select_marker', Int32, self.select_marker_callback)
        self.start_landing_sub = rospy.Subscriber('/aruco_landing/start_landing', Bool, self.start_landing_callback)
        self.goto_marker_sub = rospy.Subscriber('/aruco_landing/goto_marker', Bool, self.goto_marker_callback)
        
        # Timer for marker visualization
        self.viz_timer = rospy.Timer(rospy.Duration(0.1), self.publish_marker_visualization)
        
        rospy.loginfo("ArUco Landing Controller initialized!")
        rospy.loginfo("Processing images...")
        rospy.loginfo("Available topics:")
        rospy.loginfo("  - /aruco_landing/select_marker (Int32): Select marker ID")
        rospy.loginfo("  - /aruco_landing/goto_marker (Bool): Fly to selected marker at 2m")
        rospy.loginfo("  - /aruco_landing/start_landing (Bool): Full landing sequence")
        rospy.loginfo("-" * 50)

    def pose_callback(self, msg):
        """
        Callback for drone pose updates from motion capture system.
        
        Args:
            msg: PoseStamped message containing current drone position and orientation
        """
        self.current_pose = msg
    
    def img_callback(self, msg_in):
        """
        Main image processing callback - handles ArUco detection.
        
        This is called whenever a new compressed image is received from the camera.
        It detects ArUco markers, updates tracking, and publishes processed images.
        
        Args:
            msg_in: CompressedImage message from camera
        """
        try:
            # Convert compressed image to CV2
            frame = self.br.compressed_imgmsg_to_cv2(msg_in)
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        
        # Find ArUco markers
        frame = self.find_aruco(frame)
        
        # Publish processed frame to both topics
        self.publish_to_ros(frame)
    
    def find_aruco(self, frame):
        """
        Core ArUco detection and visualization function.
        
        Detects ArUco markers in the image, stores their positions, and adds
        visualization overlays (bounding boxes, IDs, coordinates).
        
        Args:
            frame: OpenCV image array to process
            
        Returns:
            frame: Processed image with visualization overlays
        """
        # Detect ArUco markers using OpenCV
        (corners, ids, _) = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.aruco_params)
        
        # Clean up old detections that have timed out
        # This maintains a list of "currently visible" markers
        current_time = rospy.Time.now()
        self.detected_markers = {k: v for k, v in self.detected_markers.items() 
                                if (current_time - v[2]) < self.marker_timeout}
        
        # Process each detected marker
        if len(corners) > 0:
            ids = ids.flatten()
            
            for (marker_corner, marker_ID) in zip(corners, ids):
                # Extract corner points
                corners_reshape = marker_corner.reshape((4, 2))
                (top_left, top_right, bottom_right, bottom_left) = corners_reshape
                
                # Calculate center point in pixel coordinates
                center_x = int((top_left[0] + bottom_right[0]) / 2)
                center_y = int((top_left[1] + bottom_right[1]) / 2)
                
                # Store in currently visible markers
                self.detected_markers[int(marker_ID)] = (center_x, center_y, current_time)
                
                # Calculate and permanently store world position
                # This ensures we can return to markers even after they leave view
                if self.current_pose:
                    world_pos = self.pixel_to_world(center_x, center_y)
                    if world_pos:
                        self.stored_marker_positions[int(marker_ID)] = world_pos
                
                # Convert corners to integer coordinates
                top_right = (int(top_right[0]), int(top_right[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
                top_left = (int(top_left[0]), int(top_left[1]))
                
                # Determine color based on selection
                if marker_ID == self.selected_marker_id:
                    color = (0, 255, 0)  # Green for selected
                    line_thickness = 3
                else:
                    color = (0, 255, 0)  # Keep green for compatibility with original
                    line_thickness = 2
                
                # Draw the marker boundary (original style)
                cv2.line(frame, top_left, top_right, color, line_thickness)
                cv2.line(frame, top_right, bottom_right, color, line_thickness)
                cv2.line(frame, bottom_right, bottom_left, color, line_thickness)
                cv2.line(frame, bottom_left, top_left, color, line_thickness)
                
                # Draw the center point
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Display coordinates
                coord_text = f"({center_x}, {center_y})"
                cv2.putText(frame, coord_text, (center_x + 10, center_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Don't log individual detections anymore - we'll do a summary later
                
                # Display marker ID
                cv2.putText(frame, str(marker_ID), 
                          (top_left[0], top_right[1] - 15), 
                          cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
                
                # If this is selected marker, show world coordinates
                if marker_ID == self.selected_marker_id and self.current_pose:
                    world_pos = self.pixel_to_world(center_x, center_y)
                    if world_pos:
                        cv2.putText(frame, f"World: ({world_pos[0]:.2f}, {world_pos[1]:.2f})",
                                  (center_x + 10, center_y + 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Add status text for landing controller
        if self.selected_marker_id or self.landing_active:
            status_text = f"Landing Controller - "
            if self.selected_marker_id:
                status_text += f"Selected: {self.selected_marker_id}"
            if self.landing_active:
                status_text += " | LANDING ACTIVE"
            cv2.putText(frame, status_text, (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # After processing all markers, print single-line status update
        self.print_status_update()
        
        return frame
    
    def print_status_update(self):
        """Print a single-line status update instead of spamming"""
        # Build status string
        currently_visible = list(self.detected_markers.keys())
        stored_total = list(self.stored_marker_positions.keys())
        
        if currently_visible:
            visible_str = f"Visible: {currently_visible}"
        else:
            visible_str = "Visible: none"
        
        status = f"\r[ArUco] {visible_str} | Total stored: {len(stored_total)} markers {stored_total} | Selected: {self.selected_marker_id}"
        
        # Only print if status changed
        if status != self.last_status_line:
            # Use \r to overwrite the same line
            import sys
            sys.stdout.write('\r' + ' ' * 120)  # Clear line
            sys.stdout.write(status)
            sys.stdout.flush()
            self.last_status_line = status
    
    def publish_to_ros(self, frame):
        """Publish processed frame to ROS topics"""
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.format = "jpeg"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
        
        # Publish to both topics for compatibility
        self.aruco_pub.publish(msg_out)
        self.processed_image_pub.publish(msg_out)
    
    def pixel_to_world(self, pixel_x, pixel_y):
        """
        Convert pixel coordinates to world coordinates.
        
        This function projects a pixel position to the ground plane (z=0) using:
        1. Camera field of view parameters
        2. Current drone height
        3. Camera mounting offset
        
        Args:
            pixel_x: X coordinate in image (0 to image_width)
            pixel_y: Y coordinate in image (0 to image_height)
            
        Returns:
            tuple: (world_x, world_y) position of the marker, or None if pose unavailable
        """
        if not self.current_pose:
            return None
        
        # Get current drone height above ground
        drone_height = self.current_pose.pose.position.z
        
        # Convert pixel coordinates to normalized coordinates (-1 to 1)
        # This maps the image plane to angular offsets
        norm_x = (2.0 * pixel_x / self.image_width) - 1.0
        norm_y = -((2.0 * pixel_y / self.image_height) - 1.0)  # Flip Y axis for conventional coordinates
        
        # Calculate angular offset from camera center
        # Based on camera field of view
        angle_x = np.radians(self.camera_fov_h / 2.0) * norm_x
        angle_y = np.radians(self.camera_fov_v / 2.0) * norm_y
        
        # Project to ground plane using basic trigonometry
        # Assumes marker is on the ground (z=0)
        ground_offset_x = drone_height * np.tan(angle_x)
        ground_offset_y = drone_height * np.tan(angle_y)
        
        # Add drone position and account for camera mounting offset
        world_x = self.current_pose.pose.position.x + ground_offset_x + self.camera_frame_offset_x
        world_y = self.current_pose.pose.position.y + ground_offset_y + self.camera_frame_offset_y
        
        return (world_x, world_y)
    
    def select_marker_callback(self, msg):
        """Handle marker selection"""
        marker_id = msg.data
        
        # Check if we have this marker in our permanent storage
        if marker_id in self.stored_marker_positions:
            self.selected_marker_id = marker_id
            self.selected_marker_world_pos = self.stored_marker_positions[marker_id]
            rospy.loginfo(f"\n=== Selected ArUco marker ID: {marker_id} ===")
            rospy.loginfo(f"Stored position: ({self.selected_marker_world_pos[0]:.2f}, {self.selected_marker_world_pos[1]:.2f}, 0.00)")
            
            # Check if currently visible
            if marker_id in self.detected_markers:
                rospy.loginfo("Marker is currently visible")
            else:
                rospy.loginfo("Marker not currently visible, using stored position from earlier detection")
            
            # Auto fly if enabled
            if self.auto_fly_on_select:
                rospy.loginfo(f"Auto-fly enabled: Flying to marker at {self.fly_altitude}m altitude")
                self.send_waypoint(self.selected_marker_world_pos[0], 
                                 self.selected_marker_world_pos[1], 
                                 self.fly_altitude, 0.0)
            else:
                rospy.loginfo("To fly to this marker, use:")
                rospy.loginfo("  rostopic pub /aruco_landing/goto_marker std_msgs/Bool 'data: true'")
                rospy.loginfo("OR for landing sequence:")
                rospy.loginfo("  rostopic pub /aruco_landing/start_landing std_msgs/Bool 'data: true'")
        else:
            rospy.logwarn(f"\nMarker ID {marker_id} has never been detected!")
            rospy.loginfo(f"Stored markers: {list(self.stored_marker_positions.keys())}")
            rospy.loginfo(f"Currently visible markers: {list(self.detected_markers.keys())}")
    
    def goto_marker_callback(self, msg):
        """Fly to selected marker position at 2m altitude (like ROI diversion)"""
        if msg.data:
            if self.selected_marker_id and self.selected_marker_world_pos:
                rospy.loginfo(f"\n=== Flying to marker {self.selected_marker_id} ===")
                world_x, world_y = self.selected_marker_world_pos
                
                # Fly to marker position at 2m altitude
                rospy.loginfo(f"Going to position ({world_x:.2f}, {world_y:.2f}, 2.00)")
                self.send_waypoint(world_x, world_y, 2.0, 0.0)
                
                # Don't wait for result - let it fly in background
                rospy.loginfo("Command sent. Drone flying to marker position.")
            else:
                rospy.logwarn("\nCannot go to marker: No marker selected or position unknown!")
                rospy.loginfo("First select a marker with: rostopic pub /aruco_landing/select_marker std_msgs/Int32 'data: ID'")
    
    def start_landing_callback(self, msg):
        """Start or stop landing sequence"""
        if msg.data and not self.landing_active:
            if self.selected_marker_id and self.selected_marker_world_pos:
                self.landing_active = True
                rospy.loginfo("Starting landing sequence...")
                self.execute_landing()
            else:
                rospy.logwarn("Cannot start landing: No marker selected or marker position unknown!")
                if self.selected_marker_id:
                    rospy.loginfo("Please ensure marker is visible to select it properly")
        elif not msg.data and self.landing_active:
            self.landing_active = False
            self.spar_client.cancel_goal()
            rospy.loginfo("Landing cancelled!")
    
    def execute_landing(self):
        """
        Execute the multi-phase landing sequence on selected marker.
        
        Landing phases:
        1. Navigate to stored marker position at approach height (0.5m)
        2. Refine position using visual feedback if marker visible
        3. Descend to hover height (0.15m)
        4. (Optional) Complete touchdown to ground if enabled
        
        The sequence can be aborted at any time by setting landing_active to False.
        """
        if not self.landing_active:
            return
        
        rospy.loginfo(f"Executing landing on marker {self.selected_marker_id}")
        
        # Use the stored world position from when marker was selected
        if self.selected_marker_world_pos:
            world_x, world_y = self.selected_marker_world_pos
            
            # Phase 1: Move above the stored marker position at approach height
            rospy.loginfo(f"Phase 1: Moving to position ({world_x:.2f}, {world_y:.2f}) above marker at {self.approach_height}m")
            self.send_waypoint(world_x, world_y, self.approach_height, 0.0)
            
            # Wait for completion
            result = self.spar_client.wait_for_result(rospy.Duration(30.0))
            if not result or self.spar_client.get_state() != GoalStatus.SUCCEEDED:
                rospy.logwarn("Failed to reach approach position!")
                self.landing_active = False
                return
            
            rospy.loginfo("Reached position above marker, checking for marker visibility...")
            
            # Phase 2: Now that we're above, try to detect and refine
            rospy.sleep(rospy.Duration(0.5))  # Let camera stabilize
            
            # Check if we can see the marker now
            if self.selected_marker_id in self.detected_markers:
                pixel_x, pixel_y, _ = self.detected_markers[self.selected_marker_id]
                refined_pos = self.pixel_to_world(pixel_x, pixel_y)
                
                if refined_pos:
                    rospy.loginfo(f"Phase 2: Marker detected! Refining position to ({refined_pos[0]:.2f}, {refined_pos[1]:.2f})")
                    self.send_waypoint(refined_pos[0], refined_pos[1], self.approach_height, 0.0)
                    
                    result = self.spar_client.wait_for_result(rospy.Duration(20.0))
                    if not result or self.spar_client.get_state() != GoalStatus.SUCCEEDED:
                        rospy.logwarn("Failed to refine position!")
                        self.landing_active = False
                        return
                    
                    # Update position for final descent
                    world_x, world_y = refined_pos
                else:
                    rospy.loginfo("Marker visible but couldn't refine position, using original position")
            else:
                rospy.logwarn(f"Marker {self.selected_marker_id} not visible from above, proceeding with original position")
            
            # Phase 3: Descend to landing height
            rospy.loginfo(f"Phase 3: Descending to hover height {self.landing_height}m at ({world_x:.2f}, {world_y:.2f})")
            self.send_waypoint(world_x, world_y, self.landing_height, 0.0, vel_vertical=0.1)  # Slower descent
            
            result = self.spar_client.wait_for_result(rospy.Duration(30.0))
            if result and self.spar_client.get_state() == GoalStatus.SUCCEEDED:
                rospy.loginfo(f"Reached hover altitude {self.landing_height}m")
                
                # Phase 4: Optional touchdown
                if self.touchdown_enabled:
                    rospy.loginfo("Phase 4: Executing touchdown...")
                    # Send land command using SPAR's landing motion
                    goal = FlightMotionGoal()
                    goal.motion = FlightMotionGoal.MOTION_LAND
                    goal.velocity_vertical = 0.1  # Very slow touchdown
                    self.spar_client.send_goal(goal)
                    
                    result = self.spar_client.wait_for_result(rospy.Duration(30.0))
                    if result and self.spar_client.get_state() == GoalStatus.SUCCEEDED:
                        rospy.loginfo("Touchdown complete! Drone has landed.")
                    else:
                        rospy.logwarn("Touchdown failed!")
                else:
                    rospy.loginfo("Hovering at landing height. Touchdown disabled for safety.")
                    rospy.loginfo("To enable touchdown, set 'touchdown_enabled' param to true")
            else:
                rospy.logwarn("Failed to reach landing height!")
        else:
            rospy.logerr("No stored world position for selected marker!")
        
        self.landing_active = False
    
    def send_waypoint(self, x, y, z, yaw, vel_horizontal=None, vel_vertical=None):
        """
        Send a waypoint command to SPAR flight controller.
        
        Args:
            x: Target X position in world frame (m)
            y: Target Y position in world frame (m)
            z: Target Z position (altitude) (m)
            yaw: Target yaw angle (rad)
            vel_horizontal: Optional horizontal velocity override (m/s)
            vel_vertical: Optional vertical velocity override (m/s)
        """
        goal = FlightMotionGoal()
        goal.motion = FlightMotionGoal.MOTION_GOTO
        goal.position.x = x
        goal.position.y = y
        goal.position.z = z
        goal.yaw = yaw
        goal.velocity_horizontal = vel_horizontal if vel_horizontal else self.vel_linear
        goal.velocity_vertical = vel_vertical if vel_vertical else self.vel_linear
        goal.yawrate = self.vel_yaw
        goal.wait_for_convergence = True
        goal.position_radius = self.accuracy_pos
        goal.yaw_range = self.accuracy_yaw
        
        self.spar_client.send_goal(goal)
    
    def publish_marker_visualization(self, event):
        """
        Publish marker positions for RViz visualization.
        
        Creates cylinder markers at detected ArUco positions and text labels.
        Selected markers appear in green, others in red.
        
        Args:
            event: Timer event (unused but required by ROS timer callback)
        """
        marker_array = MarkerArray()
        
        for i, (marker_id, (px, py, timestamp)) in enumerate(self.detected_markers.items()):
            world_pos = self.pixel_to_world(px, py)
            if world_pos:
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "aruco_markers"
                marker.id = marker_id
                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD
                
                marker.pose.position.x = world_pos[0]
                marker.pose.position.y = world_pos[1]
                marker.pose.position.z = 0.0
                marker.pose.orientation.w = 1.0
                
                marker.scale.x = 0.2
                marker.scale.y = 0.2
                marker.scale.z = 0.05
                
                if marker_id == self.selected_marker_id:
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                else:
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                marker.color.a = 0.8
                
                marker.lifetime = rospy.Duration(0.5)
                marker_array.markers.append(marker)
                
                # Add text label
                text_marker = Marker()
                text_marker.header = marker.header
                text_marker.ns = "aruco_labels"
                text_marker.id = marker_id + 1000
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                text_marker.pose.position.x = world_pos[0]
                text_marker.pose.position.y = world_pos[1]
                text_marker.pose.position.z = 0.3
                text_marker.scale.z = 0.2
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 1.0
                text_marker.color.a = 1.0
                text_marker.text = f"ID: {marker_id}"
                text_marker.lifetime = rospy.Duration(0.5)
                marker_array.markers.append(text_marker)
        
        self.marker_viz_pub.publish(marker_array)

def main():
    try:
        controller = ArUcoLandingController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error: {e}")

if __name__ == '__main__':
    main()