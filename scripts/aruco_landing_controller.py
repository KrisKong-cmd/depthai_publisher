#!/usr/bin/env python3

"""
ArUco Landing Controller for Drone Auto-Landing

Made by DG

This script provides automatic landing functionality on ArUco markers by integrating
ArUco detection with SPAR flight control system.

Camera Offset Configuration:
- CAMERA_FORWARD_OFFSET: Distance in meters from camera to drone center in forward direction
- Units: meters (set to 0 for camera at drone center)
- Positive values: camera is forward of drone center
- Negative values: camera is behind drone center

Operation:
1. Call the landing service with target marker ID
2. Drone will approach and hover over the marker
3. Controlled descent and landing sequence
4. Service returns success/failure status

Author: Generated for drone ArUco landing system
"""

import rospy
import actionlib
import math
import numpy as np
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.srv import SetBool, SetBoolResponse
from spar_msgs.msg import FlightMotionAction, FlightMotionGoal
from cv_bridge import CvBridge
import cv2

# Camera offset parameter - ADJUST THIS BASED ON YOUR DRONE'S CAMERA PLACEMENT
CAMERA_FORWARD_OFFSET = 0.0  # meters - distance from camera to drone center (forward direction)

class ArucoLandingController:
    def __init__(self):
        rospy.init_node('aruco_landing_controller')
        
        # Camera offset (adjustable parameter)
        self.camera_forward_offset = rospy.get_param('~camera_forward_offset', CAMERA_FORWARD_OFFSET)
        
        # Landing parameters
        self.target_marker_id = None
        self.approach_altitude = rospy.get_param('~approach_altitude', 2.0)  # meters
        self.hover_altitude = rospy.get_param('~hover_altitude', 1.0)      # meters
        self.landing_speed = rospy.get_param('~landing_speed', 0.3)        # m/s
        self.position_tolerance = rospy.get_param('~position_tolerance', 0.2)  # meters
        self.convergence_timeout = rospy.get_param('~convergence_timeout', 10.0)  # seconds
        
        # PID gains for positioning
        self.kp_xy = rospy.get_param('~kp_xy', 0.5)
        self.max_velocity = rospy.get_param('~max_velocity', 1.0)  # m/s
        
        # State management
        self.landing_active = False
        self.current_stage = 'idle'  # idle, approach, hover, descend, landed
        self.marker_detected = False
        self.marker_center = None
        self.image_center = None
        self.current_pose = None
        self.landing_start_time = None
        
        # Camera parameters (will be updated from first image)
        self.image_width = 640
        self.image_height = 480
        self.camera_fov_h = math.radians(69)  # Approximate horizontal FOV
        
        # ROS components
        self.bridge = CvBridge()
        
        # Subscribers
        self.aruco_sub = rospy.Subscriber('/processed_aruco/image/compressed', 
                                         CompressedImage, self.aruco_callback)
        self.pose_sub = rospy.Subscriber('/mavros/local_position/pose', 
                                        PoseStamped, self.pose_callback)
        
        # SPAR action client
        self.spar_client = actionlib.SimpleActionClient('spar/flight_motion', FlightMotionAction)
        rospy.loginfo("Waiting for SPAR action server...")
        self.spar_client.wait_for_server(timeout=rospy.Duration(30))
        rospy.loginfo("Connected to SPAR action server")
        
        # Services
        self.landing_service = rospy.Service('~start_landing', SetBool, self.start_landing_callback)
        self.stop_service = rospy.Service('~stop_landing', Trigger, self.stop_landing_callback)
        
        # Control timer
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        
        rospy.loginfo("ArUco Landing Controller initialized")
        rospy.loginfo(f"Camera forward offset: {self.camera_forward_offset:.2f} meters")
        rospy.loginfo("Services available:")
        rospy.loginfo("  - ~start_landing (std_srvs/SetBool) - Start landing on marker ID")
        rospy.loginfo("  - ~stop_landing (std_srvs/Trigger) - Stop landing sequence")

    def aruco_callback(self, msg):
        """Process ArUco detection data"""
        try:
            # Convert image
            frame = self.bridge.compressed_imgmsg_to_cv2(msg)
            self.image_height, self.image_width = frame.shape[:2]
            self.image_center = (self.image_width // 2, self.image_height // 2)
            
            # Detect ArUco markers
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
            aruco_params = cv2.aruco.DetectorParameters_create()
            corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
            
            self.marker_detected = False
            if ids is not None and self.target_marker_id is not None:
                # Look for target marker
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id == self.target_marker_id:
                        # Calculate marker center
                        marker_corners = corners[i].reshape((4, 2))
                        center_x = int(np.mean(marker_corners[:, 0]))
                        center_y = int(np.mean(marker_corners[:, 1]))
                        self.marker_center = (center_x, center_y)
                        self.marker_detected = True
                        break
                        
        except Exception as e:
            rospy.logerr(f"Error in ArUco callback: {e}")

    def pose_callback(self, msg):
        """Update current drone pose"""
        self.current_pose = msg

    def start_landing_callback(self, req):
        """Service to start landing on specified marker ID"""
        if req.data < 0 or req.data > 99:
            return SetBoolResponse(success=False, message="Invalid marker ID. Must be 0-99")
        
        if self.current_pose is None:
            return SetBoolResponse(success=False, message="No pose data available")
        
        if self.landing_active:
            return SetBoolResponse(success=False, message="Landing already in progress")
        
        self.target_marker_id = int(req.data)
        self.landing_active = True
        self.current_stage = 'approach'
        self.landing_start_time = rospy.Time.now()
        
        rospy.loginfo(f"Starting landing sequence for marker ID: {self.target_marker_id}")
        return SetBoolResponse(success=True, message=f"Landing started for marker {self.target_marker_id}")

    def stop_landing_callback(self, req):
        """Service to stop landing sequence"""
        if not self.landing_active:
            return TriggerResponse(success=True, message="No landing in progress")
        
        self.landing_active = False
        self.current_stage = 'idle'
        self.target_marker_id = None
        
        # Send stop command to SPAR
        self.send_stop_command()
        
        rospy.loginfo("Landing sequence stopped")
        return TriggerResponse(success=True, message="Landing stopped")

    def control_loop(self, event):
        """Main control loop for landing sequence"""
        if not self.landing_active or self.current_pose is None:
            return
        
        # Check for timeout
        if rospy.Time.now() - self.landing_start_time > rospy.Duration(self.convergence_timeout):
            rospy.logwarn("Landing timeout exceeded")
            self.stop_landing()
            return
        
        # State machine for landing stages
        if self.current_stage == 'approach':
            self.handle_approach_stage()
        elif self.current_stage == 'hover':
            self.handle_hover_stage()
        elif self.current_stage == 'descend':
            self.handle_descend_stage()

    def handle_approach_stage(self):
        """Handle approach to marker at safe altitude"""
        if not self.marker_detected:
            rospy.logwarn_throttle(2.0, f"Marker {self.target_marker_id} not detected, waiting...")
            return
        
        # Calculate target position with camera offset compensation
        target_pos = self.calculate_target_position()
        if target_pos is None:
            return
        
        # Move to approach altitude above marker
        target_pos.z = self.current_pose.pose.position.z + (self.approach_altitude - self.current_pose.pose.position.z) * 0.1
        
        # Check if close enough to marker center
        current_pos = self.current_pose.pose.position
        distance_2d = math.sqrt((target_pos.x - current_pos.x)**2 + (target_pos.y - current_pos.y)**2)
        
        if distance_2d < self.position_tolerance:
            rospy.loginfo("Approach complete, entering hover stage")
            self.current_stage = 'hover'
        else:
            self.send_goto_command(target_pos)

    def handle_hover_stage(self):
        """Handle hovering over marker for stabilization"""
        if not self.marker_detected:
            rospy.logwarn("Marker lost during hover, returning to approach")
            self.current_stage = 'approach'
            return
        
        # Maintain position over marker
        target_pos = self.calculate_target_position()
        if target_pos is None:
            return
        
        target_pos.z = self.hover_altitude
        
        # Check if stable enough to start descent
        current_pos = self.current_pose.pose.position
        distance_2d = math.sqrt((target_pos.x - current_pos.x)**2 + (target_pos.y - current_pos.y)**2)
        
        if distance_2d < self.position_tolerance * 0.5:  # Tighter tolerance for descent
            rospy.loginfo("Hover stable, starting descent")
            self.current_stage = 'descend'
        else:
            self.send_goto_command(target_pos)

    def handle_descend_stage(self):
        """Handle controlled descent to landing"""
        if not self.marker_detected:
            rospy.logwarn("Marker lost during descent, stopping landing")
            self.stop_landing()
            return
        
        # Continue descent while maintaining position over marker
        target_pos = self.calculate_target_position()
        if target_pos is None:
            return
        
        # Send landing command
        self.send_landing_command()

    def calculate_target_position(self):
        """Calculate target world position from marker center in image"""
        if not self.marker_detected or self.marker_center is None or self.image_center is None:
            return None
        
        # Calculate pixel offset from image center
        pixel_offset_x = self.marker_center[0] - self.image_center[0]
        pixel_offset_y = self.marker_center[1] - self.image_center[1]
        
        # Convert pixel offset to angular offset
        pixels_per_radian_x = self.image_width / self.camera_fov_h
        pixels_per_radian_y = pixels_per_radian_x  # Assume square pixels
        
        angle_x = pixel_offset_x / pixels_per_radian_x
        angle_y = pixel_offset_y / pixels_per_radian_y
        
        # Convert angular offset to world coordinates (assuming level flight)
        altitude = self.current_pose.pose.position.z
        world_offset_x = altitude * math.tan(angle_x)
        world_offset_y = altitude * math.tan(angle_y)
        
        # Apply camera offset compensation
        # Camera forward offset affects the Y coordinate in body frame
        world_offset_y += self.camera_forward_offset
        
        # Calculate target position
        current_pos = self.current_pose.pose.position
        target_pos = Point()
        target_pos.x = current_pos.x + world_offset_x
        target_pos.y = current_pos.y + world_offset_y
        target_pos.z = current_pos.z
        
        return target_pos

    def send_goto_command(self, target_pos):
        """Send goto command to SPAR"""
        goal = FlightMotionGoal()
        goal.motion = FlightMotionGoal.MOTION_GOTO_POS
        goal.position = target_pos
        goal.velocity_horizontal = min(self.max_velocity, self.kp_xy * 2)  # Adaptive speed
        goal.velocity_vertical = self.landing_speed
        goal.wait_for_convergence = False  # We handle convergence ourselves
        
        self.spar_client.send_goal(goal)

    def send_landing_command(self):
        """Send landing command to SPAR"""
        goal = FlightMotionGoal()
        goal.motion = FlightMotionGoal.MOTION_LAND
        goal.velocity_vertical = self.landing_speed
        
        self.spar_client.send_goal(goal)
        rospy.loginfo("Landing command sent")
        
        # Mark as completed
        self.landing_active = False
        self.current_stage = 'landed'

    def send_stop_command(self):
        """Send stop command to SPAR"""
        goal = FlightMotionGoal()
        goal.motion = FlightMotionGoal.MOTION_STOP
        
        self.spar_client.send_goal(goal)

    def stop_landing(self):
        """Stop the landing sequence"""
        self.landing_active = False
        self.current_stage = 'idle'
        self.target_marker_id = None
        self.send_stop_command()

if __name__ == '__main__':
    try:
        controller = ArucoLandingController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass