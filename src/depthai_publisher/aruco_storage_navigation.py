#!/usr/bin/env python3

"""
ArUco Marker Storage and Navigation System with ML Target Visualization
This node handles:
1. Storing ArUco marker positions (averaged over time)
2. Storing ML target positions (fire, smoke, human, bag) - one per type
3. Publishing markers and targets to RViz
4. Navigation to selected ArUco markers with landing
"""

import rospy
import actionlib
from actionlib_msgs.msg import GoalStatus
import numpy as np
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray, InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from std_msgs.msg import Header, ColorRGBA, String
from spar_msgs.msg import FlightMotionAction, FlightMotionGoal
import signal
import sys
from collections import defaultdict
import threading
import time

class ArUcoStorageNavigation:
    def __init__(self):
        rospy.init_node('aruco_storage_navigation', anonymous=True)
        
        # Parameters
        self.camera_offset_x = rospy.get_param("~camera_offset_x", 0.15)  # Camera 15cm in front of drone center
        self.camera_offset_y = rospy.get_param("~camera_offset_y", 0.0)
        self.hover_altitude = rospy.get_param("~hover_altitude", 0.5)  # Hover at 0.5m before landing
        self.approach_speed = rospy.get_param("~approach_speed", 0.3)
        self.landing_speed = rospy.get_param("~landing_speed", 0.2)
        self.position_accuracy = rospy.get_param("~position_accuracy", 0.15)
        self.yaw_accuracy = rospy.get_param("~yaw_accuracy", 0.2)
        self.averaging_window = rospy.get_param("~averaging_window", 3.0)  # Average over 3 seconds
        
        # Get the pose topic from parameter (can be /uavasr/pose for sim or /mavros/local_position/pose for real)
        self.drone_pose_topic = rospy.get_param("~drone_pose_topic", "/uavasr/pose")
        
        # Storage for ArUco markers (can have multiple)
        self.stored_markers = {}  # {marker_id: {'position': Point, 'detections': [(x,y,z,timestamp)], 'last_update': timestamp}}
        self.marker_lock = threading.Lock()
        
        # Storage for ML targets - ONLY ONE PER TYPE (latest detection replaces previous)
        self.ml_targets = {
            'bag': None,    # Will store {'position': Point, 'detections': [(x,y,z,timestamp)], 'last_update': timestamp, 'detection_count': int}
            'fire': None,
            'human': None,
            'smoke': None
        }
        self.ml_lock = threading.Lock()
        
        # ML target colors matching the YOLO visualization
        self.ml_colors = {
            'bag': ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8),      # Green
            'fire': ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8),     # Red
            'human': ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8),    # Blue
            'smoke': ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8)     # Yellow
        }
        
        # ML target marker shapes
        self.ml_shapes = {
            'bag': Marker.CUBE,
            'fire': Marker.SPHERE,
            'human': Marker.CYLINDER,
            'smoke': Marker.SPHERE
        }
        
        # Current drone position
        self.current_pose = None
        self.navigation_active = False
        
        # Action client for flight control
        action_ns = rospy.get_param("~action_topic", 'spar/flight')
        self.spar_client = actionlib.SimpleActionClient(action_ns, FlightMotionAction)
        rospy.loginfo("Waiting for SPAR action server...")
        self.spar_client.wait_for_server()
        rospy.loginfo("Connected to SPAR")
        
        # Interactive marker server for RViz (ArUco only)
        self.marker_server = InteractiveMarkerServer("aruco_markers")
        
        # Publishers
        self.pub_marker_array = rospy.Publisher('/aruco_markers/visualization', MarkerArray, queue_size=10, latch=True)
        self.pub_ml_targets = rospy.Publisher('/ml_targets/visualization', MarkerArray, queue_size=10, latch=True)
        self.pub_status = rospy.Publisher('/aruco_navigation/status', String, queue_size=10)
        
        # Subscribers for ArUco
        self.sub_aruco_roi = rospy.Subscriber('/aruco_detection/roi', PoseStamped, self.aruco_roi_callback)
        
        # Subscribers for ML targets
        self.sub_bag_roi = rospy.Subscriber('/yolo_detection/roi/bag', PoseStamped, 
                                           lambda msg: self.ml_roi_callback(msg, 'bag'))
        self.sub_fire_roi = rospy.Subscriber('/yolo_detection/roi/fire', PoseStamped, 
                                            lambda msg: self.ml_roi_callback(msg, 'fire'))
        self.sub_human_roi = rospy.Subscriber('/yolo_detection/roi/human', PoseStamped, 
                                             lambda msg: self.ml_roi_callback(msg, 'human'))
        self.sub_smoke_roi = rospy.Subscriber('/yolo_detection/roi/smoke', PoseStamped, 
                                             lambda msg: self.ml_roi_callback(msg, 'smoke'))
        
        # Subscriber for drone pose
        self.sub_drone_pose = rospy.Subscriber(self.drone_pose_topic, PoseStamped, self.pose_callback)
        
        rospy.loginfo(f"Using drone pose topic: {self.drone_pose_topic}")
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Timer to update averaged positions and publish to RViz
        self.update_timer = rospy.Timer(rospy.Duration(1.0), self.update_and_publish_all)
        
        rospy.loginfo("ArUco Storage and Navigation System with ML Target Visualization initialized")
        rospy.loginfo(f"Configured for pose topic: {self.drone_pose_topic}")
        rospy.loginfo("Visualizing: ArUco markers (interactive) and ML targets (bag, fire, human, smoke)")
        rospy.loginfo("ML targets: ONE per type - latest detection replaces previous")
        rospy.loginfo("Click on ArUco markers in RViz to navigate and land")
        rospy.loginfo("Press Ctrl+C to cancel navigation and shutdown")
        
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C interrupt gracefully"""
        rospy.loginfo("\nInterrupt received! Cancelling navigation and shutting down...")
        
        if self.navigation_active:
            self.spar_client.cancel_goal()
            self.navigation_active = False
            rospy.loginfo("Navigation cancelled")
        
        rospy.signal_shutdown("User interrupt")
        sys.exit(0)
    
    def pose_callback(self, msg):
        """Update current drone position"""
        self.current_pose = msg
    
    def aruco_roi_callback(self, msg):
        """Receive ArUco detection ROI and store it"""
        # Extract marker ID from frame_id (format: "aruco_XX")
        if msg.header.frame_id.startswith("aruco_"):
            marker_id = msg.header.frame_id.replace("aruco_", "")
            marker_key = f"ID_{marker_id}"
        else:
            # Fallback to position-based key if ID not provided
            marker_key = f"pos_{msg.pose.position.x:.2f}_{msg.pose.position.y:.2f}"
        
        with self.marker_lock:
            current_time = rospy.Time.now().to_sec()
            
            # Check if this is a new detection location or existing one
            for existing_id, data in self.stored_markers.items():
                # Check if position is close to existing marker (within 0.5m)
                dist = np.sqrt((data['position'].x - msg.pose.position.x)**2 + 
                             (data['position'].y - msg.pose.position.y)**2)
                if dist < 0.5 and not existing_id.startswith("ID_"):
                    # Update existing position-based marker if we now have an ID
                    if marker_key.startswith("ID_"):
                        # Remove old position-based key and use ID-based key
                        self.stored_markers[marker_key] = self.stored_markers.pop(existing_id)
                        break
                    else:
                        marker_key = existing_id
                        break
            
            if marker_key not in self.stored_markers:
                # New marker detected
                self.stored_markers[marker_key] = {
                    'position': Point(x=msg.pose.position.x, y=msg.pose.position.y, z=msg.pose.position.z),
                    'detections': [],
                    'last_update': current_time,
                    'first_seen': current_time
                }
                rospy.loginfo(f"New ArUco marker stored: {marker_key}")
            
            # Add detection to history
            self.stored_markers[marker_key]['detections'].append(
                (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, current_time)
            )
            self.stored_markers[marker_key]['last_update'] = current_time
            
            # Keep only recent detections (within averaging window) for position calculation
            cutoff_time = current_time - self.averaging_window
            self.stored_markers[marker_key]['detections'] = [
                d for d in self.stored_markers[marker_key]['detections'] if d[3] > cutoff_time
            ]
    
    def ml_roi_callback(self, msg, target_type):
        """Receive ML target detection ROI and store it - ONLY ONE PER TYPE"""
        with self.ml_lock:
            current_time = rospy.Time.now().to_sec()
            
            # Check if we should update the existing target or replace it
            if self.ml_targets[target_type] is None:
                # First detection of this type
                self.ml_targets[target_type] = {
                    'position': Point(x=msg.pose.position.x, y=msg.pose.position.y, z=0.0),
                    'detections': [],
                    'last_update': current_time,
                    'first_seen': current_time,
                    'detection_count': 0
                }
                rospy.loginfo(f"First {target_type} target detected at ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")
            else:
                # Check distance to existing target
                existing_pos = self.ml_targets[target_type]['position']
                dist = np.sqrt((existing_pos.x - msg.pose.position.x)**2 + 
                             (existing_pos.y - msg.pose.position.y)**2)
                
                if dist > 2.0:  # If more than 2 meters away, it's a new target location
                    # Replace with new target
                    self.ml_targets[target_type] = {
                        'position': Point(x=msg.pose.position.x, y=msg.pose.position.y, z=0.0),
                        'detections': [],
                        'last_update': current_time,
                        'first_seen': current_time,
                        'detection_count': 0
                    }
                    rospy.loginfo(f"New {target_type} location detected at ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}), replacing previous")
            
            # Add detection to history for averaging
            self.ml_targets[target_type]['detections'].append(
                (msg.pose.position.x, msg.pose.position.y, 0.0, current_time)
            )
            self.ml_targets[target_type]['last_update'] = current_time
            self.ml_targets[target_type]['detection_count'] += 1
            
            # Keep only recent detections for averaging (within averaging window)
            cutoff_time = current_time - self.averaging_window
            self.ml_targets[target_type]['detections'] = [
                d for d in self.ml_targets[target_type]['detections'] if d[3] > cutoff_time
            ]
    
    def update_and_publish_all(self, event):
        """Update averaged positions and publish everything to RViz"""
        # Update and publish ArUco markers
        self.publish_aruco_markers()
        
        # Update and publish ML targets
        self.publish_ml_targets()
    
    def publish_aruco_markers(self):
        """Update and publish ArUco markers to RViz"""
        with self.marker_lock:
            marker_array = MarkerArray()
            
            # First, add delete markers for all possible IDs to clear old ones
            for i in range(100):  # Assuming max 50 ArUco markers (2 IDs each)
                delete_marker = Marker()
                delete_marker.header.frame_id = "map"
                delete_marker.ns = "aruco_markers"
                delete_marker.id = i
                delete_marker.action = Marker.DELETE
                marker_array.markers.append(delete_marker)
                
                delete_text = Marker()
                delete_text.header.frame_id = "map"
                delete_text.ns = "aruco_labels"
                delete_text.id = i + 1000
                delete_text.action = Marker.DELETE
                marker_array.markers.append(delete_text)
            
            # Now add current markers
            marker_id = 0
            for marker_key, data in self.stored_markers.items():
                if len(data['detections']) > 0:
                    # Calculate average position from recent detections
                    avg_x = np.mean([d[0] for d in data['detections']])
                    avg_y = np.mean([d[1] for d in data['detections']])
                    avg_z = np.mean([d[2] for d in data['detections']])
                    
                    # Update stored position
                    data['position'] = Point(x=avg_x, y=avg_y, z=avg_z)
                else:
                    # Use last known position if no recent detections
                    avg_x = data['position'].x
                    avg_y = data['position'].y
                    avg_z = data['position'].z
                
                # Create visualization marker
                viz_marker = self.create_aruco_visualization_marker(marker_id, marker_key, data['position'])
                marker_array.markers.append(viz_marker)
                
                # Create text label
                time_since = rospy.Time.now().to_sec() - data['last_update']
                label = f"ArUco {marker_key}\n({avg_x:.1f}, {avg_y:.1f})\nLast: {time_since:.0f}s ago"
                text_marker = self.create_text_marker(marker_id + 1000, label, 
                                                     Point(x=avg_x, y=avg_y, z=avg_z + 0.3),
                                                     ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0),
                                                     "aruco_labels")
                marker_array.markers.append(text_marker)
                
                # Update interactive marker
                self.update_interactive_marker(marker_key, data['position'])
                
                marker_id += 1
            
            # Publish ArUco visualization
            self.pub_marker_array.publish(marker_array)
            self.marker_server.applyChanges()
    
    def publish_ml_targets(self):
        """Publish ML targets to RViz - one per type, persistent"""
        with self.ml_lock:
            ml_marker_array = MarkerArray()
            
            # Clear any old markers first
            for i in range(8):  # 4 target types * 2 markers each
                delete_marker = Marker()
                delete_marker.header.frame_id = "map"
                delete_marker.ns = "ml_targets"
                delete_marker.id = i
                delete_marker.action = Marker.DELETE
                ml_marker_array.markers.append(delete_marker)
                
                delete_text = Marker()
                delete_text.header.frame_id = "map"
                delete_text.ns = "ml_labels"
                delete_text.id = i + 5000
                delete_text.action = Marker.DELETE
                ml_marker_array.markers.append(delete_text)
            
            # Add current ML targets
            ml_marker_id = 0
            for target_type in ['bag', 'fire', 'human', 'smoke']:
                if self.ml_targets[target_type] is not None:
                    data = self.ml_targets[target_type]
                    
                    # Calculate average position if we have recent detections
                    if len(data['detections']) > 0:
                        avg_x = np.mean([d[0] for d in data['detections']])
                        avg_y = np.mean([d[1] for d in data['detections']])
                        # Update stored position with averaged value
                        data['position'] = Point(x=avg_x, y=avg_y, z=0.0)
                    else:
                        # Use last known position
                        avg_x = data['position'].x
                        avg_y = data['position'].y
                    
                    # Create ML target visualization marker
                    ml_marker = self.create_ml_target_marker(ml_marker_id, target_type, data['position'])
                    ml_marker_array.markers.append(ml_marker)
                    
                    # Create text label with info
                    time_since = rospy.Time.now().to_sec() - data['last_update']
                    label = f"{target_type.upper()}\n({avg_x:.1f}, {avg_y:.1f})\nDetections: {data['detection_count']}\nLast: {time_since:.0f}s ago"
                    text_marker = self.create_text_marker(ml_marker_id + 5000, label, 
                                                         Point(x=avg_x, y=avg_y, z=0.5),
                                                         ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0),
                                                         "ml_labels")
                    ml_marker_array.markers.append(text_marker)
                    
                    ml_marker_id += 1
            
            # Publish ML target visualization
            self.pub_ml_targets.publish(ml_marker_array)
    
    def create_aruco_visualization_marker(self, marker_id, marker_key, position):
        """Create a visualization marker for ArUco in RViz"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "aruco_markers"
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        marker.pose.position = position
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.1
        
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
        
        # No lifetime - markers persist indefinitely
        marker.lifetime = rospy.Duration(0)
        
        return marker
    
    def create_ml_target_marker(self, marker_id, target_type, position):
        """Create a visualization marker for ML targets in RViz"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "ml_targets"
        marker.id = marker_id
        marker.type = self.ml_shapes[target_type]
        marker.action = Marker.ADD
        
        marker.pose.position = position
        marker.pose.orientation.w = 1.0
        
        # Different sizes for different target types
        if target_type == 'human':
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.6
        elif target_type == 'bag':
            marker.scale.x = 0.4
            marker.scale.y = 0.3
            marker.scale.z = 0.3
        else:  # fire, smoke
            marker.scale.x = 0.4
            marker.scale.y = 0.4
            marker.scale.z = 0.4
        
        marker.color = self.ml_colors[target_type]
        
        # No lifetime - markers persist indefinitely
        marker.lifetime = rospy.Duration(0)
        
        return marker
    
    def create_text_marker(self, marker_id, text, position, color, ns):
        """Create a text label marker"""
        text_marker = Marker()
        text_marker.header.frame_id = "map"
        text_marker.header.stamp = rospy.Time.now()
        text_marker.ns = ns
        text_marker.id = marker_id
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position = position
        text_marker.scale.z = 0.15
        text_marker.color = color
        text_marker.text = text
        
        # No lifetime - text persists indefinitely
        text_marker.lifetime = rospy.Duration(0)
        
        return text_marker
    
    def update_interactive_marker(self, marker_key, position):
        """Create or update an interactive marker for clicking in RViz"""
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "map"
        int_marker.name = marker_key
        int_marker.description = f"Click to navigate to ArUco {marker_key}"
        int_marker.pose.position = position
        int_marker.scale = 0.5
        
        # Create a clickable control
        control = InteractiveMarkerControl()
        control.interaction_mode = InteractiveMarkerControl.BUTTON
        control.always_visible = True
        
        # Add visual marker to control
        viz_marker = Marker()
        viz_marker.type = Marker.CYLINDER
        viz_marker.scale.x = 0.3
        viz_marker.scale.y = 0.3
        viz_marker.scale.z = 0.1
        viz_marker.color = ColorRGBA(r=0.0, g=0.8, b=0.8, a=0.6)
        control.markers.append(viz_marker)
        
        int_marker.controls.append(control)
        
        # Add to server with feedback callback
        self.marker_server.insert(int_marker, self.marker_click_callback)
    
    def marker_click_callback(self, feedback):
        """Handle click on interactive marker"""
        if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            marker_key = feedback.marker_name
            rospy.loginfo(f"Marker {marker_key} clicked! Starting navigation...")
            
            with self.marker_lock:
                if marker_key in self.stored_markers:
                    target_position = self.stored_markers[marker_key]['position']
                    self.navigate_to_marker(target_position, marker_key)
    
    def navigate_to_marker(self, target_position, marker_key):
        """Navigate to the marker position and land"""
        if self.navigation_active:
            rospy.logwarn("Navigation already in progress!")
            return
        
        self.navigation_active = True
        status_msg = String()
        
        try:
            # Step 1: Navigate to position above marker (compensating for camera offset)
            rospy.loginfo(f"Step 1: Flying to marker {marker_key} at hover altitude...")
            status_msg.data = f"Approaching marker {marker_key}"
            self.pub_status.publish(status_msg)
            
            # Compensate for camera offset
            approach_x = target_position.x - self.camera_offset_x
            approach_y = target_position.y - self.camera_offset_y
            
            if not self.goto_position(approach_x, approach_y, self.hover_altitude):
                rospy.logwarn("Failed to reach marker position")
                return
            
            rospy.loginfo("Step 2: Reached hover position, preparing to land...")
            rospy.sleep(2.0)  # Stabilize for 2 seconds
            
            # Step 3: Land at current position
            rospy.loginfo("Step 3: Landing on marker...")
            status_msg.data = f"Landing on marker {marker_key}"
            self.pub_status.publish(status_msg)
            
            if self.land():
                rospy.loginfo(f"Successfully landed on marker {marker_key}!")
                status_msg.data = f"Landed on marker {marker_key}"
                self.pub_status.publish(status_msg)
            else:
                rospy.logwarn("Landing failed")
                status_msg.data = "Landing failed"
                self.pub_status.publish(status_msg)
                
        except Exception as e:
            rospy.logerr(f"Navigation error: {e}")
            status_msg.data = f"Navigation error: {e}"
            self.pub_status.publish(status_msg)
        finally:
            self.navigation_active = False
    
    def goto_position(self, x, y, z, yaw=0.0):
        """Send goto command to drone"""
        goal = FlightMotionGoal()
        goal.motion = FlightMotionGoal.MOTION_GOTO
        goal.position.x = x
        goal.position.y = y
        goal.position.z = z
        goal.yaw = yaw
        goal.velocity_horizontal = self.approach_speed
        goal.velocity_vertical = self.approach_speed
        goal.yawrate = 0.2
        goal.wait_for_convergence = True
        goal.position_radius = self.position_accuracy
        goal.yaw_range = self.yaw_accuracy
        
        rospy.loginfo(f"Going to position: ({x:.2f}, {y:.2f}, {z:.2f})")
        
        self.spar_client.send_goal(goal)
        self.spar_client.wait_for_result()
        
        result = self.spar_client.get_state()
        return result == GoalStatus.SUCCEEDED
    
    def land(self):
        """Execute landing sequence"""
        goal = FlightMotionGoal()
        goal.motion = FlightMotionGoal.MOTION_LAND
        goal.velocity_vertical = self.landing_speed
        
        rospy.loginfo(f"Landing at speed {self.landing_speed} m/s...")
        
        self.spar_client.send_goal(goal)
        self.spar_client.wait_for_result()
        
        result = self.spar_client.get_state()
        return result == GoalStatus.SUCCEEDED
    
    def shutdown(self):
        """Clean shutdown"""
        rospy.loginfo("Shutting down ArUco Storage and Navigation...")
        if self.navigation_active:
            self.spar_client.cancel_goal()
        self.update_timer.shutdown()
        self.marker_server.clear()
        self.marker_server.applyChanges()

def main():
    try:
        node = ArUcoStorageNavigation()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.loginfo("Keyboard interrupt received")
    finally:
        if 'node' in locals():
            node.shutdown()
        rospy.loginfo("ArUco Storage and Navigation shutdown complete")

if __name__ == '__main__':
    main()