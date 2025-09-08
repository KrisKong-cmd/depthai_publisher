#!/usr/bin/env python3

import rospy
from std_msgs.msg import String, Int32, Bool
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray, InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
import threading
import json

class PayloadConfigGUI:
    def __init__(self):
        rospy.init_node('payload_config_gui', anonymous=True)
        
        # Configuration state
        self.payload_type = "tino"  # Default to Tino's payload
        self.roi_assignments = {
            'fire': {'enabled': True, 'side': 'left'},
            'smoke': {'enabled': True, 'side': 'right'},
            'human': {'enabled': False, 'side': None},
            'bag': {'enabled': False, 'side': None}
        }
        
        # Payload specifications
        self.payload_specs = {
            'minh': {
                'name': "Minh's Payload",
                'drop1_angle': 160,
                'drop2_angle': 20,
                'speed': 120,  # Fast speed
                'description': 'Fast deployment at 160° and 20°'
            },
            'tino': {
                'name': "Tino's Payload",
                'drop1_angle': 120,
                'drop2_angle': 60,
                'speed': 60,  # Slower speed
                'description': 'Controlled deployment at 140° and 40°'
            }
        }
        
        # Interactive marker server
        self.server = InteractiveMarkerServer("payload_configuration")
        self.menu_handler = MenuHandler()
        
        # Publishers for configuration
        self.pub_config = rospy.Publisher('/payload/configuration', String, queue_size=1, latch=True)
        self.pub_display = rospy.Publisher('/payload/config_display', MarkerArray, queue_size=1, latch=True)
        
        # Setup menu entries
        self.setup_menu()
        
        # Create interactive markers
        self.create_config_marker()
        self.create_roi_markers()
        
        # Timer to update display
        self.timer = rospy.Timer(rospy.Duration(1.0), self.update_display)
        
        # Publish initial configuration immediately
        rospy.sleep(0.5)  # Small delay to ensure subscribers are ready
        self.publish_configuration()
        
        rospy.loginfo("Payload Configuration GUI initialized")
        rospy.loginfo("Right-click on CONFIG marker in RViz to configure")
        rospy.loginfo("Initial configuration published")
        
    def setup_menu(self):
        """Setup the interactive menu structure"""
        
        # Main menu - Payload Type
        self.payload_menu = self.menu_handler.insert("Select Payload Type")
        self.menu_handler.insert("Minh's Payload (160°/20° Fast)", 
                                parent=self.payload_menu,
                                callback=self.select_minh_payload)
        self.menu_handler.insert("Tino's Payload (140°/40° Controlled)", 
                                parent=self.payload_menu,
                                callback=self.select_tino_payload)
        
        self.menu_handler.insert("---")  # Separator
        
        # ROI Configuration Menu
        self.roi_menu = self.menu_handler.insert("Configure ROI Targets")
        
        # Fire configuration
        self.fire_menu = self.menu_handler.insert("🔥 Fire Detection", parent=self.roi_menu)
        self.menu_handler.insert("Disable", parent=self.fire_menu, callback=self.disable_fire)
        self.menu_handler.insert("Observe Only", parent=self.fire_menu, callback=self.observe_fire)
        self.menu_handler.insert("Deploy LEFT Side", parent=self.fire_menu, callback=self.fire_to_left)
        self.menu_handler.insert("Deploy RIGHT Side", parent=self.fire_menu, callback=self.fire_to_right)
        
        # Smoke configuration
        self.smoke_menu = self.menu_handler.insert("💨 Smoke Detection", parent=self.roi_menu)
        self.menu_handler.insert("Disable", parent=self.smoke_menu, callback=self.disable_smoke)
        self.menu_handler.insert("Observe Only", parent=self.smoke_menu, callback=self.observe_smoke)
        self.menu_handler.insert("Deploy LEFT Side", parent=self.smoke_menu, callback=self.smoke_to_left)
        self.menu_handler.insert("Deploy RIGHT Side", parent=self.smoke_menu, callback=self.smoke_to_right)
        
        # Human configuration
        self.human_menu = self.menu_handler.insert("👤 Human Detection", parent=self.roi_menu)
        self.menu_handler.insert("Disable", parent=self.human_menu, callback=self.disable_human)
        self.menu_handler.insert("Observe Only", parent=self.human_menu, callback=self.observe_human)
        self.menu_handler.insert("Deploy LEFT Side", parent=self.human_menu, callback=self.human_to_left)
        self.menu_handler.insert("Deploy RIGHT Side", parent=self.human_menu, callback=self.human_to_right)
        
        # Bag configuration
        self.bag_menu = self.menu_handler.insert("💜 Bag Detection", parent=self.roi_menu)
        self.menu_handler.insert("Disable", parent=self.bag_menu, callback=self.disable_bag)
        self.menu_handler.insert("Observe Only", parent=self.bag_menu, callback=self.observe_bag)
        self.menu_handler.insert("Deploy LEFT Side", parent=self.bag_menu, callback=self.bag_to_left)
        self.menu_handler.insert("Deploy RIGHT Side", parent=self.bag_menu, callback=self.bag_to_right)
        
        self.menu_handler.insert("---")  # Separator
        
        # Show current status
        self.menu_handler.insert("Show Configuration", callback=self.show_config)
    
    def select_minh_payload(self, feedback):
        self.payload_type = "minh"
        rospy.loginfo(f"Selected: {self.payload_specs['minh']['name']}")
        self.publish_configuration()
    
    def select_tino_payload(self, feedback):
        self.payload_type = "tino"
        rospy.loginfo(f"Selected: {self.payload_specs['tino']['name']}")
        self.publish_configuration()
    
    # Fire configuration callbacks
    def disable_fire(self, feedback):
        self.roi_assignments['fire']['enabled'] = False
        self.roi_assignments['fire']['side'] = None
        rospy.loginfo("Fire detection: DISABLED")
        self.publish_configuration()
    
    def observe_fire(self, feedback):
        self.roi_assignments['fire']['enabled'] = True
        self.roi_assignments['fire']['side'] = None
        rospy.loginfo("Fire detection: OBSERVE ONLY")
        self.publish_configuration()
    
    def fire_to_left(self, feedback):
        self.roi_assignments['fire']['enabled'] = True
        self.roi_assignments['fire']['side'] = 'left'
        rospy.loginfo("Fire detection: DEPLOY LEFT SIDE")
        self.publish_configuration()
    
    def fire_to_right(self, feedback):
        self.roi_assignments['fire']['enabled'] = True
        self.roi_assignments['fire']['side'] = 'right'
        rospy.loginfo("Fire detection: DEPLOY RIGHT SIDE")
        self.publish_configuration()
    
    # Smoke configuration callbacks
    def disable_smoke(self, feedback):
        self.roi_assignments['smoke']['enabled'] = False
        self.roi_assignments['smoke']['side'] = None
        rospy.loginfo("Smoke detection: DISABLED")
        self.publish_configuration()
    
    def observe_smoke(self, feedback):
        self.roi_assignments['smoke']['enabled'] = True
        self.roi_assignments['smoke']['side'] = None
        rospy.loginfo("Smoke detection: OBSERVE ONLY")
        self.publish_configuration()
    
    def smoke_to_left(self, feedback):
        self.roi_assignments['smoke']['enabled'] = True
        self.roi_assignments['smoke']['side'] = 'left'
        rospy.loginfo("Smoke detection: DEPLOY LEFT SIDE")
        self.publish_configuration()
    
    def smoke_to_right(self, feedback):
        self.roi_assignments['smoke']['enabled'] = True
        self.roi_assignments['smoke']['side'] = 'right'
        rospy.loginfo("Smoke detection: DEPLOY RIGHT SIDE")
        self.publish_configuration()
    
    # Human configuration callbacks
    def disable_human(self, feedback):
        self.roi_assignments['human']['enabled'] = False
        self.roi_assignments['human']['side'] = None
        rospy.loginfo("Human detection: DISABLED")
        self.publish_configuration()
    
    def observe_human(self, feedback):
        self.roi_assignments['human']['enabled'] = True
        self.roi_assignments['human']['side'] = None
        rospy.loginfo("Human detection: OBSERVE ONLY")
        self.publish_configuration()
    
    def human_to_left(self, feedback):
        self.roi_assignments['human']['enabled'] = True
        self.roi_assignments['human']['side'] = 'left'
        rospy.loginfo("Human detection: DEPLOY LEFT SIDE")
        self.publish_configuration()
    
    def human_to_right(self, feedback):
        self.roi_assignments['human']['enabled'] = True
        self.roi_assignments['human']['side'] = 'right'
        rospy.loginfo("Human detection: DEPLOY RIGHT SIDE")
        self.publish_configuration()
    
    # Bag configuration callbacks
    def disable_bag(self, feedback):
        self.roi_assignments['bag']['enabled'] = False
        self.roi_assignments['bag']['side'] = None
        rospy.loginfo("Bag detection: DISABLED")
        self.publish_configuration()
    
    def observe_bag(self, feedback):
        self.roi_assignments['bag']['enabled'] = True
        self.roi_assignments['bag']['side'] = None
        rospy.loginfo("Bag detection: OBSERVE ONLY")
        self.publish_configuration()
    
    def bag_to_left(self, feedback):
        self.roi_assignments['bag']['enabled'] = True
        self.roi_assignments['bag']['side'] = 'left'
        rospy.loginfo("Bag detection: DEPLOY LEFT SIDE")
        self.publish_configuration()
    
    def bag_to_right(self, feedback):
        self.roi_assignments['bag']['enabled'] = True
        self.roi_assignments['bag']['side'] = 'right'
        rospy.loginfo("Bag detection: DEPLOY RIGHT SIDE")
        self.publish_configuration()
    
    def show_config(self, feedback):
        """Display current configuration"""
        config_text = f"\n{'='*50}\n"
        config_text += f"CURRENT PAYLOAD CONFIGURATION\n"
        config_text += f"{'='*50}\n"
        
        spec = self.payload_specs[self.payload_type]
        config_text += f"Payload Type: {spec['name']}\n"
        config_text += f"  - Drop Angles: {spec['drop1_angle']}°/{spec['drop2_angle']}°\n"
        config_text += f"  - Speed: {spec['speed']} deg/sec\n"
        config_text += f"\nROI Assignments:\n"
        
        for roi_type, config in self.roi_assignments.items():
            if not config['enabled']:
                config_text += f"  ✗ {roi_type.upper()}: Disabled (detect only, no ROI)\n"
            elif config['side']:
                config_text += f"  ✓ {roi_type.upper()}: Deploy {config['side'].upper()} side\n"
            else:
                config_text += f"  ✓ {roi_type.upper()}: Observation only (ROI, no deploy)\n"
        
        config_text += f"{'='*50}\n"
        
        rospy.loginfo(config_text)
    
    def create_config_marker(self):
        """Create main configuration interactive marker"""
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "map"
        int_marker.name = "payload_config"
        int_marker.description = "Payload Configuration\n(Right-click for menu)"
        int_marker.pose.position.x = 0.0
        int_marker.pose.position.y = 3.0
        int_marker.pose.position.z = 0.5
        int_marker.scale = 0.5
        
        # Create a control for the marker
        control = InteractiveMarkerControl()
        control.interaction_mode = InteractiveMarkerControl.MENU
        control.always_visible = True
        
        # Add visual marker
        marker = Marker()
        marker.type = Marker.CUBE
        marker.scale.x = 0.4
        marker.scale.y = 0.4
        marker.scale.z = 0.2
        marker.color.r = 0.0
        marker.color.g = 0.8
        marker.color.b = 0.8
        marker.color.a = 0.8
        
        control.markers.append(marker)
        
        # Add text label
        text_marker = Marker()
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.text = "CONFIG"
        text_marker.scale.z = 0.15
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        text_marker.pose.position.z = 0.15
        
        control.markers.append(text_marker)
        
        int_marker.controls.append(control)
        
        self.server.insert(int_marker, self.marker_feedback)
        self.menu_handler.apply(self.server, int_marker.name)
        self.server.applyChanges()
    
    def create_roi_markers(self):
        """Create ROI status display markers"""
        positions = {
            'fire': (-1.5, 3.0),
            'smoke': (-0.5, 3.0),
            'human': (0.5, 3.0),
            'bag': (1.5, 3.0)
        }
        
        colors = {
            'fire': (1.0, 0.0, 0.0),    # Red
            'smoke': (0.5, 0.5, 0.5),    # Gray
            'human': (0.0, 0.0, 1.0),    # Blue
            'bag': (0.0, 1.0, 0.0)       # Green
        }
        
        for roi_type, pos in positions.items():
            int_marker = InteractiveMarker()
            int_marker.header.frame_id = "map"
            int_marker.name = f"roi_{roi_type}"
            int_marker.description = f"{roi_type.upper()}"
            int_marker.pose.position.x = pos[0]
            int_marker.pose.position.y = pos[1]
            int_marker.pose.position.z = 0.3
            int_marker.scale = 0.3
            
            control = InteractiveMarkerControl()
            control.interaction_mode = InteractiveMarkerControl.NONE
            control.always_visible = True
            
            # Visual marker
            marker = Marker()
            marker.type = Marker.CYLINDER
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.1
            
            color = colors[roi_type]
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 0.8
            
            control.markers.append(marker)
            int_marker.controls.append(control)
            
            self.server.insert(int_marker, self.marker_feedback)
            self.server.applyChanges()
    
    def marker_feedback(self, feedback):
        """Handle marker interactions"""
        pass
    
    def update_display(self, event):
        """Update visual display of configuration"""
        markers = MarkerArray()
        
        # Status text marker
        status_marker = Marker()
        status_marker.header.frame_id = "map"
        status_marker.type = Marker.TEXT_VIEW_FACING
        status_marker.id = 0
        status_marker.pose.position.x = 0.0
        status_marker.pose.position.y = 3.5
        status_marker.pose.position.z = 1.0
        status_marker.scale.z = 0.2
        status_marker.color.r = 1.0
        status_marker.color.g = 1.0
        status_marker.color.b = 1.0
        status_marker.color.a = 1.0
        
        spec = self.payload_specs[self.payload_type]
        status_text = f"{spec['name']}\n"
        status_text += f"{spec['drop1_angle']}°/{spec['drop2_angle']}° @ {spec['speed']}°/s"
        status_marker.text = status_text
        
        markers.markers.append(status_marker)
        
        # ROI status markers
        positions = {
            'fire': (-1.5, 3.0),
            'smoke': (-0.5, 3.0),
            'human': (0.5, 3.0),
            'bag': (1.5, 3.0)
        }
        
        for i, (roi_type, pos) in enumerate(positions.items(), 1):
            text_marker = Marker()
            text_marker.header.frame_id = "map"
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.id = i
            text_marker.pose.position.x = pos[0]
            text_marker.pose.position.y = pos[1]
            text_marker.pose.position.z = 0.6
            text_marker.scale.z = 0.12
            
            config = self.roi_assignments[roi_type]
            if not config['enabled']:
                text_marker.color.r = 0.5
                text_marker.color.g = 0.5
                text_marker.color.b = 0.5
                text_marker.text = f"{roi_type.upper()}\n✗ OFF"
            elif config['side']:
                text_marker.color.r = 0.0
                text_marker.color.g = 1.0
                text_marker.color.b = 0.0
                text_marker.text = f"{roi_type.upper()}\n✓ {config['side'].upper()}"
            else:
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 0.0
                text_marker.text = f"{roi_type.upper()}\n✓ OBS"
            
            text_marker.color.a = 1.0
            markers.markers.append(text_marker)
        
        self.pub_display.publish(markers)
    
    def publish_configuration(self):
        """Publish the current configuration"""
        config = {
            'payload_type': self.payload_type,
            'payload_specs': self.payload_specs[self.payload_type],
            'roi_assignments': self.roi_assignments
        }
        config_msg = String()
        config_msg.data = json.dumps(config)
        self.pub_config.publish(config_msg)
        rospy.loginfo("Configuration published")
    
    def shutdown(self):
        self.server.clear()
        self.server.applyChanges()
        rospy.loginfo("Payload Configuration GUI shutdown")

def main():
    try:
        gui = PayloadConfigGUI()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in Payload Config GUI: {e}")
    finally:
        if 'gui' in locals():
            gui.shutdown()

if __name__ == '__main__':
    main()