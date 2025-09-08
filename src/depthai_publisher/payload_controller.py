#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int32, String
import pigpio
import time
import threading
import json

class PayloadController:
    def __init__(self):
        rospy.init_node('payload_controller', anonymous=True)
        
        # Get parameters
        self.servo_pin = rospy.get_param("~servo_pin", 12)
        self.min_pulse = rospy.get_param("~min_pulse", 500)
        self.max_pulse = rospy.get_param("~max_pulse", 2500)
        self.neutral_angle = rospy.get_param("~neutral_angle", 90)
        
        # Default configuration (Tino's payload)
        self.config = {
            'payload_type': 'tino',
            'payload_specs': {
                'drop1_angle': 140,
                'drop2_angle': 40,
                'speed': 60
            },
            'roi_assignments': {
                'fire': {'enabled': True, 'side': 'left'},
                'smoke': {'enabled': True, 'side': 'right'},
                'human': {'enabled': False, 'side': None},
                'bag': {'enabled': False, 'side': None}
            },
            'locked': False
        }
        
        # Track which payloads have been dropped
        self.left_dropped = False
        self.right_dropped = False
        self.drop_in_progress = False
        
        # Current servo angle
        self.current_angle = self.neutral_angle
        
        # Initialize pigpio
        self.pi = pigpio.pi()
        if not self.pi.connected:
            rospy.logerr("Could not connect to pigpio daemon")
            raise RuntimeError("Could not connect to pigpio daemon")
        
        rospy.loginfo(f"Payload controller initialized on GPIO pin {self.servo_pin}")
        
        # Initialize servo to neutral position
        self.set_servo_angle(self.neutral_angle, instant=True)
        rospy.loginfo(f"Servo initialized to neutral position ({self.neutral_angle}°)")
        
        # Subscribers
        self.sub_config = rospy.Subscriber('/payload/configuration', String, self.config_callback)
        self.sub_drop_command = rospy.Subscriber('/payload/drop', Int32, self.drop_callback)
        self.sub_drop_fire = rospy.Subscriber('/payload/drop/fire', String, self.drop_fire_callback)
        self.sub_drop_smoke = rospy.Subscriber('/payload/drop/smoke', String, self.drop_smoke_callback)
        self.sub_drop_human = rospy.Subscriber('/payload/drop/human', String, self.drop_human_callback)
        self.sub_drop_bag = rospy.Subscriber('/payload/drop/bag', String, self.drop_bag_callback)
        
        # Publishers
        self.pub_status = rospy.Publisher('/payload/status', String, queue_size=10)
        
        rospy.on_shutdown(self.shutdown)
        
        rospy.loginfo("Payload controller ready with default configuration")
    
    def config_callback(self, msg):
        """Update configuration from GUI"""
        try:
            new_config = json.loads(msg.data)
            self.config = new_config
            
            if new_config['locked']:
                rospy.loginfo(f"Configuration LOCKED: {new_config['payload_type']} payload")
                rospy.loginfo(f"  Angles: {new_config['payload_specs']['drop1_angle']}° / {new_config['payload_specs']['drop2_angle']}°")
                rospy.loginfo(f"  Speed: {new_config['payload_specs']['speed']}°/s")
                
                # Log ROI assignments
                for roi, assignment in new_config['roi_assignments'].items():
                    if assignment['enabled']:
                        if assignment['side']:
                            rospy.loginfo(f"  {roi.upper()}: Deploy {assignment['side']} side")
                        else:
                            rospy.loginfo(f"  {roi.upper()}: Observation only")
        except Exception as e:
            rospy.logwarn(f"Error updating configuration: {e}")
    
    def angle_to_pulse(self, angle):
        """Convert angle to pulse width"""
        return self.min_pulse + (angle / 180.0) * (self.max_pulse - self.min_pulse)
    
    def set_servo_angle(self, target_angle, instant=False):
        """Move servo to target angle"""
        target_angle = max(0, min(180, target_angle))
        
        if instant:
            # Instant movement
            pulse = self.angle_to_pulse(target_angle)
            self.pi.set_servo_pulsewidth(self.servo_pin, pulse)
            self.current_angle = target_angle
        else:
            # Smooth movement at configured speed
            speed = self.config['payload_specs']['speed']
            angle_diff = target_angle - self.current_angle
            if abs(angle_diff) < 0.5:
                return
            
            step_time = 0.02
            angle_step = speed * step_time
            
            if angle_diff > 0:
                angle_step = abs(angle_step)
            else:
                angle_step = -abs(angle_step)
            
            steps = int(abs(angle_diff) / abs(angle_step))
            
            for i in range(steps):
                self.current_angle += angle_step
                pulse = self.angle_to_pulse(self.current_angle)
                self.pi.set_servo_pulsewidth(self.servo_pin, pulse)
                time.sleep(step_time)
            
            # Final adjustment
            self.current_angle = target_angle
            pulse = self.angle_to_pulse(target_angle)
            self.pi.set_servo_pulsewidth(self.servo_pin, pulse)
    
    def drop_payload_side(self, side):
        """Execute payload drop for specified side"""
        if self.drop_in_progress:
            rospy.logwarn("Drop already in progress, ignoring request")
            return
        
        self.drop_in_progress = True
        
        try:
            if side == 'left':
                if self.left_dropped:
                    rospy.logwarn("Left payload already dropped")
                    return
                
                angle = self.config['payload_specs']['drop1_angle']
                rospy.loginfo(f"Dropping LEFT payload at {angle}°...")
                status_msg = String()
                status_msg.data = f"Dropping LEFT payload ({self.config['payload_type']})"
                self.pub_status.publish(status_msg)
                
                # Move to drop position
                self.set_servo_angle(angle)
                time.sleep(2)  # Wait for payload to drop
                
                # Return to neutral
                self.set_servo_angle(self.neutral_angle)
                
                self.left_dropped = True
                rospy.loginfo("LEFT payload dropped successfully")
                status_msg.data = "LEFT payload dropped"
                self.pub_status.publish(status_msg)
                
            elif side == 'right':
                if self.right_dropped:
                    rospy.logwarn("Right payload already dropped")
                    return
                
                angle = self.config['payload_specs']['drop2_angle']
                rospy.loginfo(f"Dropping RIGHT payload at {angle}°...")
                status_msg = String()
                status_msg.data = f"Dropping RIGHT payload ({self.config['payload_type']})"
                self.pub_status.publish(status_msg)
                
                # Move to drop position
                self.set_servo_angle(angle)
                time.sleep(2)  # Wait for payload to drop
                
                # Return to neutral
                self.set_servo_angle(self.neutral_angle)
                
                self.right_dropped = True
                rospy.loginfo("RIGHT payload dropped successfully")
                status_msg.data = "RIGHT payload dropped"
                self.pub_status.publish(status_msg)
                
        finally:
            self.drop_in_progress = False
    
    def get_side_for_roi(self, roi_type):
        """Get which side to drop for a given ROI type"""
        if roi_type in self.config['roi_assignments']:
            assignment = self.config['roi_assignments'][roi_type]
            if assignment['enabled'] and assignment['side']:
                return assignment['side']
        return None
    
    def drop_callback(self, msg):
        """Handle generic drop command (1=left, 2=right)"""
        if msg.data == 1:
            thread = threading.Thread(target=self.drop_payload_side, args=('left',))
            thread.start()
        elif msg.data == 2:
            thread = threading.Thread(target=self.drop_payload_side, args=('right',))
            thread.start()
    
    def drop_fire_callback(self, msg):
        """Handle fire detection"""
        side = self.get_side_for_roi('fire')
        if side:
            rospy.loginfo(f"Fire detected - dropping {side} payload")
            thread = threading.Thread(target=self.drop_payload_side, args=(side,))
            thread.start()
        else:
            rospy.loginfo("Fire detected but no payload assigned or disabled")
    
    def drop_smoke_callback(self, msg):
        """Handle smoke detection"""
        side = self.get_side_for_roi('smoke')
        if side:
            rospy.loginfo(f"Smoke detected - dropping {side} payload")
            thread = threading.Thread(target=self.drop_payload_side, args=(side,))
            thread.start()
        else:
            rospy.loginfo("Smoke detected but no payload assigned or disabled")
    
    def drop_human_callback(self, msg):
        """Handle human detection (observation only)"""
        if self.config['roi_assignments']['human']['enabled']:
            rospy.loginfo("Human detected - observation only, no payload drop")
    
    def drop_bag_callback(self, msg):
        """Handle bag detection (observation only)"""
        if self.config['roi_assignments']['bag']['enabled']:
            rospy.loginfo("Bag detected - observation only, no payload drop")
    
    def shutdown(self):
        """Clean shutdown"""
        rospy.loginfo("Shutting down payload controller")
        # Return to neutral before shutdown
        self.set_servo_angle(self.neutral_angle, instant=True)
        time.sleep(0.5)
        # Stop sending pulses
        self.pi.set_servo_pulsewidth(self.servo_pin, 0)
        self.pi.stop()

def main():
    try:
        controller = PayloadController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in payload controller: {e}")

if __name__ == '__main__':
    main()