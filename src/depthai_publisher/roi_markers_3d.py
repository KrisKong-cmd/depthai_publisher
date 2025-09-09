#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


def make_color(r, g, b, a=1.0):
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c


CLASS_TOPICS = {
    "bag": "/yolo_detection/roi/bag",
    "fire": "/yolo_detection/roi/fire",
    "human": "/yolo_detection/roi/human",
    "smoke": "/yolo_detection/roi/smoke",
}

CLASS_IDS = {name: idx for idx, name in enumerate(["bag", "fire", "human", "smoke"]) }

CLASS_COLORS = {
    "bag": make_color(0.0, 1.0, 0.0, 0.95),     # green
    "fire": make_color(1.0, 0.1, 0.0, 0.95),    # red
    "human": make_color(0.0, 0.8, 1.0, 0.95),   # cyan
    "smoke": make_color(0.8, 0.8, 0.8, 0.9),    # gray
}


class ROIMarkers3D:
    def __init__(self):
        rospy.init_node("roi_markers_3d")

        # Params
        self.markers_topic = rospy.get_param("~markers_topic", "/ml_targets/visualization")
        self.ns = rospy.get_param("~namespace", "ml_targets")
        self.text_ns = self.ns + "_labels"
        self.scale_m = float(rospy.get_param("~marker_scale", 0.25))
        self.text_scale_m = float(rospy.get_param("~text_height", 0.16))
        self.text_offset_m = float(rospy.get_param("~text_z_offset", 0.30))
        self.shape = int(rospy.get_param("~shape", Marker.CYLINDER))  # CYLINDER or SPHERE/CUBE
        self.timeout_s = float(rospy.get_param("~timeout", 1.0))
        self.rate_hz = float(rospy.get_param("~rate", 10.0))

        # State for latest detections
        self.latest = {
            name: {"pose": None, "stamp": rospy.Time(0), "frame": ""}
            for name in CLASS_TOPICS.keys()
        }

        # Publishers/Subscribers
        self.pub = rospy.Publisher(self.markers_topic, MarkerArray, queue_size=10)
        self.subs = []
        for name, topic in CLASS_TOPICS.items():
            self.subs.append(rospy.Subscriber(topic, PoseStamped, self._mk_cb(name), queue_size=10))

        rospy.loginfo("roi_markers_3d: publishing MarkerArray on %s", self.markers_topic)
        for n, t in CLASS_TOPICS.items():
            rospy.loginfo("  subscribing %s -> %s", n, t)

    def _mk_cb(self, name):
        def _cb(msg):
            self.latest[name]["pose"] = msg.pose
            self.latest[name]["stamp"] = msg.header.stamp
            self.latest[name]["frame"] = msg.header.frame_id or "map"
        return _cb

    def _delete_all(self, frame_id):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = self.ns
        m.action = Marker.DELETEALL
        return m

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            ma = MarkerArray()

            # Use a consistent frame to delete; fall back to map
            any_frame = next((v["frame"] for v in self.latest.values() if v["frame"]), "map")
            ma.markers.append(self._delete_all(any_frame))

            idx = 0
            for name, data in self.latest.items():
                pose = data["pose"]
                if pose is None:
                    continue
                if (now - data["stamp"]).to_sec() > self.timeout_s:
                    continue

                frame_id = data["frame"] or "map"

                # 3D marker
                m = Marker()
                m.header.frame_id = frame_id
                m.header.stamp = now
                m.ns = self.ns
                m.id = CLASS_IDS.get(name, idx)
                m.type = self.shape
                m.action = Marker.ADD
                m.pose = pose
                m.scale.x = self.scale_m
                m.scale.y = self.scale_m
                # Cylinder needs small z to look like disk; otherwise sphere/cube use same scale
                m.scale.z = self.scale_m * (0.08 if self.shape == Marker.CYLINDER else 1.0)
                m.color = CLASS_COLORS.get(name, make_color(1.0, 1.0, 1.0, 0.95))
                ma.markers.append(m)

                # Label
                t = Marker()
                t.header.frame_id = frame_id
                t.header.stamp = now
                t.ns = self.text_ns
                t.id = CLASS_IDS.get(name, idx)
                t.type = Marker.TEXT_VIEW_FACING
                t.action = Marker.ADD
                t.pose.position = pose.position
                t.pose.position.z += self.text_offset_m
                t.scale.z = self.text_scale_m
                t.color = make_color(1.0, 1.0, 1.0, 0.95)
                t.text = name
                ma.markers.append(t)

                idx += 1

            self.pub.publish(ma)
            rate.sleep()


def main():
    node = ROIMarkers3D()
    node.spin()


if __name__ == "__main__":
    main()

