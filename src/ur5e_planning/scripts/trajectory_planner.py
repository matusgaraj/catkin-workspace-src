#!/usr/bin/env python3
"""
tomato_trajectory_planner.py:
Subscribes to /tomato_position (Marker). On startup and after every successful pick dive,
the UR5e returns to a fixed “home” joint configuration for observation:
  shoulder_pan = -90°
  shoulder_lift = -60°
  elbow        = -140°
  wrist_1      =  0°
  wrist_2      =  90°
  wrist_3      = -180°
Then when a tomato is detected, the planner waits for 3 consistent detections
within 2 cm, and then dives straight down from (radius + 0.02 m) above that point.

This version also inserts:
  – a “table” at z = 0 m (5 cm thick) to block any under-table paths  
  – two side walls at x = ±0.30 m (2 cm thick) to keep the arm from swinging too far left/right
"""

import rospy
import math
import moveit_commander
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from moveit_commander import PlanningSceneInterface
from collections import deque

class TomatoTrajectoryPlanner:
    def __init__(self):
        rospy.init_node("tomato_trajectory_planner", anonymous=True)

        # 1) MoveIt setup
        moveit_commander.roscpp_initialize([])
        self.group = moveit_commander.MoveGroupCommander("manipulator")
        self.group.set_planning_time(5.0)
        self.group.set_max_velocity_scaling_factor(0.5)
        self.group.set_max_acceleration_scaling_factor(0.5)
        self.group.allow_replanning(True)

        # 2) TF for frame conversions
        self.tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf_buffer)

        # 4) How long to hold after diving before returning home
        self.return_delay = 5.0  # seconds

        # 5) HOME JOINTS for observation
        self.home_joint_angles = [
            math.radians(-90.0),   # shoulder_pan_joint
            math.radians(-60.0),   # shoulder_lift_joint
            math.radians(-140.0),  # elbow_joint
            math.radians(0.0),     # wrist_1_joint
            math.radians(90.0),    # wrist_2_joint
            math.radians(0.0)      # wrist_3_joint
        ]

        # 6) Confirmation window for stable detection
        self._window     = deque(maxlen=3)
        self._confirm_tol = 0.02  # meters

        # 7) Marker publisher for the confirmed target
        self.target_pub = rospy.Publisher("/tomato_target", Marker, queue_size=1)

        rospy.loginfo("UR5e trajectory planner initialized.")
        self.go_to_home()

        # 3) Publish collision geometry (table + side-walls)
        self.scene = PlanningSceneInterface()
        rospy.sleep(1.0)  # allow the scene to come up

        # -- table top: 2×2 m, 5 cm thick, sitting at z = 0
        table = PoseStamped(
            header=Header(frame_id="base_link"),
            pose=Pose(
                position=Point(0.0, 0.0, -0.035),
                orientation=Quaternion(0,0,0,1)
            )
        )
        self.scene.add_box("table_top", table, size=(2.0, 2.0, 0.05))

        # -- left wall at x = +0.30 m, 2 cm thick in X, 2 m deep, 1.5 m high
        left_wall = PoseStamped(
            header=Header(frame_id="base_link"),
            pose=Pose(
                position=Point( 0.30, 0.0, 0.75),
                orientation=Quaternion(0,0,0,1)
            )
        )
        self.scene.add_box("left_wall", left_wall, size=(0.02, 2.0, 1.50))

        # -- right wall at x = -0.30 m
        right_wall = PoseStamped(
            header=Header(frame_id="base_link"),
            pose=Pose(
                position=Point(-0.30, 0.0, 0.75),
                orientation=Quaternion(0,0,0,1)
            )
        )
        self.scene.add_box("right_wall", right_wall, size=(0.02, 2.0, 1.50))


        # 8) Subscribe to incoming tomato positions
        self.busy = False
        rospy.Subscriber("/tomato_position", Marker,
                         self.coord_callback, queue_size=1)

    def go_to_home(self):
        """Send the arm to the fixed home joint configuration."""
        self.group.set_start_state_to_current_state()
        self.group.clear_pose_targets()
        rospy.loginfo("Planning to home joint configuration for observation...")
        self.group.go(self.home_joint_angles, wait=True)
        self.group.stop()
        rospy.loginfo("Reached home observation pose.")

    def coord_callback(self, marker: Marker):
        if self.busy:
            rospy.logwarn("Busy—ignoring new request.")
            return

        # extract position & radius from the marker
        tomato_radius = marker.scale.x / 2.0
        ps = PoseStamped(header=marker.header, pose=marker.pose)

        # reframe into planning frame if needed
        planning_frame = self.group.get_planning_frame()
        if ps.header.frame_id != planning_frame:
            try:
                ps = self.tf_buffer.transform(ps, planning_frame, rospy.Duration(0.5))
            except Exception as e:
                rospy.logwarn(f"TF to {planning_frame} failed: {e}")
                return

        px, py, pz = (ps.pose.position.x,
                      ps.pose.position.y,
                      ps.pose.position.z)

        # push into confirmation window
        self._window.append((px, py, pz, tomato_radius))
        if len(self._window) < self._window.maxlen:
            rospy.loginfo("Waiting for stable detection (%d/%d)",
                          len(self._window), self._window.maxlen)
            return

        # check that all three are within tolerance
        pts = list(self._window)
        stable = True
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                dz = pts[i][2] - pts[j][2]
                if dx*dx + dy*dy + dz*dz > self._confirm_tol**2:
                    stable = False
                    break
            if not stable:
                break

        if not stable:
            rospy.logwarn("Detections not stable—resetting window")
            self._window.clear()
            return

        # confirmed: use the latest
        px, py, pz, tomato_radius = self._window[-1]
        self._window.clear()
        rospy.loginfo("Confirmed tomato at x=%.3f y=%.3f z=%.3f radius=%.3f",
                      px, py, pz, tomato_radius)

        # visualize the confirmed tomato 
        target = Marker()
        target.header.frame_id = planning_frame
        target.header.stamp    = rospy.Time.now()
        target.ns              = "tomato_target"
        target.id              = 0
        target.type            = Marker.SPHERE
        target.action          = Marker.ADD

        target.pose.position.x = px
        target.pose.position.y = py
        target.pose.position.z = pz

        d = 2 * tomato_radius
        target.scale.x = target.scale.y = target.scale.z = d
        target.color.r, target.color.g, target.color.b, target.color.a = 1.0, 0.0, 0.0, 1.0

        target.lifetime = rospy.Duration(0)
        self.target_pub.publish(target)

        #reachability check on the confirmed point
        if abs(px) > 1.0 or abs(py) > 1.0 or abs(pz) > 2.0:
            rospy.logwarn("Tomato out of reach—ignoring.")
            return

        # build the dive pose
        dive = PoseStamped(
            header=Header(frame_id=planning_frame, stamp=rospy.Time.now()),
            pose=Pose(
                position=Point(px, py, pz + tomato_radius + 0.02),
                orientation=Quaternion(1,0,0,0)  # wrist straight down
            )
        )

        # plan & execute
        self.busy = True
        self.group.set_start_state_to_current_state()
        self.group.clear_pose_targets()
        ee = self.group.get_end_effector_link()
        self.group.set_pose_target(dive, ee)

        rospy.loginfo("Planning dive to (%.3f, %.3f, %.3f)...",
                      dive.pose.position.x,
                      dive.pose.position.y,
                      dive.pose.position.z)
        plan_result = self.group.plan()
        # support both tuple and single‐result APIs
        plan = (plan_result[1] if isinstance(plan_result, (tuple,list)) and len(plan_result)>1
                else plan_result)

        if not plan or not getattr(plan, "joint_trajectory", None) \
           or len(plan.joint_trajectory.points) == 0:
            rospy.logerr("No valid dive plan—aborting.")
            self.busy = False
            return

        rospy.loginfo("Executing dive...")
        success = self.group.execute(plan, wait=True)
        self.group.stop()
        self.group.clear_pose_targets()

        if success:
            rospy.loginfo("Dive complete; holding %.1f s then returning home.", self.return_delay)
            rospy.sleep(self.return_delay)
            self.go_to_home()
        else:
            rospy.logwarn("Dive execution failed.")

        self.busy = False


def main():
    try:
        TomatoTrajectoryPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()
