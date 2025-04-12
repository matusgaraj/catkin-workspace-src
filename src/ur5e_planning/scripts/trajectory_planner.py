#!/usr/bin/env python3
"""
tomato_trajectory_planner.py: 
ROS uzol, ktorý sa subscribuje na tému /tomato_position (PointStamped) a
naplánuje + exekvuje pohyb UR5e pomocou MoveIt! do prijatej polohy. 
"""

import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped, PointStamped

class TomatoTrajectoryPlanner:
    def __init__(self):
        rospy.init_node("tomato_trajectory_planner", anonymous=True)

        # Inicializácia moveit_commander
        moveit_commander.roscpp_initialize([])
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("manipulator")

        # Prípadne upravte parametre plánovania a exekúcie
        self.group.set_planning_time(5.0)
        self.group.set_max_velocity_scaling_factor(0.5)
        self.group.set_max_acceleration_scaling_factor(0.5)

        rospy.loginfo("Inicializácia UR5e trajectory planner uzla dokončená.")

        # Subscribujeme na tému s testovacími súradnicami
        self.sub = rospy.Subscriber("/tomato_position", PointStamped, self.coord_callback, queue_size=1)

        # Ochraný flag, aby sme neplánovali opakovane, kým robot vykonáva akciu
        self.busy = False

    def coord_callback(self, msg):
        """
        Keď príde PointStamped so súradnicami, naplánujeme
        pohyb koncového efektora do tejto polohy (jednoduchá neutrálna orientácia).
        """
        if self.busy:
            rospy.logwarn("Robot je zaneprázdnený, správa ignorovaná.")
            return

        rospy.loginfo("Prijaté súradnice x=%.3f, y=%.3f, z=%.3f v rámci '%s'.",
                      msg.point.x, msg.point.y, msg.point.z, msg.header.frame_id)

        self.busy = True

        # 1) Nastavíme začiatok na aktuálny stav (istota pre MoveIt!)
        self.group.set_start_state_to_current_state()

        # 2) Vytvoríme PoseStamped z PointStamped
        target_pose = PoseStamped()
        target_pose.header = msg.header  # Kopírujeme frame_id, stamp
        target_pose.pose.position = msg.point
        # Jednoduchá neutrálna orientácia (pri potrebe inej rotácie upravte)
        target_pose.pose.orientation.x = 0.0
        target_pose.pose.orientation.y = 0.0
        target_pose.pose.orientation.z = 0.0
        target_pose.pose.orientation.w = 1.0

        # 3) Nastavíme cieľ
        self.group.set_pose_target(target_pose)

        rospy.loginfo("Plánujem trajektóriu do prijatých súradníc.")
        plan_result = self.group.plan()

        # Log: nech vidíme, čo plan() vrátilo
        rospy.loginfo("Výsledok group.plan(): %s", str(plan_result))

        # Nasleduje rovnaká logika parsovania n-tice / bool / RobotTrajectory
        plan = None
        success = False

        if isinstance(plan_result, bool):
            # buď True/False
            rospy.loginfo("plan() vrátil bool: %s", plan_result)
            if plan_result:
                success = True
            else:
                success = False

        elif isinstance(plan_result, tuple):
            # Môže ísť o (True, RobotTrajectory, fraction, ...)
            rospy.loginfo("plan() vrátil n-ticu s dĺžkou %d", len(plan_result))
            if len(plan_result) >= 2:
                first_item = plan_result[0]
                second_item = plan_result[1]

                if isinstance(first_item, bool) and first_item is True:
                    rospy.loginfo("Prvá položka bool==True => plan je v druhej položke")
                    if hasattr(second_item, 'joint_trajectory'):
                        plan = second_item
                        if len(plan.joint_trajectory.points) > 0:
                            success = True
                    else:
                        rospy.logwarn("Druhá položka nie je RobotTrajectory, skúšam plan_result[2]")
                        if len(plan_result) >= 3:
                            maybe_traj = plan_result[2]
                            if hasattr(maybe_traj, 'joint_trajectory'):
                                plan = maybe_traj
                                if len(plan.joint_trajectory.points) > 0:
                                    success = True
                else:
                    # starší formát (plan, fraction) => plan je first_item
                    if hasattr(first_item, 'joint_trajectory'):
                        plan = first_item
                        if len(plan.joint_trajectory.points) > 0:
                            success = True
                    else:
                        rospy.logwarn("Nepodarilo sa extrahovať RobotTrajectory z n-tice.")
            else:
                rospy.logwarn("n-tica príliš krátka, neviem získať plan => zlyhanie.")
        else:
            # Môže to byť priamo RobotTrajectory
            if hasattr(plan_result, 'joint_trajectory'):
                plan = plan_result
                if len(plan.joint_trajectory.points) > 0:
                    success = True

        # 4) Exekúcia, ak success
        if success:
            rospy.loginfo("Trajektória bola naplánovaná, exekúcia pohybu...")
            if plan is None:
                # plan() vrátil len True, bez trajektórie => go()
                self.group.go(wait=True)
            else:
                self.group.execute(plan, wait=True)

            rospy.loginfo("Robot by mal byť v danej polohe.")
        else:
            rospy.logwarn("Plánovanie zlyhalo, exekúcia sa neuskutoční.")

        self.group.clear_pose_targets()
        self.busy = False

def main():
    planner = TomatoTrajectoryPlanner()
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
