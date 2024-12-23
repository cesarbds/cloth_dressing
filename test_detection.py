from pyrcareworld.envs.dressing_env import DressingEnv
import pyrcareworld.attributes as attr
import cv2
import numpy as np
from numpy.typing import ArrayLike
from pyrcareworld.attributes import CameraAttr
from utils_sensor import *
from pyrcareworld.attributes import PersonRandomizerAttr



intrinsics = np.eye(3)

intrinsics[0, 0] = 500
intrinsics[1, 1] = 500

intrinsics[0, 2] = 512/2
intrinsics[1, 2] = 512/2


def main(use_graphics=False):
    env, robot = setup_environment(use_graphics)
    
    # Gripper operation
    gripper = env.GetAttr(3158930)
    
    print(gripper.data['joint_positions'])
    
    # Cloth attributes and particles
    cloth = env.get_cloth()
    cloth_attr = attr.ClothAttr(env, cloth.id)
    cloth_attr.GetParticles()
    env.step()


    # Initial position
    robot.EnabledNativeIK(False)
    robot.SetJointPositionDirectly([173.68829345703125, -33.00761795043945, 179.9871063232422, -91.84587860107422, -39.07077407836914, -63.777915954589844, -136.49273681640625])
    env.step(50)
    robot.EnabledNativeIK(True)

    # Camera setup
    camera = env.get_camera()
    setup_camera(camera, gripper)
    rgb, normal, depth = process_images(env, camera)
    
    # Move to first target position
    target_position_1 = [1.8999999952316284, 1.697000012397766, 0.2689999932050705]
    particles_camera_1 = world2camera(target_position_1, camera, intrinsics)
    grasping_normal_1 = get_normal_from_camera(camera, particles_camera_1)
    move_robot_to_point_1(robot, target_position_1, grasping_normal_1, gripper, env)
    
    env.step()  
    

    # Left Arm
    execute_trajectory(robot, env)
    env.step()
    execute_robot_operations(robot, gripper, env)
    gripper.GripperOpen()
    robot.WaitDo()
    env.step()
    
    # Grasp Cloth
    move_robot(robot, [0.9892417, 1.691249212, 0.094629], [357.46221923828125, 237.6537322998047, 182.7155303955078], env)   
    robot.WaitDo()
    env.step()
    particle_index = 14
    particle = get_specific_particle(env, particle_index)
    particle[2] -= .02
    print(f"Partícula {particle_index}: {particle}")
    move_robot(robot, particle, [357.46221923828125, 237.6537322998047, 182.7155303955078], env )
    robot.WaitDo()
    env.step()
    gripper.GripperClose()
    robot.WaitDo()
    env.step()
    
    pos_inter = particle
    pos_inter[2] += 0.07
            
    move_robot(robot, pos_inter ,[357.46221923828125, 237.6537322998047, 182.7155303955078], env )
    robot.WaitDo()
    env.step()
    
    # Righ Arm
    execute_trajectory_2(robot, env)
    env.step()
    
    execute_robot_operations_2(robot, gripper, env)
    print("ACAAAAAAABOOOOOUUUU")
    gripper.GripperOpen()
    robot.WaitDo()
    env.step()
    exit(-1)

    try:
        while True:
            env.step()  # Continue advancing the environment
    except KeyboardInterrupt:
        print("Simulation ended.")

        
if __name__ == "__main__":
    main()


