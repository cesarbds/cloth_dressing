from pyrcareworld.envs.dressing_env import DressingEnv
import pyrcareworld.attributes as attr
import cv2
import numpy as np
from numpy.typing import ArrayLike
from pyrcareworld.attributes import CameraAttr
from body_pose_sensor import *

import mediapipe as mp
from pyrcareworld.envs.dressing_env import DressingEnv
import pyrcareworld.attributes as attr
import numpy as np

# Inicializa o módulo de pose do MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

intrinsics = np.eye(3)

intrinsics[0, 0] = 500
intrinsics[1, 1] = 500

intrinsics[0, 2] = 512/2
intrinsics[1, 2] = 512/2


def get_specific_particle(env, particle_index):
    # Obter o tecido do ambiente
    cloth = env.get_cloth()
    
    # Obter os atributos do tecido
    cloth_attr = attr.ClothAttr(env, cloth.id)
    cloth_attr.GetParticles()
    env.step()
    
    # Obter os dados das partículas
    particles_data = cloth.data.get('particles', None)
    
    if particles_data is None:
        print("Não foi possível obter os dados das partículas.")
        return None
    
    # Converter os dados das partículas para um array numpy
    particles = np.array(particles_data)
    
    if particle_index >= len(particles):
        print(f"Índice {particle_index} está fora do intervalo de partículas disponíveis.")
        return None
    
    # Selecionar a partícula específica pelo índice
    specific_particle = particles[particle_index]
    
    return specific_particle


def euclidean_distance(particle_pos, target_pos):
    """Calculates the Euclidean distance between two 3D points."""
    return np.linalg.norm(np.array(particle_pos) - np.array(target_pos))

def setup_environment(use_graphics):
    """Initializes the DressingEnv environment and returns the robot and gripper objects."""
    env = DressingEnv(graphics=True)
    robot = env.get_robot()
    env.step()
    return env, robot

def operate_gripper(gripper, env, open_gripper=True):
    """Opens or closes the gripper and advances the simulation."""
    if open_gripper:
        gripper.GripperOpen()
    else:
        gripper.GripperClose()
    env.step()

def setup_camera(camera, gripper):
    """Configures the camera position and retrieves RGB, normal, and depth data."""
    camera.SetTransform(position=[0.45, 2.1, 0.5], rotation=[150, -90, 180])
    # camera.SetParent(gripper.id)  # Assegurar que o ID do gripper esteja correto
    camera.GetRGB(512, 512)
    camera.GetNormal(512, 512)
    camera.GetDepth(0.1, 2.0, 512, 512)


def video(camera, env):
    camera.GetRGB(512, 512)
    env.step()
    rgb = np.frombuffer(camera.data["rgb"], dtype=np.uint8)
    rgb = cv2.imdecode(rgb, cv2.IMREAD_COLOR)
    cv2.imshow("show", rgb)
    cv2.waitKey(1)

def process_images(env, camera):
    """Processes the RGB, normal, and depth images captured by the camera."""
    env.step()
    rgb = np.frombuffer(camera.data["rgb"], dtype=np.uint8)
    rgb = cv2.imdecode(rgb, cv2.IMREAD_COLOR)
    cv2.imwrite("rgb_hand.png", rgb)

    normal = np.frombuffer(camera.data["normal"], dtype=np.uint8)
    normal = cv2.imdecode(normal, cv2.IMREAD_COLOR)
    cv2.imwrite("normal_hand.png", normal)

    depth = np.frombuffer(camera.data["depth"], dtype=np.uint8)
    depth = cv2.imdecode(depth, cv2.IMREAD_GRAYSCALE)
    depth = depth.astype(np.float32) / 255.0  # Convert to depth values in meters
    np.save("depth_hand.npy", depth)

    return rgb, normal, depth

def instance_ball(env):
    """Instances a ball object in the environment."""
    ball = env.InstanceObject(name="Rigidbody_Sphere", id=24252, attr_type=attr.RigidbodyAttr)
    ball.SetTransform(position=[3.45, 2.1, 0.5], rotation=[150, -75, 180])
    ball.SetKinematic(False)
    ball.SetScale([0.025, 0.025, 0.025])
    return ball

def calculate_distances(particles_data, pos_2):
    """Calculates the distances between the cloth particles and a target position."""
    distances = [(i, euclidean_distance(p, pos_2)) for i, p in enumerate(particles_data)]
    distances.sort(key=lambda x: x[1])
    return distances

def world2camera(world_position : ArrayLike, camera : CameraAttr, intrinsics: ArrayLike) -> np.ndarray:
    P = np.zeros((3,4))
    P[:3,:3] = intrinsics

    T = np.array(camera.data["local_to_world_matrix"])
    T = np.linalg.inv(T)

    P = P @ T

    X = np.empty(4)
    X[:3] = world_position
    X[3] = 1

    x = P @ X
    x /= x[2]

    point = x[:2].astype(int)
    point[1] = 512 - point[1]

    return point

def get_normal_from_camera(camera, particles_camera):
    """Obtains the surface normal from the normal image captured by the camera."""
    normal_image = np.frombuffer(camera.data["normal"], dtype=np.uint8)
    normal_image = cv2.imdecode(normal_image, cv2.IMREAD_COLOR)
    
    # Convert the target position from 3D coordinates to 2D image coordinates
    height, width, _ = normal_image.shape
    u = int(particles_camera[0] * width) % width
    v = int(particles_camera[1] * height) % height

    # Ensure u and v are within image bounds
    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)

    # Get the surface normal at (u, v)
    normal = normal_image[v, u, :]
    normal = (normal / 255.0) * 2 - 1  # Convert from [0, 255] to [-1, 1]
    return normal

def move_robot_to_point_1(robot, target_position, grasping_normal, gripper, env):
    """Moves the robot to the grasping point with the calculated orientation and closes the gripper."""
    # Define the gripper orientation
    y_axis = grasping_normal
    x_axis = np.array([1, 0, 0])  # Arbitray vector, adjust as needed
    z_axis = np.cross(y_axis, x_axis) 
    x_axis = np.cross(z_axis, y_axis)  # Ensrure x, y, z are mutually orthogonal

    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    rotation_angles = rotation_matrix_to_euler_angles(rotation_matrix)
    
    # Apply the calculated rotation after closing the gripper

    rotation_angles[2] -= 20 
    robot.IKTargetDoRotate(rotation=rotation_angles, duration=2.5, speed_based=False)
    robot.WaitDo()

    # Move the robot to the grasping point
    robot.IKTargetDoMove(position=target_position, duration=2.5, speed_based=False)
    robot.WaitDo()

    # Close the gripper upon reaching the point
    gripper.GripperClose()
    env.step()
    
def move_robot(robot, target_position, target_orientation, env):
    """Moves the robot to the grasping point with the calculated orientation and closes the gripper."""
    robot.IKTargetDoMove(position=target_position, duration=2.5, speed_based=False)
    robot.WaitDo()
    robot.IKTargetDoRotate(rotation=target_orientation, duration=2.5, speed_based=False)
    robot.WaitDo()
    env.step()
  
def move_robot_along_trajectory(robot, trajectory_points, env):
    """Moves the robot along a trajectory defined by multiple grasping points."""
    for target_position, target_orientation in trajectory_points:
        move_robot(robot, target_position, target_orientation, env)

def rotation_matrix_to_euler_angles(R):
    """Converts a rotation matrix to Euler angles."""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.degrees([x, y, z])

def follow_line(robot , start_point, end_point, env, num_points=10):
    """Move o robô ao longo de uma linha entre dois pontos."""
    x1, y1, z1 = start_point
    x2, y2, z2 = end_point

    trajectory_points = [
        (
            [
                x1 + (x2 - x1) * t / (num_points - 1),
                y1 + (y2 - y1) * t / (num_points - 1),
                z1 + (z2 - z1) * t / (num_points - 1),
            ],
            [357.46221923828125, 237.6537322998047, 182.7155303955078]  
        )
        for t in range(num_points)
    ]

    move_robot_along_trajectory(robot, trajectory_points, env)
    

def get_joint_positions(sensor, env, steps=15):
 
    # Atualizar o ambiente
    env.step(steps)  

    # Realizar o ciclo do sensor
    sensor.pre_step()
    env.step()
    sensor.post_step()

    # Obter os dados
    data = sensor.get_data()
    positions = data.get("positions", None)

    if positions is None or not np.isfinite(positions).all():
        print("Erro: posições inválidas retornadas pelo sensor.")
        return None

    return positions

  
def execute_robot_operations(robot, gripper, env):
    sensor = BodyPoseSensor(env)
    sensor.initialize()  # Inicializar o sensor

    # Obter as posições
    joint_positions = get_joint_positions(sensor, env)
    if joint_positions is None:
        print("Falha ao obter posições para operações do robô.")
        return

    left_wrist = joint_positions[15]
    left_wrist[2] += 0.025
    left_elbow = joint_positions[13]
    left_elbow[2] += 0.025
    left_shoulder = joint_positions[11]
    left_shoulder[2] -= 0.05

    # Movimentar o robô
    follow_line(robot, left_wrist, left_elbow, env)


    # Move from end_point to end_point_2
    follow_line(robot, left_elbow, left_shoulder, env)

    # # Optionally operate the gripper
    # gripper.GripperClose()
    # env.step()



def execute_trajectory(robot, env):
    sensor = BodyPoseSensor(env)
    sensor.initialize()  # Inicializar o sensor

    # Obter as posições
    joint_positions = get_joint_positions(sensor, env)
    if joint_positions is None:
        print("Falha ao obter posições para executar a trajetória.")
        return

    print(joint_positions)
    left_wrist = joint_positions[15]
    left_finger = left_wrist.copy()  # Evitar alterações na matriz original
    left_finger[1] += 0.095
    left_finger[2] += 0.07
    
    print(left_finger)

    trajectory_points = [
        ([1.6123123168945312, 2.1558594703674316, 0.49949676394462585], [37.966182708740234, 199.17330932617188, 79.22513580322266]),
        ([1.1766921281814575, 1.7679061889648438, 0.6753299832344055], [335.5728454589844, 316.0453796386719, 153.63223266601562]),
        ([1.0907518863677979, 1.7960761785507202, 0.6594205498695374], [357.13763427734375, 319.0745849609375, 182.0312042236328]),
        ([0.9740145802497864, 1.736168622970581, 0.5889505207538605], [357.46221923828125, 237.6537322998047, 182.7155303955078]),
        (left_finger, [357.46221923828125, 237.6537322998047, 182.7155303955078]),
    ]

    move_robot_along_trajectory(robot, trajectory_points, env)


def execute_trajectory_2(robot, env):
    sensor = BodyPoseSensor(env)
    sensor.initialize()  # Inicializar o sensor

    # Obter as posições
    joint_positions = get_joint_positions(sensor, env)
    if joint_positions is None:
        print("Falha ao obter posições para executar a trajetória.")
        return

    right_wrist = joint_positions[16]
    right_finger = right_wrist.copy()  
    right_finger[1] += 0.0095
    right_finger[2] += 0.07
    
    print(right_finger)

    trajectory_points_2 = [
        ([1.3907518863677979, 1.7960761785507202, 0.6594205498695374], [357.13763427734375, 319.0745849609375, 182.0312042236328]),
        ([1.40145802497864, 1.6736168622970581, 0.5889505207538605], [357.46221923828125, 237.6537322998047, 182.7155303955078]),
        (right_finger, [357.46221923828125, 237.6537322998047, 182.7155303955078]),
    ]

    move_robot_along_trajectory(robot, trajectory_points_2, env)

def execute_robot_operations_2(robot, gripper, env):
    sensor = BodyPoseSensor(env)
    sensor.initialize()  # Inicializar o sensor

    # Obter as posições
    joint_positions = get_joint_positions(sensor, env)
    if joint_positions is None:
        print("Falha ao obter posições para operações do robô.")
        return

    right_wrist = joint_positions[16]
    right_wrist[2] += 0.025
    right_elbow = joint_positions[14]
    right_elbow[2] += 0.025
    right_shoulder = joint_positions[12]
    right_shoulder[2] -= 0.05

    # Movimentar o robô
    follow_line(robot, right_wrist, right_elbow, env)


    # Move from end_point to end_point_2
    # follow_line(robot, right_elbow, right_shoulder, env)

    # # Optionally operate the gripper
    # gripper.GripperClose()
    # env.step()