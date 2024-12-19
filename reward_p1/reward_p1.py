from pyrcareworld.envs.dressing_env import DressingEnv
import pyrcareworld.attributes as attr
import cv2
import numpy as np
from numpy.typing import ArrayLike
from pyrcareworld.attributes import CameraAttr

intrinsics = np.eye(3)

intrinsics[0, 0] = 500
intrinsics[1, 1] = 500

intrinsics[0, 2] = 512/2
intrinsics[1, 2] = 512/2

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

    normal = np.frombuffer(camera.data["normal"], dtype=np.uint8)
    normal = cv2.imdecode(normal, cv2.IMREAD_COLOR)

    depth = np.frombuffer(camera.data["depth"], dtype=np.uint8)
    depth = cv2.imdecode(depth, cv2.IMREAD_GRAYSCALE)
    depth = depth.astype(np.float32) / 255.0  
    return rgb, normal, depth

def instance_ball(env):
    """Instances a ball object in the environment."""
    ball = env.InstanceObject(name="Rigidbody_Sphere", id=24252, attr_type=attr.RigidbodyAttr)
    ball.SetTransform(position=[3.45, 2.1, 0.5], rotation=[150, -75, 180])
    ball.SetKinematic(False)
    ball.SetScale([0.025, 0.025, 0.025])
    return ball

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

def get_normal_from_camera(env, camera, particles_camera):
    """Obtains the surface normal from the normal image captured by the camera."""
    env.step()
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

def move_robot_to_point(robot, target_position, grasping_normal, gripper, env):
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
    robot.IKTargetDoRotate(rotation=rotation_angles, duration=2, speed_based=False)
    robot.WaitDo()

    # Move the robot to the grasping point
    target_position[2] -= 0.08
    robot.IKTargetDoMove(position=target_position, duration=2, speed_based=False)
    robot.WaitDo()

    # Close the gripper upon reaching the point
    gripper.GripperClose()
    env.step()
  
def move_robot_along_trajectory(robot, trajectory_points, gripper, env):
    """Moves the robot along a trajectory defined by multiple grasping points."""
    for target_position, grasping_normal in trajectory_points:
        move_robot_to_point(robot, target_position, grasping_normal, gripper, env)

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


def get_specific_particles(particles_data, particle_indices):
 
    particles = np.array(particles_data)
    
    specific_particles = []
    
    for index in particle_indices:
        if index >= len(particles):
            print(f"indice {index} esta fora do intervalo de particulas disponiveis.")
        else:
            specific_particles.append(particles[index])
    
    return specific_particles



def calculate_distances(specific_particles, pos_arm):
    distances = [(i, euclidean_distance(p, pos_arm)) for i, p in enumerate(specific_particles)]
    distances.sort(key=lambda x: x[1])  
    return distances

def generate_plane_and_center(points):
    p1, p2, p3 = np.array(points[0]), np.array(points[1]), np.array(points[2])

    # Calcula vetores no plano
    v1 = p2 - p1
    v2 = p3 - p1

    # Produto vetorial para obter a normal do plano
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)  # Normaliza

    # Ponto central (media das particulas)
    center_point = np.mean(points, axis=0)

    return normal, center_point

def align_plane_with_finger(normal, center_point, p_finger, tolerance=0.1):
    # Calcula o vetor direcao entre o ponto central e o ponto finger
    direction_vector = np.array(p_finger) - np.array(center_point)
    direction_vector /= np.linalg.norm(direction_vector)  # Normaliza o vetor

    # Calcula o cosseno do angulo entre a normal do plano e o vetor direcao
    cos_angle = np.dot(normal, direction_vector)

    # Considera como alinhado se o angulo estiver abaixo do limite (tolerancia)
    aligned = np.arccos(np.clip(cos_angle, -1, 1)) < tolerance

    # Score de alinhamento entre 0 e 1 baseado no cosseno do angulo (1 eh o alinhamento perfeito)
    alignment_score = max(0, cos_angle)

    return aligned, alignment_score

def calculate_reward(center_point, normal, p_finger, previous_distance = None, proximity_threshold = 0.03):
    """
    Calcula a recompensa baseada em:
    1. Alinhamento com o ponto finger usando a direção da normal do plano.
    2. Aproximação do ponto finger com uma recompensa proporcional à aproximação e um bônus ao atingir a proximidade desejada.
    """
    # Inicializa a recompensa
    reward = 0

    # Etapa 1: Recompensa por alinhamento da normal do plano com o ponto finger
    #_, alignment_score = align_plane_with_finger(normal, center_point, p_finger)
    #reward += 0.7* alignment_score  # Recompensa proporcional ao score de alinhamento

    # Etapa 2: 
    dist_finger = euclidean_distance(center_point, p_finger)
    
    if previous_distance is not None:
        reward +=  (previous_distance - dist_finger )*(10)
            
    
    return reward, dist_finger

def distance_cloth_to_finger(particles_data, particles_indices=[1, 20, 109, 18, 108, 17], 
                             p_finger=[0.9859391, 1.53276789, 0.46893813], 
                             p_elbow=[1.38176648, 1.43249212, 0.23214698], 
                             p_shoulder=[1.0922417, 1.48327913, -0.0614629], previous_distance = None, proximity_threshold = 0.03):
 
    selected_particles = get_specific_particles(particles_data, particles_indices)

    normal, center_point = generate_plane_and_center(selected_particles[:3])

    reward, new_distance = calculate_reward(center_point, normal, p_finger, previous_distance, proximity_threshold)

    return normal, center_point, reward, new_distance

def fetch_particles_data(env):
    """Obtem os dados das particulas do tecido no ambiente."""
    cloth = env.get_cloth()
    cloth_attr = attr.ClothAttr(env, cloth.id)
    cloth_attr.GetParticles()
    env.step()  # Atualiza o ambiente para garantir que os dados estao prontos
    particles_data = cloth.data.get('particles', None)
    
    if particles_data is None:
        raise ValueError("Não foi possivel obter os dados das particulas.")
    
    return particles_data

def check_gripper_hold(robot, env, threshold=900):

    robot.GetJointInverseDynamicsForce()
    env.step()
    drive_forces = robot.data['drive_forces']

    #print(robot.data['drive_forces'])
    if abs(drive_forces[0]) > threshold:
        #print("Erro.")
        return False

    #print("Pode continuar.")
    return True

def complete_initial_task(env, robot, gripper, camera):
    """Completes the initial task of moving the robot to the grasping point."""

    # Initial position
    robot.EnabledNativeIK(False)
    robot.SetJointPositionDirectly([173.68829345703125, -33.00761795043945, 179.9871063232422, -91.84587860107422, -39.07077407836914, -63.777915954589844, -136.49273681640625])
    env.step(50)
    robot.EnabledNativeIK(True)


    setup_camera(camera, gripper)
    #rgb, normal, depth = process_images(env, camera)
    
    
    target_position = [1.9140679121017456, 1.7068316078186035, 0.3185024857521057]
    particles_camera = world2camera(target_position, camera, intrinsics)
    grasping_normal = get_normal_from_camera(env, camera, particles_camera)
    
    move_robot_to_point(robot, target_position, grasping_normal, gripper, env)
    env.step()    

     # Verificar se o gripper conseguiu pegar a roupa
    #return check_gripper_hold(robot, env)

def check_gripper_hold(robot, env, threshold=500):

    # Obtém as forças nas articulações
    robot.GetJointInverseDynamicsForce()
    env.step()
    drive_forces = robot.data['drive_forces']

    # Verifica o módulo do primeiro valor das forças
    if abs(drive_forces[0]) > threshold:
        #print("Erro.")
        return False

    #print("Pode continuar.")
    return True
  

