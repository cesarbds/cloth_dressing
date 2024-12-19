import numpy as np
from numpy.typing import ArrayLike
from pyrcareworld.attributes import CameraAttr
import pyrcareworld.attributes as attr
import cv2
import torch
from collections import deque, namedtuple

# Definir a classe ReplayBuffer
class ReplayBuffer_cesar:
    def __init__(self, buffer_limit):
        self.buffer = deque(maxlen=buffer_limit)
        self.transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])
        self.device = 'cuda'
    def add(self, state, action, reward, next_state, done):
        transition = self.transition(state, action, reward, next_state, done)
        self.buffer.append(transition)
       
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.tensor(states, dtype=torch.float).to(self.device), 
                torch.tensor(actions, dtype=torch.float).to(self.device),
                torch.tensor(rewards, dtype=torch.float).to(self.device), 
                torch.tensor(next_states, dtype=torch.float).to(self.device), 
                torch.tensor(dones, dtype=torch.float).to(self.device))
                
    def size(self):
        return len(self.buffer)


    
def euclidean_distance(particle_pos, target_pos):
    """Calculates the Euclidean distance between two 3D points."""
    return np.linalg.norm(np.array(particle_pos) - np.array(target_pos))

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

def is_vertex_inside(vertex, bounding_box):
    x, y, z = vertex
    min_x, max_x, min_y, max_y, min_z, max_z = bounding_box
    return min_x <= x <= max_x and min_y <= y <= max_y and min_z <= z <= max_z


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
 
    # Converter os dados das part�culas para um array numpy, se n�o for j�
    particles = np.array(particles_data)
    
    specific_particles = []
    
    for index in particle_indices:
        if index >= len(particles):
            print(f"indice {index} esta fora do intervalo de particulas disponiveis.")
        else:
            specific_particles.append(particles[index])
    
    return specific_particles



def calculate_distances(specific_particles, pos_arm):
    """Calcula as distancias entre as particulas e um ponto alvo."""
    distances = [(i, euclidean_distance(p, pos_arm)) for i, p in enumerate(specific_particles)]
    distances.sort(key=lambda x: x[1])  # Ordena por dist�ncia
    return distances

def generate_plane_and_center(points):
    """Gera um plano a partir de tres pontos e calcula o ponto central."""
    p1, p2, p3 = np.array(points[0]), np.array(points[1]), np.array(points[2])

    # Calcula vetores no plano
    v1 = p2 - p1
    v2 = p3 - p1

    # Produto vetorial para obter a normal do plano
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)  # Normaliza

    # Ponto central (m�dia das part�culas)
    center_point = np.mean(points, axis=0)

    return normal, center_point

def align_plane_with_finger(closest_particles, center_point, p_finger, tolerance=0.01):
    """Alinha o plano gerado pelas particulas ao ponto finger."""
    # Vetor de translacaoo entre o centro do plano e o ponto finger
    translation_vector = np.array(p_finger) - np.array(center_point)

    # Alinha as part�culas ao aplicar a transla��o
    aligned_particles = [np.array(p) + translation_vector for p in closest_particles]

    # Gera um novo plano ap�s o alinhamento
    normal_aligned, new_center_point = generate_plane_and_center(aligned_particles[:3])

    # Verifica se o alinhamento eh suficiente
    distance_to_finger = euclidean_distance(new_center_point, p_finger)
    aligned = distance_to_finger < tolerance

    return normal_aligned, new_center_point, aligned

def calculate_reward(center_point, p_finger, p_elbow, p_shoulder):
    """
    Calcula a recompensa em tres fases:
    1. Fase finger: recompensa por alinhar o ponto central da roupa ao ponto finger.
    2. Fase elbow: recompensa adicional por alcan�ar o ponto elbow.
    3. Fase shoulder: recompensa adicional por alcan�ar o ponto shoulder.
    """
    # Fase 1: Recompensa com base no ponto finger
    dist_finger = euclidean_distance(center_point, p_finger)
    if dist_finger < 0.03:
        reward = 1.0  # Alinhado com o ponto finger
    else:
        reward = -dist_finger  # Penaliza com base na dist�ncia do finger
        return reward  # Se ainda n�o est� alinhado com o finger, retornamos a recompensa at� aqui

    # Fase 2: Recompensa com base no ponto elbow
    dist_elbow = euclidean_distance(center_point, p_elbow)
    if dist_elbow < 0.05:
        reward += 1.0  # Recompensa adicional por alinhar com o cotovelo
    else:
        reward -= dist_elbow  # Penaliza com base na dist�ncia at� o cotovelo
        return reward  # Se n�o est� alinhado com o elbow, retorna a recompensa at� aqui

    # Fase 3: Recompensa com base no ponto shoulder
    dist_shoulder = euclidean_distance(center_point, p_shoulder)
    if dist_shoulder < 0.05:
        reward += 1.0  # Recompensa adicional por alinhar com o ombro
    else:
        reward -= dist_shoulder  # Penaliza com base na dist�ncia at� o ombro

    return reward  # Retorna a recompensa acumulada

def distance_cloth_to_finger(particles_data, particles_indices=[90, 88, 91, 89, 103, 105], 
                             p_finger=[0.9859391, 1.53276789, 0.46893813], 
                             p_elbow=[1.38176648, 1.43249212, 0.23214698], 
                             p_shoulder=[1.0922417, 1.48327913, -0.0614629]):
 
    # Obt�m as part�culas usando os �ndices fornecidos
    selected_particles = get_specific_particles(particles_data, particles_indices)

    # Gera o plano com as part�culas selecionadas
    normal, center_point = generate_plane_and_center(selected_particles[:3])

    # Calcula a recompensa levando em considera��o o alinhamento com os pontos finger, elbow e shoulder
    reward = calculate_reward(center_point, p_finger, p_elbow, p_shoulder)

    return normal, center_point, reward

def fetch_particles_data(env):
    """Obtem os dados das particulas do tecido no ambiente."""
    cloth = env.get_cloth()
    cloth_attr = attr.ClothAttr(env, cloth.id)
    cloth_attr.GetParticles()
    env.step()  # Atualiza o ambiente para garantir que os dados est�o prontos
    particles_data = cloth.data.get('particles', None)
    
    if particles_data is None:
        raise ValueError("Nao foi possivel obter os dados das particulas.")
    
    return particles_data

    
def normalize_angles(angles):
    """
    Normalize a list of angles to be within the range of -180 to 180 degrees.

    Parameters:
    angles (list of float): The list of angles in degrees to normalize.

    Returns:
    list of float: The list of normalized angles within the range of -180 to 180 degrees.
    """
    def normalize_angle(angle):
        """
        Normalize a single angle to be within the range of -180 to 180 degrees.

        Parameters:
        angle (float): The angle in degrees to normalize.

        Returns:
        float: The normalized angle within the range of -180 to 180 degrees.
        """
        angle = angle % 360
        if angle > 180:
            angle -= 360
        return angle/180.0

    return [normalize_angle(angle) for angle in angles]

def degrees_to_radians(degrees):
    return np.radians(degrees)

# Function to calculate the bounding box for a set of vertices
def calculate_bounding_box(vertices):
    min_x = min(v[0] for v in vertices)
    max_x = max(v[0] for v in vertices)
    min_y = min(v[1] for v in vertices)
    max_y = max(v[1] for v in vertices)
    min_z = min(v[2] for v in vertices)
    max_z = max(v[2] for v in vertices)
    return (min_x, max_x, min_y, max_y, min_z, max_z)

# Function to check if a vertex is inside a bounding box
def is_vertex_inside(vertex, bounding_box):
    x, y, z = vertex
    min_x, max_x, min_y, max_y, min_z, max_z = bounding_box
    return min_x <= x <= max_x and min_y <= y <= max_y and min_z <= z <= max_z

# Defining the sets of vertices for each shape
vertices_mesa = [(1.8, 0, 0.525), (0.37, 0, 0.525), (1.8, 1.325, 0.525), (0.37, 1.325, 0.525), 
                 (1.8, 0, 0.125), (0.37, 0, 0.125), (1.8, 1.325, 0.125), (0.37, 1.325, 0.125)]
vertice_braco_paciente_r = [(1.5, 1.275, 0.525), (1.5, 1.575, 0.525), (1.5, 1.275, 0.225), (1.5, 1.575, 0.225), 
                            (1.325, 1.275, 0.525), (1.325, 1.575, 0.525), (1.325, 1.275, 0.225), (1.325, 1.575, 0.225)]
vertice_braco_paciente_l = [(1.1, 1.275, 0.549), (1.1, 1.575, 0.549), (1.1, 1.275, 0.225), (1.1, 1.575, 0.225), 
                            (0.9,  1.275, 0.549), (0.9,  1.575, 0.549), (0.9, 1.275, 0.225), (0.9, 1.575, 0.225)]

# Bounding boxes for each shape
bounding_box_mesa = calculate_bounding_box(vertices_mesa)
bounding_box_braco_r = calculate_bounding_box(vertice_braco_paciente_r)
bounding_box_braco_l = calculate_bounding_box(vertice_braco_paciente_l)

# Funcao para normalizar angulos para o intervalo de -180 a 180 graus
def normalize_angle(angle):
    return (angle + 180) % 360 - 180

def is_vertex_inside_any_shape(vertex):
    if is_vertex_inside(vertex, bounding_box_mesa):
        return True
    # elif is_vertex_inside(vertex, bounding_box_braco_r):
    #     return True
    # elif is_vertex_inside(vertex, bounding_box_braco_l):
    #     return True
    else:
        return False

# Funcao para calcular o vetor baseado nos angulos
def vector_from_angles(roll, pitch, yaw, length=0.14):
    roll = normalize_angle(roll)
    pitch = normalize_angle(pitch)
    yaw = normalize_angle(yaw)
    
    roll = degrees_to_radians(roll)
    pitch = degrees_to_radians(pitch)
    yaw = degrees_to_radians(yaw)

    # Vetor unitario nas direcoes x, y, z
    x = np.cos(pitch) * np.cos(yaw)
    y = np.cos(pitch) * np.sin(yaw)
    z = np.sin(pitch)

    # Escalar o vetor para ter o comprimento desejado
    vector = np.array([x, y, z])
    vector = (vector / np.linalg.norm(vector)) * length

    return vector
