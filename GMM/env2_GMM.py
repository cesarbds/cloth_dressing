import os
import numpy as np
from gymnasium import Env
from pyrcareworld.envs.dressing_env import DressingEnv
from reward import complete_initial_task
from sklearn.preprocessing import MinMaxScaler
import joblib

GRAPHICS = True

class DressEnvGAIL(Env):
    def __init__(self, smoothed_trajectories, scaler):
        super().__init__()
        self.env = DressingEnv(graphics=GRAPHICS)
        self.smoothed_trajectories = smoothed_trajectories
        self.current_trajectory = smoothed_trajectories[6] 
        self.scaler = scaler  # Scaler carregado para normalização
        self.t = 0
        self.kinova_id = 315893
        self.robot = self.env.GetAttr(self.kinova_id)
        self.gripper = self.env.GetAttr(3158930)
        self.joint_positions = np.zeros(7)  # Inicializar posições das juntas

    def reset(self, seed=None, options=None):
        """Reinicia o ambiente e executa a tarefa inicial."""
        self.env.close()
        self.env = DressingEnv(graphics=GRAPHICS)
        self.t = 0
        self.robot = self.env.GetAttr(self.kinova_id)
        self.gripper = self.env.GetAttr(3158930)
        complete_initial_task(self.env, self.robot, self.gripper)
        self.robot.EnabledNativeIK(False)
        self.joint_positions = self.robot.data['joint_positions']  # Atualizar posições das juntas
        self.env.step(50)
        return self._get_observation()

    def step(self, action):
        """Aplica a ação da trajetória suavizada ao ambiente."""
        if self.t >= len(self.current_trajectory):
            raise ValueError("Excedeu o comprimento da trajetória suavizada")
        
        # Amostra da trajetória suavizada
        next_state = self.current_trajectory[self.t]
        self.t += 1
        next_state = next_state[1:]  # Ignorar o tempo (primeiro valor da amostra)
        next_state = self._denormalize_data(next_state)  # Desnormalizar corretamente

        # Divisão da amostra em componentes
        gripper_position = next_state[0:3]
        gripper_orientation = next_state[3:6]
        joint_increment = next_state[6:13]  # Incremento para posições das juntas

        # Atualizar posições das juntas incrementalmente
        self.joint_positions = joint_increment
        self.rotation = gripper_orientation
        self.position = gripper_position

        self.env.step(20)

        # Aplicar ações ao ambiente
        self.gripper.SetPosition(gripper_position)
        self.env.step(10)
        self.gripper.SetRotation(gripper_orientation)
        self.robot.SetJointPosition(self.joint_positions)

        # Logs para validação
        # print("Ação aplicada:")
        # print("Gripper Position:", gripper_position)
        # print("Gripper Orientation:", gripper_orientation)
        # print("Updated Joint Positions:", self.joint_positions)

        self.env.step(10)

        # Observar novo estado
        observation = self._get_observation()
        return observation, {}

    def _normalize_data(self, data):
        """Normaliza os dados com o scaler."""
        return self.scaler.transform(data.reshape(1, -1)).flatten()

    def _denormalize_data(self, data):
        """Desnormaliza os dados com o scaler."""
        if len(data.shape) == 1:
            data = data.reshape(1, -1)  # Converte para (1, n_features)
        return self.scaler.inverse_transform(data).flatten()
    
    def _get_robot_state(self):
        """Obtém o estado atual do robô (posição, orientação, etc.)."""
        gripper_position = np.array(self.gripper.data['position'])  # Garantir que seja 1D
        gripper_orientation = np.array(self.gripper.data['rotation'])
        joint_positions = np.array(self.robot.data['joint_positions'])  # Garantir que seja 1D
        state = np.hstack((joint_positions, gripper_position, gripper_orientation))
        return self._normalize_data(state)

    def _get_observation(self):
        """Coleta e retorna as observações atuais do ambiente."""
        return self._get_robot_state()


# --- Usar as trajetórias suavizadas no ambiente ---
# Carregar trajetórias suavizadas
smoothed_trajectories = np.load('smoothed_trajectories.npy', allow_pickle=True)

print("Número de trajetórias carregadas:", len(smoothed_trajectories))

# Carregar o scaler salvo
scaler = joblib.load('scaler.pkl')

# Desativar verificação dos nomes de características no scaler
scaler.feature_names_in_ = None

# Inicializar o ambiente com as trajetórias suavizadas e o scaler
env = DressEnvGAIL(smoothed_trajectories, scaler)
env.reset()

# Executar simulação até que a trajetória termine
try:
    while True:
        observation, _ = env.step(10)
except ValueError as e:
    print("Fim da trajetória:", str(e))
