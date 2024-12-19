import sys
import os

# Adiciona o diretório pai ao caminho do sistema
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from gymnasium import Env
from pyrcareworld.envs.dressing_env import DressingEnv
import reward_p1.reward_p1 as reward
from sklearn.preprocessing import MinMaxScaler
import joblib
import pyrcareworld.attributes as attr
from utils.utils import *


GRAPHICS = True

# Função para Gaussian Mixture Regression (GMR)
def gmr(gmm, current_state):
    """
    Gaussian Mixture Regression (GMR) para prever a próxima ação com base no estado atual.
    """
    n_components = gmm.n_components
    means = gmm.means_
    covariances = gmm.covariances_
    weights = gmm.weights_

    dim_input = len(current_state)
    dim_output = means.shape[1] - dim_input

    predicted_mean = np.zeros(dim_output)
    predicted_covariance = np.zeros((dim_output, dim_output))

    for i in range(n_components):
        mean_input = means[i, :dim_input]
        mean_output = means[i, dim_input:]
        cov_input = covariances[i, :dim_input, :dim_input]
        cov_output = covariances[i, dim_input:, dim_input:]
        cov_input_output = covariances[i, :dim_input, dim_input:]
        
        # Inferência usando as propriedades da distribuição normal multivariada
        inv_cov_input = np.linalg.inv(cov_input)
        predicted_mean += weights[i] * (mean_output + cov_input_output.T @ inv_cov_input @ (current_state - mean_input))
        predicted_covariance += weights[i] * (cov_output - cov_input_output.T @ inv_cov_input @ cov_input_output)

        # print(f"medias: {predicted_mean}")
        
    return predicted_mean, predicted_covariance


class DressEnvGMM(Env):
    def __init__(self, scaler_input, scaler_output, gmm_models):
        super().__init__()
        self.env = DressingEnv(graphics=GRAPHICS)
        self.scaler_input = scaler_input  # Scaler para normalizar entradas
        self.scaler_output = scaler_output  # Scaler carregado para normalização
        self.gmm_models = gmm_models  # Modelos GMM carregados
        self.t = 0
        self.kinova_id = 315893
        self.robot = self.env.GetAttr(self.kinova_id)
        self.gripper = self.env.GetAttr(3158930)
        self.joint_positions = np.zeros(7)
        self.gripper.data['velocities'][-1] = np.zeros(3)  
        
    def reset(self, seed=None, options=None):
        """Reinicia o ambiente e executa a tarefa inicial."""
        self.env.close()
        self.env = DressingEnv(graphics=GRAPHICS)
        self.t = 0
        self.camera = self.env.InstanceObject(name="Camera", id=333333, attr_type=attr.CameraAttr)
        self.camera.SetTransform(position=[0.15, 2.45, 1.7], rotation=[150, -50, 185])
        self.robot = self.env.GetAttr(self.kinova_id)
        self.gripper = self.env.GetAttr(3158930)
        reward.complete_initial_task(self.env, self.robot, self.gripper, self.camera)
        self.robot.EnabledNativeIK(False)
        self.joint_positions = self.robot.data['joint_positions']  
        self.env.step(50)
        return self._get_observation()

    def step(self):
        """Prevê e aplica a ação baseada no GMM ao ambiente."""
        # Obter estado atual normalizado (posição das juntas)
        current_state = self._get_robot_state()  # Isso retorna as posições das 7 juntas do robô
        
        # Prever a ação baseada no estado atual usando o GMR
        action = self._predict_action(current_state)  # Ação prevista de tamanho (10,): [x, y, z, roll, pitch, yaw, v_x, v_y, v_z]
        
        self.joint_positions = action
        #joint_increment = normalize_angles(joint_pos + action)
        # gripper_velocity = action[7:] 
        # Dividir a ação prevista em componentes para o gripper
        # gripper_position = action[:3]  # Posição (x, y, z)
        # gripper_orientation = action[3:6]  # Orientação (roll, pitch, yaw)
        # gripper_velocity = action[6:]  # Velocidade (v_x, v_y, v_z)
        
       

        # Atualizar a posição e a orientação do gripper
        # self.rotation = gripper_orientation
        # self.position = gripper_position
        # self.gripper.data['velocities'][-1] = gripper_velocity
        #joint_increment = [x*180 for x in joint_increment]
        #self.joint_positions = joint_increment
        
        
        # Aplicar as novas posições e velocidades ao gripper no ambiente
        # self.gripper.SetPosition(gripper_position)
        # self.gripper.SetRotation(gripper_orientation)
        # self.gripper.data['velocities'][-1] += gripper_velocity  # Atualizar velocidades
        
        self.robot.SetJointPosition(self.joint_positions)
        #self.robot.IKTargetDoMove(position=[dx, dy, dz], duration=0.1, speed_based=False, relative=True)
        self.env.step()  # Avançar o ambiente

        # Observar o novo estado após a aplicação da ação
        observation = self._get_observation()  # Obter a nova observação do estado

        return observation, {}



    def _predict_action(self, current_state):
        """Prevê a próxima ação usando o modelo GMM com GMR."""
        # Escolher o modelo GMM 
        # modelos válidos: 0, 4, 6 
        gmm_model = self.gmm_models[4]  

        # Predição com GMR
        predicted_action, predicted_covariance = gmr(gmm_model, current_state)

        # Desnormalizar a ação antes de aplicar
        predicted_action = self._denormalize_data(predicted_action, self.scaler_output)

        print(f"Ação prevista (desnormalizada): {predicted_action}")
        return predicted_action

    def _normalize_data(self, data, scaler):
        return scaler.transform(data.reshape(1, -1)).flatten()

    def _denormalize_data(self, data, scaler):
        return scaler.inverse_transform(data.reshape(1, -1)).flatten()

    def _get_robot_state(self):
        self.gripper = self.env.GetAttr(3158930)

        """Obtém o estado atual do robô."""
        #joint_positions = np.array(self.robot.data['joint_positions'])
        gripper_position = np.array(self.gripper.data['position'])  # Garantir que seja 1D
        gripper_orientation = np.array(self.gripper.data['rotation'])
        # gripper_vel = np.array(self.gripper.data['velocities'])[-1]
        # for key, value in self.gripper.data.items():
        #   print(f"{key}: {value}")        
        state = np.hstack((gripper_position, gripper_orientation))#, gripper_vel))
        return self._normalize_data(state, self.scaler_input)

    def _get_observation(self):
        """Coleta e retorna as observações atuais do ambiente."""
        return self._get_robot_state()


# --- Configuração do ambiente ---
scaler_input = joblib.load('scaler_input1.pkl')
scaler_output = joblib.load('scaler_output1.pkl')
gmm_models = joblib.load('gmm_models1.pkl')
print(scaler_input.n_features_in_)
env = DressEnvGMM(scaler_input, scaler_output, gmm_models)
env.reset()


# Executar simulação até que o ambiente seja interrompido
try:
    while True:
        observation, _ = env.step()
except KeyboardInterrupt:
    print("Execução interrompida pelo usuário.")
