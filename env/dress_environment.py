import sys
import os

# Adicionar o diretório pai ao caminho de busca
pkl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'GMM', 'scaler.pkl'))


import random
import numpy as np
import time
import cv2
from utils.utils import *
from box.box import *
from data_autoencoder.data_autoencoder import *
import cv2
import reward_p1.reward_p1  as reward
from SAC_cleanrl.sac_cleanrl import *
from TD3_cleanrl.td3_cleanrl import *
import torch
import GMM.class_GMM as GMM
import GMM.GMM_agent as GMM_agent
import joblib
from GMM.gmmf import GaussianMixtureWithTrajectories
from stable_baselines3.common.buffers import ReplayBuffer
from gymnasium import spaces
import pyrcareworld.attributes as attr
from pyrcareworld.envs.dressing_env import DressingEnv

GRAPHICS = True


class DressEnv(gym.Env):
    def __init__(self):
        super(DressEnv, self).__init__()
        # Initialize the environment
        self.env = DressingEnv(graphics=GRAPHICS)
        self.t = 0
        self.particles_t = np.zeros([170,3])
        self.velocities_particles_t = np.zeros([170,3])
        self.velocities_particles = np.zeros([170,3])
        self.image_encoder = get_encoder_network([4,256,256])
        # Load the state dict with a prefix fix
        #state_dict = torch.load('astral-wildflower-27_ENCODER.pth', weights_only=True)
        #new_state_dict = {key.replace("_encoder.", ""): value for key, value in state_dict.items()}

        # Scaler
        # Carregar o scaler salvo
        self.scaler = joblib.load(pkl_path)

        # Desativar verificação dos nomes de características no scaler
        self.scaler.feature_names_in_ = None
        # Load GMM
        #self.gmm_model = GMM.GMM(model_name="GMM/gmm_models_with_smoothed_trajectories.pkl")
        #self.agent_gmm = GMM_agent.SAC_GMM_Agent(model=self.gmm_model)
        
        # Load the modified state dict
        self.rgb_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #self.image_encoder.load_state_dict(new_state_dict, strict=True)
        self.image_encoder.to('cuda'),
        self.joint_positions = np.zeros(7)
        
        #act_high, act_low = self.agent_gmm.get_action_space()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)  # Independent joint control
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2584,), dtype=np.float32)  # LatentSpace +


    def exr_to_image(self):
        temp_file_path = "img.exr"

        with open(temp_file_path, "wb") as f:
            f.write(self.depth)
        depth = cv2.imread(temp_file_path, cv2.IMREAD_UNCHANGED)

        os.remove(temp_file_path)

        return depth
    
    def capture_scene_data(self):
        size_divider = 2

        intrinsics = np.eye(3)

        intrinsics[0, 0] = 500/size_divider
        intrinsics[1, 1] = 500/size_divider

        intrinsics[0, 2] = (512/2)/size_divider
        intrinsics[1, 2] = (512/2)/size_divider

        # Capture the camera image at the robot's hand
        self.camera.GetDepthEXR(intrinsic_matrix=intrinsics)
        self.camera.GetRGB(intrinsic_matrix=intrinsics)
        self.cloth.GetParticles()
        self.env.step()
        self.env.t += round(self.env.data['fixed_delta_time'],2) 
        rgb = np.frombuffer(self.camera.data["rgb"], dtype=np.uint8)
        rgb = cv2.imdecode(rgb, cv2.IMREAD_COLOR)
        
        self.depth = self.camera.data["depth_exr"]
        depth = self.exr_to_image()
        
        self.env.t += round(self.env.data['fixed_delta_time'],2) 
        particles = self.cloth.data.get('particles', None)
        self.velocities_particles = (np.array(particles) - np.array(self.particles_t)) / (self.env.t - self.t)
        self.acceleration_particles = (np.array(self.velocities_particles) - np.array(self.velocities_particles_t)) / (self.env.t - self.t)
        self.particles_t = particles
        self.t = self.env.t
        self.velocities_particles_t = self.velocities_particles
        # Increment the image number for the next capture
        return depth, rgb, particles, self.velocities_particles, self.acceleration_particles

    def reset(self, seed=None, options=None):
        self.previous_distance = None 
        a = time.time()
        self.env.close()
        self.env = DressingEnv(graphics=GRAPHICS, log_level=0)
        self.camera = self.env.InstanceObject(name="Camera", id=333333, attr_type=attr.CameraAttr)
        self.camera.SetTransform(position=[0.15, 2.45, 1.7], rotation=[150, -50, 185])
        self.kinova_id = 315893
        self.robot = self.env.GetAttr(self.kinova_id)
        self.gripper = self.env.GetAttr(3158930)
        camera_hand = self.env.GetAttr(654321) 
        camera_hand.Destroy()
        self.cloth = self.env.GetAttr(782563)
        self.gripper_state = 0
        self.env.step()
        self.max_episode_steps = 200
        self.env.t += round(self.env.data['fixed_delta_time'],2) 
        #Function that takes the take the cloth
        # if reward.complete_initial_task(self.env, self.robot, self.gripper, self.camera) == False:
            # return 1, False
        reward.complete_initial_task(self.env, self.robot, self.gripper, self.camera)
        self.robot.EnabledNativeIK(False)
        self.env.t += 502*round(self.env.data['fixed_delta_time'],2) 
        #Handle the seed if provided

        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)

        self.env.step()
        self.env.t += round(self.env.data['fixed_delta_time'],2) 
        depth, rgb, self.particles_t, self.velocities_particles, self.acceleration_particles  = self.capture_scene_data()
         # Convert rgb to float32 and scale values to [0, 1]
        rgb = torch.from_numpy(rgb).float()
        rgb = rgb.permute(2,0,1)
        rgb = self.rgb_transform(rgb)
        depth = torch.from_numpy(depth).unsqueeze(0).float()
        #rgb = torch.from_numpy(rgb)# Convert numpy array to PyTorch tensor
        
        rgbd_image = torch.cat((rgb, depth), dim=0)
        rgbd_image = rgbd_image.unsqueeze(0).to('cuda')
        latent_space = self.image_encoder(rgbd_image)    
        self.dynamic_latent_space = torch.stack((latent_space, latent_space, latent_space, latent_space), dim=1)
        depth, rgb, self.particles_t, self.velocities_particles, self.acceleration_particles  = self.capture_scene_data()
        
        depth = torch.from_numpy(depth).unsqueeze(0).float()
        rgb = torch.from_numpy(rgb).float()
        rgb = rgb.permute(2,0,1)
        rgb = self.rgb_transform(rgb)

        rgbd_image = torch.cat((rgb, depth), dim=0).unsqueeze(0).to('cuda')
        latent_space2 = self.image_encoder(rgbd_image).detach()
        
        self.dynamic_latent_space = torch.cat((self.dynamic_latent_space[:,:3,:], latent_space2.unsqueeze(1)), dim=1).to('cuda')
        self.dynamic_latent_space_t = self.dynamic_latent_space.reshape(1, -1).reshape(-1)
        # Define action and observation space
        self.robot_position = self.gripper.data["positions"][-1]
        
        ###Function to retrieve human joints
        ########################################################
        ########################################################
        ######       TO DO: RETRIEVE HUMAN JOINTS   ############
        ########################################################
        ########################################################
        self.human_joints = np.zeros(18)
        # Directly convert lists or scalars to tensors
        self.joint_positions = self.robot.data['joint_positions']
        joint_positions = torch.tensor(self.joint_positions).to('cuda')
        acceleration_particles = torch.tensor(self.acceleration_particles).reshape(-1).to('cuda')
        gripper_joint_positions = torch.tensor(self.gripper.data['joint_positions']).to('cuda')
        human_joints = torch.tensor(self.human_joints).to('cuda')

        normal, center_point, _, self.previous_distance = reward.distance_cloth_to_finger(self.particles_t, previous_distance = self.previous_distance, proximity_threshold = 0.03)
        gripper_position = np.array(self.gripper.data['position'])  # Garantir que seja 1D
        gripper_orientation = np.array(self.gripper.data['rotation'])
        joint_positions2 = np.array(self.robot.data['joint_positions'])  # Garantir que seja 1D
        state = np.hstack((gripper_position, gripper_orientation, joint_positions2))
        # Concatenate all tensors along the specified dimension (e.g., dim=0)
        observations = torch.cat((
            self.dynamic_latent_space_t.detach(), 
            joint_positions.detach(), 
            acceleration_particles.detach(),  
            gripper_joint_positions.detach(), 
            human_joints.detach()
        ), dim=0).cpu().numpy()  # or dim=1, as required by your model
        print("++++++++++++++")
        print("resetting")
        return observations, True,  self._normalize_data(state) # If using gymnasium, omit the info

    def _normalize_data(self, data):
        """Normaliza os dados com o scaler."""
        return self.scaler.transform(data.reshape(1, -1)).flatten()

    def _denormalize_data(self, data):
        """Desnormaliza os dados com o scaler."""
        if len(data.shape) == 1:
            data = data.reshape(1, -1)  # Converte para (1, n_features)
        return self.scaler.inverse_transform(data).flatten()

    def step(self, action):
        if not hasattr(self, 'previous_distance'):
            self.previous_distance = None 
        # Apply the action to the robot position
        self.joint_positions = normalize_angles(self.robot.data['joint_positions'] + action[:7])
        # if action[7] <= 0:
        #     self.gripper.GripperOpen()
        # else:
        #     self.gripper.GripperClose()
        result = [x*180 for x in self.joint_positions]
        self.robot.SetJointPosition(result)
        self.env.step()
        self.env.t += 20*round(self.env.data['fixed_delta_time'],2) 
        
        depth, rgb, self.particles_t, self.velocities_particles, self.acceleration_particles  = self.capture_scene_data()
        cv2.imshow("show", rgb)
        cv2.waitKey(1)
        cv2.imshow("show2", depth)
        cv2.waitKey(1)
        rgb = torch.from_numpy(rgb).float()
        rgb = rgb.permute(2,0,1)
        rgb = self.rgb_transform(rgb)
        depth = torch.from_numpy(depth).unsqueeze(0).float()
        
        rgbd_image = torch.cat((rgb, depth), dim=0)
        rgbd_image = rgbd_image.unsqueeze(0).to('cuda')

        
        
        latent_space2 = self.image_encoder(rgbd_image)
        if torch.isnan(latent_space2).any():
                    print("Warning: NaN values in latent space")
                    new_env = False
                    while  new_env  == False:
                        _, new_env = self.env.reset()
                        print("Cloth not attached")
        self.dynamic_latent_space = torch.cat((self.dynamic_latent_space[:,:3,:], latent_space2.unsqueeze(1)), dim=1)
        self.dynamic_latent_space_t = self.dynamic_latent_space.reshape(1, -1).reshape(-1)
        # Define action and observation space
        self.robot_position = self.gripper.data["positions"][-1]
        
        ###Function to retrieve human joints
        ########################################################
        ########################################################
        ######       TO DO: RETRIEVE HUMAN JOINTS   ############
        ########################################################
        ########################################################
        self.human_joints = np.zeros(18)
        # Directly convert lists or scalars to tensors
        joint_positions = torch.tensor(self.robot.data['joint_positions']).to('cuda')
        acceleration_particles = torch.tensor(self.acceleration_particles).reshape(-1).to('cuda')
        gripper_joint_positions = torch.tensor(self.gripper.data['joint_positions']).to('cuda')
        human_joints = torch.tensor(self.human_joints).to('cuda')
        gripper_position = np.array(self.gripper.data['position'])  # Garantir que seja 1D
        gripper_orientation = np.array(self.gripper.data['rotation'])
        joint_positions2 = np.array(self.robot.data['joint_positions'])  # Garantir que seja 1D
        state = np.hstack((gripper_position, gripper_orientation, joint_positions2))
        # Concatenate all tensors along the specified dimension (e.g., dim=0)
        observations = torch.cat((
            self.dynamic_latent_space_t, 
            joint_positions, 
            acceleration_particles,  
            gripper_joint_positions, 
            human_joints
        ), dim=0).cpu().detach().numpy()  # or dim=1, as required by your model
        if np.isnan(observations).any():
            print("Warning: NaN values in observations")
        done = False
        # # Calculate reward
        prism_vertices_list = []
        #balls = list()
        joint_pos = self.robot.data['positions'][3:10]
        #print((joint_pos))
        for i in range(1, len(joint_pos)):
            p1 = joint_pos[i-1]
            p2 = joint_pos[i]
            direction = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
            # Prism dimensions
            width = .1
            height = .1
            #height = 3
            length = (distance_3d(p1,p2))
            #print(height)
        
        # Generate vertices of the prism
            base_vertices, top_vertices = generate_prism_vertices(p1, direction, width, height, length)
            #print(base_vertices, top_vertices)
            #print(len(prisms))
            prism_vertices_list.append((base_vertices, top_vertices))
            
            vert = generate_prisms(prism_vertices_list) 
        for h in range(len(vert)):       
            if is_vertex_inside_any_shape(vert[h]):
                done = True
        if done:
            return observations, -100, done, False, self._normalize_data(state)
        normal, center_point, _, _ = reward.distance_cloth_to_finger(self.particles_t, previous_distance = self.previous_distance, proximity_threshold = 0.03)

        
        step_reward, new_distance = reward.calculate_reward(center_point, normal, p_finger=[0.9859391, 1.53276789, 0.46893813], previous_distance=self.previous_distance, proximity_threshold=0.03)#, p_elbow=[1.38176648, 1.43249212, 0.23214698], p_shoulder=[1.0922417, 1.48327913, -0.0614629])
        self.previous_distance = new_distance
       
        # Define the reward  - Cloth proximity with finger, alignment with arm....
        if abs(new_distance) < 0.05:
             print("reached")
             done = True
             step_reward += 100
        if self.particles_t[1][1] < 0.2:
            step_reward -= 1
        
        
        # Check if the target is reached ( #distance to shoulder)
        #

        # Prepare the next observation
        

        return observations, step_reward, done, False, self._normalize_data(state)
    
    def render(self, mode='human'):
        pass

    def close(self):
        self.env.close()
