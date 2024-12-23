from typing import Optional, Any
import numpy as np
from numpy.typing import ArrayLike
import mediapipe as mp
import pyrcareworld.attributes as attr
from pyrcareworld.envs.base_env import RCareWorld
from utils_camera import get_color, get_depth_exr, get_positions
from utils_math import reject_outliers
from sensor import Sensor
import cv2 as cv
import matplotlib.pyplot as plt  

class BodyPoseSensor(Sensor):
    def __init__(self, env: RCareWorld, intrinsics: Optional[ArrayLike] = None) -> None:
        super().__init__(env)

        if intrinsics is None:
            size_divider = 2

            intrinsics = np.eye(3)            

            intrinsics[0, 0] = 500 / size_divider
            intrinsics[1, 1] = 500 / size_divider

            intrinsics[0, 2] = (512 / 2) / size_divider
            intrinsics[1, 2] = (512 / 2) / size_divider

        intrinsics = np.asarray(intrinsics, np.float32)
    
        assert intrinsics.shape == (3, 3)
        self._intrinsics = intrinsics
        self._intrinsics_inv = np.linalg.inv(intrinsics)

        self._positions: np.ndarray | None = None

    def initialize(self) -> None:
        camera_positions = [[1.2, 2.5, 0.2], [0.7, 2.5, 0.2], [1.7, 2.5, 0.2]]
        camera_rotations = [[90.0, 0.0, 0.0], [60, 90, 0.0], [60, -90, 0.0]]

        cameras: list[attr.CameraAttr] = []

        for i in range(len(camera_positions)):
            camera: attr.CameraAttr = self._env.InstanceObject("Camera", attr_type=attr.CameraAttr)
            cameras.append(camera)

            camera.SetPosition(camera_positions[i])
            camera.SetRotation(camera_rotations[i])
            
        self._cameras = cameras

        # Inicializando o MediaPipe Pose
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        
        self._inferencer = mp_pose.Pose()

    def pre_step(self) -> None:
        for camera in self._cameras:
            camera.GetRGB(intrinsic_matrix=self._intrinsics)
            camera.GetDepthEXR(intrinsic_matrix=self._intrinsics)

        self._positions = None

    def post_step(self) -> None:
        imgs = []
        imgs_depth = []

        for camera in self._cameras:
            img = get_color(camera)
            depth = get_depth_exr(camera)

            imgs.append(img)
            imgs_depth.append(depth)

        self._imgs = imgs
        self._imgs_depth = imgs_depth

    def get_data(self) -> dict[str, np.ndarray]:
        if self._positions is None:
            # Processar as imagens para a detecção de pose
            results = [self._inferencer.process(cv.cvtColor(img, cv.COLOR_BGR2RGB)) for img in self._imgs]
            
            all_positions = np.empty((3, 17, 3), np.float32)  

            for i, (camera, result, img_depth) in enumerate(zip(self._cameras, results, self._imgs_depth)):
                if result.pose_landmarks:
                    keypoints = result.pose_landmarks.landmark
                    coordinates = np.array([[keypoint.x * img_depth.shape[1], keypoint.y * img_depth.shape[0]] for keypoint in keypoints[:17]])

                    positions = get_positions(camera, self._intrinsics_inv, img_depth, coordinates)

                    all_positions[i] = positions

                    # # Desenhar pontos na imagem 
                    # img_draw = cv.cvtColor(self._imgs[i], cv.COLOR_BGR2RGB)
                    # for idx in [11, 13, 15]:
                    #     x, y = int(coordinates[idx][0]), int(coordinates[idx][1])
                    #     cv.circle(img_draw, (x, y), 1, (255, 0, 0), -1)

                    # # Exibir a imagem com os pontos desenhados
                    # plt.imshow(img_draw)
                    # plt.show()

            positions = np.empty((17, 3), np.float32)

            # Aplicar rejeição de outliers e calcular a média
            for i_keypoint in range(all_positions.shape[1]):
                for j_axis in range(3):
                    positions[i_keypoint, j_axis] = reject_outliers(all_positions[:, i_keypoint, j_axis]).mean()

            self._positions = positions

        return {"positions": self._positions.copy()}

    def close(self) -> None:
        """Finaliza o sensor, liberando recursos do Mediapipe."""
        self._inferencer.close()
