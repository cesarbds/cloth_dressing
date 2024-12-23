import abc
from typing import Any

from pyrcareworld.envs.base_env import RCareWorld

class Sensor(abc.ABC):
    def __init__(self, env:RCareWorld) -> None:
        self._env = env

    @abc.abstractmethod
    def initialize(self) -> None:
        ...

    @abc.abstractmethod
    def pre_step(self) -> None:
        ...

    @abc.abstractmethod
    def post_step(self) -> None:
        ...

    @abc.abstractmethod
    def get_data(self) -> dict[str, Any]:
        ...