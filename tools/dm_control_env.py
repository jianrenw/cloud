from dm_control import suite
from dm_control.suite.wrappers import pixels
from dm_env.specs import Array, BoundedArray

import numpy as np
import os
import copy
from collections import namedtuple, OrderedDict
from utils.utils import namedarraytuple, Env, EnvStep, EnvSpaces
from dm_control import viewer
import cv2

State = None
def grabFrame(env):
    rgbArr = env.physics.render(480, 600, camera_id="video")
    return cv2.cvtColor(rgbArr, cv2.COLOR_BGR2RGB)

EnvInfo = None
Observation = None

def init_namedtuples(info_keys=None, state_keys=None):
    global EnvInfo, Observation, State

    if info_keys is None:
        info_keys = ['traj_done']

    if state_keys is None:
        state_keys = ['pixels']

    EnvInfo = namedtuple('EnvInfo', info_keys)
    Observation = namedarraytuple('Observation', state_keys)
    State = namedtuple('State', state_keys)

class DMControlEnv(Env):

    def __init__(self,
                 domain,
                 task,
                 frame_skip=1,
                 normalize=False,
                 pixel_wrapper_kwargs=None,
                 task_kwargs={},
                 environment_kwargs={},
                 max_path_length=1200,
                 ):
        self._max_path_length = max_path_length
        print(">>> domain")
        print(domain)
        domain = "rope_dr"
        print(">>> task")
        print(task)
        print("env kwargs")
        print(environment_kwargs)
        env = suite.load(domain_name=domain,
                         task_name=task,
                         task_kwargs=task_kwargs,
                         environment_kwargs=environment_kwargs)
        self._video_name = 'visualization/video.mp4'
        frame = grabFrame(env)
        height, width, layers = frame.shape
        self._video = cv2.VideoWriter(self._video_name, cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (width, height))
        
        if normalize:
            np.testing.assert_equal(env.action_spec().minimum, -1)
            np.testing.assert_equal(env.action_spec().maximum, 1)
        if pixel_wrapper_kwargs is not None:
            env = pixels.Wrapper(env, **pixel_wrapper_kwargs)
        self._env = env

        # viewer.launch(self._env)
        self._observation_keys = tuple(env.observation_spec().keys())

        observation_space = env.observation_spec()
        self._observation_space = observation_space

        action_space = env.action_spec()
        if len(action_space.shape) > 1:
            raise NotImplementedError(
                "Shape of the action space ({}) is not flat, make sure to"
                " check the implemenation.".format(action_space))
        self._action_space = action_space

        self._step_count = 0

    def reset(self):
        self._step_count = 0
        time_step = self._env.reset()
        observation = self._filter_observation(time_step.observation)

        global Observation
        if Observation is None:
            Observation = namedarraytuple("Observation", list(observation.keys()))
        observation = Observation(**{k: v for k, v in observation.items()
                                     if k in self._observation_keys})
        return observation

    def step(self, action):
        self._video.write(grabFrame(self._env))
        time_step = self._env.step(action)
        reward = time_step.reward
        terminal = time_step.last()

        observation = self._filter_observation(time_step.observation)

        self._step_count += 1

        global Observation
        if Observation is None:
            Observation = namedarraytuple("Observation", list(observation.keys()))
        observation = Observation(**{k: v.copy() for k, v in observation.items()
                                     if k in self._observation_keys})

        return EnvStep(observation, reward, terminal, dict())

    def render(self, *args, mode='rgb_array', width=64, height=64,
               cameria_id=0, **kwargs):
        cameria_id = "fixed"
        return self._env.physics.render(width=width, height=height,
                                        camera_id=cameria_id, **kwargs)
        raise NotImplementedError(mode)

    def get_obs(self):
        obs = self._env.task.get_observation(self._env.physics)
        obs['pixels'] = self._env.physics.render(**self._env._render_kwargs)
        obs = self._filter_observation(obs)
        obs = Observation(**{k: v for k, v in obs.items()
                             if k in self._observation_keys})
        return obs

    def get_state(self, ignore_step=True):
        if ignore_step:
            return self._env.physics.get_state()
        return self._env.physics.get_state(), self._step_count

    def set_state(self, state, ignore_step=True):
        if ignore_step:
            self._env.physics.set_state(state)
            self._env.step(np.zeros(self.action_space.shape))
        else:
            self._env.physics.set_state(state[0])
            self._env.step(np.zeros(self.action_space.shape))
            self._step_count = state[1]

    def get_geoms(self):
        return self._env.task.get_geoms(self._env.physics)

    @property
    def spaces(self):
        return EnvSpaces(
            observation=self._observation_space,
            action=self._action_space,
        )

    def _filter_observation(self, observation):
        observation = type(observation)([
            (name, value)
            for name, value in observation.items()
            if name in self._observation_keys
        ])
        return observation
