from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz

class Tinker(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

    def compute_observations(self):
        """ Computes observations

        """
        self.obs_buf = torch.cat((  
            self.commands[:, :3] * self.commands_scale,  # 3 命令
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3 基座角速递
            self.base_euler_xyz,  # 3 基座欧拉角
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 10 所有自由度的角度
            self.dof_vel * self.obs_scales.dof_vel,  # 10 所有自由度的角速度
            self.actions * self.cfg.control.action_scale,  # 10 上一轮输出的action
            ),dim=-1)

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1) # 1
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.*contacts, dim=1)==1
        return 1.*single_contact # 奖励只有一只脚接触地面的情况
