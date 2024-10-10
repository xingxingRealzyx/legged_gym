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
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)  # 步态生成器-生成的参考姿势
        for i in range(self.num_envs):
            self.ref_dof_pos[i] = self.default_dof_pos
        print("------------------------------------------------")
        print(self.ref_dof_pos[0])
        print("------------------------------------------------")

    def compute_observations(self):
        """ Computes observations

        """
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        # self.obs_buf shape: [num_envs, num_observations] , 这里就是 [4096, 39]
        # dim = -1 表示在最后一个维度上进行拼接
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
            # print("0------------" + str(self.obs_buf.size()))
            # print("1------------" + str(self.root_states[:, 2].size()))
            # print("2------------" + str(self.root_states[:, 2].unsqueeze(1).size()))
            # print("3------------" + str(self.measured_heights.size()))

            # self.root_states shape: [num_envs, 13]
            # 取 z 轴, self.root_states[:, 2]的形状是[num_envs],unsqueeze(1)之后的形状是 [num_envs, 1]
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            # print("0-height-----------" + str(heights.size()))

            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1) # 1
            # print("1------------" + str(self.obs_buf.size()))

        # add noise if needed
        if self.add_noise:
            # print("2------------" + str(self.obs_buf.size()))
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
    def check_termination(self):
        super().check_termination()
        measured_heights = torch.sum(self.rigid_state[:, self.feet_indices, 2], dim=1) / 2
        base_height = self.root_states[:, 2] - (measured_heights - 0.05)
        self.reset_buf2 = base_height < self.cfg.asset.terminate_body_height
        # self.reset_buf2 = self.root_states[:, 2] < self.cfg.asset.terminate_body_height  # 0.3!!!!!!!!!!!!!!!!!
        self.reset_buf |= self.reset_buf2

    # ------------------------ rewards --------------------------------------------------------------------------------

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.*contacts, dim=1)==1
        return 1.*single_contact # 奖励只有一只脚接触地面的情况

    def _reward_target_joint_pos_l(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff_l = torch.sum((self.dof_pos[:, 5:10] - self.ref_dof_pos[:, 5:10]) ** 2, dim=1)
        imitate_reward = torch.exp(-7 * joint_diff_l)
        return imitate_reward # 奖励左侧关节幅度小的情况
    
    def _reward_target_joint_pos_r(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff_r = torch.sum((self.dof_pos[:, 0:5] - self.ref_dof_pos[:, 0:5]) ** 2, dim=1)
        imitate_reward = torch.exp(-7*joint_diff_r)  # positive reward, not the penalty
        return imitate_reward # 奖励右侧关节幅度小的情况
    
    def _reward_orientation(self):
        # 奖励机器人倾角小
        return torch.exp(-10. * torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1))
    
    def _reward_tracking_lin_x_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.abs(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_y_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.abs(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_target_ankle_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        diff = torch.abs(self.dof_pos[:, 4:6] - self.ref_dof_pos[:, 4:6])
        diff += torch.abs(self.dof_pos[:, 10:] - self.ref_dof_pos[:, 10:])
        ankle_imitate_reward = torch.exp(-20*torch.sum(diff, dim=1))  # positive reward, not the penalty
        return ankle_imitate_reward

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penilize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        # torch.norm 
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist # 0.2
        max_df = self.cfg.rewards.max_dist # 0.5
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2
