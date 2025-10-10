

# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.base.legged_robot import LeggedRobot, euler_from_quaternion
from .humanoid_config import HumanoidCfg

class Humanoid(LeggedRobot):
    def __init__(self, cfg: HumanoidCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self._parse_cfg(self.cfg)
        self.domain_rand_general = self.cfg.domain_rand.domain_rand_general
        
        # Pre init for motion loading
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'
        
        BaseTask.__init__(self, self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        self.global_counter = 0
        self.total_env_steps_counter = 0

    def _init_buffers(self):
        super()._init_buffers()
        
        self.feet_force_sum = torch.ones(self.num_envs, 2, device=self.device)

    def _create_envs(self):
        super()._create_envs()
        if self.cfg.env.record_video:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 720*2
            camera_props.height = 480*2
            self._rendering_camera_handles = []
            for i in range(self.num_envs):
                cam_pos = np.array([2, 0, 0.3])
                camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
                self._rendering_camera_handles.append(camera_handle)
                self.gym.set_camera_location(camera_handle, self.envs[i], gymapi.Vec3(*cam_pos), gymapi.Vec3(*0*cam_pos))
                
    def render_record(self, mode="rgb_array"):
        self.gym.step_graphics(self.sim)
        self.gym.clear_lines(self.viewer)
        self.gym.render_all_camera_sensors(self.sim)
        imgs = []
        for i in range(self.num_envs):
            cam = self._rendering_camera_handles[i]
            root_pos = self.root_states[i, :3].cpu().numpy()
            cam_pos = root_pos + np.array([0, -2, 0.3])
            self.gym.set_camera_location(cam, self.envs[i], gymapi.Vec3(*cam_pos), gymapi.Vec3(*root_pos))
            img = self.gym.get_camera_image(self.sim, self.envs[i], cam, gymapi.IMAGE_COLOR)
            w, h = img.shape
            imgs.append(img.reshape([w, h // 4, 4]))
                
        return imgs
    
                              
    def draw_goal(self, id=None):
        sphere_geom_lin = gymutil.WireframeSphereGeometry(0.01, 32, 32, None, color=(1, 0, 0))
        sphere_geom_cmd = gymutil.WireframeSphereGeometry(0.01, 32, 32, None, color=(0, 1, 0))
        sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 32, 32, None, color=(0.5, 0.0, 0.0))
        sphere_geom_cmd_arrow = gymutil.WireframeSphereGeometry(0.02, 32, 32, None, color=(0.0, 0.3, 0.0))
        
        sphere_geom_ang_vel = gymutil.WireframeSphereGeometry(0.01, 32, 32, None, color=(0, 0, 1))
        sphere_geom_ang_cmd = gymutil.WireframeSphereGeometry(0.01, 32, 32, None, color=(1, 1, 0))
        sphere_geom_ang_vel_arrow = gymutil.WireframeSphereGeometry(0.02, 32, 32, None, color=(0.0, 0.0, 0.5))
        sphere_geom_ang_cmd_arrow = gymutil.WireframeSphereGeometry(0.02, 32, 32, None, color=(0.5, 0.5, 0.0))
        
        current_orientation = self.root_states[:, 3:7]
        cmd_dir = torch.zeros((self.num_envs, 3), device=self.device)
        cmd_dir[:, 0] = self.commands[:, 0]
        cmd_dir[:, 1] = self.commands[:, 1]
        vel_dir = self.root_states[:, 7:10].clone()
        base_dir = quat_rotate(current_orientation, cmd_dir)
        
        ang_vel_yaw_dir = torch.zeros((self.num_envs, 3), device=self.device)
        ang_vel_yaw_dir[:, 2] = self.base_ang_vel[:, 2]
        ang_cmd_dir = torch.zeros((self.num_envs, 3), device=self.device)
        ang_cmd_dir[:, 2] = self.commands[:, 2]
        
        for id in range(self.num_envs):
            for i in range(10):
                pose = gymapi.Transform(gymapi.Vec3(self.root_states[id, 0] + i*0.1*base_dir[id, 0], self.root_states[id, 1] + i*0.1*base_dir[id, 1], self.root_states[id, 2]), r=None)
                vel_pose = gymapi.Transform(gymapi.Vec3(self.root_states[id, 0] + i*0.1*vel_dir[id, 0], self.root_states[id, 1] + i*0.1*vel_dir[id, 1], self.root_states[id, 2]), r=None)
                gymutil.draw_lines(sphere_geom_cmd, self.gym, self.viewer, self.envs[id], pose)
                gymutil.draw_lines(sphere_geom_lin, self.gym, self.viewer, self.envs[id], vel_pose)
                
                ang_cmd_pose = gymapi.Transform(gymapi.Vec3(self.root_states[id, 0], self.root_states[id, 1] + i*0.1*ang_cmd_dir[id, 2], self.root_states[id, 2] + 0.5), r=None)
                ang_vel_yaw_pose = gymapi.Transform(gymapi.Vec3(self.root_states[id, 0], self.root_states[id, 1] + i*0.1*ang_vel_yaw_dir[id, 2], self.root_states[id, 2] + 0.5), r=None)
                gymutil.draw_lines(sphere_geom_ang_cmd, self.gym, self.viewer, self.envs[id], ang_cmd_pose)
                gymutil.draw_lines(sphere_geom_ang_vel, self.gym, self.viewer, self.envs[id], ang_vel_yaw_pose)
            gymutil.draw_lines(sphere_geom_cmd_arrow, self.gym, self.viewer, self.envs[id], pose)
            gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[id], vel_pose)
            
            gymutil.draw_lines(sphere_geom_ang_cmd_arrow, self.gym, self.viewer, self.envs[id], ang_cmd_pose)
            gymutil.draw_lines(sphere_geom_ang_vel_arrow, self.gym, self.viewer, self.envs[id], ang_vel_yaw_pose)
    
    def step(self, actions):
        actions = self.reindex(actions)
        actions.to(self.device)
        action_tensor = actions.clone()
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), action_tensor[:, None, :].clone()], dim=1)

        if self.cfg.domain_rand.action_delay:
            if self.total_env_steps_counter <= self.cfg.noise.noise_increasing_steps * 24:
                self.delay = torch.tensor(0, device=self.device, dtype=torch.float)
            else:
                self.delay = torch.tensor(np.random.randint(2), device=self.device, dtype=torch.float)
            indices = -self.delay - 1
            action_tensor = self.action_history_buf[:, indices.long()]

        self.global_counter += 1
        self.total_env_steps_counter += 1
        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        self.actions = torch.clip(action_tensor, -clip_actions, clip_actions).to(self.device)
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset_idx(self, env_ids, init=False):
        if len(env_ids) == 0:
            return

        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
            
        dof_pos = self.default_dof_pos_all.clone()

        # reset robot states
        self._reset_dofs(env_ids, dof_pos, torch.zeros_like(dof_pos))
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)  # no resample commands
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.  # reset obs history buffer TODO no 0s
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        self.feet_land_time[env_ids] = 0.
        self._reset_buffers_extra(env_ids)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['metric_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] * self.reward_scales[key]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        return
    
    def _reset_buffers_extra(self, env_ids):
        pass
                                                                                                                                                                                                                                                                                                                                                                   
    def _reset_dofs(self, env_ids, dof_pos, dof_vel):
        self.dof_pos[env_ids] = dof_pos[env_ids] * torch_rand_float(0.8, 1.2, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = dof_vel[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        self.episode_length[env_ids] = self.episode_length_buf[env_ids].float()

        self.reset_idx(env_ids)

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]

        if self.cfg.rewards.regularization_scale_curriculum:
            # import ipdb; ipdb.set_trace()
            if torch.mean(self.episode_length.float()).item()> 420.:
                self.cfg.rewards.regularization_scale *= (1. + self.cfg.rewards.regularization_scale_gamma)
            elif torch.mean(self.episode_length.float()).item() < 50.:
                self.cfg.rewards.regularization_scale *= (1. - self.cfg.rewards.regularization_scale_gamma)
            self.cfg.rewards.regularization_scale = max(min(self.cfg.rewards.regularization_scale, self.cfg.rewards.regularization_scale_range[1]), self.cfg.rewards.regularization_scale_range[0])

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.draw_goal()
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0)
        self._resample_commands(env_ids.nonzero(as_tuple=False).flatten())

        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
    
    def _randomize_gravity(self, external_force = None):
        if self.cfg.domain_rand.randomize_gravity and external_force is None:
            min_gravity, max_gravity = self.cfg.domain_rand.gravity_range
            external_force = torch.rand(3, dtype=torch.float, device=self.device,
                                        requires_grad=False) * (max_gravity - min_gravity) + min_gravity


        sim_params = self.gym.get_sim_params(self.sim)
        if external_force is None:
            gravity = torch.Tensor([0, 0, -9.81]).to(self.device)
        else:
            gravity = external_force + torch.Tensor([0, 0, -9.81]).to(self.device)
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)
    
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.cfg.domain_rand.gravity_rand_interval = np.ceil(self.cfg.domain_rand.gravity_rand_interval_s / self.dt)
    
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self.root_states[env_ids, 0:3] = root_pos
        self.root_states[env_ids, 3:7] = root_rot
        self.root_states[env_ids, 7:10] = root_vel
        self.root_states[env_ids, 10:13] = root_ang_vel

        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel
        return

    def check_termination(self):
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        height_cutoff = self.root_states[:, 2] < self.cfg.rewards.termination_height
        
        roll_cut = torch.abs(self.roll) > self.cfg.rewards.termination_roll
        pitch_cut = torch.abs(self.pitch) > self.cfg.rewards.termination_pitch

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= height_cutoff
        self.reset_buf |= roll_cut
        self.reset_buf |= pitch_cut

    def get_first_contact(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        first_contact = (self.feet_air_time > 0.) * contact_filt
        return first_contact
        
    def update_feet_force_sum(self):
        self.feet_force_sum += self.contact_forces[:, self.feet_indices, 2] * self.dt

    def  _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase
    
    def _get_gait_phase(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        stance_mask[:, 0] = sin_pos >= 0
        stance_mask[:, 1] = sin_pos < 0
        stance_mask[torch.abs(sin_pos) < self.cfg.rewards.double_support_threshold] = 1
        stance_mask[~self.get_walking_cmd_mask()] = 1

        return stance_mask
    
    def compute_ref_state(self):
        phase = self._get_phase()
        _sin_pos_l = torch.sin(2 * torch.pi * phase)
        _sin_pos_r = torch.sin(2 * torch.pi * phase + torch.pi)
        sin_pos_l = _sin_pos_l.clone()
        sin_pos_r = _sin_pos_r.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, 2] = sin_pos_l * scale_1
        self.ref_dof_pos[:, 3] = -sin_pos_l * scale_2
        self.ref_dof_pos[:, 4] = sin_pos_l * scale_1
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r > 0] = 0
        self.ref_dof_pos[:, 8] = sin_pos_r * scale_1
        self.ref_dof_pos[:, 9] = -sin_pos_r * scale_2
        self.ref_dof_pos[:, 10] = sin_pos_r * scale_1
        # Double support phase
        indices = (torch.abs(sin_pos_l) < self.cfg.rewards.double_support_threshold) & (torch.abs(sin_pos_r) < self.cfg.rewards.double_support_threshold)
        self.ref_dof_pos[indices] = 0
        # self.ref_dof_pos[~self.get_walking_cmd_mask()] = 0
        self.ref_dof_pos += self.default_dof_pos_all

        self.ref_action = 2 * self.ref_dof_pos
    
    def _get_noise_scale_vec(self, cfg):
        noise_scale_vec = torch.zeros(1, self.cfg.env.n_proprio, device=self.device)
        if not self.cfg.noise.add_noise:
            return noise_scale_vec
        noise_start_dim = 2 + self.cfg.commands.num_commands
        noise_scale_vec[:, noise_start_dim:noise_start_dim+3] = self.cfg.noise.noise_scales.ang_vel
        noise_scale_vec[:, noise_start_dim+3:noise_start_dim+5] = self.cfg.noise.noise_scales.imu
        noise_scale_vec[:, noise_start_dim+5:noise_start_dim+5+self.num_dof] = self.cfg.noise.noise_scales.dof_pos
        noise_scale_vec[:, noise_start_dim+5+self.num_dof:noise_start_dim+5+2*self.num_dof] = self.cfg.noise.noise_scales.dof_vel
        return noise_scale_vec
    
    def compute_observations(self):

        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)
        
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(0*self.yaw, 0*self.yaw, self.yaw)
        # self.commands[:] = 0.
        obs_buf = torch.cat((
                            sin_pos,
                            cos_pos,
                            self.commands,  # 3 dims
                            self.base_ang_vel  * self.obs_scales.ang_vel,   # 3 dims
                            imu_obs,    # 2 dims
                            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
                            self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                            self.reindex(self.action_history_buf[:, -1]),
                            ),dim=-1)
        if self.cfg.noise.add_noise and self.headless:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * min(self.total_env_steps_counter / (self.cfg.noise.noise_increasing_steps * 24),  1.)
        elif self.cfg.noise.add_noise and not self.headless:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec
        else:
            obs_buf += 0.

        if self.cfg.domain_rand.domain_rand_general:
            priv_latent = torch.cat((
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                self.motor_strength[0] - 1, 
                self.motor_strength[1] - 1,
                self.base_lin_vel,
            ), dim=-1)
        else:
            priv_latent = torch.zeros((self.num_envs, self.cfg.env.n_priv_latent), device=self.device)

        self.obs_buf = torch.cat([obs_buf, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)


        if self.cfg.env.history_len > 0:
            self.obs_history_buf = torch.where(
                (self.episode_length_buf <= 1)[:, None, None], 
                torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
                torch.cat([
                    self.obs_history_buf[:, 1:],
                    obs_buf.unsqueeze(1)
                ], dim=1)
            )

        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([
                self.contact_buf[:, 1:],
                self.contact_filt.float().unsqueeze(1)
            ], dim=1)
        )
        
    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if self.cfg.env.teleop_mode:
            self.commands[env_ids] = 0.
            return
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 0] *= torch.abs(self.commands[env_ids, 0]) > self.cfg.commands.lin_vel_clip
        self.commands[env_ids, 1] *= torch.abs(self.commands[env_ids, 1]) > self.cfg.commands.lin_vel_clip
        self.commands[env_ids, 2] *= torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_clip
        
        if_stand = torch_rand_float(0, 1, (len(env_ids), 1), device=self.device) < self.cfg.commands.stand_prob
        self.commands[env_ids] *= ~if_stand
    
    def get_walking_cmd_mask(self, env_ids=None, return_all=False):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        walking_mask0 = torch.abs(self.commands[env_ids, 0]) > self.cfg.commands.lin_vel_clip
        walking_mask1 = torch.abs(self.commands[env_ids, 1]) > self.cfg.commands.lin_vel_clip
        walking_mask2 = torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_clip
        walking_mask = walking_mask0 | walking_mask1 | walking_mask2
        if return_all:
            return walking_mask0, walking_mask1, walking_mask2, walking_mask
        return walking_mask
    
    ######### utils #########
    
    def get_episode_log(self, env_ids=0):
        log = {
            "ang vel": self.base_ang_vel[env_ids].cpu().numpy().tolist(),
            "dof pos": self.dof_pos[env_ids].cpu().numpy().tolist(),
            "dof vel": self.dof_vel[env_ids].cpu().numpy().tolist(),
            "action": self.action_history_buf[env_ids, -1].cpu().numpy().tolist(),
            "torque": self.torques[env_ids].cpu().numpy().tolist(),
        }
        
        return log
    
    
    ######### Rewards #########
    def compute_reward(self):
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() 
            if name in self.cfg.rewards.regularization_names:
                self.rew_buf += rew * self.reward_scales[name] * self.cfg.rewards.regularization_scale
            else: 
                self.rew_buf += rew * self.reward_scales[name]
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        if self.cfg.rewards.clip_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=-0.5)
        
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        
    def _reward_alive(self):
        return 1.
    
    def _reward_tracking_lin_vel_exp(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma_ang)
    
    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)
    
    def _reward_lin_vel_z(self):
        rew = torch.square(self.base_lin_vel[:, 2])
        return rew
    
    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        rew = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return rew
    
    def _reward_collision(self):
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_dof_torque_limits(self):
        out_of_limits = torch.sum((torch.abs(self.torques) / self.torque_limits - self.cfg.rewards.soft_torque_limit).clip(min=0), dim=1)
        return out_of_limits
    
    def _reward_feet_contact_forces(self):
        rew = torch.norm(self.contact_forces[:, self.feet_indices, 2], dim=-1)
        rew[rew < self.cfg.rewards.max_contact_force] = 0
        rew[rew > self.cfg.rewards.max_contact_force] -= self.cfg.rewards.max_contact_force
        rew[~self.get_walking_cmd_mask()] = 0
        return rew
    
    def _reward_torque_penalty(self):
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_error
    
    def _reward_dof_error_upper(self):
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.cfg.asset.n_lower_body_dofs:], dim=1)
        return dof_error
    
    def _reward_feet_stumble(self):
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             4 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        return rew.float()
    
    def _reward_feet_clearance(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        feet_z = self.rigid_body_states[:, self.feet_indices, 2] - 0.05
        
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z
        swing_mask = 1 - self._get_gait_phase()
        rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos
    
    def _reward_feet_contact_number(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_body_states[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1) 
    
    def _reward_feet_distance(self):
        foot_pos = self.rigid_body_states[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2
    
    def _reward_knee_distance(self):
        foot_pos = self.rigid_body_states[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_knee_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_joint_pos(self):
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r
    
    def _reward_tracking_vel_hard(self):
        lin_vel_error = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)
        
        ang_vel_error = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)
        
        linear_error = 0.2 * (lin_vel_error + ang_vel_error)
        
        return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error
    
    def _reward_low_speed(self):
        base_lin_vel = self.base_lin_vel[:, 0]
        absolute_speed = torch.abs(base_lin_vel)
        absolute_command = torch.abs(self.commands[:, 0])
        
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)
        
        sign_mismatch = torch.sign(base_lin_vel) != torch.sign(self.commands[:, 0])
        
        reward = torch.zeros_like(self.base_lin_vel[:, 0])
        
        reward[speed_too_low] = -1.0
        reward[speed_too_high] = 0.
        reward[speed_desired] = 1.2
        reward[sign_mismatch] = -2.0
        return reward
    
    def _reward_stand_still(self):
        # dof_error = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
        dof_error = torch.norm(self.dof_pos - self.default_dof_pos, dim=1)
        rew = torch.exp(-dof_error*2)
        rew[self.get_walking_cmd_mask()] = 0.
        return rew
    

    
    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        return torch.norm(self.last_actions - self.actions, dim=1)
    
    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_stand_dof_acc(self):
        dof_acc = torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
        dof_acc[self.get_walking_cmd_mask()] = 0.
        return dof_acc
    
    def _reward_stand_dof_vel(self):
        dof_vel = torch.sum(torch.square(self.dof_vel), dim=1)
        dof_vel[self.get_walking_cmd_mask()] = 0.
        return dof_vel
