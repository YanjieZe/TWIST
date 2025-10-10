import numpy as np

from isaacgym.torch_utils import *
from isaacgym import gymtorch

import torch

from legged_gym.envs.base.legged_robot import euler_from_quaternion
from .humanoid_mimic_config import HumanoidMimicCfg
from .humanoid_char import HumanoidChar, convert_to_global_root_body_pos, convert_to_local_root_body_pos

from pose.utils import torch_utils
from pose.utils.motion_lib_pkl import MotionLib

import time
from termcolor import cprint

import torch


class HumanoidMimic(HumanoidChar):
    def __init__(self, cfg: HumanoidMimicCfg, sim_params, physics_engine, sim_device, headless):
        self._enable_early_termination = cfg.env.enable_early_termination
        self._pose_termination = cfg.env.pose_termination
        self._pose_termination_dist = cfg.env.pose_termination_dist
        self._root_tracking_termination_dist = cfg.env.root_tracking_termination_dist
        self._tar_obs_steps = cfg.env.tar_obs_steps
        self._tar_obs_steps = torch.tensor(self._tar_obs_steps, device=sim_device, dtype=torch.int)
        self._rand_reset = cfg.env.rand_reset
        self._ref_char_offset = torch.tensor(cfg.env.ref_char_offset, device=sim_device, dtype=torch.float)
        self._track_root = cfg.env.track_root
        self.global_obs = cfg.env.global_obs
        cprint(f"[HumanoidMimic] global_obs: {self.global_obs}")

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.episode_length = torch.zeros((self.num_envs), device=self.device)
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        num_motions = self._motion_lib.num_motions()
        self.motion_difficulty = 9 * torch.ones((num_motions), device=self.device, dtype=torch.float)
        self.mean_motion_difficulty = 9.
        self.motion_termination_dist = torch.ones((num_motions), device=self.device, dtype=torch.float)
        self.motion_names = self._motion_lib.get_motion_names()
        self.deviate_tracking_frames = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)
        self.deviate_vel_tracking_frames = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
    
    def _get_max_motion_len(self):
        max_len = 0
        num_motions = self._motion_lib.num_motions()
        for i in range(num_motions):
            curr_len = self._motion_lib.get_motion_length(i)
            max_len = max(max_len, curr_len)
            
        return max_len
        
    def _init_buffers(self):
        self._load_motions()
        if self.viewer is None:
            self.max_episode_length_s = self._get_max_motion_len().item()
            self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        super()._init_buffers()
        self._init_motion_buffers()
        
    def _load_motions(self):
        self._motion_lib = MotionLib(motion_file=self.cfg.motion.motion_file, device=self.device)
        return
    
    def _init_motion_buffers(self):
        self._motion_ids = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self._motion_time_offsets = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        self._ref_root_pos = torch.zeros_like(self.root_states[:, 0:3])
        self._ref_root_rot = torch.zeros_like(self.root_states[:, 3:7])
        self._ref_root_vel = torch.zeros_like(self.root_states[:, 7:10])
        self._ref_root_ang_vel = torch.zeros_like(self.root_states[:, 10:13])
        self._ref_body_pos = torch.zeros_like(self.rigid_body_states[..., 0:3])
        self._ref_dof_pos = torch.zeros_like(self.dof_pos)
        self._ref_dof_vel = torch.zeros_like(self.dof_vel)
        
        self._dof_err_w = self.cfg.env.dof_err_w
        if self._dof_err_w is None:
            self._dof_err_w = torch.ones(self.num_dof, device=self.device, dtype=torch.float)
        else:
            self._dof_err_w = torch.tensor(self._dof_err_w, device=self.device, dtype=torch.float)
        
        self._key_body_ids_motion = self._motion_lib.get_key_body_idx(key_body_names=self.cfg.motion.key_bodies)
        # compare two tensors are same
        # assert torch.equal(self._key_body_ids, torch.tensor(key_body_ids_motion, device=self.device, dtype=torch.long)), \
        #     f"Key body ids mismatch: {self._key_body_ids} vs {key_body_ids_motion}"
    
    def _reset_ref_motion(self, env_ids, motion_ids=None):
        n = len(env_ids)
        if motion_ids is None:
            motion_ids = self._motion_lib.sample_motions(n, motion_difficulty=self.motion_difficulty)
        
        if self._rand_reset:
            motion_times = self._motion_lib.sample_time(motion_ids)
        else:
            motion_times = torch.zeros(motion_ids.shape, device=self.device, dtype=torch.float)
        
        self._motion_ids[env_ids] = motion_ids
        self._motion_time_offsets[env_ids] = motion_times
        
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        root_pos[:, 2] += self.cfg.motion.height_offset
        
        self._ref_root_pos[env_ids] = root_pos
        self._ref_root_rot[env_ids] = root_rot
        self._ref_root_vel[env_ids] = root_vel
        self._ref_root_ang_vel[env_ids] = root_ang_vel
        self._ref_dof_pos[env_ids] = dof_pos
        self._ref_dof_vel[env_ids] = dof_vel
        self._ref_body_pos[env_ids] = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=body_pos)
    
    def _get_motion_times(self, env_ids=None):
        if env_ids is None:
            motion_times = self.episode_length_buf * self.dt + self._motion_time_offsets
        else:
            motion_times = self.episode_length_buf[env_ids] * self.dt + self._motion_time_offsets[env_ids]
        return motion_times
    
    def _update_ref_motion(self):
        motion_ids = self._motion_ids
        motion_times = self._get_motion_times()
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        root_pos[:, 2] += self.cfg.motion.height_offset
        root_pos[:, :2] += self.episode_init_origin[:, :2]
        
        self._ref_root_pos[:] = root_pos
        self._ref_root_rot[:] = root_rot
        self._ref_root_vel[:] = root_vel
        self._ref_root_ang_vel[:] = root_ang_vel
        self._ref_dof_pos[:] = dof_pos
        self._ref_dof_vel[:] = dof_vel
        self._ref_body_pos[:] = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=body_pos)
            
    def _reset_root_states(self, env_ids, root_vel=None, root_quat=None, root_pos=None, root_ang_vel=None):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if self.cfg.env.randomize_start_pos:
                rand_pos = torch_rand_float(-0.3, 0.3, (len(env_ids), 2), device=self.device)
                self.root_states[env_ids, :2] += rand_pos # xy position within 1m of the center
                self.episode_init_origin[env_ids, :2] = self.env_origins[env_ids, :2] + rand_pos
            if self.cfg.env.randomize_start_yaw:
                rand_yaw = torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                quat = quat_from_euler_xyz(0*rand_yaw, 0*rand_yaw, rand_yaw) 
                self.root_states[env_ids, 3:7] = quat[:, :]
            
            if root_vel is not None:
                self.root_states[env_ids, 7:10] = root_vel[env_ids, :]
            if root_quat is not None:
                self.root_states[env_ids, 3:7] = root_quat[env_ids, :]
            if root_pos is not None:
                self.root_states[env_ids, 2] = root_pos[env_ids, 2] + 0.05 # always higher a bit to avoid foot penetration
                self.root_states[env_ids, :2] += root_pos[env_ids, :2]
            if root_ang_vel is not None:
                self.root_states[env_ids, 10:13] = root_ang_vel[env_ids, :]
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
            
    def reset_idx(self, env_ids, motion_ids=None):
        if len(env_ids) == 0:
            return
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['metric_' + key] = torch.mean(self.episode_sums[key][env_ids] / self._motion_lib.get_motion_length(self._motion_ids[env_ids]))
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] * self.reward_scales[key] / self._motion_lib.get_motion_length(self._motion_ids[env_ids]))
            self.episode_sums[key][env_ids] = 0.
        
        if self.cfg.motion.motion_curriculum:
            self._update_motion_difficulty(env_ids)
        self._reset_ref_motion(env_ids=env_ids, motion_ids=motion_ids)
        
        # vel_factor = torch_rand_float(0.0, 0.6, (self.num_envs, 1), device=self.device)
        # vel_factor[vel_factor < 0.2] = 0.0
        vel_factor = 0.8

        # RSI
        self._reset_dofs(env_ids, self._ref_dof_pos, self._ref_dof_vel*vel_factor)
        self._reset_root_states(env_ids=env_ids, root_vel=self._ref_root_vel*vel_factor, root_quat=self._ref_root_rot,
                                root_pos=self._ref_root_pos, root_ang_vel=self._ref_root_ang_vel*vel_factor)

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
        self.deviate_tracking_frames[env_ids] = 0.
        self.deviate_vel_tracking_frames[env_ids] = 0.
        self._reset_buffers_extra(env_ids)

        self.episode_length_buf[env_ids] = 0
        
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        
        if self.cfg.motion.motion_curriculum:
            self.mean_motion_difficulty = torch.mean(self.motion_difficulty)
            
        _, _, y = euler_from_quaternion(self.root_states[:, 3:7])
        self.init_yaw[env_ids] = y[env_ids]
        return
    
    def _hard_sync_motion_loop(self):
        motion_times = self._get_motion_times()
        motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)
        hard_sync_envs = (motion_times >= motion_lengths) & (torch.abs(motion_times - motion_lengths) < self.dt)
        hard_sync_env_ids = hard_sync_envs.nonzero(as_tuple=False).flatten()
        if len(hard_sync_env_ids) == 0:
            return
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos = self._motion_lib.calc_motion_frame(self._motion_ids, motion_times*0)
        self._reset_dofs(hard_sync_env_ids, dof_pos, dof_vel*0.8)
        self._reset_root_states(env_ids=hard_sync_env_ids, root_vel=root_vel*0.8, root_quat=root_rot, root_pos=root_pos, root_ang_vel=root_ang_vel*0.8)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
    
    def _update_motion_difficulty(self, env_ids):
        reset_motion_ids = self._motion_ids[env_ids] # get motion ids that are being reset
        completion_rate = self.episode_length_buf[env_ids] * self.dt / self._motion_lib.get_motion_length(reset_motion_ids) # get completion rate of each environment
        # get mean completion rate of each motion
        motion_completion_rate_sum = torch.zeros(self._motion_lib.num_motions(), device=self.device, dtype=torch.float).scatter_add(0, reset_motion_ids, completion_rate)
        motion_completion_rate_count = torch.zeros(self._motion_lib.num_motions(), device=self.device, dtype=torch.float).scatter_add(0, reset_motion_ids, torch.ones_like(completion_rate, dtype=torch.float))
        motion_completion_rate = motion_completion_rate_sum / torch.clamp(motion_completion_rate_count, min=1)
        motion_completion_rate[motion_completion_rate_count == 0] = 0.7
        # update motion difficulty
        add_idx = motion_completion_rate <= 0.5
        sub_idx = motion_completion_rate >= 0.95
        self.motion_difficulty[add_idx] *= (1 + self.cfg.motion.motion_curriculum_gamma)
        self.motion_difficulty[sub_idx] *= (1 - self.cfg.motion.motion_curriculum_gamma)
        self.motion_difficulty = torch.clamp(self.motion_difficulty, min=1., max=9.)
        
        motion_difficulty_ratio = (self.motion_difficulty - 1) / 8
        self.motion_termination_dist = (self._pose_termination_dist - 0.1) * motion_difficulty_ratio + 0.1 # when motion difficulty is 1, termination dist is 0.1
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        self._update_ref_motion()
        # self._hard_sync_motion_loop()

        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        
        if self.cfg.domain_rand.push_end_effector and (self.common_step_counter % self.cfg.domain_rand.push_end_effector_interval == 0):
            self._push_end_effector()
            
    def check_termination(self):
        contact_force_termination = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf = contact_force_termination
        
        # height_cutoff = self.root_states[:, 2] < self.cfg.rewards.termination_height
        height_cutoff = torch.abs(self.root_states[:, 2] - self._ref_root_pos[:, 2]) > self.cfg.rewards.root_height_diff_threshold

        roll_cut = torch.abs(self.roll) > self.cfg.rewards.termination_roll
        pitch_cut = torch.abs(self.pitch) > self.cfg.rewards.termination_pitch
        self.reset_buf |= roll_cut
        self.reset_buf |= pitch_cut
        motion_end = self.episode_length_buf * self.dt >= self._motion_lib.get_motion_length(self._motion_ids)
        self.reset_buf |= height_cutoff
        
        if self.viewer is None:
            self.reset_buf |= motion_end
        
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        if self.viewer is None:
            self.time_out_buf |= motion_end
        
        self.reset_buf |= self.time_out_buf
        
        vel_too_large = torch.norm(self.root_states[:, 7:10], dim=-1) > 5.
        self.reset_buf |= vel_too_large
        
        if self._pose_termination:
            body_pos = self.rigid_body_states[:, self._key_body_ids, 0:3] - self.rigid_body_states[:, 0:1, 0:3]
            tar_body_pos = self._ref_body_pos[:, self._key_body_ids] - self._ref_root_pos[:, None, :] 
            
            if not self.global_obs:
                body_pos = convert_to_local_root_body_pos(self.root_states[:, 3:7], body_pos)
                tar_body_pos = convert_to_local_root_body_pos(self._ref_root_rot, tar_body_pos)
            
            body_pos_diff = tar_body_pos - body_pos # (envs, bodies, 3)
            body_pos_dist = torch.sum(body_pos_diff * body_pos_diff, dim=-1) # (envs, bodies)
            body_pos_dist = torch.max(body_pos_dist, dim=-1)[0] # (envs)
            # if lose tracking for 50 frames continuously, reset (corresponds to 1 second)
            # lose_tracking = body_pos_dist > self.motion_termination_dist[self._motion_ids] ** 2
            # self.deviate_tracking_frames[lose_tracking] += 1
            # self.deviate_tracking_frames[~lose_tracking] = 0
            # pose_fail = self.deviate_tracking_frames >= self.cfg.motion.reset_consec_frames # 50 frames = 1 second
            
            pose_fail = body_pos_dist > self._pose_termination_dist ** 2
            
            # pose_fail = body_pos_dist > self.motion_termination_dist[self._motion_ids] ** 2
            
            if self._track_root:
                root_pos_diff = self._ref_root_pos - self.root_states[:, 0:3]
                root_pos_dist = torch.sum(root_pos_diff * root_pos_diff, dim=-1)
                root_pos_fail = root_pos_dist > self._root_tracking_termination_dist ** 2
                root_pos_fail = root_pos_fail.squeeze(-1)
                pose_fail |= root_pos_fail
            self.reset_buf |= pose_fail
        
        first_step = self.episode_length_buf == 0

        
        self.reset_buf[first_step] = 0 # Do not reset on first step
        

    def _get_mimic_obs(self):
        num_steps = self._tar_obs_steps.shape[0]
        assert num_steps > 0, "Invalid number of target observation steps"
        motion_times = self._get_motion_times().unsqueeze(-1)
        obs_motion_times = self._tar_obs_steps * self.dt + motion_times
        motion_ids_tiled = torch.broadcast_to(self._motion_ids.unsqueeze(-1), obs_motion_times.shape)
        motion_ids_tiled = motion_ids_tiled.flatten()
        obs_motion_times = obs_motion_times.flatten()
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos = self._motion_lib.calc_motion_frame(motion_ids_tiled, obs_motion_times)
        roll, pitch, _ = euler_from_quaternion(root_rot)
        roll = roll.reshape(self.num_envs, num_steps, 1)
        pitch = pitch.reshape(self.num_envs, num_steps, 1)
        
        if not self.global_obs:
            root_vel = quat_rotate_inverse(root_rot, root_vel)
            root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
        
        root_pos = root_pos.reshape(self.num_envs, num_steps, root_pos.shape[-1])
        root_vel = root_vel.reshape(self.num_envs, num_steps, root_vel.shape[-1])
        root_rot = root_rot.reshape(self.num_envs, num_steps, root_rot.shape[-1])
        root_ang_vel = root_ang_vel.reshape(self.num_envs, num_steps, root_ang_vel.shape[-1])
        dof_pos = dof_pos.reshape(self.num_envs, num_steps, dof_pos.shape[-1])
        
        mimic_obs_buf = torch.cat((
            root_pos[..., 0:3], # 3 dims @Yanjie: for tracking human root position
            roll, pitch, # 2 dims
            root_vel, # 3 dims
            root_ang_vel[..., 2:3], # 1 dim, yaw only
            dof_pos, # num_dof dims
        ), dim=-1) # shape: (num_envs, num_steps, 7 + num_dof)
        
        return mimic_obs_buf.reshape(self.num_envs, -1)
        
    def compute_observations(self):
        # imu_obs = torch.stack((self.roll, self.pitch, self.yaw - self.init_yaw), dim=1)
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        
        self.base_yaw_quat = quat_from_euler_xyz(0*self.yaw, 0*self.yaw, self.yaw)
        
        mimic_obs = self._get_mimic_obs()
        obs_buf = torch.cat((
                            mimic_obs, # (11 + num_dof) * num_steps
                            self.base_ang_vel  * self.obs_scales.ang_vel,   # 3 dims
                            imu_obs,    # 3 dims
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
            priv_latent = torch.cat((priv_latent, self.base_lin_vel), dim=-1)

 
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
        
            
    def _get_noise_scale_vec(self, cfg):
        noise_scale_vec = torch.zeros(1, self.cfg.env.n_proprio, device=self.device)
        if not self.cfg.noise.add_noise:
            return noise_scale_vec
        ang_vel_dim = 3
        imu_dim = 2
        noise_start_dim = self.cfg.env.n_mimic_obs * len(self._tar_obs_steps)
        noise_scale_vec[:, noise_start_dim:noise_start_dim+ang_vel_dim] = self.cfg.noise.noise_scales.ang_vel
        noise_scale_vec[:, noise_start_dim+ang_vel_dim:noise_start_dim+ang_vel_dim+imu_dim] = self.cfg.noise.noise_scales.imu
        noise_scale_vec[:, noise_start_dim+(ang_vel_dim+imu_dim):noise_start_dim+(ang_vel_dim+imu_dim)+self.num_dof] = self.cfg.noise.noise_scales.dof_pos
        noise_scale_vec[:, noise_start_dim+(ang_vel_dim+imu_dim)+self.num_dof:noise_start_dim+(ang_vel_dim+imu_dim)+2*self.num_dof] = self.cfg.noise.noise_scales.dof_vel
        return noise_scale_vec
    
    
    # ================== rewards ==================
    def _reward_alive(self):
        return 1.
    
    def _reward_tracking_joint_dof(self):
        dof_diff = self._ref_dof_pos - self.dof_pos
        dof_err = torch.sum(self._dof_err_w * dof_diff * dof_diff, dim=-1)
        
        pos_scale = 0.15
        return torch.exp(-pos_scale * dof_err)
    
    def _error_tracking_joint_dof(self):
        dof_diff = self._ref_dof_pos - self.dof_pos
        # compute L1 error
        dof_err = torch.mean(torch.abs(dof_diff), dim=-1)
        return dof_err
    
    def _reward_tracking_joint_vel(self):
        vel_diff = self._ref_dof_vel - self.dof_vel
        vel_err = torch.sum(self._dof_err_w * vel_diff * vel_diff, dim=-1)
        
        vel_scale = 0.01
        return torch.exp(-vel_scale * vel_err)
    
    def _error_tracking_joint_vel(self):
        vel_diff = self._ref_dof_vel - self.dof_vel
        # compute L1 error
        vel_err = torch.mean(torch.abs(vel_diff), dim=-1)
        return vel_err
    
    def _reward_tracking_root_pose(self):
        if self.global_obs:
            root_pos_diff = self._ref_root_pos - self.root_states[:, 0:3]
        else:
            root_pos_diff = self._ref_root_pos[:, 2:3] - self.root_states[:, 2:3]

        # root_pos_diff = self._ref_root_pos - self.root_states[:, 0:3]
        
        root_pos_err = torch.sum(root_pos_diff * root_pos_diff, dim=-1)
        
        root_rot_err = torch_utils.quat_diff_angle(self.root_states[:, 3:7], self._ref_root_rot)
        root_rot_err *= root_rot_err
        
        root_pose_scale = 5.0
        
        return torch.exp(-root_pose_scale * (root_pos_err + 0.1 * root_rot_err))
    
    def _error_tracking_root_translation(self):
        root_pos_diff = self._ref_root_pos - self.root_states[:, 0:3]
        # compute L1 error
        root_pos_err = torch.mean(torch.abs(root_pos_diff), dim=-1)
        return root_pos_err
    
    def _error_tracking_root_rotation(self):
        root_rot_err = torch_utils.quat_diff_angle(self.root_states[:, 3:7], self._ref_root_rot)
        # compute L1 error
        root_rot_err = torch.mean(torch.abs(root_rot_err), dim=-1)
        return root_rot_err
    
    def _reward_tracking_root_vel(self):
            
        if self.global_obs:
            root_vel_diff = self._ref_root_vel - self.root_states[:, 7:10]
            root_ang_vel_diff = self._ref_root_ang_vel - self.root_states[:, 10:13]
        else:
            local_ref_root_vel = quat_rotate_inverse(self._ref_root_rot, self._ref_root_vel)
            root_vel_diff = local_ref_root_vel - self.base_lin_vel
            local_ref_root_ang_vel = quat_rotate_inverse(self._ref_root_rot, self._ref_root_ang_vel)
            root_ang_vel_diff = local_ref_root_ang_vel - self.base_ang_vel
        
        
        root_vel_err = torch.sum(root_vel_diff * root_vel_diff, dim=-1)
        root_ang_vel_err = torch.sum(root_ang_vel_diff * root_ang_vel_diff, dim=-1)
        root_vel_scale = 1.0
        
        # return torch.exp(-root_vel_scale * (root_vel_err + 0.1 * root_ang_vel_err))
        return torch.exp(-root_vel_scale * (root_vel_err + 0.5 * root_ang_vel_err))
    
    
    def _error_tracking_root_vel(self):
        local_ref_root_vel = quat_rotate_inverse(self._ref_root_rot, self._ref_root_vel)
        root_vel_diff = local_ref_root_vel - self.base_lin_vel
        # compute L1 error
        root_vel_err = torch.mean(torch.abs(root_vel_diff), dim=-1)
        return root_vel_err
    
    def _error_tracking_root_ang_vel(self):
        local_ref_root_ang_vel = quat_rotate_inverse(self._ref_root_rot, self._ref_root_ang_vel)
        root_ang_vel_diff = local_ref_root_ang_vel - self.base_ang_vel
        # compute L1 error
        root_ang_vel_err = torch.mean(torch.abs(root_ang_vel_diff), dim=-1)
        return root_ang_vel_err
    
    def _reward_tracking_keybody_pos(self):
        key_body_pos = self.rigid_body_states[:, self._key_body_ids, 0:3] # (num_envs, num_key_bodies, 3)
        key_body_pos = key_body_pos - self.root_states[:, 0:3].unsqueeze(1)
        if not self.global_obs:
            base_yaw_quat = quat_from_euler_xyz(0*self.yaw, 0*self.yaw, self.yaw)
            # key_body_pos = convert_to_local_root_body_pos(self.root_states[:, 3:7], key_body_pos)
            key_body_pos = convert_to_local_root_body_pos(base_yaw_quat, key_body_pos)
        tar_key_body_pos = self._ref_body_pos[:, self._key_body_ids, :]
        tar_key_body_pos = tar_key_body_pos - self._ref_root_pos.unsqueeze(1)
        if not self.global_obs:
            _, _, ref_yaw = euler_from_quaternion(self._ref_root_rot)
            ref_yaw_quat = quat_from_euler_xyz(0*ref_yaw, 0*ref_yaw, ref_yaw)
            # tar_key_body_pos = convert_to_local_root_body_pos(self._ref_root_rot, tar_key_body_pos)
            tar_key_body_pos = convert_to_local_root_body_pos(ref_yaw_quat, tar_key_body_pos)
        key_body_pos_diff = key_body_pos - tar_key_body_pos
        key_body_pos_err = torch.sum(key_body_pos_diff * key_body_pos_diff, dim=-1)
        key_body_pos_err = torch.sum(key_body_pos_err, dim=-1)
        
        key_body_pos_scale = 10.0
        
        return torch.exp(-key_body_pos_scale * key_body_pos_err)
    
    def _error_tracking_keybody_pos(self):
        key_body_pos = self.rigid_body_states[:, self._key_body_ids, 0:3] # (num_envs, num_key_bodies, 3)
        key_body_pos = key_body_pos - self.root_states[:, 0:3].unsqueeze(1)
        if not self.global_obs:
            base_yaw_quat = quat_from_euler_xyz(0*self.yaw, 0*self.yaw, self.yaw)
            # key_body_pos = convert_to_local_root_body_pos(self.root_states[:, 3:7], key_body_pos)
            key_body_pos = convert_to_local_root_body_pos(base_yaw_quat, key_body_pos)
        tar_key_body_pos = self._ref_body_pos[:, self._key_body_ids, :]
        tar_key_body_pos = tar_key_body_pos - self._ref_root_pos.unsqueeze(1)
        if not self.global_obs:
            _, _, ref_yaw = euler_from_quaternion(self._ref_root_rot)
            ref_yaw_quat = quat_from_euler_xyz(0*ref_yaw, 0*ref_yaw, ref_yaw)
            tar_key_body_pos = convert_to_local_root_body_pos(ref_yaw_quat, tar_key_body_pos)
        key_body_pos_diff = torch.mean(torch.abs(key_body_pos - tar_key_body_pos), dim=-1)
        # compute L1 error
        key_body_pos_err = torch.mean(key_body_pos_diff, dim=-1)
        return key_body_pos_err, key_body_pos_diff
    
    def _reward_tracking_feet_height(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        ref_feet_height = self._ref_body_pos[:, self.feet_indices, 2]
        feet_z = self.rigid_body_states[:, self.feet_indices, 2]
        
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z
        
        rew_pos = torch.abs(self.feet_height - ref_feet_height) < 0.05
        
        in_place_flag = torch.norm(self._ref_root_vel[:, :2], dim=1) < 0.1
        rew_pos[in_place_flag] = 0.
        self.feet_height *= ~contact
        return torch.sum(rew_pos, dim=1)
    
    def _reward_collision(self):
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_dof_torque_limits(self):
        out_of_limits = torch.sum((torch.abs(self.torques) / self.torque_limits - self.cfg.rewards.soft_torque_limit).clip(min=0), dim=1)
        return out_of_limits
    
    def _reward_feet_stumble(self):
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             4 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        return rew.float()
    
    def _reward_feet_contact_forces(self):
        rew = torch.norm(self.contact_forces[:, self.feet_indices, 2], dim=-1)
        rew[rew < self.cfg.rewards.max_contact_force] = 0
        rew[rew > self.cfg.rewards.max_contact_force] -= self.cfg.rewards.max_contact_force
        return rew
    
    def _reward_feet_height(self):
        # from OmniH2O
        feet_height = self.rigid_body_states[:,self.feet_indices, 2]
        dif = torch.abs(feet_height - self.cfg.rewards.feet_height_target)
        dif = torch.min(dif, dim=1).values # [num_env], # select the foot closer to target 
        return torch.clip(dif - 0.02, min=0.) # target - 0.02 ~ target + 0.02 is acceptable 
        
    def _reward_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_body_states[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)
    
    def _error_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_body_states[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)
    
    def _reward_lin_vel_z(self):
        rew = torch.square(self.base_lin_vel[:, 2])
        return rew
    
    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        rew = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return rew
    
    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        return torch.norm(self.last_actions - self.actions, dim=1)
    
    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_torque_penalty(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        tgt_air_time = self.cfg.rewards.feet_air_time_target
        air_time = (self.feet_air_time - tgt_air_time) * first_contact
        air_time = air_time.clamp(max=0.)
        self.feet_air_time *= ~self.contact_filt
        rew_airtime = air_time.sum(dim=1)
        rew_airtime *= torch.norm(self._ref_root_vel[:, :2], dim=1) > 0.05
        return rew_airtime
