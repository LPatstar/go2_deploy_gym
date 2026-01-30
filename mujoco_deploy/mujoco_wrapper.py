import torch as th 
import numpy as np 
from mujoco_deploy.mujoco_sensors.mujoco_raycaster import MujocoRaycaster
from mujoco_deploy.mujoco_sensors.mujoco_contact_sensor import MujocoContactSensor
from mujoco_deploy.mujoco_sensors.mujoco_depth_camera import MujocoDepthCamera
from mujoco_deploy.utils_maths  import euler_xyz_from_quat, wrap_to_pi
from mujoco_deploy.mujoco_env import MujocoEnv
from mujoco_deploy.mujoco_sensors.mujoco_joystick_controller import MujocoJoystick
import copy, cv2, torchvision
from core.utils import isaac_to_mujoco, mujoco_to_isaac, stand_down_joint_pos
import math
from typing import Optional
import os
from datetime import datetime


from tqdm import tqdm
class MujocoWrapper():
    """
    Apply Reward, Oberservation, Termination in here 
    """
    def __init__(
        self, 
        env_cfg,
        agent_cfg,
        model_xml_path:str,
        use_camera: bool,
        use_joystick: bool = False,
        ):
        self._mujoco_env = MujocoEnv(env_cfg, model_xml_path, use_camera)
        self._use_camera = use_camera
        self._use_joystick = use_joystick
        self.device = self._mujoco_env.articulation.device
        self.common_step_counter = 0 
        self.decimation = env_cfg.decimation 
        self._history_length = env_cfg.observations.policy.extreme_parkour_observations.params.history_length
        self._clip = th.tensor([[*env_cfg.actions.joint_pos.clip['.*']]], device=self.device).repeat(
            1, self._mujoco_env.articulation.num_motor, 1
            )
        self._observation_clip = env_cfg.observations.policy.extreme_parkour_observations.clip[-1]
        self._agent_cfg = agent_cfg
        self._stand_down_joint_pos = th.tensor(stand_down_joint_pos,dtype=float).to('cuda:0')[mujoco_to_isaac]
        # print("_stand_down_joint_pos:", self._stand_down_joint_pos)
        self.estimator = None 
        self._init_sensors()
        self._init_commands()
        self._init_action_buffers()


    def _init_commands(self):
        if self._use_joystick:
             self._joystick = MujocoJoystick(self._mujoco_env.env_cfg, self.device)
             self._joystick.start_listening()
        else:
             print("Joystick disabled by argument.")
             self._joystick = None

    def _init_sensors(self):
        self._sensor_term = []
        if not self._use_camera:
            self._height_scanner = MujocoRaycaster(self._mujoco_env.env_cfg, 
                            self._mujoco_env.articulation, 
                            self._mujoco_env.model, 
                            self._mujoco_env.data)
            self._sensor_term.append(self._height_scanner)
        else: 
            self._depth_camera = MujocoDepthCamera(self._mujoco_env.env_cfg, 
                    self._mujoco_env.articulation.device,
                    self._mujoco_env.model, 
                    self._mujoco_env.data)
            self._sensor_term.append(self._depth_camera)
            
            # Load clip parameters from config or use defaults matching training
            self.near_clip = self._depth_camera.sensor_cfg.near_clip if hasattr(self._depth_camera.sensor_cfg, 'near_clip') else 0.15
            print("Near clip:", self.near_clip)
            self.far_clip = self._depth_camera.sensor_cfg.far_clip if hasattr(self._depth_camera.sensor_cfg, 'far_clip') else 2.0
            
            self.depth_buffer = th.zeros(self.num_envs,  
                                self._mujoco_env.env_cfg.observations.depth_camera.depth_cam.params.buffer_len, 
                                *self._mujoco_env.env_cfg.observations.depth_camera.depth_cam.params.resize).to(self.device)

        self._contact_sensor = MujocoContactSensor(self._mujoco_env.env_cfg, 
                            self._mujoco_env.articulation, 
                            self._mujoco_env.model, 
                            self._mujoco_env.data)
        
        self._sensor_term.append(self._contact_sensor)
        
    def _init_action_buffers(self):
        joint_pos_cfg = self._mujoco_env.env_cfg.actions.joint_pos
        self._action_history_length = joint_pos_cfg.history_length
        self._delay_update_global_steps = int(joint_pos_cfg.delay_update_global_steps)
        self._use_delay = joint_pos_cfg.use_delay
        self._action_delay_steps = joint_pos_cfg.action_delay_steps
        self.delay = th.tensor(0.0, device=self.device, dtype=th.float)
        self._action_history_buf = th.zeros(1, self._action_history_length, self._mujoco_env.articulation.num_motor, device=self.device, dtype=th.float)
        self._actions = th.zeros(1, self._mujoco_env.articulation.num_motor, device=self.device)
        self._processed_actions = th.zeros(1, self._mujoco_env.articulation.num_motor, device=self.device)
        self._obs_history_buffer = th.zeros(1, self._history_length, self._agent_cfg.estimator.num_prop, device=self.device)
        self.episode_length_buf = th.zeros(1, device=self.device, dtype=th.long)
        self._delta_yaw = th.zeros(1,1).to(self.device)
        self._delta_next_yaw = th.zeros(1,1).to(self.device)
        self._priv_explicit = th.zeros(1, self._agent_cfg.estimator.num_priv_explicit, device=self.device, dtype=th.float32)
        self._priv_latent = th.zeros(1, self._agent_cfg.estimator.num_priv_latent, device=self.device, dtype=th.float32)
        self._dummy_scan = th.zeros(1, self._agent_cfg.estimator.num_scan, device=self.device, dtype=th.float32)

    def _init_pose_stand_up(self):
        runing_time = 0.0
        with tqdm(total=3.0, desc="[INFO] Setting up the initial posture ...") as pbar:   
            while runing_time < 3.0:
                self.sensor_update()
                self.sensor_render()
                runing_time += self._mujoco_env.env_cfg.sim.dt
                pbar.update(min(self._mujoco_env.env_cfg.sim.dt, 3.0 - pbar.n)) 
                phase = th.tanh(th.tensor([runing_time / 1.2]).to('cuda:0'))
                cur_pose = phase * self._mujoco_env.default_joint_pose + (1-phase) * self._stand_down_joint_pos
                self._init_actions = self._mujoco_env.articulation.joint_stiffness*(cur_pose - self._mujoco_env.articulation.joint_pos) + \
                    self._mujoco_env.articulation.joint_dampings * (self._mujoco_env.articulation.control_joint_velocities - self._mujoco_env.articulation.joint_vel)
                processed_action_np = self._init_actions.detach().cpu().numpy()
                self._mujoco_env.step(processed_action_np)


    def _init_pose_stand_down(self):
        runing_time = 0.0
        init_pose = self._mujoco_env.articulation.joint_pos
        with tqdm(total=3.0, desc="[INFO] Setting up the stand down posture ...") as pbar:   
            while runing_time < 3.0:
                self.sensor_update()
                self.sensor_render()
                runing_time += 0.1
                pbar.update(min(0.1, 3.0 - pbar.n)) 
                phase = th.tanh(th.tensor([runing_time / 1.2]).to('cuda:0'))
                cur_pose = phase * self._stand_down_joint_pos + (1-phase) * init_pose
                self._init_actions = self._mujoco_env.articulation.joint_stiffness*(cur_pose - self._mujoco_env.articulation.joint_pos) + \
                    self._mujoco_env.articulation.joint_dampings * (self._mujoco_env.articulation.control_joint_velocities - self._mujoco_env.articulation.joint_vel)
                processed_action_np = self._init_actions.detach().cpu().numpy()
                self._mujoco_env.step(processed_action_np)

    def reset(self):
        self.common_step_counter = 0
        self._mujoco_env.reset()
        self.episode_length_buf = th.zeros(1, device=self.device, dtype=th.long)
        for i in range(100):
            self._mujoco_env.step()
        self._init_pose_stand_down()
        self._init_pose_stand_up()
        self.sensor_update()
        self.sensor_render()
        print('[INFO] Initial posture setting complete')

    def get_observations(self):
        self.roll, self.pitch, yaw = euler_xyz_from_quat(self._mujoco_env.articulation.root_quat_w)
        imu_obs = th.stack((wrap_to_pi(self.roll), wrap_to_pi(self.pitch)), dim=1).to(self.device)
        height_scan = th.clip(self._height_scanner.sensor_data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.sensor_data.ray_hits_w[..., 2] - 0.3, -1, 1).to(self.device) \
                        if not self._use_camera  else\
                        self._dummy_scan
        height_scan = th.clip(height_scan, -1, 1).to(self.device)
        env_idx_tensor = th.tensor([True]).to(dtype = th.bool, device=self.device)
        invert_env_idx_tensor = ~env_idx_tensor
        
        if self._joystick is not None:
             commands = self._joystick.velocity_cmd
        else:
             # Default command if no joystick: 0.5 m/s forward
             commands = th.zeros((1, 3), device=self.device)
             commands[:, 0] = 0.5
             commands[:, 2] = 0


        self._delta_next_yaw[:] = self._delta_yaw[:] = -wrap_to_pi(yaw)[:,None]
        # obs_buf = th.cat((
        #                     self._mujoco_env.articulation.root_ang_vel_b* 0.25,   #[1,4]
        #                     imu_obs,    #[1,2]
        #                     0*self._delta_yaw, 
        #                     self._delta_yaw,
        #                     self._delta_next_yaw,
        #                     0*commands[:, 0:2],                   # 2
        #                     commands[:, 0:1],                     # 1
        #                     env_idx_tensor.float()[:, None],
        #                     invert_env_idx_tensor.float()[:, None],
        #                     self.reindex_to_MJ(self._mujoco_env.articulation.joint_pos - self._mujoco_env.default_joint_pose),
        #                     self.reindex_to_MJ(self._mujoco_env.articulation.joint_vel* 0.05),
        #                     self.reindex_to_MJ(self._action_history_buf[:, -1]),
        #                     self.reindex_feet(self._get_contact_fill()),
        #                     ),dim=-1)
        obs_buf = th.cat((
                            self._mujoco_env.articulation.root_ang_vel_b* 0.25,   #[1,4]
                            imu_obs,    #[1,2]
                            wrap_to_pi(yaw)[:,None],                    # 1
                            commands[:, 2:3],                     # 1
                            0*self._delta_next_yaw,               # 1
                            0*commands[:, 0:2],                   # 2
                            commands[:, 0:1],                     # 1
                            env_idx_tensor.float()[:, None],
                            invert_env_idx_tensor.float()[:, None],
                            self.reindex_to_MJ(self._mujoco_env.articulation.joint_pos - self._mujoco_env.default_joint_pose),
                            self.reindex_to_MJ(self._mujoco_env.articulation.joint_vel* 0.05),
                            self.reindex_to_MJ(self._action_history_buf[:, -1]),
                            self.reindex_feet(self._get_contact_fill()),
                            ),dim=-1)
        # print("self._mujoco_env.articulation.root_ang_vel_b:", self._mujoco_env.articulation.root_ang_vel_b)
        # print("self._delta_yaw:", self._delta_yaw)
        # print("commands[:, 0:1]", commands[:, 0:1])
        # print("obs_buf:", obs_buf)

        obs_buf_for_history = obs_buf.clone()
        obs_buf_for_history[:, 6:8] = 0
        current_history = th.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            th.stack([obs_buf_for_history] * self._history_length, dim=1),
            th.cat([
                self._obs_history_buffer[:, 1:],
                obs_buf_for_history.unsqueeze(1)
            ], dim=1)
        )

        self._priv_explicit[:] = self.estimator(obs_buf.float())

        observations = th.cat([obs_buf, #53 
                                height_scan, #132 
                                self._priv_explicit, # 9
                                self._priv_latent, #29
                                self._obs_history_buffer.view(1, -1)
                                ], dim=-1)
        
        self._obs_history_buffer = current_history

        if self._use_camera:
            depth_image = self._get_depth_image()
        else:
            depth_image = th.zeros(1, device=self.device)

        observations = th.clip(observations, min = -self._observation_clip, max = self._observation_clip)
        extras = {'observations':{"policy":observations.to(th.float),
                                  'depth_camera':depth_image.to(th.float)}}
        return observations.to(th.float), extras
    
    def _process_depth_image(self, depth_image):
        # Align pipeline sequence with training (LeggedRobot.process_depth_image):
        # 1. Crop
        # 2. Clip & Normalize
        # 3. Resize (Use AdaptiveAvgPool2d to match training's resize2d)
        # 4. Gaussian Blur
        
        depth_image = self._crop_depth_image(depth_image)
        depth_image = self._normalize_depth_image(depth_image)
        
        # Resize using AdaptiveAvgPool2d (H, W) -> (1, H, W) -> (1, 58, 87) -> (58, 87)
        target_size = self._mujoco_env.env_cfg.observations.depth_camera.depth_cam.params.resize # [58, 87]
        depth_image = th.nn.functional.adaptive_avg_pool2d(depth_image.unsqueeze(0), target_size).squeeze(0)
        
        depth_image = self._gaussian_blur(depth_image)
        return depth_image
        
    def _gaussian_blur(self, depth_image):
        # Aligned with training: kernel=3, sigma=1.0, replicate padding
        k = 3
        sigma = 1.0
        
        if not hasattr(self, "_gaussian_kernel") or self._gaussian_kernel.device != depth_image.device:
            x = th.arange(k, dtype=th.float32, device=depth_image.device) - k // 2
            gauss_1d = th.exp(-x ** 2 / (2 * sigma ** 2))
            gauss_1d = gauss_1d / gauss_1d.sum()
            gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
            self._gaussian_kernel = gauss_2d.view(1, 1, k, k)
            
        # Input shape [H, W] -> [1, 1, H, W] for conv2d
        depth_input = depth_image.unsqueeze(0).unsqueeze(0)
        pad = k // 2
        
        # Manual pad with replicate to match training
        depth_padded = th.nn.functional.pad(depth_input, (pad, pad, pad, pad), mode='replicate')
        
        depth_blurred = th.nn.functional.conv2d(depth_padded, self._gaussian_kernel, padding=0)
        
        return depth_blurred.squeeze()
    
    def _crop_depth_image(self, depth_image):
        # Align with training crop: LeggedRobot.crop_depth_image (batch_depth_images[..., 2:-2, 15:-2])
        # Training: 2 from top/bottom, 15 from left, 2 from right
        return depth_image[2:-2, 15:-2]
    
    def _normalize_depth_image(self, depth_image):
        # Align with training logic: LeggedRobot.normalize_depth_image
        # 1. Clip to valid range [near, far]
        depth_image = th.clip(depth_image, min=self.near_clip, max=self.far_clip)
        
        # 2. Normalize: (val - near) / (far - near) - 0.5 -> [-0.5, 0.5]
        depth_image = (depth_image - self.near_clip) / (self.far_clip - self.near_clip) - 0.5
        return depth_image

    def _get_depth_image(
        self,
        is_reset: bool = False 
        ):
        depth_image = self._depth_camera.sensor_data.output["distance_to_camera"].squeeze(-1)[:]
        processed_image = self._process_depth_image(depth_image)
        if is_reset:
            self.depth_buffer[0] = th.stack([processed_image]* 2, dim=0)
        if self.common_step_counter % 5 ==0:
            self.depth_buffer[0] = th.cat([self.depth_buffer[0, 1:], 
                                    processed_image.to(self.device).unsqueeze(0)], dim=0)
            # Add 0.5 for visualization because data is normalized to [-0.5, 0.5]
            # Without this, all values < 0 (i.e. distance < mid-range) appear black in imshow
            cv2.imshow('processed_image', processed_image.detach().cpu().numpy() + 0.5)
            cv2.waitKey(1)
        return self.depth_buffer[:, -2].to(self.device)
    
    def _get_contact_fill(
        self,
        ):
        foot_ids = self._contact_sensor.sensor_data.foot_ids
        contact_forces = self._contact_sensor.sensor_data.net_forces_w_history[:, 0, foot_ids] #(N, 4, 3)
        previous_contact_forces = self._contact_sensor.sensor_data.net_forces_w_history[:, -1, foot_ids] # N, 4, 3
        contact = th.norm(contact_forces, dim=-1) > 2.
        last_contacts = th.norm(previous_contact_forces, dim=-1) > 2.
        contact_filt = th.logical_or(contact, last_contacts) 
        return (contact_filt.float()-0.5).to(self.device)

    def _apply_action(self):
        """Applies the current action to the robot."""
        """ class LocomotionEnv(DirectRLEnv)"""
        """ 1. make processed actions"""
        # print("Processed actions:", self._processed_actions[0])
        error_pos = self._processed_actions - self._mujoco_env.articulation.joint_pos
        error_vel = self._mujoco_env.articulation.control_joint_velocities - self._mujoco_env.articulation.joint_vel # type: ignore
        self.computed_effort = self._mujoco_env.articulation.joint_stiffness * error_pos \
                               + self._mujoco_env.articulation.joint_dampings * error_vel 
        

        # """DCMotor _clip_effort"""
        max_effort = self._mujoco_env.articulation.saturation_effort \
            * (1.0 - self._mujoco_env.articulation.joint_vel/ self._mujoco_env.articulation.velocity_limit)
        max_effort = th.clip(max_effort, min=self._mujoco_env.articulation.zeros_effort, max=self._mujoco_env.articulation.effort_limit)
        min_effort = self._mujoco_env.articulation.saturation_effort \
            * (-1.0 - self._mujoco_env.articulation.joint_vel / self._mujoco_env.articulation.velocity_limit)
        min_effort = th.clip(min_effort, min=-self._mujoco_env.articulation.effort_limit, max=self._mujoco_env.articulation.zeros_effort)
        self._applied_effort = th.clip(self.computed_effort, min=min_effort, max=max_effort)
        # print("Applied effort:", self._applied_effort[0])

        # # Record efforts
        # effort_np = self._applied_effort.detach().cpu().numpy()
        # with open(self.effort_file, "a") as f:
        #     np.savetxt(f, effort_np, fmt="%.6f", delimiter=" ")


    def _process_actions(self):
        if self.common_step_counter % self._delay_update_global_steps == 0:
            if len(self._action_delay_steps) != 0:
                self.delay = th.tensor(self._action_delay_steps.pop(0), device=self.device, dtype=th.float)
        self._action_history_buf = th.cat([self._action_history_buf[:, 1:].clone(), self._actions[:, None, :].clone()], dim=1)
        indices = -1 - self.delay
        if self._use_delay:
            self._actions = self._action_history_buf[:, indices.long()]

        if self._mujoco_env.env_cfg.actions.joint_pos.clip is not None:
            self._actions = th.clamp(
                self._actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        if self._mujoco_env.env_cfg.actions.joint_pos.use_default_offset:
            """ JointPositionAction """
            """ process_actions: self._raw_actions * self._scale + self._offset """
            self._processed_actions = self._actions * self._mujoco_env.env_cfg.actions.joint_pos.scale + self._mujoco_env.default_joint_pose

        else:
            self._processed_actions = self._actions * self._mujoco_env.env_cfg.actions.joint_pos.scale 
    
    def reindex_feet(self, vec):
        return vec[:, [1, 0, 3, 2]]

    def reindex_to_ILAB(self, vec):
        return vec[:, mujoco_to_isaac]
    
    def reindex_to_MJ(self, vec):
        return vec[:, isaac_to_mujoco]
    
    def step(self, actions: Optional[th.Tensor] = None):
        ### actions is isaacsim based
        self._actions = self.reindex_to_ILAB(actions.clone())
        # print("action:", self._actions)    
        self._process_actions()
        self.common_step_counter += 1
        self.episode_length_buf += 1  # step in current episode (per env)

        for _ in range(self.decimation):
            self._apply_action()
            applied_effort_np = self._applied_effort.detach().cpu().numpy()
            self._mujoco_env.step(applied_effort_np)
            self.sensor_render()
            self.sensor_update()
        obs, extras = self.get_observations()
        termination = self._termination()
        max_episode_length = math.ceil(self._mujoco_env.env_cfg.episode_length_s/(self._mujoco_env.env_cfg.sim.dt  * self.decimation))
        time_out_buf = self.episode_length_buf >= max_episode_length
        return obs , termination , time_out_buf, extras
    
    def _termination(self):
        reset_buf = th.zeros((1, ), dtype=th.bool, device=self._mujoco_env.articulation.device)
        roll_cutoff = th.abs(wrap_to_pi(self.roll)) > 1.5
        pitch_cutoff = th.abs(wrap_to_pi(self.pitch)) > 1.5
        height_cutoff = self._mujoco_env.articulation.root_state_w[:, 2] < -0.25
        reset_buf |= roll_cutoff
        reset_buf |= pitch_cutoff
        reset_buf |= height_cutoff  
        return reset_buf

    def sensor_render(self):
        for sensor_term in self._sensor_term:
            if hasattr(sensor_term, 'render'): 
                sensor_term.render(self._mujoco_env.viewer)
            else:
                continue

    def sensor_update(self):
        for sensor_term in self._sensor_term:
            sensor_term.update(self._mujoco_env.env_cfg.sim.dt)
        
    @property
    def sim(self):
        return self._mujoco_env

    @property
    def num_actions(self):
        return self._mujoco_env.articulation.num_motor

    @property
    def num_envs(self):
        return 1

    def close(self):
        self._mujoco_env.close()

