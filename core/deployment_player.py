import torch as th
from mujoco_deploy.mujoco_wrapper import MujocoWrapper
import os
import core
from core.my_modules.actor_critic import Actor, get_activation
from core.my_modules.depth_backbone import RecurrentDepthBackbone, DepthOnlyFCBackbone58x87
from legged_gym.utils import task_registry # Use task_registry from installed legged_gym
from typing import Dict, Optional
import torch.nn as nn
import numpy as np
from datetime import datetime

class EmpiricalNormalization(nn.Module):
    def __init__(self, shape, update=True):
        super(EmpiricalNormalization, self).__init__()
        self.shape = shape
        self.mean = nn.Parameter(th.zeros(shape), requires_grad=False)
        self.std = nn.Parameter(th.ones(shape), requires_grad=False)
        self.update = update

    def forward(self, x):
        return (x - self.mean) / (self.std + 1e-8)

class DeploymentPlayer:
    def __init__(
        self,
        env_cfg,
        agent_cfg, 
        network_interface,
        model_path=None,
        use_joystick=False, 
    ):
        use_camera = False
        try: 
            if env_cfg.scene.depth_camera and not isinstance(env_cfg.scene.depth_camera, bool):
                use_camera = True 
        except: 
            pass

        if network_interface.lower() =='lo':
            self.env = MujocoWrapper(env_cfg, agent_cfg, os.path.join(core.__path__[0],'go2/scene_terrain.xml'), use_camera, use_joystick)
        self._use_camera = use_camera
        
        # Initialize Normalizer (default Identity)
        # Assuming observation shape from env_cfg or hardcode if necessary?
        # Obs dimension is usually calculated in env.
        # But here we don't have easy access to obs dict size until we run?
        # Let's instantiate it with a placeholder or wait until first obs?
        # Better: Load it, and valid shape check will happen or broadcast.
        self.obs_normalizer = None

        # Load policy from specified path or default locations
        if model_path is not None and os.path.exists(model_path):
            print(f"Loading model from: {model_path}")

            # Reconstruct configs to use task_registry like in play.py
            # 假设 model_path 上级目录类似 .../parkour_new/021-g2-teacher/model_1200.pt
            # 我们需要构建 log_root (run_dir)
            log_root = os.path.dirname(model_path) 
            # 有可能 model_path 指向了具体的文件，而 task_registry 需要的是 experiment name 和 log_dir
            # 这里我们尝试从 model_path 解析出 run_name
            # 例如: .../logs/parkour_new/EXP_NAME/model_xxx.pt
            # 我们需要的是 log_dir = .../logs/parkour_new, run_name = EXP_NAME
            
            # 由于参数只传入了 env_cfg, agent_cfg，这里我们直接从文件加载
            # 但为了完全复用 play.py 的逻辑，我们需要调用 make_alg_runner
            
            # 1. 构造一个临时的 env 对象 (MujocoWrapper)
            # 注意: make_alg_runner 需要一个 env 对象来初始化 runner
            # 但是 MujocoWrapper 可能不完全兼容 IsaacGym 的接口，特别是 get_observations 的返回
            # play.py 中使用的是 LeggedRobot
            
            # 这里我们只为了加载权重，所以可以使用原来的加载方式，但是用 task_registry 来获取 runner 类
            
            # 使用与 play.py 一致的逻辑:
            # train_cfg.runner.resume = True
            # ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(...)
            
            # 为了适配 make_alg_runner，我们需要 task_name。
            # 这里假设 task_name 可以从 env_cfg 中推断，或者由于是 deploy，我们可能已知是 "go2" 
            # 但 env_cfg 已经是解析后的对象。
            
            # 简化方案：直接使用 deployment 原有的加载逻辑，但确保 Normalizer 的加载正确
            # 或者，如果一定要完全一致：
            
            import torch
            
            # 加载 checkpoint
            loaded_dict = torch.load(model_path, map_location=self.env.device)
            
            # 使用 task_registry 获取 train_cfg (如果能知道 task name)
            # 这里我们假设 task name 是 'go2'，你可以根据实际情况修改
            # 或者尝试从 checkpoint 中的 config 恢复？通常 checkpoint 不带完整 config。
            # 我们使用当前传入的 agent_cfg 构建 runner。
            
            # 根据 play.py:
            # ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(...)
            # policy = ppo_runner.get_inference_policy(device=env.device)
            # estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
            # depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)
            
            # 关键在于 ppo_runner 的实例化。
            # 我们可以直接实例化 OnPolicyRunner (来自 my_runner) 并 load
            # 因为 my_runner.on_policy_runner 已经被复制过来了
            
            from core.my_runner.on_policy_runner import OnPolicyRunner
            
            # 我们需要重构 train_cfg 字典，因为 OnPolicyRunner 需要它
            # agent_cfg 看起来是 train_cfg 的一部分或转换版
            # 这里直接使用 loaded_dict 来加载到 runner 
            
            # 实例化 Runner (需要 env，alg_cfg, log_dir, device)
            # 注意：这里的 self.env 是 MujocoWrapper
            # OnPolicyRunner.__init__ 会用到 env.num_envs, env.num_obs 等
            
            # 为了避免复杂的 Runner 初始化依赖，我们手动提取 Runner 中的加载逻辑
            # 参考 OnPolicyRunner.load() 和 get_inference_policy()
            
            print("Loading via manual reconstruction (mimicking Runner)...")
            
            # 1. 确定算法类 (PPO) 并实例化
            # 这需要完整的 train_cfg。部署时通常只有 agent_cfg (部分参数)
            # 如果没有完整的 train_cfg，直接实例化 PPO 会缺参数。
            # 但是，我们可以直接加载 model_state_dict 到对应的 ActorCritic 结构中，
            # 只要结构定义一致 (core.my_modules vs legged_gym.modules)
            
            # 由于你确认了 my_modules 已经复制且一致，我们继续用 deployment_player 原有的手动实例化方式，
            # 但是要 **修正 Normalizer 的加载逻辑**，使其和 OnPolicyRunner 一致。
            
            # --- 修正后的加载逻辑 ---
            
            # A. 加载字典
            raw_loaded_dict = th.load(model_path, map_location=self.env.device)
            if 'model_state_dict' in raw_loaded_dict:
                model_dict = raw_loaded_dict['model_state_dict']
            else:
                model_dict = raw_loaded_dict

            # Handle key prefixes (stripping 'actor.' if present)
            new_model_dict = {}
            for k, v in model_dict.items():
                if k.startswith('actor.'):
                    new_model_dict[k[6:]] = v # Remove 'actor.'
                else:
                    new_model_dict[k] = v
            model_dict = new_model_dict

            # B. 实例化 Actor (Policy)
            # 确保参数名与训练时一致
            est_cfg = agent_cfg['estimator']
            activation_name = est_cfg.activation
            activation_module = get_activation(activation_name)
            
            # 使用 core.my_modules.actor_critic.Actor (你确认这是训练用的代码)
            self.policy = Actor(
                num_prop=est_cfg.num_prop,
                num_scan=est_cfg.num_scan,
                num_actions=12,
                scan_encoder_dims=est_cfg.scan_encoder_dims,
                actor_hidden_dims=est_cfg.actor_hidden_dims,
                priv_encoder_dims=est_cfg.priv_encoder_dims,
                num_priv_latent=est_cfg.num_priv_latent,
                num_priv_explicit=est_cfg.num_priv_explicit,
                num_hist=est_cfg.num_hist,
                activation=activation_module,
            ).to(self.env.device)
            
            # 加载权重
            missing_keys, unexpected_keys = self.policy.load_state_dict(model_dict, strict=False)
            print("Policy loaded.")
            if missing_keys:
                print(f"  Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"  Unexpected keys: {unexpected_keys}")
            
            self.policy.eval()

            # C. 加载 Depth Encoder (如果使用相机)
            if self._use_camera:
                print("Loading Depth Encoder...")
                # 1. 初始化 Backbone
                # DepthOnlyFCBackbone58x87(prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1)
                # hidden_state_dim 在原始实现中其实未被使用，但需要传递占位符
                
                scan_dot_dim = est_cfg.scan_encoder_dims[-1] if hasattr(est_cfg, 'scan_encoder_dims') else 32
                
                depth_backbone = DepthOnlyFCBackbone58x87(
                    prop_dim=est_cfg.num_prop, 
                    scandots_output_dim=scan_dot_dim, 
                    hidden_state_dim=512, # Dummy value, not used in Original implementation
                )
                
                # 2. 初始化 Recurrent Encoder (包装器)
                # RecurrentDepthBackbone(base_backbone, env_cfg)
                self.depth_encoder = RecurrentDepthBackbone(depth_backbone, env_cfg).to(self.env.device)
                
                depth_dict = {}
                if 'depth_encoder_state_dict' in raw_loaded_dict:
                    depth_dict = raw_loaded_dict['depth_encoder_state_dict']
                else:
                    # Look in model_dict for 'depth_encoder.' or 'alg.depth_encoder.'
                    # OnPolicyRunner model_state_dict often puts it in module directly or via wrapper?
                    # Actually OnPolicyRunner save() puts it in 'depth_encoder_state_dict' explicitly.
                    # But if we are loading some other checkpoint format:
                    full_dict = raw_loaded_dict.get('model_state_dict', raw_loaded_dict)
                    for k, v in full_dict.items():
                        if 'depth_encoder.' in k:
                            depth_dict[k.replace('depth_encoder.', '')] = v
                
                if depth_dict:
                     self.depth_encoder.load_state_dict(depth_dict, strict=False)
                     self.depth_encoder.eval()
                     print("Depth Encoder loaded using RecurrentDepthBackbone wrapper.")
                else:
                     print("WARNING: Depth Encoder requested but not found in checkpoint!")

            
            # D. 加载 Normalizer (关键修改)
            # OnPolicyRunner 保存时，normalizer 通常在 alg.actor_critic.obs_normalizer 或 alg.obs_normalizer
            # 这里的 model_dict 通常是 actor_critic 的 state_dict
            
            print("Loading Normalizer...")
            self.obs_normalizer = EmpiricalNormalization(shape=[1])
            
            # 尝试从 model_dict 中直接找 obs_normalizer (如果它作为子模块注册在 AC 中)
            # 比如 key 可能是 "obs_std_normalizer.mean" 或 "obs_normalizer.mean"
            # 搜索所有包含 "running_mean" 的键
            # USE raw_loaded_dict (full checkpoint) first if possible, but also check model_dict (which is state dict)
            # Note: model_dict is now stripped of 'actor.'
            
            norm_keys = [k for k in raw_loaded_dict.get('model_state_dict', raw_loaded_dict).keys() if "running_mean" in k]
            print(f"Found running_mean keys in state_dict: {norm_keys}")
            
            obs_norm_key = None
            # 优先匹配含有 'obs' 的名字
            for k in norm_keys:
                if 'obs' in k:
                    obs_norm_key = k
                    break
            # 如果没找到带 obs 的，但确实有 mean (比如可能是只有唯一的 normalizer)，取第一个
            if obs_norm_key is None and len(norm_keys) > 0:
                obs_norm_key = norm_keys[0]
            
            if obs_norm_key:
                base_key = obs_norm_key.replace("running_mean", "")
                print(f"Detected normalizer prefix: {base_key}")
                full_dict = raw_loaded_dict.get('model_state_dict', raw_loaded_dict)
                self.obs_normalizer.mean.data = full_dict[base_key + "running_mean"]
                self.obs_normalizer.std.data = th.sqrt(full_dict[base_key + "running_var"] + 1e-8)
                print(f"Loaded Normalizer: Mean[0]=%.4f, Std[0]=%.4f" % (self.obs_normalizer.mean.data[0], self.obs_normalizer.std.data[0]))
            else:
                # 检查是否在顶层 raw_loaded_dict 中 (RSL_RL 风格)
                if 'obs_norm_state_dict' in raw_loaded_dict:
                     print("Loading from obs_norm_state_dict...")
                     obs_norm_algo_dict = raw_loaded_dict['obs_norm_state_dict']
                     # Usually structure is just mean/var/count
                     if 'mean' in obs_norm_algo_dict:
                        self.obs_normalizer.mean.data = obs_norm_algo_dict['mean']
                        self.obs_normalizer.std.data = th.sqrt(obs_norm_algo_dict['var'] + 1e-8)
                     else:
                        # Sometimes it's the module state dict
                        self.obs_normalizer.load_state_dict(obs_norm_algo_dict)
                else:
                     print("WARNING: Could not find Normalizer parameters! Running with Identity Normalization.")

            # D. 加载 Depth Encoder (如果需要)
            if self._use_camera:
                print("Instantiating Depth Encoder...")
                backbone = DepthOnlyFCBackbone58x87(
                        est_cfg.num_prop,
                        est_cfg.scan_encoder_dims[-1], 
                        est_cfg.actor_hidden_dims[0],
                        "elu", 
                        1
                )
                self.depth_encoder = RecurrentDepthBackbone(backbone, env_cfg).to(self.env.device)
                
                # 尝试从 model_dict 加载 (如果是联合训练)
                # 检查是否有 depth_encoder 前缀
                depth_prefix = "depth_encoder."
                depth_keys = [k for k in model_dict.keys() if k.startswith(depth_prefix)]
                
                if len(depth_keys) > 0:
                     self.depth_encoder.load_state_dict(model_dict, strict=False) # 前缀通常已经在 model_dict 里了？
                     # 如果 model_dict 的 key 是 "depth_encoder.weight"，load_state_dict(..., strict=False) 会匹配上属性名吗？
                     # 通常需要去掉前缀如果 depth_encoder 是独立实例
                     # 但这里 depth_encoder 不是 self.policy 的子模块，所以我们需要构建 subset dict 并 strip prefix
                     
                     depth_subset = {k.replace(depth_prefix, ""): model_dict[k] for k in depth_keys}
                     self.depth_encoder.load_state_dict(depth_subset)
                     print("Loaded Depth Encoder from model_state_dict.")
                
                # 检查是否有 depth_encoder_state_dict (RSL_RL 风格)
                elif 'depth_encoder_state_dict' in raw_loaded_dict:
                    self.depth_encoder.load_state_dict(raw_loaded_dict['depth_encoder_state_dict'])
                    print("Loaded Depth Encoder from depth_encoder_state_dict.")
                
                else:
                    # 尝试加载外部文件
                     depth_path = os.path.join(os.path.dirname(model_path), 'depth_latest.pt')
                     if os.path.exists(depth_path):
                         print(f"Loading external depth file: {depth_path}")
                         depth_sd = th.load(depth_path, map_location=self.env.device)
                         if 'model_state_dict' in depth_sd: depth_sd = depth_sd['model_state_dict']
                         # 移除可能的前缀
                         depth_sd = {k.replace('depth_encoder.', ''): v for k, v in depth_sd.items()}
                         self.depth_encoder.load_state_dict(depth_sd, strict=False)
                     else:
                         print("WARNING: Depth Encoder weights not found!")
                
                self.depth_encoder.eval()

        else:
            # Fallback for old hardcoded paths (unchanged)
            self.policy = th.jit.load(os.path.join(logs_path,'exported_teacher','policy.pt'), map_location=self.env.device)
            self.policy.eval()
            self.depth_encoder = None

        self._clip_actions = agent_cfg['clip_actions']
        estimator_paras = agent_cfg["estimator"]
        self.num_prop = estimator_paras["num_prop"]
        self.num_scan = estimator_paras["num_scan"]
        self.num_priv_explicit = estimator_paras["num_priv_explicit"]
        self.history_len = 10 
        self.cnt = 0 
        self._call_cnt = 0
        self._maximum_iteration = float('inf')
        
        # # Setup action recording
        # self.action_dir = os.path.join(os.getcwd(), "actions")
        # os.makedirs(self.action_dir, exist_ok=True)
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # self.action_file = os.path.join(self.action_dir, f"actions_{timestamp}.txt")
        # print(f"Actions will be recorded to: {self.action_file}")

        print(self.policy)
        
    def play(self):
        """Advances the environment one time step after generating observations"""
        obs, extras = self.env.get_observations()
        with th.inference_mode():
            # Apply Normalization
            if self.obs_normalizer is not None:
                # Ensure device match
                if self.obs_normalizer.mean.device != obs.device:
                    self.obs_normalizer.to(obs.device)
                obs = self.obs_normalizer(obs)

            if not self._use_camera:
                actions = self.policy(obs , hist_encoding=True)
            else:
                if self.env.common_step_counter %5 == 0:
                    depth_image = extras["observations"]['depth_camera']
                    proprioception = obs[:, :self.num_prop].clone()
                    proprioception[:, 6:8] = 0
                    depth_latent_and_yaw = self.depth_encoder(depth_image , proprioception )
                    self.depth_latent = depth_latent_and_yaw[:, :-2]
                    self.yaw = depth_latent_and_yaw[:, -2:]
                
                # In training (on_policy_runner.py), overwriting obs[:, 6:8] with inferred yaw is COMMENTED OUT.
                # So we must NOT overwrite it here either. The policy expects the original command/yaw data in these slots.
                # obs[:, 6:8] = 1.5*self.yaw <--- REMOVED
                
                actions = self.policy(obs , hist_encoding=True, scandots_latent=self.depth_latent)
        
        # # Record actions
        # act_np = actions.detach().cpu().numpy()
        # with open(self.action_file, "a") as f:
        #     np.savetxt(f, act_np, fmt="%.6f", delimiter=" ")

        if self._clip_actions is not None:
            actions = th.clamp(actions, -self._clip_actions, self._clip_actions)
        
        obs, terminated, timeout, extras = self.env.step(actions)  # For HW, this internally just does forward

        self.cnt += 1
        return obs, terminated, timeout, extras
    
    def reset(self, maximum_iteration: Optional[int] = None, extras: Optional[Dict[str, str]] = None):
        self._call_cnt +=1 
        if type(maximum_iteration) == int:
            self.maximum_iteration = maximum_iteration
        if self.alive():
            self.env.reset() 
            print('[Current eval iter]: ', self._call_cnt, '[Left]: ', self.maximum_iteration-self._call_cnt)

    def alive(self):
        if self._call_cnt <= self.maximum_iteration:
            return True
        else:
            self.env.close()
            return False 
        
