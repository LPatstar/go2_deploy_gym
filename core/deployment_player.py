import torch as th
from mujoco_deploy.mujoco_wrapper import MujocoWrapper
import os
import core
from core.my_modules.actor_critic import Actor, get_activation
from core.my_modules.estimator import Estimator
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

        self.obs_normalizer = None

        # Load policy from specified path or default locations
        if model_path is not None and os.path.exists(model_path):
            print(f"Loading model from: {model_path}")

            log_root = os.path.dirname(model_path) 
            # 有可能 model_path 指向了具体的文件，而 task_registry 需要的是 experiment name 和 log_dir
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
            
            # 简化方案：直接使用 deployment 原有的加载逻辑
            
            import torch
            
            # 加载 checkpoint
            loaded_dict = torch.load(model_path, map_location=self.env.device)
            
            # 使用 task_registry 获取 train_cfg (如果能知道 task name)
            # 这里我们假设 task name 是 'go2'，你可以根据实际情况修改
            # 或者尝试从 checkpoint 中的 config 恢复？通常 checkpoint 不带完整 config。
            # 我们使用当前传入的 agent_cfg 构建 runner。
            
            # 关键在于 ppo_runner 的实例化。
            # 我们可以直接实例化 OnPolicyRunner (来自 my_runner) 并 load
            # 因为 my_runner.on_policy_runner 已经被复制过来了
            
            from core.my_runner.on_policy_runner import OnPolicyRunner
            
            # 为了避免复杂的 Runner 初始化依赖，我们手动提取 Runner 中的加载逻辑
            # 参考 OnPolicyRunner.load() 和 get_inference_policy()
            
            print("Loading via manual reconstruction (mimicking Runner)...")
            
            # 由于确认了 my_modules 已经复制且一致，继续用 deployment_player 原有的手动实例化方式，

            # 1. 加载原始字典
            raw_loaded_dict = th.load(model_path, map_location=self.env.device)
            est_cfg = agent_cfg['estimator']
            activation_name = est_cfg.activation
            activation_module = get_activation(activation_name)

            # 2. 加载Actor
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

            # A. 处理 State Dict
            if self._use_camera and 'depth_actor_state_dict' in raw_loaded_dict:
                model_dict = raw_loaded_dict['depth_actor_state_dict']
                print("Loading parameters from depth_actor_state_dict...")
            elif 'model_state_dict' in raw_loaded_dict:
                model_dict = raw_loaded_dict['model_state_dict']
                # 处理 Key 前缀（清洗 'actor.' 前缀）
                new_model_dict = {}
                for k, v in model_dict.items():
                    # 检查是否以 'actor.' 开头
                    if k.startswith('actor.'):
                        new_model_dict[k[6:]] = v  # 移除前 6 个字符 'actor.'
                    else:
                        new_model_dict[k] = v
                model_dict = new_model_dict
                print("depth_actor_state_dict not found, using model_state_dict instead.")
            else:
                model_dict = raw_loaded_dict
                print("Target key not found, loading raw dictionary.")

            # B. 实例化 Actor (Policy)
            missing_keys, unexpected_keys = self.policy.load_state_dict(model_dict, strict=False)
            print("Policy loaded.")
            if missing_keys:
                print(f"  Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"  Unexpected keys: {unexpected_keys}")
            
            self.policy.eval()

            # 3. Load Estimator
            print("Loading Estimator...")
            self.estimator = Estimator(
                input_dim=est_cfg.num_prop, 
                output_dim=est_cfg.num_priv_explicit,
                hidden_dims=est_cfg.estimator_hidden_dims,
                activation=est_cfg.activation
            ).to(self.env.device)
            
            if 'estimator_state_dict' in raw_loaded_dict:
                self.estimator.load_state_dict(raw_loaded_dict['estimator_state_dict'])
                print("Estimator loaded from estimator_state_dict.")
            else:
                print("WARNING: estimator_state_dict not found in checkpoint! Using random initialization.")
            
            self.estimator.eval()
            self.env.estimator = self.estimator

            # 4. 加载 Depth Encoder
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
                
                if 'depth_encoder_state_dict' in raw_loaded_dict:
                    self.depth_encoder.load_state_dict(raw_loaded_dict['depth_encoder_state_dict'])
                    print("Loaded Depth Encoder from depth_encoder_state_dict.")
                else:
                    print("WARNING: Depth Encoder weights not found!")
                
                self.depth_encoder.eval()

        self._clip_actions = agent_cfg['clip_actions']
        estimator_paras = agent_cfg["estimator"]
        self.num_prop = estimator_paras["num_prop"]
        self.num_scan = estimator_paras["num_scan"]
        self.num_priv_explicit = estimator_paras["num_priv_explicit"]
        self.history_len = 10 
        self.cnt = 0 
        self._call_cnt = 0
        self._maximum_iteration = float('inf')

        print(self.policy)
        if self._use_camera:
            print(self.depth_encoder)
        
    def play(self):
        """Advances the environment one time step after generating observations"""
        obs, extras = self.env.get_observations()
        with th.inference_mode():

            if not self._use_camera:
                actions = self.policy(obs, hist_encoding=True)
            else:
                depth_image = extras["observations"]['depth_camera']
                proprioception = obs[:, :self.num_prop].clone()
                proprioception[:, 6:8] = 0
                depth_latent_and_yaw = self.depth_encoder(depth_image, proprioception)
                self.depth_latent = depth_latent_and_yaw[:, :-2]
                self.yaw = depth_latent_and_yaw[:, -2:]

                actions = self.policy(obs, hist_encoding=True, scandots_latent=self.depth_latent)
        

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
        
