import sys
import os

# Try to import isaacgym at the very start to avoid "PyTorch imported before isaacgym" error
try:
    import isaacgym
except ImportError:
    pass # If not available, we might not need it if not loading legacy configs

# import parkour_isaaclab # Removed to break dependency
from scripts.utils import load_local_cfg
from scripts.config_adapter import get_configs
from core.deployment_player import DeploymentPlayer

def main(args):
    """Play with RSL-RL agent."""
    
    # Load Configuration
    if args.params_path and os.path.exists(args.params_path):
        print(f"Loading YAML config from {args.params_path}")
        env_cfg = load_local_cfg(args.params_path, 'env')
        agent_cfg = load_local_cfg(args.params_path, 'agent')
        env_cfg.scene.num_envs = 1
    else:
        print("Loading Config from Python Classes (go2_parkour)...")
        env_cfg, agent_cfg = get_configs(use_camera=args.use_camera)
    
    # Use user provided logs path or construct a default relative one
    logs_path = args.logs_path if args.logs_path else os.path.join('logs', args.rl_lib, args.task, args.expid)
    
    model_file = args.load_model if args.load_model else os.path.join(logs_path, 'exported_deploy', 'policy.pt')

    player = DeploymentPlayer(
        env_cfg=env_cfg,
        agent_cfg = agent_cfg, 
        network_interface= args.interface,
        model_path = model_file,
        use_joystick = args.use_joystick
    )
    
    player.reset(maximum_iteration = args.n_eval)
    while player.alive():
        _, terminated, timeout, extras = player.play()
        if terminated or timeout:
           player.reset(extras = extras)
    print('Eval Done')
    
    sys.exit()

if __name__ == "__main__":
    import argparse
    # mp.set_start_method("spawn")
    parser = argparse.ArgumentParser(description='sim_2_sim')
    parser.add_argument("--rl_lib", type=str, default='rsl_rl')
    parser.add_argument("--task", type=str, default='unitree_go2_parkour')
    parser.add_argument("--expid", type=str, default='2025-09-03_12-07-56')
    parser.add_argument("--interface", type=str, default='lo')
    parser.add_argument("--use_joystick", action='store_true', default=False)
    parser.add_argument("--use_camera", action='store_true', default=False, help="Use depth camera for student policy")
    parser.add_argument("--n_eval", type=int, default=10)
    
    # New arguments for flexible paths
    parser.add_argument("--logs_path", type=str, default=None, help="Root path to logs/params if not using defaults")
    parser.add_argument("--params_path", type=str, default=None, help="Path to params directory containing env.yaml and agent.yaml")
    parser.add_argument("--load_model", type=str, default=None, help="Path to the specific .pt file to load")
    
    args = parser.parse_args()
    main(args)
