
import sys
import os
from easydict import EasyDict

# Add the go2_parkour directory to sys.path to allow importing legged_gym
# Assuming this script is run from go2_parkour_deploy/scripts/.. or similar 
# We need to find the root of go2_parkour relative to go2_parkour_deploy
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(os.path.dirname(current_dir))
parkour_root = os.path.join(workspace_root, 'go2_parkour')

if parkour_root not in sys.path:
    sys.path.append(parkour_root)

# Try importing the config
import_success = False
try:
    import isaacgym # Import isaacgym before torch/legged_gym to avoid order issues
    from legged_gym.envs.go2.go2_config import Go2RoughCfg, Go2RoughCfgPPO
    import_success = True
except ImportError as e:
    print(f"Failed to import training config: {e}")
    # Define dummy classes if import fails to avoid crashing immediately, though it will likely fail later
    class Go2RoughCfg: pass
    class Go2RoughCfgPPO: pass

def get_configs(use_camera=False, use_delay=False):
    """
    Converts Go2RoughCfg and Go2RoughCfgPPO to the format expected by DeploymentPlayer
    """
    env_cfg = EasyDict()
    agent_cfg = EasyDict()

    # --- Env Config Conversion ---
    env_cfg.scene = EasyDict()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.depth_camera = Go2RoughCfg.depth.use_camera
    
    # Robot / Scene params for Mujoco - mimicking IsaacLab structure expected by MujocoArticulation
    env_cfg.scene.robot = EasyDict()
    env_cfg.scene.robot.actuators = EasyDict()
    env_cfg.scene.robot.actuators.base_legs = EasyDict()
    # Extract scalar values from 'joint' key which seems to apply to all joints in Go2 config
    env_cfg.scene.robot.actuators.base_legs.stiffness = Go2RoughCfg.control.stiffness.get('joint', 40.0)
    env_cfg.scene.robot.actuators.base_legs.damping = Go2RoughCfg.control.damping.get('joint', 1.0)
    # The following keys need to be populated from configs or reasonable defaults if missing
    # saturation_effort, velocity_limit, effort_limit are dicts mapping '.*' or specific joints
    
    # Defaults based on Go2 hardware specs if not in config
    default_effort_limit = 23.7 
    default_velocity_limit = 30.1 # approx rad/s
    
    # Try to find these in Go2RoughCfg, otherwise use defaults
    # In LeggedRobotCfg, torque_limits might be defined, or we assume uniform
    # MujocoArticulation expects a dictionary mapping regex keys to values.
    
    env_cfg.scene.robot.actuators.base_legs.saturation_effort = {'.*': default_effort_limit}
    env_cfg.scene.robot.actuators.base_legs.velocity_limit = {'.*': default_velocity_limit}
    env_cfg.scene.robot.actuators.base_legs.effort_limit = {'.*': default_effort_limit}

    # Init State
    env_cfg.scene.robot.init_state = EasyDict()
    env_cfg.scene.robot.init_state.pos = Go2RoughCfg.init_state.pos
    env_cfg.scene.robot.init_state.rot = Go2RoughCfg.init_state.rot
    env_cfg.scene.robot.init_state.joint_pos = Go2RoughCfg.init_state.default_joint_angles
    env_cfg.scene.robot.init_state.lin_vel = Go2RoughCfg.init_state.lin_vel
    env_cfg.scene.robot.init_state.ang_vel = Go2RoughCfg.init_state.ang_vel

    # Simulation params
    env_cfg.sim = EasyDict()
    env_cfg.sim.device = 'cuda' if import_success else 'cpu' # Use what we have
    env_cfg.sim.dt = Go2RoughCfg.sim.dt
    env_cfg.sim.gravity = Go2RoughCfg.sim.gravity
    env_cfg.decimation = Go2RoughCfg.control.decimation
    env_cfg.episode_length_s = Go2RoughCfg.env.episode_length_s

    # Actions
    env_cfg.actions = EasyDict()
    env_cfg.actions.joint_pos = EasyDict()
    env_cfg.actions.joint_pos.scale = Go2RoughCfg.control.action_scale
    env_cfg.actions.joint_pos.use_default_offset = True # Standard PPO uses this
    env_cfg.actions.joint_pos.history_length = 3 # Buffer size for action history
    env_cfg.actions.joint_pos.use_delay = use_delay # Simplify for deployment
    env_cfg.actions.joint_pos.delay_update_global_steps = 1 # Prevent ZeroDivisionError
    # Align with training: use action_delay_view as the delay step value
    # action_delay_view is the fixed delay used during visualization/inference in training
    env_cfg.actions.joint_pos.action_delay_steps = [Go2RoughCfg.domain_rand.action_delay_view] if use_delay else [] # Must be a list
    
    # Using a large value for clip if not explicitly defined in joint_pos clip, or use action clip
    # MujocoWrapper expects a list at env_cfg.actions.joint_pos.clip['.*'] because it uses unpack operator [*...]
    env_cfg.actions.joint_pos.clip = {'.*': [-100.0, 100.0]} 

    # Sensors Configs
    env_cfg.scene.contact_forces = EasyDict()
    env_cfg.scene.contact_forces.update_period = env_cfg.sim.dt # Update every step
    env_cfg.scene.contact_forces.history_length = 0

    env_cfg.scene.height_scanner = EasyDict()
    env_cfg.scene.height_scanner.update_period = env_cfg.sim.dt
    env_cfg.scene.height_scanner.attach_yaw_only = True # Standard for height scanners
    env_cfg.scene.height_scanner.drift_range = [0.0, 0.0] # Default to no drift
    env_cfg.scene.height_scanner.max_distance = 3.0 # Standard ray cast distance
    env_cfg.scene.height_scanner.pattern_cfg = EasyDict()
    # Replicating Go2RoughCfg measured_points logic
    # x: [-0.45 ... 1.2], length 1.65, center 0.375. 1.65/11 steps = 0.15 resolution
    # y: [-0.75 ... 0.75], length 1.5, center 0.0. 1.5/10 steps = 0.15 resolution
    env_cfg.scene.height_scanner.pattern_cfg.size = [1.65, 1.5]
    env_cfg.scene.height_scanner.pattern_cfg.resolution = 0.15
    env_cfg.scene.height_scanner.pattern_cfg.ordering = 'xy'
    env_cfg.scene.height_scanner.pattern_cfg.direction = [0, 0, -1]
    
    env_cfg.scene.height_scanner.offset = EasyDict()
    env_cfg.scene.height_scanner.offset.pos = [0.375, 0, 0]
    env_cfg.scene.height_scanner.offset.rot = [1, 0, 0, 0]

    # Commands
    env_cfg.commands = EasyDict()
    env_cfg.commands.base_velocity = EasyDict()
    env_cfg.commands.base_velocity.metrics = EasyDict() # placeholders
    
    # Terrain Generator (To prevent crash in mujoco_terrain_generator.py if accessed)
    env_cfg.scene.terrain = EasyDict()
    env_cfg.scene.terrain.terrain_generator = EasyDict()
    env_cfg.scene.terrain.terrain_generator.sub_terrains = EasyDict()
    env_cfg.scene.terrain.terrain_generator.sub_terrains.parkour_demo = EasyDict() # Empty dict

    # Observations
    env_cfg.observations = EasyDict()
    env_cfg.observations.policy = EasyDict()

    env_cfg.observations.policy.extreme_parkour_observations = EasyDict()
    env_cfg.observations.policy.extreme_parkour_observations.params = EasyDict()
    env_cfg.observations.policy.extreme_parkour_observations.params.history_length = Go2RoughCfg.env.history_len
    env_cfg.observations.policy.extreme_parkour_observations.clip = [Go2RoughCfg.normalization.clip_observations]

    # Env params needed by Depth Backbone
    env_cfg.env = EasyDict()
    env_cfg.env.n_proprio = Go2RoughCfg.env.n_proprio

    # Depth Camera Configs (if used)
    # Enable depth camera if requested
    env_cfg.scene.depth_camera = use_camera 
    if env_cfg.scene.depth_camera:
        env_cfg.scene.depth_camera = EasyDict() # Re-init as dict if bool true above
        env_cfg.scene.depth_camera.update_period = Go2RoughCfg.depth.update_interval * env_cfg.sim.dt
        env_cfg.scene.depth_camera.max_distance = 4.0 
        
        # Add clip parameters for normalization alignment
        env_cfg.scene.depth_camera.near_clip = Go2RoughCfg.depth.near_clip if hasattr(Go2RoughCfg.depth, 'near_clip') else 0.15
        env_cfg.scene.depth_camera.far_clip = Go2RoughCfg.depth.far_clip if hasattr(Go2RoughCfg.depth, 'far_clip') else 2.0

        # Force resize to [58, 87] (H, W) for DepthOnlyFCBackbone58x87
        Go2RoughCfg.depth.resized = [58, 87]

        # Add pattern_cfg for MujocoDepthCamera
        env_cfg.scene.depth_camera.pattern_cfg = EasyDict()
        
        # Determine width and height
        if hasattr(Go2RoughCfg.depth, 'original'):
             env_cfg.scene.depth_camera.pattern_cfg.width = Go2RoughCfg.depth.original[0]
             env_cfg.scene.depth_camera.pattern_cfg.height = Go2RoughCfg.depth.original[1]
        else:
             env_cfg.scene.depth_camera.pattern_cfg.width = 160
             env_cfg.scene.depth_camera.pattern_cfg.height = 108
             
        # Add intrinsics parameters for MujocoDepthCamera (Realsense D435 approximations)
        import math
        # Use horizontal_fov from training config (Go2RoughCfg.depth.horizontal_fov)
        fov_h_deg = Go2RoughCfg.depth.horizontal_fov if hasattr(Go2RoughCfg.depth, 'horizontal_fov') else 79.0
        env_cfg.scene.depth_camera.pattern_cfg.horizontal_aperture = 10.0 # arbitrary units
        # Maintain aspect ratio assuming square pixels
        aspect_ratio = env_cfg.scene.depth_camera.pattern_cfg.height / env_cfg.scene.depth_camera.pattern_cfg.width
        env_cfg.scene.depth_camera.pattern_cfg.vertical_aperture = env_cfg.scene.depth_camera.pattern_cfg.horizontal_aperture * aspect_ratio
        
        # focal_length from horizontal FOV
        env_cfg.scene.depth_camera.pattern_cfg.focal_length = (env_cfg.scene.depth_camera.pattern_cfg.horizontal_aperture / 2) / math.tan(math.radians(fov_h_deg / 2))
        
        # Calculate Vertical FOV for MuJoCo (fovy)
        # tan(fovy/2) = (h/2) / f
        # fovy = 2 * atan(h/(2f))
        vertical_fov_rad = 2 * math.atan(env_cfg.scene.depth_camera.pattern_cfg.vertical_aperture / (2 * env_cfg.scene.depth_camera.pattern_cfg.focal_length))
        env_cfg.scene.depth_camera.fovy_deg = math.degrees(vertical_fov_rad)

        env_cfg.scene.depth_camera.pattern_cfg.horizontal_aperture_offset = 0.0
        env_cfg.scene.depth_camera.pattern_cfg.vertical_aperture_offset = 0.0
             
        env_cfg.scene.depth_camera.data_types = ['distance_to_camera'] # Default needed by MujocoDepthCamera
        env_cfg.scene.depth_camera.depth_clipping_behavior = 'max'

        env_cfg.scene.depth_camera.depth_cam = EasyDict()
        env_cfg.scene.depth_camera.depth_cam.params = EasyDict()
        env_cfg.scene.depth_camera.depth_cam.params.resize = Go2RoughCfg.depth.resized
        env_cfg.scene.depth_camera.depth_cam.params.buffer_len = Go2RoughCfg.depth.buffer_len if hasattr(Go2RoughCfg.depth, 'buffer_len') else 2
        
        # Also map env_cfg.observations.depth_camera... as referenced in wrapper
        env_cfg.observations.depth_camera = env_cfg.scene.depth_camera

    # --- Agent Config Conversion ---
    agent_cfg.clip_actions = Go2RoughCfg.normalization.clip_actions
    
    # Estimator / Net params
    agent_cfg.estimator = EasyDict()
    agent_cfg.estimator.num_prop = Go2RoughCfg.env.n_proprio
    agent_cfg.estimator.num_scan = Go2RoughCfg.env.n_scan
    agent_cfg.estimator.num_priv_explicit = Go2RoughCfg.env.n_priv
    agent_cfg.estimator.num_priv_latent = Go2RoughCfg.env.n_priv_latent
    agent_cfg.estimator.num_hist = Go2RoughCfg.env.history_len
    
    # These are usually hardcoded in RSL_RL PPO or derived. 
    # We provide standard defaults if not in Cfg
    agent_cfg.estimator.scan_encoder_dims = [128, 64, 32]
    agent_cfg.estimator.actor_hidden_dims = [512, 256, 128]
    agent_cfg.estimator.priv_encoder_dims = [64, 20]
    agent_cfg.estimator.activation = 'elu'
    agent_cfg.estimator.estimator_hidden_dims = [128, 64]

    # Try to load from Config if available
    try:
        agent_cfg.estimator.scan_encoder_dims = Go2RoughCfgPPO.policy.scan_encoder_dims
        agent_cfg.estimator.actor_hidden_dims = Go2RoughCfgPPO.policy.actor_hidden_dims
        agent_cfg.estimator.priv_encoder_dims = Go2RoughCfgPPO.policy.priv_encoder_dims
        agent_cfg.estimator.activation = Go2RoughCfgPPO.policy.activation
        agent_cfg.estimator.estimator_hidden_dims = Go2RoughCfgPPO.estimator.hidden_dims
    except:
        pass
    agent_cfg.estimator.actor_hidden_dims = [512, 256, 128]
    agent_cfg.estimator.priv_encoder_dims = [64, 20]
    agent_cfg.estimator.activation = 'elu'

    return env_cfg, agent_cfg
