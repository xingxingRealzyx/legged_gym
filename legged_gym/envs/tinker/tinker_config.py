from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TinkerRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 4096 
        num_observations = 39
        num_actions = 10
        env_spacing = 1.

    class terrain( LeggedRobotCfg.terrain):
        # measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] # 1mx1m rectangle (without center line)
        # measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        measure_heights = False # 测量高度
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.5] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'joint_l_yaw': 0., 
            'joint_l_roll': 0., 
            'joint_l_pitch': 0., 
            'joint_l_knee': 0., 
            'joint_l_ankle':0., 

            'joint_r_yaw': 0.,
            'joint_r_roll': 0., 
            'joint_r_pitch': 0., 
            'joint_r_knee': 0., 
            'joint_r_ankle': 0.
        }
        

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = { 'joint': 100.0 }  # [N*m/rad]
        damping = { 'joint': 3.0 }  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.1
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 2

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 50.
        clip_actions = 100.

    class commands(LeggedRobotCfg.commands):
        step_joint_offset = 0.30  # rad
        step_freq = 1.5  # HZ （e.g. cycle-time=0.66）

        class ranges(LeggedRobotCfg.commands.ranges):
            #lin_vel_x = [-0.3, 0.5]  # min max [m/s]
            #lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            #ang_vel_yaw = [-0.3, 0.3]    # min max [rad/s]
            #heading = [-3.14, 3.14]
            lin_vel_x = [-0.0, 0.7]  # min max [m/s]
            lin_vel_y = [-0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-0.0, 0.0]  # min max [rad/s]
            heading = [-0, 0]
        heading_command = False  # if true: compute ang vel command from heading error

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/tinker/urdf/tinker_urdf.urdf'
        name = "foot"
        foot_name = 'foot' # 脚的刚体的名字
        # terminate_after_contacts_on = ['pelvis']
        flip_visual_attachments = False # 模型显示不正确可能是因为这个
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        terminate_body_height = 0.4

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.
        only_positive_rewards = True
        min_dist = 0.2 # 两只脚的最小距离
        max_dist = 0.5 # 两只脚的最大距离
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -5.
            target_joint_pos_r = 10.0  # 3.  reference joint imitation reward
            target_joint_pos_l = 10.0
            orientation = 10.0  # 
            feet_distance = 1.0

class TinkerRoughCfgPPO( LeggedRobotCfgPPO ):
    init_noise_std = 0.12
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'tanh'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'tinker'

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
