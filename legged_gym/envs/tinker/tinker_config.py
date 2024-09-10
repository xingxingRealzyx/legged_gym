from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TinkerRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 1024 
        num_observations = 39
        num_actions = 10

    class terrain( LeggedRobotCfg.terrain):
        # measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] # 1mx1m rectangle (without center line)
        # measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        measure_heights = False
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.] # x,y,z [m]
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
        stiffness = {   'yaw': 100.0, 'roll': 100.0, 'pitch': 100.0, 
                        'knee': 200., 'ankle': 40.}  # [N*m/rad]
        damping = { 'yaw': 3.0, 'roll': 3.0, 'pitch' : 3.0,
                    'knee': 6., 'ankle': 6 }  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/tinker/urdf/tinker_urdf.urdf'
        name = "tinker"
        # foot_name = 'foot'
        # terminate_after_contacts_on = ['pelvis']
        flip_visual_attachments = False # 模型显示不正确可能是因为这个
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -200.
            tracking_ang_vel = 1.0
            torques = -5.e-6
            dof_acc = -2.e-7
            lin_vel_z = -0.5
            feet_air_time = 5.
            dof_pos_limits = -1.
            no_fly = 0.25
            dof_vel = -0.0
            ang_vel_xy = -0.0
            feet_contact_forces = -0.

class TinkerRoughCfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_tinker'

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
