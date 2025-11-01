import sys
import random
import numpy as np
import xml.etree.ElementTree as ET
from mujoco_env.y_env import SimpleEnv
from mujoco_env.utils import prettify, sample_xyzs, rotation_matrix, add_title_to_img
from mujoco_env.ik import solve_ik
from mujoco_env.transforms import rpy2r, r2rpy
import os
import copy
import glfw

class SO100Env(SimpleEnv):
    """
    专门用于 SO100 机械臂的环境类
    继承自 SimpleEnv，但使用正确的关节名称
    """
    def __init__(self, 
                 xml_path,
                action_type='eef_pose', 
                state_type='joint_angle',
                seed=None):
        # 先初始化父类的环境解析器
        from mujoco_env.mujoco_parser import MuJoCoParserClass
        self.env = MuJoCoParserClass(name='Tabletop', rel_xml_path=xml_path)
        self.action_type = action_type
        self.state_type = state_type

        # SO100 机械臂的实际关节名称
        self.joint_names = ['Rotation',
                           'Pitch', 
                           'Elbow',
                           'Wrist_Pitch',
                           'Wrist_Roll',
                           'Jaw']
        
        # SO100 的夹爪关节名称
        self.gripper_joint_name = 'Jaw'
        
        self.init_viewer()
        self.reset(seed)
        # Cache actuator indices for efficient control updates
        self.ctrl_idxs = self.env.get_idxs_step(joint_names=self.joint_names)
        self.env.data.ctrl[self.ctrl_idxs] = self.q

    def reset(self, seed=None):
        '''
        Reset the environment for SO100 arm
        '''
        if seed != None: 
            np.random.seed(seed=0) 
        
        # SO100 的初始关节角度 (使用 SO100 的实际关节)
        q_init = np.array([0, -np.pi/2, np.pi/2, 0, 0, 0])  # 适合 SO100 的初始姿态
        
        # 使用逆运动学求解初始位置
        q_zero, ik_err_stack, ik_info = solve_ik(
            env=self.env,
            joint_names_for_ik=self.joint_names[:-1],  # 不包括夹爪关节
            body_name_trgt='Fixed_Jaw',  # SO100 的末端执行器
            q_init=q_init[:-1],  # 不包括夹爪关节
            p_trgt=np.array([0.3, 0.0, 1.0]),
            R_trgt=rpy2r(np.deg2rad([90, -0., 90])),
        )
        
        # 添加夹爪关节状态
        q_zero_with_gripper = np.concatenate([q_zero, [0.0]])
        self.env.forward(q=q_zero_with_gripper, joint_names=self.joint_names, increase_tick=False)

        # Set object positions - 确保block在plate的右侧
        obj_names = self.env.get_body_names(prefix='body_obj_')
        n_obj = len(obj_names)
        
        # 简化的位置设置 - block的初始位置已在XML中定义
        for obj_idx, obj_name in enumerate(obj_names):
            if 'plate' in obj_name:
                # Plate放在左侧，保持原有设置
                obj_pos = np.array([0.3, -0.25, 0.82])
                self.env.set_p_base_body(body_name=obj_name, p=obj_pos)
                self.env.set_R_base_body(body_name=obj_name, R=np.eye(3, 3))
            elif 'body_obj_block_red' in obj_name or 'cube' in obj_name:
                # Block/Cube位置由XML中的初始位置定义，这里可以添加小的随机扰动
                current_pos = self.env.get_p_body(obj_name)
                # 添加小范围随机扰动 (±1cm)
                random_offset = np.random.uniform(-0.01, 0.01, 3)
                random_offset[2] = 0  # Z轴不添加随机扰动，保持在桌面上
                # obj_pos = current_pos + random_offset
                obj_pos = np.array([0.3, 0.1, 0.82])
                self.env.set_p_base_body(body_name=obj_name, p=obj_pos)
                self.env.set_R_base_body(body_name=obj_name, R=np.eye(3, 3))
            else:
                # 其他物体使用原来的随机位置
                obj_xyzs = sample_xyzs(
                    1,
                    x_range=[+0.24, +0.4],
                    y_range=[-0.2, +0.2],
                    z_range=[0.82, 0.82],
                    min_dist=0.15,
                    xy_margin=0.0
                )
                obj_pos = obj_xyzs[0]
                self.env.set_p_base_body(body_name=obj_name, p=obj_pos)
                self.env.set_R_base_body(body_name=obj_name, R=np.eye(3, 3))
        self.env.forward(increase_tick=False)

        # Set the initial pose of the robot
        self.last_q = copy.deepcopy(q_zero)
        self.q = q_zero_with_gripper
        self.p0, self.R0 = self.env.get_pR_body(body_name='Fixed_Jaw')
        block_init_pose, plate_init_pose = self.get_obj_pose()
        self.obj_init_pose = np.concatenate([block_init_pose, plate_init_pose], dtype=np.float32)
        if hasattr(self, "ctrl_idxs"):
            self.env.data.ctrl[self.ctrl_idxs] = self.q
        for _ in range(100):
            self.step_env()
        print("DONE INITIALIZATION")
        self.gripper_state = False
        self.past_chars = []

    def get_obj_pose(self):
        '''
        重写父类方法，处理block环境
        returns: 
            p_block: np.array, position of the block
            p_plate: np.array, position of the plate
        '''
        p_block = self.env.get_p_body('body_obj_block_red')
        p_plate = self.env.get_p_body('body_obj_plate_11')
        return p_block, p_plate

    def set_obj_pose(self, p_block, p_plate):
        '''
        重写父类方法，处理block环境
        Set the object poses
        args:
            p_block: np.array, position of the block
            p_plate: np.array, position of the plate
        '''
        self.env.set_p_base_body(body_name='body_obj_block_red', p=p_block)
        self.env.set_R_base_body(body_name='body_obj_block_red', R=np.eye(3,3))
        self.env.set_p_base_body(body_name='body_obj_plate_11', p=p_plate)
        self.env.set_R_base_body(body_name='body_obj_plate_11', R=np.eye(3,3))
        self.step_env()

    def step(self, action):
        '''
        Take a step in the environment for SO100
        '''
        if self.action_type == 'eef_pose':
            q = self.env.get_qpos_joints(joint_names=self.joint_names[:-1])  # 不包括夹爪
            self.p0 += action[:3]
            self.R0 = self.R0.dot(rpy2r(action[3:6]))
            q, ik_err_stack, ik_info = solve_ik(
                env=self.env,
                joint_names_for_ik=self.joint_names[:-1],  # 不包括夹爪关节
                body_name_trgt='Fixed_Jaw',  # SO100 的末端执行器
                q_init=q,
                p_trgt=self.p0,
                R_trgt=self.R0,
                max_ik_tick=50,
                ik_stepsize=1.0,
                ik_eps=1e-2,
                ik_th=np.radians(5.0),
                render=False,
                verbose_warning=False,
            )
        elif self.action_type == 'delta_joint_angle':
            q = action[:-1] + self.last_q
        elif self.action_type == 'joint_angle':
            q = action[:-1]
        else:
            raise ValueError('action_type not recognized')
        
        # SO100 夹爪控制 (只有一个关节) - 优化防穿模
        gripper_cmd = action[-1]
        
        # 获取当前夹爪状态，实现平滑过渡
        current_gripper = self.env.get_qpos_joint(self.gripper_joint_name)[0]
        
        # 优化的夹爪控制：0.0 -> 完全打开, 1.2 -> 适度关闭（适合小物体）
        target_angle = 1.2 if gripper_cmd > 0.5 else 0.0
        
        # 更精细的控制，适应不同阶段
        angle_diff = target_angle - current_gripper
        
        # 根据当前位置动态调整变化速度
        if target_angle > 0.5:  # 关闭过程
            if current_gripper < 0.3:
                max_change = 0.08  # 初始快速接近
            elif current_gripper < 0.8:
                max_change = 0.04  # 中等速度
            else:
                max_change = 0.02  # 接近时减速，精细控制
        else:  # 打开过程
            max_change = 0.06  # 较快的打开速度
            
        if abs(angle_diff) > max_change:
            gripper_angle = current_gripper + np.sign(angle_diff) * max_change
        else:
            gripper_angle = target_angle
            
        # 匹配新的夹爪范围，略微增大关闭角度
        gripper_angle = np.clip(gripper_angle, -0.2, 1.6)
        
        self.compute_q = q
        q_cmd = np.concatenate([q, [gripper_angle]])
        self.q = q_cmd

        # Update actuator targets without bypassing physics so contacts remain valid
        if self.ctrl_idxs is not None:
            self.env.data.ctrl[self.ctrl_idxs] = q_cmd
        else:
            self.env.data.ctrl[:] = q_cmd
        if self.state_type == 'joint_angle':
            return self.get_joint_state()
        elif self.state_type == 'ee_pose':
            return self.get_ee_pose()
        elif self.state_type == 'delta_q' or self.action_type == 'delta_joint_angle':
            dq = self.get_delta_q()
            return dq
        else:
            raise ValueError('state_type not recognized')

    def get_joint_state(self):
        '''
        Get the joint state of the SO100 robot
        returns:
            q: np.array, joint angles of the robot + gripper state (0 for open, 1 for closed)
            [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw, gripper_cmd] = 7维
        '''
        # 获取机械臂前5个关节的状态（不包括夹爪关节 Jaw）
        arm_qpos = self.env.get_qpos_joints(joint_names=self.joint_names[:-1])
        
        # 获取夹爪关节的原始状态
        jaw_qpos = self.env.get_qpos_joint(self.gripper_joint_name)
        
        # 优化的夹爪状态检测：更精确的阈值判断
        gripper_cmd = 1.0 if jaw_qpos[0] > 0.6 else 0.0
        
        # 返回：[5个机械臂关节, 1个夹爪关节角度, 1个夹爪命令] = 7维
        return np.concatenate([arm_qpos, jaw_qpos, [gripper_cmd]], dtype=np.float32)

    def get_delta_q(self):
        '''
        Get the delta joint angles of the SO100 robot
        '''
        # 计算机械臂关节的变化（前5个关节）
        delta = self.compute_q - self.last_q
        self.last_q = copy.deepcopy(self.compute_q)
        
        # 获取当前夹爪状态
        jaw_qpos = self.env.get_qpos_joint(self.gripper_joint_name)
        gripper_cmd = 1.0 if jaw_qpos[0] > 0.6 else 0.0
        
        # 返回：[5个机械臂关节变化, 1个夹爪关节变化, 1个夹爪命令] = 7维
        jaw_delta = 0.0  # 夹爪变化量（简化处理）
        return np.concatenate([delta, [jaw_delta], [gripper_cmd]], dtype=np.float32)

    def check_success(self):
        '''
        Check if the block is placed on the plate for SO100
        + Gripper should be open and move upward above 0.9
        '''
        p_block = self.env.get_p_body('body_obj_block_red')
        p_plate = self.env.get_p_body('body_obj_plate_11')
        # 使用 SO100 的夹爪关节检查
        gripper_pos = self.env.get_qpos_joint(self.gripper_joint_name)[0]
        # 由于block更小，使用更精确的检测阈值
        if np.linalg.norm(p_block[:2] - p_plate[:2]) < 0.08 and np.linalg.norm(p_block[2] - p_plate[2]) < 0.5 and gripper_pos < 0.1:
            p = self.env.get_p_body('Fixed_Jaw')[2]  # SO100 的末端执行器
            if p > 0.9:
                return True
        return False

    def render(self, teleop=False):
        '''
        Render the environment for SO100
        '''
        self.env.plot_time()
        # 移除末端执行器的可视化标记，避免干扰观察
        # p_current, R_current = self.env.get_pR_body(body_name='Fixed_Jaw')  # SO100 的末端执行器
        # R_current = R_current @ np.array([[1,0,0],[0,0,1],[0,1,0 ]])
        # self.env.plot_sphere(p=p_current, r=0.02, rgba=[0.95,0.05,0.05,0.5])
        # self.env.plot_capsule(p=p_current, R=R_current, r=0.01, h=0.2, rgba=[0.05,0.95,0.05,0.5])
        
        # 确保图像已经被获取
        if not hasattr(self, 'rgb_ego') or not hasattr(self, 'rgb_agent'):
            self.grab_image()
            
        rgb_egocentric_view = add_title_to_img(self.rgb_ego,text='Egocentric View',shape=(640,480))
        rgb_agent_view = add_title_to_img(self.rgb_agent,text='Agent View',shape=(640,480))
        
        self.env.viewer_rgb_overlay(rgb_agent_view,loc='top right')
        self.env.viewer_rgb_overlay(rgb_egocentric_view,loc='bottom right')
        if teleop:
            rgb_side_view = add_title_to_img(self.rgb_side,text='Side View',shape=(640,480))
            self.env.viewer_rgb_overlay(rgb_side_view, loc='top left')
            self.env.viewer_text_overlay(text1='Key Pressed',text2='%s'%(self.env.get_key_pressed_list()))
            self.env.viewer_text_overlay(text1='Key Repeated',text2='%s'%(self.env.get_key_repeated_list()))
        self.env.render()

    def grab_image(self):
        '''
        grab images from the SO100 environment
        returns:
            rgb_agent: np.array, rgb image from the agent's view
            rgb_ego: np.array, rgb image from the egocentric (wrist) view
        '''
        self.rgb_agent = self.env.get_fixed_cam_rgb(cam_name='agentview')
        # 使用新添加的 egocentric 摄像头
        self.rgb_ego = self.env.get_fixed_cam_rgb(cam_name='egocentric')
        # 保留其他摄像头用于渲染
        self.rgb_side = self.env.get_fixed_cam_rgb(cam_name='sideview')
        return self.rgb_agent, self.rgb_ego

    def get_ee_pose(self):
        '''
        get the end effector pose of the SO100 robot + gripper state
        '''
        p, R = self.env.get_pR_body(body_name='Fixed_Jaw')  # SO100 的末端执行器
        rpy = r2rpy(R)
        return np.concatenate([p, rpy], dtype=np.float32)
