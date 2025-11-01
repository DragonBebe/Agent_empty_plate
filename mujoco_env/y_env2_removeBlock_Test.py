import sys
import random
import numpy as np
import xml.etree.ElementTree as ET
from mujoco_env.mujoco_parser import MuJoCoParserClass
from mujoco_env.utils import prettify, sample_xyzs, rotation_matrix, add_title_to_img
from mujoco_env.ik import solve_ik
from mujoco_env.transforms import rpy2r, r2rpy
import os
import copy
import glfw
import time

class EnvRemoveBlock_test:
    def __init__(self, 
                 xml_path,
               action_type='eef_pose', 
               state_type='joint_angle',
               seed = None):
        """
        args:
            xml_path: str, path to the xml file
            action_type: str, type of action space, 'eef_pose','delta_joint_angle' or 'joint_angle'
            state_type: str, type of state space, 'joint_angle' or 'ee_pose'
            seed: int, seed for random number generator
        """
        # Load the xml file
        self.env = MuJoCoParserClass(name='Tabletop',rel_xml_path=xml_path)
        self.action_type = action_type
        self.state_type = state_type

        self.joint_names = ['joint1',
                    'joint2',
                    'joint3',
                    'joint4',
                    'joint5',
                    'joint6',]
        
        # 双任务状态管理
        self.task_states = {
            'remove_red_block': False,  # 红块移除任务完成状态
            'place_blue_mug': False     # 蓝杯放置任务完成状态
        }
        self.current_task = 'remove_red_block'  # 当前执行的任务
        self.is_moving_to_initial = False  # 是否正在移动到初始位置
        self.policy_switch_count = 0  # policy切换计数器，用于调试
        
        # 测试循环相关变量
        self.test_stats = {
            'total_tests': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'timeout_tests': 0,
            'success_times': []  # 记录每次成功的时间
        }
        self.current_test_round = 0
        self.max_test_rounds = 10  # 默认测试轮数
        self.task_start_time = None
        self.task_timeout = 120.0  # 2分钟超时
        self.is_testing = False
        
        self.init_viewer()
        self.reset(seed)

    def init_viewer(self):
        '''
        Initialize the viewer
        '''
        self.env.reset()
        self.env.init_viewer(
            distance          = 2.0,
            elevation         = -30, 
            transparent       = False,
            black_sky         = True,
            use_rgb_overlay = False,
            loc_rgb_overlay = 'top right',
        )
    def reset(self, seed = None):
        '''
        Reset the environment
        Move the robot to a initial position, set the object positions based on the seed
        '''
        # 重置任务状态
        self.task_states = {
            'remove_red_block': False,  # 红块移除任务完成状态
            'place_blue_mug': False     # 蓝杯放置任务完成状态
        }
        self.current_task = 'remove_red_block'  # 重置为第一个任务
        self.is_moving_to_initial = False  # 重置移动状态
        self.policy_switch_count = 0  # 重置policy切换计数器
        
        # 重置任务开始时间
        if self.is_testing:
            self.task_start_time = time.time()
        
        if seed != None: np.random.seed(seed=0) 
        q_init = np.deg2rad([0,0,0,0,0,0])
        q_zero,ik_err_stack,ik_info = solve_ik(
            env = self.env,
            joint_names_for_ik = self.joint_names,
            body_name_trgt     = 'tcp_link',
            q_init       = q_init, # ik from zero pose
            p_trgt       = np.array([0.3,0.0,1.0]),
            R_trgt       = rpy2r(np.deg2rad([90,-0.,90 ])),
        )
        self.env.forward(q=q_zero,joint_names=self.joint_names,increase_tick=False)
        
        # set plate position
        plate_xyz = np.array([0.3, -0.25, 0.82])
        self.env.set_p_base_body(body_name='body_obj_plate_11',p=plate_xyz)
        self.env.set_R_base_body(body_name='body_obj_plate_11',R=np.eye(3,3))
        # Set object positions
        # set red block position from scene (you can change here)
        red_block_xyz = np.array([0.32, -0.25, 0.83])
        self.env.set_p_base_body(body_name='body_obj_block_red',p=red_block_xyz)
        self.env.set_R_base_body(body_name='body_obj_block_red',R=np.eye(3,3))
        obj_xyzs = sample_xyzs(
            1,
            x_range   = [+0.32,+0.33],
            y_range   = [-0.00,+0.02],
            z_range   = [0.83,0.83],
            min_dist  = 0.16,
            xy_margin = 0.0
        )
        # default to blue block available; will be updated below
        self.has_blue = True
        obj_xyzs = sample_xyzs(
            1,
            x_range   = [+0.29,+0.3],
            y_range   = [0.19,+0.21],
            z_range   = [0.83,0.83],
            min_dist  = 0.16,
            xy_margin = 0.0
        )
        try:
            self.env.set_p_base_body(body_name='body_obj_mug_6',p=obj_xyzs[0,:])
            self.env.set_R_base_body(body_name='body_obj_mug_6',R=np.eye(3,3))
            self.has_mug = True
        except KeyError:
            # scene may not contain mug
            self.has_mug = False
        self.env.forward(increase_tick=False)

        # Set the initial pose of the robot
        self.last_q = copy.deepcopy(q_zero)
        self.q = np.concatenate([q_zero, np.array([0.0]*4)])
        self.p0, self.R0 = self.env.get_pR_body(body_name='tcp_link')
        block_red_init_pose, mug_init_pose, plate_init_pose = self.get_obj_pose()
        self.obj_init_pose = np.concatenate([block_red_init_pose, mug_init_pose, plate_init_pose],dtype=np.float32)
        for _ in range(100):
            self.step_env()
        self.set_instruction()
        print("DONE INITIALIZATION")
        self.gripper_state = False
        self.past_chars = []

    def set_instruction(self, given = None):
        """
        Set the instruction for the task
        """
        if given is None:
            # 根据当前任务设置指令，避免关键词冲突
            if self.current_task == 'remove_red_block':
                self.instruction = 'Remove the red block from the plate.'
                self.obj_target = 'body_obj_block_red'
            elif self.current_task == 'place_blue_mug':
                # 使用更明确的指令，避免与red相关的混淆
                self.instruction = 'Place the blue mug on the plate.'
                self.obj_target = 'body_obj_mug_6'
        else:
            self.instruction = given
            text = self.instruction.lower()
            if 'remove' in text and 'red' in text:
                # e.g., "Remove the red block from plate/the plate."
                self.obj_target = 'body_obj_block_red'
                self.current_task = 'remove_red_block'
            elif 'place' in text and 'mug' in text and 'blue' in text:
                # e.g., "Place the blue mug on the plate."
                if not getattr(self, 'has_mug', True):
                    raise ValueError('Scene does not contain mug, but instruction refers to mug.')
                self.obj_target = 'body_obj_mug_6'
                self.current_task = 'place_blue_mug'
            else:
                # 默认回退到移除红块任务
                self.instruction = 'Remove the red block from the plate.'
                self.obj_target = 'body_obj_block_red'
                self.current_task = 'remove_red_block'

    def step(self, action):
        '''
        Take a step in the environment
        args:
            action: np.array of shape (7,), action to take
        returns:
            state: np.array, state of the environment after taking the action
                - ee_pose: [px,py,pz,r,p,y]
                - joint_angle: [j1,j2,j3,j4,j5,j6]

        '''
        if self.action_type == 'eef_pose':
            q = self.env.get_qpos_joints(joint_names=self.joint_names)
            self.p0 += action[:3]
            self.R0 = self.R0.dot(rpy2r(action[3:6]))
            q ,ik_err_stack,ik_info = solve_ik(
                env                = self.env,
                joint_names_for_ik = self.joint_names,
                body_name_trgt     = 'tcp_link',
                q_init             = q,
                p_trgt             = self.p0,
                R_trgt             = self.R0,
                max_ik_tick        = 50,
                ik_stepsize        = 1.0,
                ik_eps             = 1e-2,
                ik_th              = np.radians(5.0),
                render             = False,
                verbose_warning    = False,
            )
        elif self.action_type == 'delta_joint_angle':
            q = action[:-1] + self.last_q
        elif self.action_type == 'joint_angle':
            q = action[:-1]
        else:
            raise ValueError('action_type not recognized')
        
        gripper_cmd = np.array([action[-1]]*4)
        gripper_cmd[[1,3]] *= 0.8
        self.compute_q = q
        q = np.concatenate([q, gripper_cmd])

        self.q = q
        if self.state_type == 'joint_angle':
            return self.get_joint_state()
        elif self.state_type == 'ee_pose':
            return self.get_ee_pose()
        elif self.state_type == 'delta_q' or self.action_type == 'delta_joint_angle':
            dq =  self.get_delta_q()
            return dq
        else:
            raise ValueError('state_type not recognized')

    def step_env(self):
        self.env.step(self.q)

    def grab_image(self):
        '''
        grab images from the environment
        returns:
            rgb_agent: np.array, rgb image from the agent's view
            rgb_ego: np.array, rgb image from the egocentric
        '''
        self.rgb_agent = self.env.get_fixed_cam_rgb(
            cam_name='agentview')
        self.rgb_ego = self.env.get_fixed_cam_rgb(
            cam_name='egocentric')
        # self.rgb_top = self.env.get_fixed_cam_rgbd_pcd(
        #     cam_name='topview')
        self.rgb_side = self.env.get_fixed_cam_rgb(
            cam_name='sideview')
        return self.rgb_agent, self.rgb_ego
        

    def render(self, teleop=False, idx = 0):
        '''
        Render the environment
        '''
        self.env.plot_time()
        p_current, R_current = self.env.get_pR_body(body_name='tcp_link')
        R_current = R_current @ np.array([[1,0,0],[0,0,1],[0,1,0 ]])
        self.env.plot_sphere(p=p_current, r=0.02, rgba=[0.95,0.05,0.05,0.5])
        self.env.plot_capsule(p=p_current, R=R_current, r=0.01, h=0.2, rgba=[0.05,0.95,0.05,0.5])
        rgb_egocentric_view = add_title_to_img(self.rgb_ego,text='Egocentric View',shape=(640,480))
        rgb_agent_view = add_title_to_img(self.rgb_agent,text='Agent View',shape=(640,480))
        self.env.plot_T(p = np.array([0.1,0.0,1.0]), label=f"Episode {idx}", plot_axis=False, plot_sphere=False)
        self.env.viewer_rgb_overlay(rgb_agent_view,loc='top right')
        self.env.viewer_rgb_overlay(rgb_egocentric_view,loc='bottom right')
        if teleop:
            rgb_side_view = add_title_to_img(self.rgb_side,text='Side View',shape=(640,480))
            self.env.viewer_rgb_overlay(rgb_side_view, loc='top left')
            self.env.viewer_text_overlay(text1='Key Pressed',text2='%s'%(self.env.get_key_pressed_list()))
            self.env.viewer_text_overlay(text1='Key Repeated',text2='%s'%(self.env.get_key_repeated_list()))
        if getattr(self, 'instruction', None) is not None:
            language_instructions = self.instruction
            self.env.viewer_text_overlay(text1='Language Instructions',text2=language_instructions)
        self.env.render()

    def get_joint_state(self):
        '''
        Get the joint state of the robot
        returns:
            q: np.array, joint angles of the robot + gripper state (0 for open, 1 for closed)
            [j1,j2,j3,j4,j5,j6,gripper]
        '''
        qpos = self.env.get_qpos_joints(joint_names=self.joint_names)
        gripper = self.env.get_qpos_joint('rh_r1')
        gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0
        return np.concatenate([qpos, [gripper_cmd]],dtype=np.float32)
    
    def teleop_robot(self):
        '''
        Teleoperate the robot using keyboard
        returns:
            action: np.array, action to take
            done: bool, True if the user wants to reset the teleoperation
        
        Keys:
            ---------     -----------------------
               w       ->        backward
            s  a  d        left   forward   right
            ---------      -----------------------
            In x, y plane

            ---------
            R: Moving Up
            F: Moving Down
            ---------
            In z axis

            ---------
            Q: Tilt left
            E: Tilt right
            UP: Look Upward
            Down: Look Donward
            Right: Turn right
            Left: Turn left
            ---------
            For rotation

            ---------
            z: reset
            SPACEBAR: gripper open/close
            ---------   


        '''
        # char = self.env.get_key_pressed()
        dpos = np.zeros(3)
        drot = np.eye(3)
        if self.env.is_key_pressed_repeat(key=glfw.KEY_S):
            dpos += np.array([0.007,0.0,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_W):
            dpos += np.array([-0.007,0.0,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_A):
            dpos += np.array([0.0,-0.007,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_D):
            dpos += np.array([0.0,0.007,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_R):
            dpos += np.array([0.0,0.0,0.007])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_F):
            dpos += np.array([0.0,0.0,-0.007])
        if  self.env.is_key_pressed_repeat(key=glfw.KEY_LEFT):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        if  self.env.is_key_pressed_repeat(key=glfw.KEY_RIGHT):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_DOWN):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_UP):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_Q):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_E):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_once(key=glfw.KEY_Z):
            return np.zeros(7, dtype=np.float32), True
        if self.env.is_key_pressed_once(key=glfw.KEY_SPACE):
            self.gripper_state =  not  self.gripper_state
        drot = r2rpy(drot)
        action = np.concatenate([dpos, drot, np.array([self.gripper_state],dtype=np.float32)],dtype=np.float32)
        return action, False
    
    def get_delta_q(self):
        '''
        Get the delta joint angles of the robot
        returns:
            delta: np.array, delta joint angles of the robot + gripper state (0 for open, 1 for closed)
            [dj1,dj2,dj3,dj4,dj5,dj6,gripper]
        '''
        delta = self.compute_q - self.last_q
        self.last_q = copy.deepcopy(self.compute_q)
        gripper = self.env.get_qpos_joint('rh_r1')
        gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0
        return np.concatenate([delta, [gripper_cmd]],dtype=np.float32)

    def check_success(self):
        '''
        ['body_obj_block_red', 'body_obj_mug_5', 'body_obj_plate_11']
        Check success condition depending on target object
        '''
        p_block = self.env.get_p_body(self.obj_target)
        p_plate = self.env.get_p_body('body_obj_plate_11')
        
        # 检查红块移除任务
        if self.obj_target == 'body_obj_block_red':
            dist = np.linalg.norm(p_block - p_plate)
            gripper_open = self.env.get_qpos_joint('rh_r1') < 0.1
            # Use plate z as table-top reference; allow small tolerance
            table_z = p_plate[2]
            on_table = (p_block[2] > table_z - 0.05) and (p_block[2] < table_z + 0.03)
            success = bool((dist > 0.2) and gripper_open and on_table)
            if success and not self.task_states['remove_red_block']:
                self.task_states['remove_red_block'] = True
                print("红块移除任务完成！")
            return success
            
        # 检查蓝杯放置任务
        elif self.obj_target == 'body_obj_mug_6':
            # Place task (e.g., mug): original success criterion
            if np.linalg.norm(p_block[:2] - p_plate[:2]) < 0.1 and np.linalg.norm(p_block[2] - p_plate[2]) < 0.6 and self.env.get_qpos_joint('rh_r1') < 0.1:
                p = self.env.get_p_body('tcp_link')[2]
                if p > 0.9:
                    if not self.task_states['place_blue_mug']:
                        self.task_states['place_blue_mug'] = True
                        print("蓝杯放置任务完成！")
                    return True
        return False
    
    def check_all_tasks_completed(self):
        """
        检查所有任务是否都已完成
        """
        return self.task_states['remove_red_block'] and self.task_states['place_blue_mug']
    
    def move_to_initial_position(self):
        """
        将机械臂移动到初始位置，以便能够看到蓝杯
        """
        print("正在移动机械臂到初始位置...")
        self.is_moving_to_initial = True
        
        # 获取初始关节角度
        q_init = np.deg2rad([0,0,0,0,0,0])
        q_zero,ik_err_stack,ik_info = solve_ik(
            env = self.env,
            joint_names_for_ik = self.joint_names,
            body_name_trgt     = 'tcp_link',
            q_init       = q_init, # ik from zero pose
            p_trgt       = np.array([0.3,0.0,1.0]),
            R_trgt       = rpy2r(np.deg2rad([90,-0.,90 ])),
        )
        
        # 平滑移动到初始位置
        current_q = self.env.get_qpos_joints(joint_names=self.joint_names)
        target_q = q_zero
        
        # 分步移动，确保平滑过渡
        num_steps = 100  # 增加移动步数，使移动更平滑
        for i in range(num_steps):
            # 使用sigmoid函数进行更自然的插值
            alpha = 1 / (1 + np.exp(-10 * (i / num_steps - 0.5)))
            interpolated_q = current_q + alpha * (target_q - current_q)
            
            # 设置关节角度
            gripper_cmd = np.array([0.0]*4)  # 保持夹爪打开
            q = np.concatenate([interpolated_q, gripper_cmd])
            self.q = q
            
            # 执行一步
            self.step_env()
            self.render()
            
            # 短暂暂停以确保平滑移动
            import time
            time.sleep(0.005)  # 减少暂停时间，使移动更流畅
        
        self.is_moving_to_initial = False
        print("机械臂已移动到初始位置，可以开始蓝杯放置任务")
        return True
    
    def switch_to_next_task(self):
        """
        切换到下一个任务
        """
        if self.current_task == 'remove_red_block' and self.task_states['remove_red_block']:
            # 先移动机械臂到初始位置
            self.move_to_initial_position()
            
            # 然后切换任务
            self.current_task = 'place_blue_mug'
            self.set_instruction()  # 重新设置指令
            print("切换到蓝杯放置任务")
            return True
        return False
    
    def get_obj_pose(self):
        '''
        returns: 
            p_block_red: np.array, position of the red block
            p_mug: np.array, position of the mug
            p_plate: np.array, position of the plate
        '''
        p_block_red = self.env.get_p_body('body_obj_block_red')
        if getattr(self, 'has_mug', True):
            p_mug = self.env.get_p_body('body_obj_mug_6')
        else:
            p_mug = np.zeros(3, dtype=np.float32)
        p_plate = self.env.get_p_body('body_obj_plate_11')

        return p_block_red, p_mug, p_plate
    
    def set_obj_pose(self, p_block_red, p_mug, p_plate):
        '''
        Set the object poses
        args:
            p_block_red: np.array, position of the red block
            p_mug: np.array, position of the mug
            p_plate: np.array, position of the plate
        '''
        self.env.set_p_base_body(body_name='body_obj_block_red',p=p_block_red)
        self.env.set_R_base_body(body_name='body_obj_block_red',R=np.eye(3,3))
        if getattr(self, 'has_mug', True):
            self.env.set_p_base_body(body_name='body_obj_mug_6',p=p_mug)
            self.env.set_R_base_body(body_name='body_obj_mug_6',R=np.eye(3,3))
        self.env.set_p_base_body(body_name='body_obj_plate_11',p=p_plate)
        self.env.set_R_base_body(body_name='body_obj_plate_11',R=np.eye(3,3))
        self.step_env()


    def get_ee_pose(self):
        '''
        get the end effector pose of the robot + gripper state
        '''
        p, R = self.env.get_pR_body(body_name='tcp_link')
        rpy = r2rpy(R)
        return np.concatenate([p, rpy],dtype=np.float32)
    
    def start_testing(self, max_rounds=10):
        """
        开始测试循环
        args:
            max_rounds: int, 最大测试轮数
        """
        self.max_test_rounds = max_rounds
        self.current_test_round = 0
        self.is_testing = True
        self.test_stats = {
            'total_tests': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'timeout_tests': 0,
            'success_times': []  # 记录每次成功的时间
        }
        print(f"开始测试循环，总共将进行 {max_rounds} 轮测试")
        self.reset()
    
    def check_timeout(self):
        """
        检查任务是否超时
        returns:
            bool: True if timeout, False otherwise
        """
        if not self.is_testing or self.task_start_time is None:
            return False
        
        elapsed_time = time.time() - self.task_start_time
        return elapsed_time > self.task_timeout
    
    def handle_task_completion(self, success=False, timeout=False):
        """
        处理任务完成（成功、失败或超时）
        args:
            success: bool, 任务是否成功
            timeout: bool, 任务是否超时
        returns:
            bool: True if all tests completed, False otherwise
        """
        if not self.is_testing:
            return False
        
        # 更新统计信息
        self.test_stats['total_tests'] += 1
        if success:
            self.test_stats['successful_tests'] += 1
            # 记录成功时间
            if self.task_start_time:
                success_time = time.time() - self.task_start_time
                self.test_stats['success_times'].append(success_time)
                print(f"第 {self.current_test_round + 1} 轮测试成功！用时: {success_time:.1f}秒")
            else:
                print(f"第 {self.current_test_round + 1} 轮测试成功！")
        elif timeout:
            self.test_stats['timeout_tests'] += 1
            self.test_stats['failed_tests'] += 1
            print(f"第 {self.current_test_round + 1} 轮测试超时失败！")
        else:
            self.test_stats['failed_tests'] += 1
            print(f"第 {self.current_test_round + 1} 轮测试失败！")
        
        self.current_test_round += 1
        
        # 检查是否完成所有测试
        if self.current_test_round >= self.max_test_rounds:
            self.is_testing = False
            self.print_test_results()
            # 自动关闭MuJoCo窗口
            self.close_viewer()
            return True
        
        # 重置环境进行下一轮测试
        print(f"准备进行第 {self.current_test_round + 1} 轮测试...")
        self.reset()
        return False
    
    def close_viewer(self):
        """
        关闭MuJoCo仿真窗口
        """
        try:
            if hasattr(self.env, 'viewer') and self.env.viewer is not None:
                self.env.viewer.close()
                print("MuJoCo仿真窗口已关闭")
        except Exception as e:
            print(f"关闭窗口时出现错误: {e}")
    
    def print_test_results(self):
        """
        打印测试结果
        """
        total = self.test_stats['total_tests']
        success = self.test_stats['successful_tests']
        failed = self.test_stats['failed_tests']
        timeout = self.test_stats['timeout_tests']
        success_times = self.test_stats['success_times']
        
        success_rate = (success / total * 100) if total > 0 else 0
        
        print("\n" + "="*50)
        print("测试结果统计")
        print("="*50)
        print(f"总测试轮数: {total}")
        print(f"成功次数: {success}")
        print(f"失败次数: {failed}")
        print(f"超时次数: {timeout}")
        print(f"成功率: {success_rate:.2f}%")
        
        # 显示成功时间统计
        if success_times:
            avg_time = sum(success_times) / len(success_times)
            min_time = min(success_times)
            max_time = max(success_times)
            print(f"\n成功任务时间统计:")
            print(f"  平均用时: {avg_time:.1f}秒")
            print(f"  最短用时: {min_time:.1f}秒")
            print(f"  最长用时: {max_time:.1f}秒")
            print(f"  所有成功时间: {[f'{t:.1f}s' for t in success_times]}")
        
        print("="*50)
        
        return {
            'total_tests': total,
            'successful_tests': success,
            'failed_tests': failed,
            'timeout_tests': timeout,
            'success_rate': success_rate,
            'success_times': success_times
        }
    
    def get_test_progress(self):
        """
        获取当前测试进度
        returns:
            dict: 测试进度信息
        """
        if not self.is_testing:
            return None
        
        elapsed_time = 0
        if self.task_start_time:
            elapsed_time = time.time() - self.task_start_time
        
        return {
            'current_round': self.current_test_round + 1,
            'total_rounds': self.max_test_rounds,
            'elapsed_time': elapsed_time,
            'timeout_limit': self.task_timeout,
            'current_task': self.current_task,
            'stats': self.test_stats.copy()
        }