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
try:
    from scipy.stats import qmc  # 用于拉丁超立方采样
    HAS_SCIPY = True
except ImportError:
    print("警告: scipy未安装，拉丁超立方采样将不可用")
    HAS_SCIPY = False

class SimpleEnvRemoveBlockCollectDataset:
    def __init__(self, 
                 xml_path,
                 action_type='eef_pose', 
                 state_type='joint_angle',
                 seed=None,
                 random_block_position=True,
                 plate_flat_radius=0.102,  # 基于用户精确分析的平整区域半径
                 sampling_method='uniform'):  # 采样方法: 'uniform', 'latin_hypercube'
        """
        支持随机block位置生成的数据收集环境
        
        Args:
            xml_path: str, path to the xml file
            action_type: str, type of action space, 'eef_pose','delta_joint_angle' or 'joint_angle'
            state_type: str, type of state space, 'joint_angle' or 'ee_pose'
            seed: int, seed for random number generator
            random_block_position: bool, whether to randomize block position
            plate_flat_radius: float, radius of the flat area on the plate (in meters)
            sampling_method: str, sampling method for random positions
                           'uniform': 均匀随机采样
                           'latin_hypercube': 拉丁超立方采样
        """
        # Load the xml file
        self.env = MuJoCoParserClass(name='Tabletop', rel_xml_path=xml_path)
        self.action_type = action_type
        self.state_type = state_type
        self.random_block_position = random_block_position
        self.plate_flat_radius = plate_flat_radius
        self.sampling_method = sampling_method
        
        # Block参数
        self.block_half_size = 0.015  # block中心到边的距离
        self.block_corner_distance = self.block_half_size * np.sqrt(2)  # 中心到角点距离
        
        # 计算安全生成区域半径
        self.safe_generation_radius = self.plate_flat_radius - self.block_corner_distance
        
        # 拉丁超立方采样器初始化
        self.lhs_sampler = None
        self.lhs_samples = None
        self.lhs_index = 0
        
        print(f"环境配置:")
        print(f"  随机block位置: {'启用' if random_block_position else '禁用'}")
        if random_block_position:
            print(f"  采样方法: {self.sampling_method}")
            print(f"  Plate平整半径: {self.plate_flat_radius:.4f}m")
            print(f"  Block角点距离: {self.block_corner_distance:.4f}m")
            print(f"  安全生成半径: {self.safe_generation_radius:.4f}m")
            if self.safe_generation_radius <= 0:
                print(f"  警告: 安全半径 <= 0，将使用固定位置")
                self.random_block_position = False

        self.joint_names = ['joint1',
                    'joint2',
                    'joint3',
                    'joint4',
                    'joint5',
                    'joint6',]
        
        self.init_viewer()
        self.reset(seed)

    def generate_random_block_position(self, plate_center):
        """
        在plate的平整圆形区域内生成随机block位置
        基于同心圆思路：外圆(plate边界) -> 内圆(平整区域) -> 生成圆(block安全区域)
        
        Args:
            plate_center: plate中心位置 [x, y, z]
            
        Returns:
            np.array: [x, y, z] block位置
        """
        if not self.random_block_position or self.safe_generation_radius <= 0:
            # 回退到固定位置，同样添加安全间隙
            safety_gap = 0.005  # 5mm安全间隙，确保block稳定放置在plate表面
            return np.array([plate_center[0] + 0.02, plate_center[1], plate_center[2] + self.block_half_size + safety_gap])
        
        if self.sampling_method == 'uniform':
            x_rel, y_rel = self._uniform_sampling()
        elif self.sampling_method == 'latin_hypercube':
            if HAS_SCIPY:
                x_rel, y_rel = self._latin_hypercube_sampling()
            else:
                print("警告: scipy未安装，回退到uniform采样")
                x_rel, y_rel = self._uniform_sampling()
        else:
            print(f"警告: 未知采样方法 {self.sampling_method}，使用uniform采样")
            x_rel, y_rel = self._uniform_sampling()
        
        # 转换到世界坐标
        world_x = plate_center[0] + x_rel
        world_y = plate_center[1] + y_rel
        
        # 计算安全的block高度，避免镶嵌在plate中
        # 考虑到物理引擎会让block自然落到plate表面，我们需要更大的安全间隙
        # 同时考虑plate的实际厚度和物理碰撞检测的精度
        safety_gap = 0.005  # 5mm安全间隙，确保block稳定放置在plate表面
        world_z = plate_center[2] + self.block_half_size + safety_gap
        
        return np.array([world_x, world_y, world_z])
    
    def _uniform_sampling(self):
        """
        均匀随机采样：在圆形区域内均匀分布
        
        Returns:
            tuple: (x_rel, y_rel) 相对于plate中心的坐标
        """
        # 在安全圆形区域内均匀随机采样
        r = np.sqrt(np.random.random()) * self.safe_generation_radius
        theta = np.random.random() * 2 * np.pi
        
        x_rel = r * np.cos(theta)
        y_rel = r * np.sin(theta)
        
        return x_rel, y_rel
    
    def _latin_hypercube_sampling(self):
        """
        拉丁超立方采样：在圆形区域内进行拉丁超立方采样
        提供更好的空间覆盖性
        
        Returns:
            tuple: (x_rel, y_rel) 相对于plate中心的坐标
        """
        # 如果采样器未初始化或样本用完，重新生成
        if self.lhs_sampler is None or self.lhs_index >= len(self.lhs_samples):
            self._initialize_lhs_sampler()
        
        # 获取下一个拉丁超立方样本
        sample = self.lhs_samples[self.lhs_index]
        self.lhs_index += 1
        
        # 将[0,1]²的样本转换为圆形区域内的点
        # 使用逆变换采样将均匀分布转换为圆内均匀分布
        r = np.sqrt(sample[0]) * self.safe_generation_radius
        theta = sample[1] * 2 * np.pi
        
        x_rel = r * np.cos(theta)
        y_rel = r * np.sin(theta)
        
        return x_rel, y_rel
    
    def _initialize_lhs_sampler(self, n_samples=100):
        """
        初始化拉丁超立方采样器
        
        Args:
            n_samples: 生成的样本数量
        """
        if not HAS_SCIPY:
            print("警告: scipy未安装，无法初始化拉丁超立方采样器")
            return
            
        # 创建2维拉丁超立方采样器
        self.lhs_sampler = qmc.LatinHypercube(d=2, seed=np.random.randint(0, 10000))
        
        # 生成样本
        self.lhs_samples = self.lhs_sampler.random(n=n_samples)
        self.lhs_index = 0
        
        print(f"初始化拉丁超立方采样器: {n_samples}个样本")

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
        
    def reset(self, seed=None):
        '''
        Reset the environment
        Move the robot to a initial position, set the object positions based on the seed
        '''
        if seed != None: 
            np.random.seed(seed)
        
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
        
        # Set red block position - 随机或固定
        red_block_xyz = self.generate_random_block_position(plate_xyz)
        
        if self.random_block_position:
            print(f"随机生成red block位置: [{red_block_xyz[0]:.3f}, {red_block_xyz[1]:.3f}, {red_block_xyz[2]:.3f}]")
        else:
            print(f"使用固定red block位置: [{red_block_xyz[0]:.3f}, {red_block_xyz[1]:.3f}, {red_block_xyz[2]:.3f}]")
        
        self.env.set_p_base_body(body_name='body_obj_block_red',p=red_block_xyz)
        self.env.set_R_base_body(body_name='body_obj_block_red',R=np.eye(3,3))
        
        # Set mug position (保持原有逻辑)
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
        self.compute_q = copy.deepcopy(q_zero)  # 添加compute_q变量
        self.q = np.concatenate([q_zero, np.array([0.0]*4)])
        self.p0, self.R0 = self.env.get_pR_body(body_name='tcp_link')
        block_red_init_pose, mug_init_pose, plate_init_pose = self.get_obj_pose()
        # 完整的初始pose（用于内部使用）
        self.obj_init_pose_full = np.concatenate([block_red_init_pose, mug_init_pose, plate_init_pose],dtype=np.float32)
        # 根据数据集配置，obj_init应该是(9,)形状，包含block和mug的位置信息
        self.obj_init_pose = np.concatenate([block_red_init_pose[:3], mug_init_pose[:3], plate_init_pose[:3]]).astype(np.float32)
        for _ in range(100):
            self.step_env()
        
        self.set_instruction()
        print("DONE INITIALIZATION")
        self.gripper_state = False
        self.past_chars = []

    def set_instruction(self, given=None):
        """
        Set the instruction for the task
        """
        if given is None:
            # 固定指令用于数据采集
            self.instruction = "Remove the red block from the plate"
        else:
            self.instruction = given
        print(f"Task instruction: {self.instruction}")

    def get_obj_pose(self):
        """
        Get the pose of the objects
        """
        # Get the pose of the red block
        try:
            block_red_p, block_red_R = self.env.get_pR_body(body_name='body_obj_block_red')
            block_red_pose = np.concatenate([block_red_p, r2rpy(block_red_R)])
        except:
            block_red_pose = np.zeros(6)
        
        # Get the pose of the mug
        try:
            mug_p, mug_R = self.env.get_pR_body(body_name='body_obj_mug_6')
            mug_pose = np.concatenate([mug_p, r2rpy(mug_R)])
        except:
            mug_pose = np.zeros(6)
        
        # Get the pose of the plate
        try:
            plate_p, plate_R = self.env.get_pR_body(body_name='body_obj_plate_11')
            plate_pose = np.concatenate([plate_p, r2rpy(plate_R)])
        except:
            plate_pose = np.zeros(6)
        
        return block_red_pose, mug_pose, plate_pose

    def step_env(self):
        self.env.step(self.q)

    def get_state(self):
        """
        Get the state of the environment
        """
        if self.state_type == 'joint_angle':
            return self.q[:6].astype(np.float32)  # 确保返回float32类型
        elif self.state_type == 'ee_pose':
            p, R = self.env.get_pR_body(body_name='tcp_link')
            return np.concatenate([p, r2rpy(R)]).astype(np.float32)  # 确保返回float32类型

    def check_success(self):
        """
        Check success condition for remove red block task
        Success criteria:
        1) Block is far from plate (> 0.2 m)
        2) Gripper is open (released the block)  
        3) Block is placed on the table (z near table/plate height)
        """
        try:
            p_block = self.env.get_p_body('body_obj_block_red')
            p_plate = self.env.get_p_body('body_obj_plate_11')
            
            # Remove task: red block success when
            # 1) far from plate (> 0.2 m),
            # 2) gripper open, and
            # 3) block placed on the table (z near table/plate height)
            dist = np.linalg.norm(p_block - p_plate)
            gripper_open = self.env.get_qpos_joint('rh_r1') < 0.1
            # Use plate z as table-top reference; allow small tolerance
            table_z = p_plate[2]
            on_table = (p_block[2] > table_z - 0.05) and (p_block[2] < table_z + 0.03)
            
            return bool((dist > 0.2) and gripper_open and on_table)
        except:
            return False

    def teleop_robot(self):
        """
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
        """
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

    def step(self, action):
        """
        Take a step in the environment
        args:
            action: np.array of shape (7,), action to take
        returns:
            state: np.array, state of the environment after taking the action
                - ee_pose: [px,py,pz,r,p,y]
                - joint_angle: [j1,j2,j3,j4,j5,j6]
        """
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

    def get_joint_state(self):
        """
        Get the joint state of the robot
        returns:
            q: np.array, joint angles of the robot + gripper state (0 for open, 1 for closed)
            [j1,j2,j3,j4,j5,j6,gripper]
        """
        qpos = self.env.get_qpos_joints(joint_names=self.joint_names)
        gripper = self.env.get_qpos_joint('rh_r1')
        gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0
        return np.concatenate([qpos, [gripper_cmd]],dtype=np.float32)

    def get_delta_q(self):
        """
        Get the delta joint angles of the robot
        returns:
            delta: np.array, delta joint angles of the robot + gripper state (0 for open, 1 for closed)
            [dj1,dj2,dj3,dj4,dj5,dj6,gripper]
        """
        delta = self.compute_q - self.last_q
        self.last_q = copy.deepcopy(self.compute_q)
        gripper = self.env.get_qpos_joint('rh_r1')
        gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0
        return np.concatenate([delta, [gripper_cmd]],dtype=np.float32)

    def get_ee_pose(self):
        """
        get the end effector pose of the robot + gripper state
        """
        p, R = self.env.get_pR_body(body_name='tcp_link')
        rpy = r2rpy(R)
        return np.concatenate([p, rpy],dtype=np.float32)

    def grab_image(self):
        """
        grab images from the environment
        returns:
            rgb_agent: np.array, rgb image from the agent's view
            rgb_ego: np.array, rgb image from the egocentric
        """
        self.rgb_agent = self.env.get_fixed_cam_rgb(
            cam_name='agentview')
        self.rgb_ego = self.env.get_fixed_cam_rgb(
            cam_name='egocentric')
        # self.rgb_top = self.env.get_fixed_cam_rgbd_pcd(
        #     cam_name='topview')
        self.rgb_side = self.env.get_fixed_cam_rgb(
            cam_name='sideview')
        return self.rgb_agent, self.rgb_ego

    def render(self, teleop=False, idx=0):
        """
        Render the environment
        """
        # First grab the latest images
        self.grab_image()
        
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

    def close(self):
        """
        Close the environment
        """
        # MuJoCoParserClass doesn't have a close method
        # Just clean up any resources if needed
        pass

    def get_random_position_info(self):
        """
        获取当前随机位置生成的信息
        """
        if not self.random_block_position:
            return "Random position generation is disabled"
        
        info = {
            'enabled': self.random_block_position,
            'sampling_method': self.sampling_method,
            'plate_flat_radius': self.plate_flat_radius,
            'block_half_size': self.block_half_size,
            'block_corner_distance': self.block_corner_distance,
            'safe_generation_radius': self.safe_generation_radius,
        }
        
        # 添加拉丁超立方采样器信息
        if self.sampling_method == 'latin_hypercube':
            info['lhs_samples_total'] = len(self.lhs_samples) if self.lhs_samples is not None else 0
            info['lhs_samples_used'] = self.lhs_index
            info['lhs_samples_remaining'] = max(0, len(self.lhs_samples) - self.lhs_index) if self.lhs_samples is not None else 0
        
        return info

# 便捷函数，用于快速创建环境
def create_collect_dataset_env(xml_path, random_block=True, seed=None, sampling_method='uniform'):
    """
    便捷函数：创建数据收集环境
    
    Args:
        xml_path: XML文件路径
        random_block: 是否启用随机block位置
        seed: 随机种子
        sampling_method: 采样方法 ('uniform' 或 'latin_hypercube')
        
    Returns:
        SimpleEnvRemoveBlockCollectDataset: 环境实例
    """
    return SimpleEnvRemoveBlockCollectDataset(
        xml_path=xml_path,
        seed=seed,
        random_block_position=random_block,
        plate_flat_radius=0.102,  # 基于用户精确分析
        sampling_method=sampling_method
    )

# 测试函数
def test_random_positions(xml_path, num_tests=10, sampling_method='uniform'):
    """
    测试随机位置生成
    
    Args:
        xml_path: XML文件路径
        num_tests: 测试次数
        sampling_method: 采样方法
    """
    print(f"测试随机block位置生成 - {sampling_method}采样...")
    
    env = create_collect_dataset_env(xml_path, random_block=True, seed=None, sampling_method=sampling_method)
    
    print(f"环境信息:")
    info = env.get_random_position_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print(f"\n生成{num_tests}个随机位置:")
    positions = []
    
    for i in range(num_tests):
        env.reset()
        block_pose, _, _ = env.get_obj_pose()
        positions.append(block_pose[:3])
        print(f"  位置 {i+1}: [{block_pose[0]:.3f}, {block_pose[1]:.3f}, {block_pose[2]:.3f}]")
    
    # 分析位置分布
    positions = np.array(positions)
    print(f"\n位置分布统计:")
    print(f"  X范围: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
    print(f"  Y范围: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
    print(f"  Z范围: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")
    
    # 计算到plate中心的距离
    plate_center = np.array([0.3, -0.25])
    distances = np.linalg.norm(positions[:, :2] - plate_center, axis=1)
    print(f"  到plate中心距离: [{distances.min():.3f}, {distances.max():.3f}]")
    print(f"  平均距离: {distances.mean():.3f}")
    
    env.close()
    return positions

def compare_sampling_methods(xml_path, num_tests=20):
    """
    比较不同采样方法的效果
    
    Args:
        xml_path: XML文件路径
        num_tests: 每种方法的测试次数
    """
    print("=" * 60)
    print("比较不同采样方法")
    print("=" * 60)
    
    methods = ['uniform', 'latin_hypercube']
    results = {}
    
    for method in methods:
        print(f"\n🔍 测试 {method} 采样方法:")
        print("-" * 40)
        positions = test_random_positions(xml_path, num_tests, method)
        results[method] = positions
        
        # 计算空间分布均匀性
        plate_center = np.array([0.3, -0.25])
        distances = np.linalg.norm(positions[:, :2] - plate_center, axis=1)
        
        print(f"\n📊 {method} 采样统计:")
        print(f"  距离标准差: {distances.std():.4f}")
        print(f"  距离变异系数: {distances.std()/distances.mean():.4f}")
    
    print("\n" + "=" * 60)
    print("采样方法比较完成")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    # 测试代码
    xml_path = './asset/scene_remove_block.xml'
    
    print("🧪 测试不同采样方法")
    print("=" * 50)
    
    # 测试uniform采样
    print("\n1️⃣ 测试uniform采样:")
    test_random_positions(xml_path, num_tests=5, sampling_method='uniform')
    
    print("\n2️⃣ 测试拉丁超立方采样:")
    test_random_positions(xml_path, num_tests=5, sampling_method='latin_hypercube')
    
    print("\n3️⃣ 比较两种采样方法:")
    compare_sampling_methods(xml_path, num_tests=10)