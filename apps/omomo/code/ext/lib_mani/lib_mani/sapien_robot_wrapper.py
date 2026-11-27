import torch

class RobotKinematicsConverter:
    """
    Fast robot kinematics converter that pre-computes all expensive joint pattern analysis.
    
    This class moves all the expensive testing and analysis from the conversion functions
    to the initialization phase, making the actual conversions super fast.
    """
    
    def __init__(self, robot, test_delta=0.001):
        """
        Initialize the converter by pre-computing all joint patterns and reference data.
        
        Args:
            robot: SAPIEN robot object
            test_delta: Small value for joint motion testing during analysis
        """
        self.robot = robot
        self.test_delta = test_delta
        
        # Get robot structure
        self.links = robot.get_links()
        self.joints = robot.get_active_joints()
        self.num_joints = len(self.joints)
        self.num_links = len(self.links)
        
        print(f"🔧 Initializing FastRobotKinematicsConverter...")
        print(f"   Robot: {robot.name}")
        print(f"   Active joints: {self.num_joints}")
        print(f"   Links: {self.num_links}")
        
        # Pre-compute all expensive data
        self._precompute_joint_patterns()
        print(f"✅ FastRobotKinematicsConverter ready!")
        print(f"   Pre-computed patterns for {len(self.joint_patterns)} joints")
        
    def _precompute_joint_patterns(self):
        """Pre-compute joint motion patterns and reference poses."""
        
        # Get zero position reference (this is expensive, do once)
        zero_state = torch.zeros(self.num_joints, dtype=torch.float32)
        self.robot.set_qpos(zero_state.unsqueeze(0))
        self.zero_link_poses = self._get_link_poses()
        
        # Pre-compute joint patterns
        self.joint_patterns = {}
        self.joint_parent_child_mapping = {}
        
        for joint_idx in range(self.num_joints):
            joint = self.joints[joint_idx]
            
            # Get parent and child links
            parent_link = joint.get_parent_link()
            child_link = joint.get_child_link()
            
            # Find parent/child indices
            parent_idx = child_idx = None
            for link_idx, link in enumerate(self.links):
                if link == parent_link:
                    parent_idx = link_idx
                if link == child_link:
                    child_idx = link_idx
            
            if parent_idx is None or child_idx is None:
                continue
                
            # Store the mapping
            self.joint_parent_child_mapping[joint_idx] = (parent_idx, child_idx)
            
            # Test small positive motion to understand joint's motion pattern
            test_qpos = torch.zeros(self.num_joints)
            test_qpos[joint_idx] = self.test_delta
            self.robot.set_qpos(test_qpos.unsqueeze(0))
            test_link_poses = self._get_link_poses()
            
            # Compute how this joint's motion affects the relative transformation
            T_rel_zero = torch.linalg.inv(self.zero_link_poses[parent_idx]) @ self.zero_link_poses[child_idx]
            T_rel_test = torch.linalg.inv(test_link_poses[parent_idx]) @ test_link_poses[child_idx]
            T_motion = torch.linalg.inv(T_rel_zero) @ T_rel_test
            
            # Record the motion pattern for this joint
            joint_type = joint.type[0] if isinstance(joint.type, list) else joint.type
            if joint_type in ['revolute', 'revolute_unwrapped']:
                # For revolute joints: record rotation angle per unit qpos
                angle = torch.atan2(T_motion[1, 0], T_motion[0, 0])
                self.joint_patterns[joint_idx] = ('rotation', angle / self.test_delta)
            elif joint_type == 'prismatic':
                # For prismatic joints: record translation per unit qpos
                trans = T_motion[:3, 3]
                abs_trans = torch.abs(trans)
                dominant_axis = torch.argmax(abs_trans).item()
                motion_per_qpos = trans[dominant_axis] / self.test_delta
                self.joint_patterns[joint_idx] = ('translation', dominant_axis, motion_per_qpos)
                
        # Reset robot to zero state after analysis
        self.robot.set_qpos(zero_state.unsqueeze(0))
    
    def _get_link_poses(self):
        """Get all link poses as a tensor. Internal helper function."""
        link_list = self.robot.get_links()
        T_list = [lk.pose.to_transformation_matrix() for lk in link_list]
        T_list = torch.stack(T_list, dim=0)
        # Remove batch dimension if present
        if T_list.dim() == 4 and T_list.shape[1] == 1:
            T_list = T_list.squeeze(1)
        return T_list
    
    def qpos_to_link_poses(self, qpos):
        """
        Fast conversion from joint positions to link poses.
        
        Args:
            qpos: Joint positions (tensor or array)
        
        Returns:
            torch.Tensor: Link poses as (num_links, 4, 4) transformation matrices
        """
        state_tensor = torch.as_tensor(qpos, dtype=torch.float32)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        self.robot.set_qpos(state_tensor)
        return self._get_link_poses()
    
    def get_robot_root_pose(self, qpos):
        """Get the root link pose from the robot."""
        state_tensor = torch.as_tensor(qpos, dtype=torch.float32)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        self.robot.set_qpos(state_tensor)
        return self.robot.root_pose.to_transformation_matrix()
    
    def link_poses_to_qpos(self, link_poses, device=None, dtype=None, verbose=False):
        """
        Super-fast conversion from link poses to joint positions using pre-computed patterns.
        
        Args:
            link_poses: Link poses as (num_links, 4, 4) transformation matrices
            device: Target device for output tensor
            dtype: Target dtype for output tensor  
            verbose: Print debug information
        
        Returns:
            torch.Tensor: Joint positions
        """
        # Handle input
        link_Ts = torch.as_tensor(link_poses, dtype=dtype or torch.float32)
        if device is not None:
            link_Ts = link_Ts.to(device)
            
        # Handle batch dimension if present
        if link_Ts.dim() == 4 and link_Ts.shape[1] == 1:
            link_Ts = link_Ts.squeeze(1)
        
        # Initialize output
        qpos = torch.zeros(self.num_joints, dtype=link_Ts.dtype, device=link_Ts.device)
        
        if verbose:
            print(f"Converting {len(link_Ts)} link poses to qpos...")
        
        # Use pre-computed patterns for super-fast conversion
        for joint_idx, pattern in self.joint_patterns.items():
            if joint_idx not in self.joint_parent_child_mapping:
                continue
                
            parent_idx, child_idx = self.joint_parent_child_mapping[joint_idx]
            
            # Get reference poses (move to same device/dtype as input)
            zero_parent = self.zero_link_poses[parent_idx].to(dtype=link_Ts.dtype, device=link_Ts.device)
            zero_child = self.zero_link_poses[child_idx].to(dtype=link_Ts.dtype, device=link_Ts.device)
            
            # Compute actual motion from zero to target pose
            T_rel_zero = torch.linalg.inv(zero_parent) @ zero_child
            T_rel_actual = torch.linalg.inv(link_Ts[parent_idx]) @ link_Ts[child_idx]
            T_motion_actual = torch.linalg.inv(T_rel_zero) @ T_rel_actual
            
            # Recover qpos using pre-computed pattern
            if pattern[0] == 'rotation':
                # For revolute joints
                angle_actual = torch.atan2(T_motion_actual[1, 0], T_motion_actual[0, 0])
                motion_per_qpos = pattern[1]
                if isinstance(motion_per_qpos, torch.Tensor):
                    motion_per_qpos = motion_per_qpos.to(dtype=link_Ts.dtype, device=link_Ts.device)
                qpos[joint_idx] = angle_actual / motion_per_qpos
                
            elif pattern[0] == 'translation':
                # For prismatic joints
                axis = pattern[1]
                motion_per_qpos = pattern[2]
                if isinstance(motion_per_qpos, torch.Tensor):
                    motion_per_qpos = motion_per_qpos.to(dtype=link_Ts.dtype, device=link_Ts.device)
                trans_actual = T_motion_actual[:3, 3]
                qpos[joint_idx] = trans_actual[axis] / motion_per_qpos
            
            if verbose:
                joint_type = pattern[0]
                print(f"Joint {joint_idx} ({joint_type}): parent={parent_idx}, child={child_idx}, value={qpos[joint_idx].item():.4f}")
        
        if verbose:
            print(f"Computed qpos: {qpos}")
            
        return qpos
    
    def __repr__(self):
        return f"FastRobotKinematicsConverter(robot={self.robot.name}, joints={self.num_joints}, links={self.num_links})"