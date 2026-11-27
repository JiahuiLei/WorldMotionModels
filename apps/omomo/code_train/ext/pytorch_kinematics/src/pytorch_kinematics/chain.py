from functools import lru_cache
from typing import Optional, Sequence

import copy
import numpy as np
import time
import torch

import pytorch_kinematics.transforms as tf
from pytorch_kinematics import jacobian
from pytorch_kinematics.frame import Frame, Link, Joint
from pytorch_kinematics.transforms.rotation_conversions import (
    axis_and_angle_to_matrix_44,
    axis_and_d_to_pris_matrix,
)
from tqdm import tqdm


def get_n_joints(th):
    """

    Args:
        th: A dict, list, numpy array, or torch tensor of joints values. Possibly batched

    Returns: The number of joints in the input

    """
    if isinstance(th, torch.Tensor) or isinstance(th, np.ndarray):
        return th.shape[-1]
    elif isinstance(th, list) or isinstance(th, dict):
        return len(th)
    else:
        raise NotImplementedError(f"Unsupported type {type(th)}")


def get_batch_size(th):
    if isinstance(th, torch.Tensor) or isinstance(th, np.ndarray):
        return th.shape[0]
    elif isinstance(th, dict):
        elem_shape = get_dict_elem_shape(th)
        return elem_shape[0]
    elif isinstance(th, list):
        # Lists cannot be batched. We don't allow lists of lists.
        return 1
    else:
        raise NotImplementedError(f"Unsupported type {type(th)}")


def ensure_2d_tensor(th, dtype, device):
    if not torch.is_tensor(th):
        th = torch.tensor(th, dtype=dtype, device=device)
    if len(th.shape) <= 1:
        N = 1
        th = th.reshape(1, -1)
    else:
        N = th.shape[0]
    return th, N


def get_dict_elem_shape(th_dict):
    elem = th_dict[list(th_dict.keys())[0]]
    if isinstance(elem, np.ndarray):
        return elem.shape
    elif isinstance(elem, torch.Tensor):
        return elem.shape
    else:
        return ()


class Chain:
    """
    Robot model that may be constructed from different descriptions via their respective parsers.
    Fundamentally, a robot is modelled as a chain (not necessarily serial) of frames, with each frame
    having a physical link and a number of child frames each connected via some joint.
    """

    def __init__(self, root_frame, dtype=torch.float32, device="cpu"):
        print("DEBUG: CUSTOMIZED VERSION")
        self._root = root_frame
        self.dtype = dtype
        self.device = device

        self.identity = torch.eye(4, device=self.device, dtype=self.dtype).unsqueeze(0)

        low, high = self.get_joint_limits()
        self.low = torch.tensor(low, device=self.device, dtype=self.dtype)
        self.high = torch.tensor(high, device=self.device, dtype=self.dtype)

        # As we traverse the kinematic tree, each frame is assigned an index.
        # We use this index to build a flat representation of the tree.
        # parents_indices and joint_indices all use this indexing scheme.
        # The root frame will be index 0 and the first frame of the root frame's children will be index 1,
        # then the child of that frame will be index 2, etc. In other words, it's a depth-first ordering.
        self.parents_indices = []  # list of indices from 0 (root) to the given frame
        self.joint_indices = []
        self.n_joints = len(self.get_joint_parameter_names())
        self.axes = torch.zeros(
            [self.n_joints, 3], dtype=self.dtype, device=self.device
        )
        self.link_offsets = []
        self.joint_offsets = []
        self.joint_type_indices = []
        queue = []
        queue.insert(-1, (self._root, -1, 0))  # the root has no parent so we use -1.
        idx = 0
        self.frame_to_idx = {}
        self.idx_to_frame = {}
        while len(queue) > 0:
            root, parent_idx, depth = queue.pop(0)
            name_strip = root.name.strip("\n")
            self.frame_to_idx[name_strip] = idx
            self.idx_to_frame[idx] = name_strip
            if parent_idx == -1:
                self.parents_indices.append([idx])
            else:
                self.parents_indices.append(self.parents_indices[parent_idx] + [idx])

            is_fixed = root.joint.joint_type == "fixed"

            if root.link.offset is None:
                self.link_offsets.append(None)
            else:
                self.link_offsets.append(root.link.offset.get_matrix())

            if root.joint.offset is None:
                self.joint_offsets.append(None)
            else:
                self.joint_offsets.append(root.joint.offset.get_matrix())

            if is_fixed:
                self.joint_indices.append(-1)
            else:
                jnt_idx = self.get_joint_parameter_names().index(root.joint.name)
                self.axes[jnt_idx] = root.joint.axis
                self.joint_indices.append(jnt_idx)

            # these are integers so that we can use them as indices into tensors
            # FIXME: how do we know the order of these types in C++?
            self.joint_type_indices.append(Joint.TYPES.index(root.joint.joint_type))

            for child in root.children:
                queue.append((child, idx, depth + 1))

            idx += 1

        self._rk_plan_cache = {}
        self.joint_type_indices = torch.tensor(self.joint_type_indices)
        self.joint_indices = torch.tensor(self.joint_indices)
        # We need to use a dict because torch.compile doesn't list lists of tensors
        self.parents_indices = [
            torch.tensor(p, dtype=torch.long, device=self.device)
            for p in self.parents_indices
        ]

    def to(self, dtype=None, device=None):
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
        self._root = self._root.to(dtype=self.dtype, device=self.device)

        self.identity = self.identity.to(device=self.device, dtype=self.dtype)
        self.parents_indices = [
            p.to(dtype=torch.long, device=self.device) for p in self.parents_indices
        ]
        self.joint_type_indices = self.joint_type_indices.to(
            dtype=torch.long, device=self.device
        )
        self.joint_indices = self.joint_indices.to(dtype=torch.long, device=self.device)
        self.axes = self.axes.to(dtype=self.dtype, device=self.device)
        self.link_offsets = [
            l if l is None else l.to(dtype=self.dtype, device=self.device)
            for l in self.link_offsets
        ]
        self.joint_offsets = [
            j if j is None else j.to(dtype=self.dtype, device=self.device)
            for j in self.joint_offsets
        ]
        self.low = self.low.to(dtype=self.dtype, device=self.device)
        self.high = self.high.to(dtype=self.dtype, device=self.device)

        return self

    def __str__(self):
        return str(self._root)

    @staticmethod
    def _find_frame_recursive(name, frame: Frame) -> Optional[Frame]:
        for child in frame.children:
            if child.name == name:
                return child
            ret = Chain._find_frame_recursive(name, child)
            if not ret is None:
                return ret
        return None

    def find_frame(self, name) -> Optional[Frame]:
        if self._root.name == name:
            return self._root
        return self._find_frame_recursive(name, self._root)

    @staticmethod
    def _find_link_recursive(name, frame) -> Optional[Link]:
        for child in frame.children:
            if child.link.name == name:
                return child.link
            ret = Chain._find_link_recursive(name, child)
            if not ret is None:
                return ret
        return None

    @staticmethod
    def _get_joints(frame, exclude_fixed=True):
        joints = []
        if exclude_fixed and frame.joint.joint_type != "fixed":
            joints.append(frame.joint)
        for child in frame.children:
            joints.extend(Chain._get_joints(child))
        return joints

    def get_joints(self, exclude_fixed=True):
        joints = self._get_joints(self._root, exclude_fixed=exclude_fixed)
        return joints

    @lru_cache()
    def get_joint_parameter_names(self, exclude_fixed=True):
        names = []
        for j in self.get_joints(exclude_fixed=exclude_fixed):
            if exclude_fixed and j.joint_type == "fixed":
                continue
            names.append(j.name)
        return names

    @staticmethod
    def _find_joint_recursive(name, frame):
        for child in frame.children:
            if child.joint.name == name:
                return child.joint
            ret = Chain._find_joint_recursive(name, child)
            if not ret is None:
                return ret
        return None

    def find_link(self, name) -> Optional[Link]:
        if self._root.link.name == name:
            return self._root.link
        return self._find_link_recursive(name, self._root)

    def find_joint(self, name):
        if self._root.joint.name == name:
            return self._root.joint
        return self._find_joint_recursive(name, self._root)

    @staticmethod
    def _get_joint_parent_frame_names(frame, exclude_fixed=True):
        joint_names = []
        if not (exclude_fixed and frame.joint.joint_type == "fixed"):
            joint_names.append(frame.name)
        for child in frame.children:
            joint_names.extend(
                Chain._get_joint_parent_frame_names(child, exclude_fixed)
            )
        return joint_names

    def get_joint_parent_frame_names(self, exclude_fixed=True):
        names = self._get_joint_parent_frame_names(self._root, exclude_fixed)
        return sorted(set(names), key=names.index)

    @staticmethod
    def _get_frame_names(frame: Frame, exclude_fixed=True) -> Sequence[str]:
        names = []
        if not (exclude_fixed and frame.joint.joint_type == "fixed"):
            names.append(frame.name)
        for child in frame.children:
            names.extend(Chain._get_frame_names(child, exclude_fixed))
        return names

    def get_frame_names(self, exclude_fixed=True):
        names = self._get_frame_names(self._root, exclude_fixed)
        return sorted(set(names), key=names.index)

    @staticmethod
    def _get_links(frame):
        links = [frame.link]
        for child in frame.children:
            links.extend(Chain._get_links(child))
        return links

    def get_links(self):
        links = self._get_links(self._root)
        return links

    @staticmethod
    def _get_link_names(frame):
        link_names = [frame.link.name]
        for child in frame.children:
            link_names.extend(Chain._get_link_names(child))
        return link_names

    def get_link_names(self):
        names = self._get_link_names(self._root)
        return sorted(set(names), key=names.index)

    @lru_cache
    def get_frame_indices(self, *frame_names):
        return torch.tensor(
            [self.frame_to_idx[n] for n in frame_names],
            dtype=torch.long,
            device=self.device,
        )

    def print_tree(self, do_print=True):
        tree = str(self._root)
        if do_print:
            print(tree)
        return tree

    def forward_kinematics(self, th, frame_indices: Optional = None):
        """
        Compute forward kinematics for the given joint values.

        Args:
            th: A dict, list, numpy array, or torch tensor of joints values. Possibly batched.
            frame_indices: A list of frame indices to compute transforms for. If None, all frames are computed.
                Use `get_frame_indices` to convert from frame names to frame indices.

        Returns:
            A dict of Transform3d objects for each frame.

        """
        if frame_indices is None:
            frame_indices = self.get_all_frame_indices()

        th = self.ensure_tensor(th)
        th = torch.atleast_2d(th)

        b = th.shape[0]
        axes_expanded = self.axes.unsqueeze(0).repeat(b, 1, 1)

        # compute all joint transforms at once first
        # in order to handle multiple joint types without branching, we create all possible transforms
        # for all joint types and then select the appropriate one for each joint.
        rev_jnt_transform = axis_and_angle_to_matrix_44(axes_expanded, th)
        pris_jnt_transform = axis_and_d_to_pris_matrix(axes_expanded, th)

        frame_transforms = {}
        b = th.shape[0]
        for frame_idx in frame_indices:
            frame_transform = torch.eye(4).to(th).unsqueeze(0).repeat(b, 1, 1)

            # iterate down the list and compose the transform
            for chain_idx in self.parents_indices[frame_idx.item()]:
                if chain_idx.item() in frame_transforms:
                    frame_transform = frame_transforms[chain_idx.item()]
                else:
                    link_offset_i = self.link_offsets[chain_idx]
                    if link_offset_i is not None:
                        frame_transform = frame_transform @ link_offset_i

                    joint_offset_i = self.joint_offsets[chain_idx]
                    if joint_offset_i is not None:
                        frame_transform = frame_transform @ joint_offset_i

                    jnt_idx = self.joint_indices[chain_idx]
                    jnt_type = self.joint_type_indices[chain_idx]
                    if jnt_type == 0:
                        pass
                    elif jnt_type == 1:
                        jnt_transform_i = rev_jnt_transform[:, jnt_idx]
                        frame_transform = frame_transform @ jnt_transform_i
                    elif jnt_type == 2:
                        jnt_transform_i = pris_jnt_transform[:, jnt_idx]
                        frame_transform = frame_transform @ jnt_transform_i

            frame_transforms[frame_idx.item()] = frame_transform

        frame_names_and_transform3ds = {
            self.idx_to_frame[frame_idx]: tf.Transform3d(matrix=transform)
            for frame_idx, transform in frame_transforms.items()
        }

        return frame_names_and_transform3ds

    def reverse_kinematics_v2(self, frame_names_and_transform3ds):
        """
        Compute reverse kinematics for the given frame names and transform3ds.

        Given a dictionary of frame names to Transform3d objects, this function computes
        the joint angles that would produce those transforms. This is the inverse operation
        of forward_kinematics.

        Args:
            frame_names_and_transform3ds: A dict mapping frame names to Transform3d objects

        Returns:
            th: A tensor of joint values that would produce the given transforms
        """
        # Initialize joint values tensor
        first_transform = list(frame_names_and_transform3ds.values())[0]
        if isinstance(first_transform, torch.Tensor):
            b = first_transform.shape[0]
        else:
            b = len(first_transform)

        th = torch.zeros([b, self.n_joints], dtype=self.dtype, device=self.device)

        # Convert frame names to indices
        frame_transforms = {}
        for frame_name, transform3d in frame_names_and_transform3ds.items():
            frame_idx = self.frame_to_idx[frame_name]
            if hasattr(transform3d, "get_matrix"):
                frame_transforms[frame_idx] = transform3d.get_matrix()
            else:
                frame_transforms[frame_idx] = transform3d

        # Cache for computed link transforms (frame_idx -> Tensor [b, 4, 4])
        # We start with the root (identity) implicitly handled.
        # But we can cache explicit indices.
        computed_transforms = {}

        # Process each frame in order to extract joint values
        # We assume keys are topologically sorted (by design of URDF/indices)
        for frame_idx in sorted(frame_transforms.keys()):
            # Get the parent chain indices (path from root to current frame)
            parent_indices = self.parents_indices[frame_idx]

            # Find the furthest ancestor that is already cached
            # Parent indices are [root, ..., parent, current]
            # We want to find T_{parent}.

            # We construct the path up to the parent of the current frame
            path_up_to_parent = parent_indices[:-1]

            # Find start point from cache
            curr_transform = (
                torch.eye(4, dtype=self.dtype, device=self.device)
                .unsqueeze(0)
                .repeat(b, 1, 1)
            )
            start_k = 0

            # Optimization: Check cache from the end of the path backwards
            for k in range(len(path_up_to_parent) - 1, -1, -1):
                ancestor_idx = path_up_to_parent[k]
                if ancestor_idx in computed_transforms:
                    curr_transform = computed_transforms[ancestor_idx]
                    start_k = k + 1
                    break

            # Walk forward from the cached ancestor to the immediate parent
            for k in range(start_k, len(path_up_to_parent)):
                chain_idx = path_up_to_parent[k]

                # Apply offsets/joints to get T_{this_node} from T_{parent_of_this_node}
                # Note: chain_idx here is the node we are computing the transform FOR.
                # In the original code:
                # for chain_idx in parent_indices[:-1]: ... parent_transform @= offsets
                # The offsets associated with `chain_idx` move us from `chain_idx`'s parent to `chain_idx`.

                link_offset_i = self.link_offsets[chain_idx]
                if link_offset_i is not None:
                    curr_transform = curr_transform @ link_offset_i

                joint_offset_i = self.joint_offsets[chain_idx]
                if joint_offset_i is not None:
                    curr_transform = curr_transform @ joint_offset_i

                jnt_idx = self.joint_indices[chain_idx]
                jnt_type = self.joint_type_indices[chain_idx]

                if jnt_type == 1:  # revolute
                    # Use the already computed joint value (th should be populated if it was targeted)
                    axis = self.axes[jnt_idx].unsqueeze(0).repeat(b, 1)
                    angle = th[:, jnt_idx]
                    jnt_transform = axis_and_angle_to_matrix_44(
                        axis.unsqueeze(1), angle.unsqueeze(1)
                    ).squeeze(1)
                    curr_transform = curr_transform @ jnt_transform
                elif jnt_type == 2:  # prismatic
                    axis = self.axes[jnt_idx].unsqueeze(0).repeat(b, 1)
                    d = th[:, jnt_idx]
                    jnt_transform = axis_and_d_to_pris_matrix(
                        axis.unsqueeze(1), d.unsqueeze(1)
                    ).squeeze(1)
                    curr_transform = curr_transform @ jnt_transform

                # Cache this ancestor's transform
                computed_transforms[chain_idx] = curr_transform

            # Now curr_transform is T_{parent}
            parent_transform = curr_transform

            # Current frame kinematics setup
            current_chain_idx = parent_indices[-1]  # This is frame_idx

            current_local_transform = (
                torch.eye(4, dtype=self.dtype, device=self.device)
                .unsqueeze(0)
                .repeat(b, 1, 1)
            )

            link_offset_i = self.link_offsets[current_chain_idx]
            if link_offset_i is not None:
                current_local_transform = current_local_transform @ link_offset_i

            joint_offset_i = self.joint_offsets[current_chain_idx]
            if joint_offset_i is not None:
                current_local_transform = current_local_transform @ joint_offset_i

            # Target transform
            target_transform = frame_transforms[frame_idx]

            # Solve for joint transform: T_parent @ T_local_offset @ T_joint = T_target
            # T_joint = (T_parent @ T_local_offset)^-1 @ T_target
            parent_offset = parent_transform @ current_local_transform
            parent_offset_inv = torch.inverse(parent_offset)
            joint_transform = parent_offset_inv @ target_transform

            # Extract joint value based on joint type
            jnt_idx = self.joint_indices[current_chain_idx]
            jnt_type = self.joint_type_indices[current_chain_idx]

            if jnt_type == 0:  # fixed
                pass
            elif jnt_type == 1:  # revolute
                R = joint_transform[:, :3, :3]
                axis = self.axes[jnt_idx]

                trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
                cos_theta = (trace - 1) / 2
                cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

                R_minus_RT = R - R.transpose(-2, -1)
                skew_x = R_minus_RT[:, 2, 1]
                skew_y = R_minus_RT[:, 0, 2]
                skew_z = R_minus_RT[:, 1, 0]

                sin_theta = (skew_x * axis[0] + skew_y * axis[1] + skew_z * axis[2]) / 2
                theta = torch.atan2(sin_theta, cos_theta)
                th[:, jnt_idx] = theta

                # Update T_current with the solved joint angle and cache it
                # T_current = T_parent @ T_local_offset @ T_joint_solved

                # Recompute joint transform from theta
                axis_batch = axis.unsqueeze(0).repeat(b, 1)
                jnt_transform_solved = axis_and_angle_to_matrix_44(
                    axis_batch.unsqueeze(1), theta.unsqueeze(1)
                ).squeeze(1)

                computed_transforms[frame_idx] = parent_offset @ jnt_transform_solved

            elif jnt_type == 2:  # prismatic
                translation = joint_transform[:, :3, 3]
                axis = self.axes[jnt_idx]
                d = (translation * axis).sum(dim=-1)
                th[:, jnt_idx] = d

                # Update T_current
                axis_batch = axis.unsqueeze(0).repeat(b, 1)
                jnt_transform_solved = axis_and_d_to_pris_matrix(
                    axis_batch.unsqueeze(1), d.unsqueeze(1)
                ).squeeze(1)

                computed_transforms[frame_idx] = parent_offset @ jnt_transform_solved
            else:
                # Fixed joint, T_current = parent_offset
                computed_transforms[frame_idx] = parent_offset

        return th

    def reverse_kinematics(self, frame_names_and_transform3ds):
        """
        Compute reverse kinematics for the given frame names and transform3ds.

        Given a dictionary of frame names to Transform3d objects (or 4x4 matrices),
        this function computes the joint angles that would produce those transforms.

        Args:
            frame_names_and_transform3ds: A dict mapping frame names to Transform3d objects or 4x4 tensors

        Returns:
            th: A tensor of joint values that would produce the given transforms
        """
        # Initialize joint values tensor
        first_transform = list(frame_names_and_transform3ds.values())[0]
        if hasattr(first_transform, "get_matrix"):
            input_tensor = first_transform.get_matrix()
        else:
            input_tensor = first_transform

        b = input_tensor.shape[0]
        input_dtype = input_tensor.dtype
        compute_dtype = torch.float64

        th = torch.zeros([b, self.n_joints], dtype=compute_dtype, device=self.device)

        # Convert frame names to indices and cache flattened matrices
        frame_transforms = {}
        for frame_name, transform in frame_names_and_transform3ds.items():
            frame_idx = self.frame_to_idx[frame_name]
            if hasattr(transform, "get_matrix"):
                # Handle pure Transform3d or wrapped
                val = transform.get_matrix()
            else:
                val = transform
            frame_transforms[frame_idx] = val.to(dtype=compute_dtype)

        # Cache for computed link transforms (frame_idx -> Tensor [b, 4, 4])
        computed_transforms = {}

        # Process each frame in order
        for frame_idx in sorted(frame_transforms.keys()):
            # Get the parent chain indices (path from root to current frame inclusive)
            parent_indices = self.parents_indices[frame_idx]
            path_up_to_parent = parent_indices[:-1]

            # Initialize current transform (parent of current frame)
            # Try to find the furthest ancestor that is already cached
            curr_transform = (
                torch.eye(4, dtype=compute_dtype, device=self.device)
                .unsqueeze(0)
                .repeat(b, 1, 1)
            )
            start_k = 0

            # Optimization: Check cache from backward
            for k in range(len(path_up_to_parent) - 1, -1, -1):
                ancestor_idx = path_up_to_parent[k]
                if ancestor_idx in computed_transforms:
                    curr_transform = computed_transforms[ancestor_idx]
                    start_k = k + 1
                    break

            # Walk forward from the cached ancestor
            for k in range(start_k, len(path_up_to_parent)):
                chain_idx = path_up_to_parent[k]

                # Apply offsets
                link_offset_i = self.link_offsets[chain_idx]
                if link_offset_i is not None:
                    curr_transform = curr_transform @ link_offset_i.to(
                        dtype=compute_dtype
                    )

                joint_offset_i = self.joint_offsets[chain_idx]
                if joint_offset_i is not None:
                    curr_transform = curr_transform @ joint_offset_i.to(
                        dtype=compute_dtype
                    )

                jnt_idx = self.joint_indices[chain_idx]
                jnt_type = self.joint_type_indices[chain_idx]

                if jnt_type == 1:  # revolute
                    axis = (
                        self.axes[jnt_idx]
                        .to(dtype=compute_dtype)
                        .unsqueeze(0)
                        .repeat(b, 1)
                    )
                    angle = th[:, jnt_idx]
                    jnt_transform = axis_and_angle_to_matrix_44(
                        axis.unsqueeze(1), angle.unsqueeze(1)
                    ).squeeze(1)
                    curr_transform = curr_transform @ jnt_transform
                elif jnt_type == 2:  # prismatic
                    axis = (
                        self.axes[jnt_idx]
                        .to(dtype=compute_dtype)
                        .unsqueeze(0)
                        .repeat(b, 1)
                    )
                    d = th[:, jnt_idx]
                    jnt_transform = axis_and_d_to_pris_matrix(
                        axis.unsqueeze(1), d.unsqueeze(1)
                    ).squeeze(1)
                    curr_transform = curr_transform @ jnt_transform

                # Cache this ancestor
                computed_transforms[chain_idx] = curr_transform

            # Now curr_transform is parent_transform
            parent_transform = curr_transform

            # Current frame setup
            current_chain_idx = parent_indices[-1]  # == frame_idx

            current_local_transform = (
                torch.eye(4, dtype=compute_dtype, device=self.device)
                .unsqueeze(0)
                .repeat(b, 1, 1)
            )

            link_offset_i = self.link_offsets[current_chain_idx]
            if link_offset_i is not None:
                current_local_transform = current_local_transform @ link_offset_i.to(
                    dtype=compute_dtype
                )

            joint_offset_i = self.joint_offsets[current_chain_idx]
            if joint_offset_i is not None:
                current_local_transform = current_local_transform @ joint_offset_i.to(
                    dtype=compute_dtype
                )

            target_transform = frame_transforms[frame_idx]  # already float64

            # Solve T_joint = (T_parent @ T_local_offset)^-1 @ T_target
            parent_offset = parent_transform @ current_local_transform
            parent_offset_inv = torch.inverse(parent_offset)
            joint_transform = parent_offset_inv @ target_transform

            # Extract joint value
            jnt_idx = self.joint_indices[current_chain_idx]
            jnt_type = self.joint_type_indices[current_chain_idx]

            if jnt_type == 1:  # revolute
                R = joint_transform[:, :3, :3]
                axis = self.axes[jnt_idx].to(dtype=compute_dtype)

                trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
                cos_theta = (trace - 1) / 2
                cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

                R_minus_RT = R - R.transpose(-2, -1)
                skew_x = R_minus_RT[:, 2, 1]
                skew_y = R_minus_RT[:, 0, 2]
                skew_z = R_minus_RT[:, 1, 0]

                sin_theta = (skew_x * axis[0] + skew_y * axis[1] + skew_z * axis[2]) / 2
                theta = torch.atan2(sin_theta, cos_theta)
                th[:, jnt_idx] = theta

                # Update T_current
                axis_batch = axis.unsqueeze(0).repeat(b, 1)
                jnt_transform_solved = axis_and_angle_to_matrix_44(
                    axis_batch.unsqueeze(1), theta.unsqueeze(1)
                ).squeeze(1)
                computed_transforms[frame_idx] = parent_offset @ jnt_transform_solved

            elif jnt_type == 2:  # prismatic
                translation = joint_transform[:, :3, 3]
                axis = self.axes[jnt_idx].to(dtype=compute_dtype)
                d = (translation * axis).sum(dim=-1)
                th[:, jnt_idx] = d

                axis_batch = axis.unsqueeze(0).repeat(b, 1)
                jnt_transform_solved = axis_and_d_to_pris_matrix(
                    axis_batch.unsqueeze(1), d.unsqueeze(1)
                ).squeeze(1)
                computed_transforms[frame_idx] = parent_offset @ jnt_transform_solved
            else:
                computed_transforms[frame_idx] = parent_offset

        return th.to(dtype=input_dtype)

    def batch_fk(self, th, key_bodies: list = [], profile_dict=None):
        """
        Clean wrapper for forward_kinematics with pure PyTorch tensor I/O.

        Args:
            th: Joint angles tensor of shape (B, N_joints)
            key_bodies: List of frame/link names to compute transforms for.
                       If empty, returns transforms for all links.
            profile_dict: Optional dict to collect profiling information.
                         If provided, will accumulate timing for:
                         - 'preprocess': tensor preprocessing
                         - 'get_indices': getting frame indices
                         - 'forward_kinematics': FK computation
                         - 'dict_to_tensor': converting dict to tensor

        Returns:
            ret_T: Transformation matrices of shape (B, len(key_bodies), 4, 4)
            key_bodies: List of body names corresponding to the transforms
        """
        profile_enabled = profile_dict is not None
        t_start = 0.0
        t_end = 0.0

        # Ensure th is a tensor
        if profile_enabled:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_start = time.time()

        th = self.ensure_tensor(th)
        th = torch.atleast_2d(th)

        # Determine which bodies to return
        if len(key_bodies) == 0:
            key_bodies = self.get_link_names()

        if profile_enabled:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_end = time.time()
            profile_dict["preprocess"] = profile_dict.get("preprocess", 0.0) + (
                t_end - t_start
            )

        # Get frame indices for the requested bodies
        if profile_enabled:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_start = time.time()

        frame_indices = self.get_frame_indices(*key_bodies)

        if profile_enabled:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_end = time.time()
            profile_dict["get_indices"] = profile_dict.get("get_indices", 0.0) + (
                t_end - t_start
            )

        # Run forward kinematics
        if profile_enabled:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_start = time.time()

        frame_transforms_dict = self.forward_kinematics(th, frame_indices)

        if profile_enabled:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_end = time.time()
            profile_dict["forward_kinematics"] = profile_dict.get(
                "forward_kinematics", 0.0
            ) + (t_end - t_start)

        # Convert dict to tensor in the order of key_bodies
        if profile_enabled:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_start = time.time()

        B = th.shape[0]
        N = len(key_bodies)
        ret_T = torch.zeros(B, N, 4, 4, dtype=self.dtype, device=self.device)

        for i, body_name in enumerate(key_bodies):
            if body_name in frame_transforms_dict:
                ret_T[:, i] = frame_transforms_dict[body_name].get_matrix()

        if profile_enabled:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_end = time.time()
            profile_dict["dict_to_tensor"] = profile_dict.get("dict_to_tensor", 0.0) + (
                t_end - t_start
            )

        return ret_T, key_bodies

    def batch_finetune(
        self, T, names, th, lrs=[1.0] * 5, profile_flag=False, weights={}
    ):
        # use differentialable FK to optimize the joint angles to match the positional information in the T
        # weights: optional dict mapping name -> weight, if not specified for a name, weight is 1.0
        th = th.clone().detach()
        th.requires_grad = True
        # optimizer = torch.optim.SGD([th], lr=lr)
        gt_position = T[:, :, :3, 3]

        # Build weight tensor from weights dict
        weight_tensor = torch.tensor(
            [weights.get(name, 1.0) for name in names],
            dtype=self.dtype,
            device=self.device,
        )
        weight_tensor = abs(weight_tensor)
        weight_tensor = weight_tensor / weight_tensor.sum()

        # Initialize profiling variables
        total_fk_time = 0.0
        total_backward_time = 0.0
        fk_start = 0.0
        backward_start = 0.0
        fk_profile_dict = {} if profile_flag else None
        loss_list = []

        for step in range(len(lrs)):
            # optimizer.zero_grad()
            if th.grad is not None:
                th.grad.zero_()

            if profile_flag:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                fk_start = time.time()

            T_pred, key_bodies = self.batch_fk(th, names, profile_dict=fk_profile_dict)

            if profile_flag:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                fk_end = time.time()
                total_fk_time += fk_end - fk_start

            pred_position = T_pred[:, :, :3, 3]
            loss_per_name = (pred_position - gt_position).norm(dim=-1)  # Shape: (B, N)
            # Apply weights: broadcast weight_tensor to (1, N) and multiply
            weighted_loss = (loss_per_name * weight_tensor.unsqueeze(0)).sum(dim=-1)
            loss = weighted_loss.mean()

            if profile_flag:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                backward_start = time.time()

            loss.backward()

            if profile_flag:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                backward_end = time.time()
                total_backward_time += backward_end - backward_start

            # optimizer.step()
            th.data -= float(lrs[step]) * th.grad
            # print(f"Step {step}: loss={loss.item():.6f}")
            loss_list.append(loss.item())

        if profile_flag:
            print(f"\nProfiling Results:")
            steps = len(lrs)
            print(
                f"  Total FK time: {total_fk_time:.4f}s (avg: {total_fk_time/steps:.4f}s/step)"
            )
            if fk_profile_dict:
                print(
                    f"    └─ Preprocess: {fk_profile_dict.get('preprocess', 0.0):.4f}s (avg: {fk_profile_dict.get('preprocess', 0.0)/steps:.4f}s/step)"
                )
                print(
                    f"    └─ Get indices: {fk_profile_dict.get('get_indices', 0.0):.4f}s (avg: {fk_profile_dict.get('get_indices', 0.0)/steps:.4f}s/step)"
                )
                print(
                    f"    └─ Forward kinematics: {fk_profile_dict.get('forward_kinematics', 0.0):.4f}s (avg: {fk_profile_dict.get('forward_kinematics', 0.0)/steps:.4f}s/step)"
                )
                print(
                    f"    └─ Dict to tensor: {fk_profile_dict.get('dict_to_tensor', 0.0):.4f}s (avg: {fk_profile_dict.get('dict_to_tensor', 0.0)/steps:.4f}s/step)"
                )
            print(
                f"  Total Backward time: {total_backward_time:.4f}s (avg: {total_backward_time/steps:.4f}s/step)"
            )

        th = th.detach()
        return th, loss_list

    def batch_rk(self, T, names):
        """
        Clean wrapper for reverse_kinematics with pure PyTorch tensor I/O.
        Uses cached execution plan for speed.
        """
        names_tuple = tuple(names)
        if names_tuple in self._rk_plan_cache:
            plan = self._rk_plan_cache[names_tuple]
        else:
            plan = self.compile_rk_plan(names, T.device, torch.float64)
            self._rk_plan_cache[names_tuple] = plan

        return self.execute_rk_plan(T, plan)

    def compile_rk_plan(self, names, device, dtype):
        """Pre-compute indices and offsets for vectorized RK."""
        # Maps name -> index in T
        name_to_idx = {name: i for i, name in enumerate(names)}

        parent_indices_in_T = []  # -1 indicates root (identity)
        child_indices_in_T = []
        th_indices = []
        target_joint_types = []
        target_axes = []
        target_link_offsets = []

        for i, name in enumerate(names):
            if name not in self.frame_to_idx:
                continue
            frame_idx = self.frame_to_idx[name]

            # Parent check
            # Find the nearest ancestor that is in our input 'names' (or is root)
            parent_indices_chain = self.parents_indices[frame_idx]

            ancestor_frame_idx = -1
            found_ancestor_in_names = False

            # Walk up the chain from immediate parent
            # parent_indices_chain ends with current frame, so -2 is immediate parent
            # We iterate backwards from immediate parent
            chain_path = []

            if len(parent_indices_chain) >= 2:
                for p_idx_in_chain in reversed(parent_indices_chain[:-1]):
                    p_iframe = int(p_idx_in_chain)
                    p_name = self.idx_to_frame[p_iframe]

                    if p_name in name_to_idx:
                        # Found an ancestor in the input!
                        ancestor_frame_idx = p_iframe
                        found_ancestor_in_names = True
                        break
                    elif p_iframe == self.frame_to_idx[self._root.name]:
                        # Reached root
                        ancestor_frame_idx = p_iframe
                        # Root is always "available" as identity
                        found_ancestor_in_names = (
                            False  # It's not in names, but it is valid root
                        )
                        break
                    else:
                        # This frame is missing from input, add to path for offset computation
                        chain_path.append(p_iframe)

            # If still -1, it means we couldn't find anything valid in names
            # But check if frame itself is root or if we hit root in the loop above?

            # The loop above sets ancestor_frame_idx IF it hits root.
            # But if len(parent_indices_chain) < 2, loop doesn't run.
            # Usually happens if frame_idx IS root.
            if len(parent_indices_chain) < 2:
                if frame_idx == self.frame_to_idx[self._root.name]:
                    ancestor_frame_idx = frame_idx
                    found_ancestor_in_names = False

            # Determine index in T
            if found_ancestor_in_names:
                p_idx_in_T = name_to_idx[self.idx_to_frame[ancestor_frame_idx]]
            elif ancestor_frame_idx == self.frame_to_idx[self._root.name]:
                p_idx_in_T = -1
            else:
                # Should not happen if root is reachable
                raise ValueError(f"Could not find valid ancestor for {name}")

            parent_indices_in_T.append(p_idx_in_T)
            child_indices_in_T.append(i)

            # Compute composite offset from Ancestor -> Current
            # We need T_ancestor_to_current = (T_ancestor_to_missing @ T_missing_to_missing2 ...)
            # Actually we usually store T_parent_to_child = offset. So T_child = T_parent @ offset.
            # So T_current = T_ancestor @ offset_composite.
            # offset_composite = (L_a @ J_a) @ (L_m @ J_m) ... @ (L_curr @ J_curr)
            # The chain_path contains the MISSING intermediate link indices in reverse order (closest to current first?)
            # Wait, chain_path was built walking UP efficiently:
            # if parent is missing, append to chain_path.
            # So chain_path[0] is immediate parent (if missing), chain_path[1] is grandparent...
            # The order of transforms from Ancestor to Current is:
            # Ancestor -> Missing_Top -> ... -> Missing_Bot -> Current

            # We need to construct the combined offset matrix.
            # Start with Identity.
            # Multiply from Right?
            # T_current = T_parent @ (link_off @ joint_off)

            # Let's rebuild the path from Ancestor DOWN to Current.
            # Ancestor is not in chain_path.
            # chain_path has intermediate missing frames.
            # We also need to process the Current frame itself!

            # Path to process: [Missing_Top, ..., Missing_Bot, Current]
            # chain_path currently has [Missing_Bot, ..., Missing_Top] (because we walked up)

            path_down = list(reversed(chain_path)) + [frame_idx]

            # Compute composite offset
            # We assume missing joints are at 0, so their J matrix is just joint_offset.
            # (Variables are 0).

            composite_transform = torch.eye(4, dtype=dtype, device=device)

            for f_idx in path_down:
                # Get offsets for this frame
                lo = self.link_offsets[f_idx]
                if lo is None:
                    lo = torch.eye(4, dtype=dtype, device=device)
                else:
                    lo = lo.to(dtype=dtype, device=device)
                    if lo.ndim == 3:
                        lo = lo.squeeze(0)

                jo = self.joint_offsets[f_idx]
                if jo is None:
                    jo = torch.eye(4, dtype=dtype, device=device)
                else:
                    jo = jo.to(dtype=dtype, device=device)
                    if jo.ndim == 3:
                        jo = jo.squeeze(0)

                # Combine
                # T_this = T_parent @ (lo @ jo)
                # So we accumulate: T_accum = T_accum @ (lo @ jo)

                step_transform = lo @ jo
                composite_transform = composite_transform @ step_transform

            # Store INVERSE of this composite offset
            # T_joint_val = offset_inv @ T_ancestor_inv @ T_current
            target_link_offsets.append(torch.inverse(composite_transform))

            # Joint info for current frame
            jnt_idx = self.joint_indices[frame_idx]
            th_indices.append(jnt_idx)
            target_joint_types.append(self.joint_type_indices[frame_idx])

            # Axes
            target_axes.append(self.axes[jnt_idx].to(dtype=dtype, device=device))

            # Note: We DON'T add offsets for current here, we already included them in composite!
            # The 'target_link_offsets' list is used for the T_joint calculation.
            # The logic was: local_offset = lo @ jo; target_link_offsets.append(inv(local_offset))
            # Now we use composite.

        # Compile to tensors
        # Cache tensors on device
        if len(th_indices) == 0:
            return None

        plan = {
            "p_idx": torch.tensor(parent_indices_in_T, device=device, dtype=torch.long),
            "c_idx": torch.tensor(child_indices_in_T, device=device, dtype=torch.long),
            "th_idx": torch.tensor(th_indices, device=device, dtype=torch.long),
            "axes": torch.stack(target_axes),
            "offsets": torch.stack(target_link_offsets),
            "types": torch.tensor(target_joint_types, device=device),
            "dtype": dtype,
        }

        # Pre-compute masks for types
        plan["rev_mask"] = plan["types"] == 1
        plan["pris_mask"] = plan["types"] == 2

        # Mask for root/non-root parents
        plan["root_p_mask"] = plan["p_idx"] == -1
        plan["non_root_p_mask"] = ~plan["root_p_mask"]
        plan["valid_p_indices"] = plan["p_idx"][plan["non_root_p_mask"]]

        return plan

    def execute_rk_plan(self, T, plan):
        """Execute pre-computed RK plan."""
        B = T.shape[0]
        device = T.device
        original_dtype = T.dtype

        if plan is None:
            return torch.zeros([B, self.n_joints], dtype=original_dtype, device=device)

        dtype = plan["dtype"]
        T = T.to(dtype=dtype)

        # 1. GatherChild transforms
        T_child = T[:, plan["c_idx"]]  # (B, K, 4, 4)

        # 2. Gather Parent transforms
        # T_parent = torch.zeros_like(T_child) # Allocation overhead?
        # Better: create T_parent initialized with identity for root, then scatter?
        # Or masked fill matches what we did before.

        T_parent = (
            torch.eye(4, dtype=dtype, device=device)
            .unsqueeze(0)
            .expand(B, len(plan["p_idx"]), 4, 4)
            .clone()
        )
        # The above 'expand' creates view, 'clone' makes it contiguous/writable -> slightly slower allocation

        # Just fill non-root parts
        # T_parent[:, plan["non_root_p_mask"]] = T[:, plan["valid_p_indices"]]
        # Using fancy indexing for batch and dim 1
        # plan["valid_p_indices"] has shape (M,) where M is number of non-root parents
        # We need to assign to (B, M, 4, 4) from (B, M, 4, 4)

        if plan["non_root_p_mask"].any():
            T_parent[:, plan["non_root_p_mask"]] = T[:, plan["valid_p_indices"]]

        # 3. Compute relative transform
        T_parent_inv = torch.inverse(T_parent)

        # Expand offsets (1, K, 4, 4) -> (B, K, 4, 4)
        local_offsets_inv_b = plan["offsets"].unsqueeze(0).expand(B, -1, -1, -1)

        T_joint = local_offsets_inv_b @ T_parent_inv @ T_child

        # 4. Extract angles
        th = torch.zeros([B, self.n_joints], dtype=dtype, device=device)

        # Revolute
        if plan["rev_mask"].any():
            R = T_joint[:, plan["rev_mask"], :3, :3]
            axes = plan["axes"][plan["rev_mask"]]

            trace = R[:, :, 0, 0] + R[:, :, 1, 1] + R[:, :, 2, 2]
            cos_theta = (trace - 1) / 2
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

            R_minus_RT = R - R.transpose(-2, -1)
            skew_x = R_minus_RT[:, :, 2, 1]
            skew_y = R_minus_RT[:, :, 0, 2]
            skew_z = R_minus_RT[:, :, 1, 0]

            # axes is (K_rev, 3). Need to broadcast against B?
            # axes.unsqueeze(0) -> (1, K_rev, 3) broadcast ok with (B, K_rev)
            axes_b = axes.unsqueeze(0)
            sin_theta = (
                skew_x * axes_b[..., 0]
                + skew_y * axes_b[..., 1]
                + skew_z * axes_b[..., 2]
            ) / 2

            angles = torch.atan2(sin_theta, cos_theta)

            # Scatter
            # th[:, plan["th_idx"][plan["rev_mask"]]] = angles
            # plan["th_idx"] is (K,). Masked is (K_rev,).
            th[:, plan["th_idx"][plan["rev_mask"]]] = angles

        # Prismatic
        if plan["pris_mask"].any():
            trans = T_joint[:, plan["pris_mask"], :3, 3]
            axes = plan["axes"][plan["pris_mask"]].unsqueeze(0)
            d = (trans * axes).sum(dim=-1)
            th[:, plan["th_idx"][plan["pris_mask"]]] = d

        return th.to(dtype=original_dtype)

    def get_important_link_names(self):
        """
        Returns the frame names that are necessary to compute reverse kinematics.
        These are frames with non-fixed joints (revolute or prismatic) that contribute
        to the joint configuration. Frames with fixed joints or dummy links are excluded.

        Returns:
            List of frame names corresponding to frames with non-fixed joints
        """
        important_names = []
        for idx in range(len(self.joint_type_indices)):
            jnt_type = self.joint_type_indices[idx]
            # Only include frames with non-fixed joints (revolute=1 or prismatic=2)
            if jnt_type != 0:  # 0 is fixed joint
                frame_name = self.idx_to_frame[idx]
                important_names.append(frame_name)
        return important_names

    def ensure_tensor(self, th):
        """
        Converts a number of possible types into a tensor. The order of the tensor is determined by the order
        of self.get_joint_parameter_names(). th must contain all joints in the entire chain.
        """
        if isinstance(th, np.ndarray):
            th = torch.tensor(th, device=self.device, dtype=self.dtype)
        elif isinstance(th, list):
            th = torch.tensor(th, device=self.device, dtype=self.dtype)
        elif isinstance(th, dict):
            # convert dict to a flat, complete, tensor of all joints values. Missing joints are filled with zeros.
            th_dict = th
            elem_shape = get_dict_elem_shape(th_dict)
            th = (
                torch.ones(
                    [*elem_shape, self.n_joints], device=self.device, dtype=self.dtype
                )
                * torch.nan
            )
            joint_names = self.get_joint_parameter_names()
            for joint_name, joint_position in th_dict.items():
                jnt_idx = joint_names.index(joint_name)
                th[..., jnt_idx] = joint_position
            if torch.any(torch.isnan(th)):
                msg = "Missing values for the following joints:\n"
                for joint_name, th_i in zip(self.get_joint_parameter_names(), th):
                    msg += joint_name + "\n"
                raise ValueError(msg)
        return th

    def get_all_frame_indices(self):
        frame_indices = self.get_frame_indices(
            *self.get_frame_names(exclude_fixed=False)
        )
        return frame_indices

    def clamp(self, th):
        """

        Args:
            th: Joint configuration

        Returns: Always a tensor in the order of self.get_joint_parameter_names(), possibly batched.

        """
        th = self.ensure_tensor(th)
        return torch.clamp(th, self.low, self.high)

    def get_joint_limits(self):
        return self._get_joint_limits("limits")

    def get_joint_velocity_limits(self):
        return self._get_joint_limits("velocity_limits")

    def get_joint_effort_limits(self):
        return self._get_joint_limits("effort_limits")

    def _get_joint_limits(self, param_name):
        low = []
        high = []
        for joint in self.get_joints():
            val = getattr(joint, param_name)
            if val is None:
                # NOTE: This changes the previous default behavior of returning
                # +/- np.pi for joint limits to be more natural for both
                # revolute and prismatic joints
                low.append(-np.inf)
                high.append(np.inf)
            else:
                low.append(val[0])
                high.append(val[1])
        return low, high

    @staticmethod
    def _get_joints_and_child_links(frame):
        joint = frame.joint

        me_and_my_children = [frame.link]
        for child in frame.children:
            recursive_child_links = yield from Chain._get_joints_and_child_links(child)
            me_and_my_children.extend(recursive_child_links)

        if joint is not None and joint.joint_type != "fixed":
            yield joint, me_and_my_children

        return me_and_my_children

    def get_joints_and_child_links(self):
        yield from Chain._get_joints_and_child_links(self._root)


class SerialChain(Chain):
    """
    A serial Chain specialization with no branches and clearly defined end effector.
    Serial chains can be generated from subsets of a Chain.
    """

    def __init__(self, chain, end_frame_name, root_frame_name="", **kwargs):
        root_frame = (
            chain._root if root_frame_name == "" else chain.find_frame(root_frame_name)
        )
        if root_frame is None:
            raise ValueError("Invalid root frame name %s." % root_frame_name)
        chain = Chain(root_frame, **kwargs)

        # make a copy of those frames that includes only the chain up to the end effector
        end_frame_idx = chain.get_frame_indices(end_frame_name)
        ancestors = chain.parents_indices[end_frame_idx]

        frames = []
        # first pass create copies of the ancestor nodes
        for idx in ancestors:
            this_frame_name = chain.idx_to_frame[idx.item()]
            this_frame = copy.deepcopy(chain.find_frame(this_frame_name))
            if idx == end_frame_idx:
                this_frame.children = []
            frames.append(this_frame)
        # second pass assign correct children (only the next one in the frame list)
        for i in range(len(ancestors) - 1):
            frames[i].children = [frames[i + 1]]

        self._serial_frames = frames
        super().__init__(frames[0], **kwargs)

    def jacobian(self, th, locations=None, **kwargs):
        if locations is not None:
            locations = tf.Transform3d(pos=locations)
        return jacobian.calc_jacobian(self, th, tool=locations, **kwargs)

    def forward_kinematics(self, th, end_only: bool = True):
        """Like the base class, except `th` only needs to contain the joints in the SerialChain, not all joints."""
        frame_indices, th = self.convert_serial_inputs_to_chain_inputs(th, end_only)

        mat = super().forward_kinematics(th, frame_indices)

        if end_only:
            return mat[self._serial_frames[-1].name]
        else:
            return mat

    def convert_serial_inputs_to_chain_inputs(self, th, end_only: bool):
        # th = self.ensure_tensor(th)
        th_b = get_batch_size(th)
        th_n_joints = get_n_joints(th)
        if isinstance(th, list):
            th = torch.tensor(th, device=self.device, dtype=self.dtype)

        if end_only:
            frame_indices = self.get_frame_indices(self._serial_frames[-1].name)
        else:
            # pass through default behavior for frame indices being None, which is currently
            # to return all frames.
            frame_indices = None
        if th_n_joints < self.n_joints:
            # if th is only a partial list of joints, assume it's a list of joints for only the serial chain.
            partial_th = th
            nonfixed_serial_frames = list(
                filter(lambda f: f.joint.joint_type != "fixed", self._serial_frames)
            )
            if th_n_joints != len(nonfixed_serial_frames):
                raise ValueError(
                    f"Expected {len(nonfixed_serial_frames)} joint values, got {th_n_joints}."
                )
            th = torch.zeros(
                [th_b, self.n_joints], device=self.device, dtype=self.dtype
            )
            for i, frame in enumerate(nonfixed_serial_frames):
                joint_name = frame.joint.name
                if isinstance(partial_th, dict):
                    partial_th_i = partial_th[joint_name]
                else:
                    partial_th_i = partial_th[..., i]
                k = self.frame_to_idx[frame.name]
                jnt_idx = self.joint_indices[k]
                if frame.joint.joint_type != "fixed":
                    th[..., jnt_idx] = partial_th_i
        return frame_indices, th
