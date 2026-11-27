import h5py
import torch
import numpy as np

def load_h5_recursive(group):
    res = {}
    for key in group.keys():
        if isinstance(group[key], h5py.Group):
            res[key] = load_h5_recursive(group[key])
        else:
            res[key] = group[key][:]
    return res

def to_numpy(x):
    if torch.is_tensor(x):
        return x.cpu().numpy()
    return x

def get_total_pose_matrix(obj):
    mat = obj.pose.to_transformation_matrix()
    if torch.is_tensor(mat):
        mat = mat.cpu().numpy()
    if mat.ndim == 3: # GPU sim batch
        mat = mat[0]
    return mat
