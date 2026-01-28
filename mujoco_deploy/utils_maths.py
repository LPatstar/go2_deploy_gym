import torch

def wrap_to_pi(angles):
    angles %= 2*torch.pi
    angles -= 2*torch.pi * (angles > torch.pi)
    return angles

def euler_xyz_from_quat(quat):
    # quat: (w, x, y, z) -> mujoco order is usually (w, x, y, z)
    # This implementation expects (w, x, y, z)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1.0, 1.0)
    pitch_y = torch.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z

def quat_apply(b_q, b_v):
     shape = b_q.shape
     if b_q.shape[-1] != 4:
         print("quat apply error, input shape should be ?*4, but got", b_q.shape)
         return
    
    # get w x y z
     w, x, y, z = b_q[..., 0], b_q[..., 1], b_q[..., 2], b_q[..., 3]
     num_points = x.numel()
     w = w.view(num_points)
     x = x.view(num_points)
     y = y.view(num_points)
     z = z.view(num_points)
     
     vx, vy, vz = b_v[..., 0].view(num_points), b_v[..., 1].view(num_points), b_v[..., 2].view(num_points)
     
     return torch.stack([
         (1 - 2 * (y*y + z*z)) * vx + 2 * (x*y - z*w) * vy + 2 * (x*z + y*w) * vz,
         2 * (x*y + z*w) * vx + (1 - 2 * (x*x + z*z)) * vy + 2 * (y*z - x*w) * vz,
         2 * (x*z - y*w) * vx + 2 * (y*z + x*w) * vy + (1 - 2 * (x*x + y*y)) * vz
     ], dim=-1).view(shape[:-1] + (3,))

def quat_apply_yaw(b_q, b_v):
    """Rotate a vector only by the yaw component of the quaternion."""
    roll, pitch, yaw = euler_xyz_from_quat(b_q)
    # create yaw quaternion
    zeros = torch.zeros_like(yaw)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    # quaternion format (w, x, y, z)
    yaw_q = torch.stack([cy, zeros, zeros, sy], dim=-1)
    return quat_apply(yaw_q, b_v)

def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((a[:, 0:1], -a[:, 1:]), dim=-1).view(shape)

def quat_rotate_inverse(q, v):
    q_inv = quat_conjugate(q)
    return quat_apply(q_inv, v)