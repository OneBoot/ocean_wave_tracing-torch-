import torch

# All Metric related stuff
def find_g_im_h(position):
    g_11 = 1.
    g_12 = 0.
    g_21 = 0.
    g_22 = 1.
    return torch.stack([
        torch.tensor([g_11, g_12]),
        torch.tensor([g_21, g_22])
    ])

def find_gnj(position):
    g_11 = 1.
    g_12 = 0.
    g_21 = 0.
    g_22 = 1.
    return torch.stack([
        torch.tensor([g_11, g_12]),
        torch.tensor([g_21, g_22])
    ])

def find_J_jm(position):
    J_11 = 1.
    J_12 = 0.
    J_21 = 0.
    J_22 = 1.
    return torch.stack([
        torch.tensor([J_11, J_12]),
        torch.tensor([J_21, J_22])
    ])