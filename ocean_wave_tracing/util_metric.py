import torch

class CartesianMetric():
# All Metric related stuff for the Cartesian case

    def find_g_im(self, position):
        N = position.shape[0]
        g_11 = torch.ones(N, dtype=position.dtype, device=position.device)
        g_12 = torch.zeros(N, dtype=position.dtype, device=position.device)
        g_21 = torch.zeros(N, dtype=position.dtype, device=position.device)
        g_22 = torch.ones(N, dtype=position.dtype, device=position.device)
        return torch.stack([
            torch.stack([g_11, g_12], dim=-1),
            torch.stack([g_21, g_22], dim=-1)
        ], dim=-2)

    def find_gnj(self, position):
        N = position.shape[0]
        g_11 = torch.ones(N, dtype=position.dtype, device=position.device)
        g_12 = torch.zeros(N, dtype=position.dtype, device=position.device)
        g_21 = torch.zeros(N, dtype=position.dtype, device=position.device)
        g_22 = torch.ones(N, dtype=position.dtype, device=position.device)
        return torch.stack([
            torch.stack([g_11, g_12], dim=-1),
            torch.stack([g_21, g_22], dim=-1)
        ], dim=-2)

    def find_J_jm(self, position):
        N = position.shape[0]
        J_11 = torch.ones(N, dtype=position.dtype, device=position.device)
        J_12 = torch.zeros(N, dtype=position.dtype, device=position.device)
        J_21 = torch.zeros(N, dtype=position.dtype, device=position.device)
        J_22 = torch.ones(N, dtype=position.dtype, device=position.device)
        return torch.stack([
            torch.stack([J_11, J_12], dim=-1),
            torch.stack([J_21, J_22], dim=-1)
        ], dim=-2)

    def find_h_i(self, position):
        N = position.shape[0]
        h_1 = torch.ones(N, dtype=position.dtype, device=position.device)
        h_2 = torch.ones(N, dtype=position.dtype, device=position.device)
        return torch.stack([h_1, h_2], dim=-1)


class SphericalMetric():

# All Metric related stuff for the spherical case
    def __init__(self):
        self.R = 6371e3
        pass

    def find_g_im(self, position):
        N = position.shape[0]
        _, lat = position.T
#        lat = torch.deg2rad(lat)
        g_11 = (self.R*torch.cos(lat))**2 
        g_12 = torch.zeros(N, dtype=position.dtype, device=position.device)
        g_21 = torch.zeros(N, dtype=position.dtype, device=position.device)
        g_22 = torch.ones(N, dtype=position.dtype, device=position.device)*(self.R**2)
        return torch.stack([
            torch.stack([g_11, g_12], dim=-1),
            torch.stack([g_21, g_22], dim=-1)
        ], dim=-2)

    def find_gnj(self, position):
        N = position.shape[0]
        _, lat = position.T
#        lat = torch.deg2rad(lat)
        g_11 = 1/(self.R*torch.cos(lat))**2
        g_12 = torch.zeros(N, dtype=position.dtype, device=position.device)
        g_21 = torch.zeros(N, dtype=position.dtype, device=position.device)
        g_22 = torch.ones(N, dtype=position.dtype, device=position.device)/(self.R**2)
        return torch.stack([
            torch.stack([g_11, g_12], dim=-1),
            torch.stack([g_21, g_22], dim=-1)
        ], dim=-2)

    def find_J_jm(self, position):
        N = position.shape[0]
        J_11 = torch.ones(N, dtype=position.dtype, device=position.device)
        J_12 = torch.zeros(N, dtype=position.dtype, device=position.device)
        J_21 = torch.zeros(N, dtype=position.dtype, device=position.device)
        J_22 = torch.ones(N, dtype=position.dtype, device=position.device)
        return torch.stack([
            torch.stack([J_11, J_12], dim=-1),
            torch.stack([J_21, J_22], dim=-1)
        ], dim=-2)

    def find_h_i(self, position):
        N = position.shape[0]
        _, lat = position.T
#        lat = torch.deg2rad(lat)
        h_1 = 1/(self.R*torch.cos(lat))
        h_2 = torch.ones(N, dtype=position.dtype, device=position.device)/self.R
        return torch.stack([h_1, h_2], dim=-1)
