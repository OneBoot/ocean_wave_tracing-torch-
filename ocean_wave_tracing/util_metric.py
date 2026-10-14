import torch

class CartesianMetric():
# All Metric related stuff for the Cartesian case

    def find_g_im_h(self, position):
        g_11 = 1.
        g_12 = 0.
        g_21 = 0.
        g_22 = 1.
        return torch.stack([
            torch.tensor([g_11, g_12]),
            torch.tensor([g_21, g_22])
        ])

    def find_gnj(self, position):
        g_11 = 1.
        g_12 = 0.
        g_21 = 0.
        g_22 = 1.
        return torch.stack([
            torch.tensor([g_11, g_12]),
            torch.tensor([g_21, g_22])
        ])

    def find_J_jm(self, position):
        J_11 = 1.
        J_12 = 0.
        J_21 = 0.
        J_22 = 1.
        return torch.stack([
            torch.tensor([J_11, J_12]),
            torch.tensor([J_21, J_22])
        ])


class SphericalMetric():

# All Metric related stuff for the spherical case
    def __init__(self):
        self.R = 6371e6
        pass

    def find_g_im_h(self, position):
        lon, lat = position
        g_h11 = self.R*torch.cos(lat) 
        g_h12 = 0.
        g_h21 = 0.
        g_h22 = self.R
        return torch.stack([
            torch.tensor([g_h11, g_h12]),
            torch.tensor([g_h21, g_h22])
        ])
    
    def find_gnj(self, position):
        lon, lat = position
        g_11 = 1/(self.R*torch.cos(lat))**2
        g_12 = 0.
        g_21 = 0.
        g_22 = 1/self.R**2
        return torch.stack([
            torch.tensor([g_11, g_12]),
            torch.tensor([g_21, g_22])
        ])

    def find_J_jm(self, position):
        J_11 = 1.
        J_12 = 0.
        J_21 = 0.
        J_22 = 1.
        return torch.stack([
            torch.tensor([J_11, J_12]),
            torch.tensor([J_21, J_22])
        ])