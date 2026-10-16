"""
Standalone methods for computing the time derivative of the wave vector
"""
import torch

class ray_properties():
    # Helper class that keeps all interesting ray properties needed for generating the end dataset in one place
    def __init__(self, k, k_norm, d, U, dU, cg=None):
        self.k = k
        self.k_norm = k_norm
        self.d = d
        self.U = U
        self.dU = dU
        self.cg = cg

    # The following should be put in a separate solver file, along with legacy solver code that avoids autograd
    # The legacy code might then run faster on the GPU


class Autoderivative():
    """
    Class that contains all the functionality needed to solve the ray-tracing equations "by brute force" without 
    needing to analytically solve for anything except the metric tensors input into it
    """
    def __init__(self, wave_tracing):
        self.wt = wave_tracing

    def find_sigma(self, k_norm, d):
        return torch.sqrt(self.wt.g * k_norm * torch.tanh(k_norm * d))

    def find_U_and_d(self, positions, idt):
        # positions: (N, 2)
        device = positions.device
        dtype = positions.dtype

        X0 = self.wt.domain_X0.to(device=device, dtype=dtype)
        XN = self.wt.domain_XN.to(device=device, dtype=dtype)
        Y0 = self.wt.domain_Y0.to(device=device, dtype=dtype)
        YN = self.wt.domain_YN.to(device=device, dtype=dtype)

        x = 2 * (positions[:, 0] - X0) / (XN - X0) - 1
        y = 2 * (positions[:, 1] - Y0) / (YN - Y0) - 1

        grid = torch.stack([x, y], dim=-1).reshape(1, -1, 1, 2)

        interpolated = torch.nn.functional.grid_sample(
            self.wt.field[idt].unsqueeze(0),
            grid,
            mode="bilinear",
            align_corners=True,
        )

        interpolated = interpolated[0, :, :, 0].T

        local_d = interpolated[:, 0]
        local_U = interpolated[:, 1:3]
        local_dU = interpolated[:, 3:7]

        return local_d, local_U, local_dU


    def Omega(self, rays, idt):
        # rays: (N, 4)
        #       x, y, kx, ky

        positions = rays[:, :2]
        Theta = rays[:, 2:]

        d, U, dU = self.find_U_and_d(positions, idt)

        # In the future, these could be aggregated within the metric class to avoid multiplying over inverses every time
        g_im = self.wt.metric.find_g_im(positions)
        g_nj = self.wt.metric.find_gnj(positions)
        J_jm = self.wt.metric.find_J_jm(positions)
        h_i = self.wt.metric.find_h_i(positions)

        k = torch.einsum("ni,nij->nj", Theta, g_nj)
        k_norm = torch.linalg.vector_norm(k, dim=1)

        sigma = self.find_sigma(k_norm, d)

        current_term = torch.einsum("nim,ni,nmj,ni,nj->n", g_im, h_i, J_jm, U, k)

        # Neatly pack away all the additional properties in a new data structure
        properties = ray_properties(k, k_norm, d, U, dU)

        return sigma + current_term, properties


    def dOmega(self, rays, idt):
        rays = rays.clone().detach().requires_grad_(True)

        omega, properties = self.Omega(rays, idt)

        sigma = self.find_sigma(properties.k_norm, properties.d)

        dOmega = torch.autograd.grad(omega.sum(), rays, create_graph=False, allow_unused=False)[0]
        properties.cg = torch.autograd.grad(sigma.sum(), properties.k_norm, create_graph=False, allow_unused=False)[0]

        return dOmega, properties


    def evolution(self, active_rays, idt):
        """
        Function that returns the time derivative of the rays to be input to the solver
        """

        dOmega, properties = self.dOmega(active_rays, idt)
        # dOmega is ordered [dkx, dky, dx, dy].
        derivative = dOmega[:, [2, 3, 0, 1]]
        derivative = derivative * torch.tensor([1., 1., -1., -1.], dtype=active_rays.dtype, device=active_rays.device)

        return derivative, properties
