"""
Standalone methods for computing the time derivative of the wave vector
"""
import torch

from pyproj import CRS, Transformer

proj4Norkyst = "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=70 +x_0=3369600 +y_0=1844800 +a=6378137 +b=6356752.3142 +units=m +no_defs +type=crs"

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
        interpolated = self.wt.interpolate(positions)

        # Select the requested time snapshot for every ray.
        # Shape: (number_of_rays, 7)
        interpolated = interpolated[:, :, idt, 0]

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
        # these metric tensors need to be specified further to relate to either the field or the solving metric
        gmj = self.wt.metric.find_gmj(positions)
        K_ji = self.wt.projection.find_K_ji(positions)

        # Physical wavenumber magnitude: sqrt(Theta_i g^ij Theta_j).
        k_sq = torch.einsum("nj,nm,nmj->n", Theta, Theta, gmj )
        k = torch.sqrt(torch.clamp(k_sq, min=0.0))

        sigma = self.find_sigma(k, d)

        current_term = torch.einsum("ni,nj,nji->n", U, Theta, K_ji)

        # Neatly pack away all the additional properties in a new data structure
        properties = ray_properties(k=k, k_norm=k, d=d, U=U, dU=dU)

        return sigma + current_term, properties

    def dOmega(self, rays, idt):
        rays = rays.clone().detach().requires_grad_(True)

        Omega, properties = self.Omega(rays, idt)

        sigma = self.find_sigma(properties.k, properties.d)

        dOmega = torch.autograd.grad(Omega.sum(), rays, create_graph=False, allow_unused=False)[0]
        properties.cg = torch.autograd.grad(sigma.sum(), properties.k, create_graph=False, allow_unused=False)[0]

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
    
class Legacyderivative():
    """
    unfinished class that should serve as the parent class of the two legacy derivative solvers, but with 
    Advection and Wavenumber evolution converted to work with the new 4-vector workflow
    """
    def __init__(self, wave_tracing):
        self.wt = wave_tracing