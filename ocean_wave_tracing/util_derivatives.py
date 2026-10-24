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

        #This can be computed once, and never again after that
        # positions: (N, 2)

        field = self.wt.field
        positions = positions.to(dtype=field.dtype, device=field.device)

        X0 = self.wt.domain_X0.to(device=field.device, dtype=field.dtype)
        XN = self.wt.domain_XN.to(device=field.device, dtype=field.dtype)
        Y0 = self.wt.domain_Y0.to(device=field.device, dtype=field.dtype)
        YN = self.wt.domain_YN.to(device=field.device, dtype=field.dtype)

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

        # Contravariant components, retained for the current term.
        k = torch.einsum("ni,nij->nj", Theta, g_nj)

        # Physical wavenumber magnitude: sqrt(Theta_i g^ij Theta_j).
        k_norm_sq = torch.einsum("ni,nij,nj->n", Theta, g_nj, Theta)
        k_norm = torch.sqrt(torch.clamp(k_norm_sq, min=0.0))

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

        print(derivative)

        return derivative, properties

"""#Class that allows for implementation in the curvilinear grid while maintaining lat, lon coordinates
class Autoderivative_ESPG(Autoderivative):

    def __init__(self, wave_tracing, projection=proj4Norkyst):
        Autoderivative.__init__(self, wave_tracing)
        #initialize the transformer
        self.transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_proj4(projection), always_xy=True)
        self.backtransformer = Transformer.from_crs(CRS.from_proj4(projection), CRS.from_epsg(4326), always_xy=True)

    def find_U_and_d(self, positions, idt):
        positions_transformed = self.transform(positions)
        return super().find_U_and_d(positions_transformed, idt)

    def transform(self, positions):
        positions_detached = positions.detach().numpy()
        positions_transformed = torch.tensor(self.transformer.transform(positions_detached[:,0], positions_detached[:,1]), dtype=torch.float32).T
        return positions_transformed
#    def Omega(self, rays, idt):
#        omega, properties = super().Omega(rays, idt)
#        return torch.rad2deg(omega), properties
"""
class Legacyderivative():
    """
    unfinished class that should serve as the parent class of the two legacy derivative solvers, but with 
    Advection and Wavenumber evolution converted to work with the new 4-vector workflow
    """
    def __init__(self, wave_tracing):
        self.wt = wave_tracing