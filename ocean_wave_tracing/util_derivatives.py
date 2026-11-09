"""
Standalone methods for computing the time derivative of the wave vector
"""
import torch

from pyproj import CRS, Transformer

import logging

logger = logging.getLogger(__name__)

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
        self._diagnostic_context = {
            "timestep": None,
            "stage": "UNSPECIFIED",
            "ray_ids": None,
        }
        self._logged_nonfinite_rays = set()

    def set_diagnostic_context(self, *, timestep=None, stage=None, ray_ids=None):
        """Attach integration metadata to derivative evaluations."""

        if timestep is not None and timestep != self._diagnostic_context["timestep"]:
            self._logged_nonfinite_rays.clear()

        if timestep is not None:
            self._diagnostic_context["timestep"] = int(timestep)

        if stage is not None:
            self._diagnostic_context["stage"] = str(stage)

        if ray_ids is not None:
            self._diagnostic_context["ray_ids"] = ray_ids

    @staticmethod
    def _tensor_row(value, index):
        """Convert a tensor row to CPU-readable Python values."""
        if value is None:
            return None

        if torch.is_tensor(value):
            value = value.detach().cpu()

            if value.ndim == 0:
                return value.item()

            row = value[index]

            if row.ndim == 0:
                return row.item()

            return row.tolist()

        try:
            row = value[index]
            return row.tolist() if hasattr(row, "tolist") else row
        except (TypeError, IndexError, KeyError):
            return repr(value)

    def _log_nonfinite_derivatives(self, rays, properties, dOmega, derivative):
        context = self._diagnostic_context
        timestep = context["timestep"]

        # Ignore first integration interval.
        if timestep == 1:
            return

        ray_ids = context["ray_ids"]

        diagnostics = {
            "state": rays,
            "position": rays[:, :2],
            "wavenumber": rays[:, 2:],
            "depth": properties.d,
            "current_UV": properties.U,
            "current_gradient_dU": properties.dU,
            "inverse_metric_gij": properties.metric,
            "inverse_projection_jacobian_Kji": properties.projection_inverse,
            "k_squared": properties.k_squared,
            "k": properties.k,
            "sigma": properties.sigma,
            "current_term": properties.current_term,
            "Omega": properties.omega,
            "dOmega_dstate": dOmega,
            "d_sigma_dstate": getattr(
                properties, "d_sigma_dstate", None
            ),
            "d_current_term_dstate": getattr(
                properties, "d_current_term_dstate", None
            ),
            "group_velocity_cg": properties.cg,
            "final_derivative_dx_dy_dkx_dky": derivative,
        }

        for i in range(rays.shape[0]):
            if torch.isfinite(derivative[i]).all():
                continue

            ray_id = int(ray_ids[i].item()) if ray_ids is not None else i

            if ray_id in self._logged_nonfinite_rays:
                continue

            self._logged_nonfinite_rays.add(ray_id)

            nonfinite_fields = []
            detail_lines = []

            for name, value in diagnostics.items():
                if value is None:
                    continue

                row = self._tensor_row(value, i)
                detail_lines.append(f"{name}={row}")

                if torch.is_tensor(value) and not torch.isfinite(value[i]).all():
                    nonfinite_fields.append(name)

            logger.warning(
                "DERIVATIVE NON-FINITE: timestep=%s, stage=%s, "
                "ray_id=%s, ray_index=%s, nonfinite_fields=%s\n%s",
                timestep,
                context["stage"],
                ray_id,
                i,
                nonfinite_fields,
                "\n".join(detail_lines),
            )

    def find_sigma(self, k_norm, d):
        return torch.sqrt(self.wt.g * k_norm * torch.tanh(k_norm * d))


    def find_U_and_d(self, positions, idt):
        interpolated = self.wt.interpolate(positions, idt=idt)

        # Only one time snapshot is returned.
        interpolated = interpolated[:, :, 0, 0]

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

        properties.k_squared = k_sq
        properties.sigma = sigma
        properties.current_term = current_term
        properties.omega = sigma + current_term
        properties.metric = gmj
        properties.projection_inverse = K_ji

        return sigma + current_term, properties
    
    def dOmega(self, rays, idt):
        rays = rays.clone().detach().requires_grad_(True)

        Omega, properties = self.Omega(rays, idt)

        sigma = self.find_sigma(properties.k, properties.d)

        dOmega = torch.autograd.grad(Omega.sum(), rays, create_graph=False, retain_graph=True, allow_unused=False)[0]
        properties.cg = torch.autograd.grad(sigma.sum(), properties.k, create_graph=False, allow_unused=False)[0]

        components = {
            "depth": properties.d,
            "U": properties.U[:, 0],
            "V": properties.U[:, 1],
            "k": properties.k,
        }

        bad = ~torch.isfinite(dOmega).all(dim=1)

        for name, value in components.items():
            grad = torch.autograd.grad(
                value.sum(),
                rays,
                retain_graph=True,
                allow_unused=True
            )[0]

            if grad is not None:
                logger.warning(
                    "%s spatial derivatives: %s",
                    name,
                    grad[bad, :2].detach().cpu()
                )

        """if not torch.isfinite(dOmega).all():
            print("Non-finite Hamiltonian derivatives")
            print("Rays:", rays)
            print("dOmega:", dOmega)"""

        if not torch.isfinite(dOmega).all():
            properties.d_sigma_dstate = torch.autograd.grad(
                properties.sigma.sum(),
                rays,
                create_graph=False,
                retain_graph=True,
                allow_unused=True
            )[0]

            properties.d_current_term_dstate = torch.autograd.grad(
                properties.current_term.sum(),
                rays,
                create_graph=False,
                retain_graph=True,
                allow_unused=True
            )[0]

        return dOmega, properties

    def evolution(self, active_rays, idt):
        """
        Function that returns the time derivative of the rays to be input to the solver
        """

        dOmega, properties = self.dOmega(active_rays, idt)
        # dOmega is ordered [dkx, dky, dx, dy].
        derivative = dOmega[:, [2, 3, 0, 1]]
        derivative = derivative * torch.tensor([1., 1., -1., -1.], dtype=active_rays.dtype, device=active_rays.device)

        if not torch.isfinite(derivative).all():
            self._log_nonfinite_derivatives(
                active_rays, properties, dOmega, derivative
            )

        return derivative, properties
    
class Legacyderivative():
    """
    unfinished class that should serve as the parent class of the two legacy derivative solvers, but with 
    Advection and Wavenumber evolution converted to work with the new 4-vector workflow
    """
    def __init__(self, wave_tracing):
        self.wt = wave_tracing