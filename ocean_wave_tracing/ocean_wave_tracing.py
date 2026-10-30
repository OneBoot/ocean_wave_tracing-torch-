import numpy as np
import matplotlib.pyplot as plt
import logging
import xarray as xa
import pyproj # type: ignore
import sys
import cmocean.cm as cm
from netCDF4 import Dataset
import json
#from importlib import resources
from importlib_resources import files, as_file

from .util_solvers import Advection, WaveNumberEvolution, RungeKutta4
from .util_methods import make_xarray_dataArray, to_xarray_ds, check_velocity_field, check_bathymetry

from .util_metric import CartesianMetric, Identity
from .util_derivatives import Autoderivative#, Autoderivative_ESPG
import torch


logger = logging.getLogger(__name__)
logging.basicConfig(filename='ocean_wave_tracing.log', level=logging.INFO)
logging.info('\nStarted')

class Wave_tracing():
    """ Class computing the path of ocean wave rays according to the geometrical
    optics approximation.
    """
    def __init__(self, U, V,  nx, ny, nt, T, dx, dy,
                 nb_wave_rays, domain_X0, domain_XN, domain_Y0, domain_YN,
                 temporal_evolution=False, metric=CartesianMetric(), projection=Identity(), derivative_scheme=Autoderivative,
                 d=None,DEBUG=False, **kwargs):
        """
        Args:
            U (float): eastward velocity 2D field
            V (float): northward velocity 2D field
            nx (int): number of points in x-direction of velocity domain
            ny (int): number of points in y-direction of velocity domain
            nt (int): number of time steps for computation
            T (int): Seconds. Duration of wave tracing
            dx (int): Spatial resolution in x-direction. Units conforming to U
            dy (int): Spatial resolution in y-direction. Units conforming to V
            nb_wave_rays (int): Number of wave rays to track.
            domain_*0 (float): start value of domain area in X and Y direction
            domain_*N (float): end value of domain area in X and Y direction
            temporal_evolution (bool): flag if velocity field should change in time
            d (float): 2D bathymetry field
            **kwargs
        """

        #Should be changeable later
        self.device = torch.device("cpu")#
        self.dtype = torch.float32

        self.metric = metric
        self.derivative_scheme = derivative_scheme(self)
        self.projection = projection
        #self.ESPG_needed = ESPG_needed

        self.g = 9.81
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.dx = dx
        self.dy = dy
        self.nb_wave_rays = nb_wave_rays
        self.C = -1
        assert nb_wave_rays > 0, "Number of wave rays must be larger than zero"

        #Indexer, keeping track of the ray ids at a given timestep
        self.active_ray_indices = [torch.arange(self.nb_wave_rays, device=self.device, dtype=torch.long)]

        # specify the space and time domains
        self.domain_X0 = torch.as_tensor(domain_X0, dtype=self.dtype, device=self.device) # left side
        self.domain_XN = torch.as_tensor(domain_XN, dtype=self.dtype, device=self.device) # right side
        self.domain_Y0 = torch.as_tensor(domain_Y0, dtype=self.dtype, device=self.device) # bottom
        self.domain_YN = torch.as_tensor(domain_YN, dtype=self.dtype, device=self.device) # top
        self.T = T

        #specify whether or not the fields are changable
        self.temporal_evolution = temporal_evolution
        self.debug = DEBUG


        # Setting up X and Y domain
        self.x = np.linspace(domain_X0, domain_XN, nx)
        self.y = np.linspace(domain_Y0, domain_YN, ny)

        self.U = check_velocity_field(U,temporal_evolution,x=self.x,y=self.y).values
        self.V = check_velocity_field(V,temporal_evolution,x=self.x,y=self.y).values

        # make torch tensor of velocity field
        U_t = torch.as_tensor(check_velocity_field(U,temporal_evolution,x=self.x,y=self.y).values, dtype=self.dtype, device=self.device)
        V_t = torch.as_tensor(check_velocity_field(V,temporal_evolution,x=self.x,y=self.y).values, dtype=self.dtype, device=self.device)
        
        # Spatial gradients for every time snapshot
        dU_dy, dU_dx = torch.gradient(U_t, spacing=(self.dy, self.dx), dim=(-2, -1))
        dV_dy, dV_dx = torch.gradient(V_t, spacing=(self.dy, self.dx), dim=(-2, -1))


        # Check the bathymetry and make it into a pytorch compatible tensor
        if d is not None:
            self.d = check_bathymetry(d=d,x=self.x,y=self.y).values
            d_t = torch.as_tensor(self.d, dtype=self.dtype, device=self.device)
        else:
            d_static = 1e5
            logging.info(f'Hardcoding bathymetry to {d_static}m since not given.')
            self.d = check_bathymetry(d=np.ones((ny,nx))*d_static,x=self.x,y=self.y).values
            d_t = torch.as_tensor(self.d, dtype=self.dtype, device=self.device)
        d_t = torch.broadcast_to(d_t, U_t.shape)

        #Initialize the fields that are later interpolated for the ray time-integration
        self.field = torch.stack([d_t, U_t, V_t, dU_dx, dU_dy, dV_dx, dV_dy], dim=1)

        #designate space for all the rays
        self.initial_rays = torch.zeros(
            (nb_wave_rays, 4),
            dtype=self.dtype,
            device=self.device,
        )

        # Time
        self.dt = T/nt
        self.nb_velocity_time_steps = len(U_t)
        if not temporal_evolution:
            self.velocity_idt = np.zeros(nt,dtype=int)
            #logging.info('Vel idt : {}'.format(self.velocity_idt))
        else:
            t_velocity_field = U.time.data
            self.T0 = t_velocity_field[0]
            t_wr = np.arange(self.T0, self.T0+np.timedelta64(T,'s'),np.timedelta64(int(((T/nt)*1e3)),'ms'))
            self.velocity_idt = np.array([self.find_nearest(t_velocity_field,t_wr[i]) for i in range(len(t_wr))])

        self.kwargs = kwargs

    #Method for interpolating the field
    
    def interpolate(self, position):
        """Bilinearly interpolate fields at one or more positions.

        Positions outside the domain return NaNs.

        Returns:
            Tensor of shape (N, 7, nt_velocity, 1).
        """

        # Normalize to a batch: (N, 2).
        if position.ndim == 1:
            position = position.unsqueeze(0)

        if position.ndim != 2 or position.shape[1] != 2:
            raise ValueError(f"Expected positions with shape (N, 2), got {tuple(position.shape)}")

        # Transform coordinates.
        x = self.projection.a_1(position[:, 0], position[:, 1])
        y = self.projection.a_2(position[:, 0], position[:, 1])

        x = x.reshape(-1)
        y = y.reshape(-1)

        # Identify positions inside the domain BEFORE clamping.
        valid = (torch.isfinite(x) & torch.isfinite(y)
                  & (x >= self.domain_X0) & (x <= self.domain_XN) & (y >= self.domain_Y0) & (y <= self.domain_YN))

        # Convert coordinates to fractional grid indices.
        ix = (x - self.domain_X0) * (self.nx - 1) / (self.domain_XN - self.domain_X0)
        iy = (y - self.domain_Y0) * (self.ny - 1) / (self.domain_YN - self.domain_Y0)

        # Clamp indices so indexing remains safe, including for NaN inputs.
        ix = torch.nan_to_num(ix, nan=0.0, posinf=0.0, neginf=0.0).clamp(0, self.nx - 1)
        iy = torch.nan_to_num(iy, nan=0.0, posinf=0.0, neginf=0.0).clamp(0, self.ny - 1)

        x0 = torch.floor(ix).long()
        y0 = torch.floor(iy).long()

        x1 = (x0 + 1).clamp(max=self.nx - 1)
        y1 = (y0 + 1).clamp(max=self.ny - 1)

        wx = ix - x0.to(self.dtype)
        wy = iy - y0.to(self.dtype)

        # self.field shape: (time, field, y, x).
        f00 = self.field[..., y0, x0]
        f10 = self.field[..., y0, x1]
        f01 = self.field[..., y1, x0]
        f11 = self.field[..., y1, x1]

        wx = wx[None, None, :]
        wy = wy[None, None, :]

        result = ((1 - wx) * (1 - wy) * f00
                   + wx * (1 - wy) * f10
                   + (1 - wx) * wy * f01
                   + wx * wy * f11)

        # Convert (time, field, N) to (N, field, time, 1).
        result = result.permute(2, 1, 0).unsqueeze(-1)

        # Replace every interpolated field with NaN for invalid positions.
        result = result.masked_fill((~valid)[:, None, None, None], float("nan"))

        return result

    def check_CFL(self, cg, max_speed):
        """ Method for checking the Courant, Friedrichs, and Lewy
            condition for numerical intergration
        """
        dt = self.dt
        DX = np.abs(np.min([self.dx,self.dy])) 

        assert cg>=0, "Group velocity must be positive. Currently {}".format(cg)
        assert max_speed>=0,  "Maximum current speed must be positive. Currently {}".format(max_speed)

        u = cg+max_speed

        C = u*(dt/DX)

        if C<=1:
            logger.info('Courant number is {}'.format(np.round(C,2)))
        else:
            logger.warning('Courant number is {}'.format(np.round(C,2)))

        self.C = C


    def find_nearest(self,array, value):
        """ Method finding nearest indices to a position in array

        Args:
            array: Array containg to be compared with the value
            value (float): value to which index in array should be found

        Returns:
            idx (int): Index of the array which is closest to the value
        """

        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def inside_domain(self, position):
        x = self.projection.a_1(position[:, 0], position[:, 1])
        y = self.projection.a_2(position[:, 0], position[:, 1])

        return (torch.isfinite(x)
                & torch.isfinite(y)
                & (x >= self.domain_X0)
                & (x <= self.domain_XN)
                & (y >= self.domain_Y0)
                & (y <= self.domain_YN))
    

    def wave(self, T, theta, position, idt=0, max_iter=80):
        """
        Calculate initial wavenumber components.
        Invalid rays receive NaN values and are excluded from the solution.
        """

        device = self.field.device
        dtype = self.field.dtype

        position = torch.as_tensor(position, dtype=dtype, device=device)

        if position.ndim == 1:
            position = position.unsqueeze(0)

        n_rays = position.shape[0]

        theta = torch.as_tensor(theta, dtype=dtype, device=device)
        theta = torch.broadcast_to(theta, (n_rays,))

        target = torch.full_like(theta, 2 * torch.pi / T)

        # Outputs default to NaN. Only successfully solved rays are filled.
        kx = torch.full_like(theta, float("nan"))
        ky = torch.full_like(theta, float("nan"))

        # Obtain local depth and current.
        d, U, _ = self.derivative_scheme.find_U_and_d(position, idt)

        # Initial validity check.
        valid = (torch.isfinite(position).all(dim=1)
                  & torch.isfinite(theta)
                  & torch.isfinite(d)
                  & (d > 0)
                  & torch.isfinite(U).all(dim=1))

        # Normalize propagation direction using the inverse metric.
        direction = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

        gmj = self.metric.find_gmj(position)

        direction_norm = torch.sqrt(torch.einsum("ni,nj,nij->n", direction, direction, gmj))

        valid &= (torch.isfinite(gmj).all(dim=(1, 2))
                  & torch.isfinite(direction_norm)
                  & (direction_norm > 0))

        direction = direction / direction_norm.unsqueeze(-1)

        # Eckart approximation.
        k0 = target**2 / self.g
        alpha = k0 * d
        k_approx = (alpha / torch.sqrt(torch.tanh(alpha))) / d

        valid &= torch.isfinite(k_approx) & (k_approx > 0)

        # Keep only initially valid rays.
        ids = valid.nonzero(as_tuple=True)[0]

        if len(ids) == 0:
            return kx, ky

        pos = position[ids]
        direction = direction[ids]
        target = target[ids]
        k_approx = k_approx[ids]

        def evaluate_omega(k_magnitude):
            k_components = k_magnitude.unsqueeze(-1) * direction
            trial_rays = torch.cat([pos, k_components], dim=-1)

            omega, _ = self.derivative_scheme.Omega(trial_rays, idt)
            return omega

        # Initial bracket.
        k_low = torch.zeros_like(k_approx)
        k_high = k_approx.clone()

        alive = torch.ones_like(k_approx, dtype=torch.bool)

        # Expand upper bound independently for each ray.
        for _ in range(64):

            omega_high = evaluate_omega(k_high)

            finite = torch.isfinite(omega_high)
            alive &= finite

            bracketed = alive & (omega_high >= target)

            # Stop when every remaining ray is bracketed.
            if torch.all(bracketed | ~alive):
                break

            increase = alive & ~bracketed
            k_high = torch.where(increase, 2.0 * k_high, k_high)

        # Verify the final upper bound.
        omega_high = evaluate_omega(k_high)

        alive &= torch.isfinite(omega_high)
        alive &= omega_high >= target

        # Keep only rays with a valid bracket.
        k_low = k_low[alive]
        k_high = k_high[alive]
        target = target[alive]
        direction = direction[alive]
        pos = pos[alive]
        ids = ids[alive]

        if len(ids) == 0:
            return kx, ky

        # Vectorized bisection.
        for _ in range(max_iter):

            k_mid = 0.5 * (k_low + k_high)

            omega_mid = evaluate_omega(k_mid)

            finite = torch.isfinite(omega_mid)

            lower_half = finite & (omega_mid < target)

            k_low = torch.where(lower_half, k_mid, k_low)
            k_high = torch.where(
                finite & ~lower_half,
                k_mid,
                k_high
            )

            # A ray with a non-finite Omega is discarded.
            alive &= finite

        k = 0.5 * (k_low + k_high)

        k_components = k.unsqueeze(-1) * direction

        # Store only successfully solved rays.
        successful_ids = ids[alive]

        kx[successful_ids] = k_components[alive, 0]
        ky[successful_ids] = k_components[alive, 1]

        return kx, ky
    

    def set_initial_condition(self, wave_period, theta0, **kwargs):
        """Set initial positions and wavenumbers for all wave rays."""

        device = self.field.device
        dtype = self.field.dtype
        nb_wave_rays = self.nb_wave_rays

        # --------------------------------------------------
        # 1. Initial ray positions
        # --------------------------------------------------

        valid_sides = ["left", "right", "top", "bottom"]

        if "incoming_wave_side" in kwargs:
            side = kwargs["incoming_wave_side"]

            if side not in valid_sides:
                logger.info("Invalid initial side. Left will be used.")
                side = "left"

            if side in ("left", "right"):
                x_value = self.domain_X0 if side == "left" else self.domain_XN

                xs = torch.full((nb_wave_rays,), x_value.item(), dtype=dtype, device=device)
                ys = torch.linspace(self.domain_Y0.item(), self.domain_YN.item(), nb_wave_rays, dtype=dtype, device=device)

            else:
                y_value = self.domain_Y0 if side == "bottom" else self.domain_YN

                xs = torch.linspace(self.domain_X0.item(), self.domain_XN.item(), nb_wave_rays, dtype=dtype, device=device)
                ys = torch.full((nb_wave_rays,), y_value.item(), dtype=dtype, device=device)

        else:
            ipx = kwargs.get("ipx")
            ipy = kwargs.get("ipy")

            if ipx is None and ipy is None:
                logger.info("No initial positions given. Using left boundary.")

                xs = torch.full((nb_wave_rays,), self.domain_X0.item(), dtype=dtype, device=device)
                ys = torch.linspace(self.domain_Y0.item(), self.domain_YN.item(), nb_wave_rays, dtype=dtype, device=device)

            else:
                if ipx is None or ipy is None:
                    raise ValueError("Provide both ipx and ipy, or neither.")

                xs = torch.as_tensor(ipx, dtype=dtype, device=device)
                ys = torch.as_tensor(ipy, dtype=dtype, device=device)

                if xs.ndim == 0:
                    xs = xs.expand(nb_wave_rays)

                if ys.ndim == 0:
                    ys = ys.expand(nb_wave_rays)

                if xs.shape != (nb_wave_rays,) or ys.shape != (nb_wave_rays,):
                    raise ValueError("ipx and ipy must each be scalar or have nb_wave_rays values.")

        # --------------------------------------------------
        # 2. Store initial positions
        # --------------------------------------------------

        b_1 = self.projection.b_1(xs, ys)
        b_2 = self.projection.b_2(xs, ys)

        positions = torch.stack([b_1, b_2], dim=-1)
        idt = int(self.velocity_idt[0])

        # Check that all initial positions lie inside the domain.
        valid_positions = (torch.isfinite(positions).all(dim=1)
            & (xs >= self.domain_X0) & (xs <= self.domain_XN)
            & (ys >= self.domain_Y0) & (ys <= self.domain_YN))

        if not torch.all(valid_positions):
            raise ValueError("Initial ray positions must be finite and inside the domain.")

        self.initial_rays[:, :2] = positions

        # --------------------------------------------------
        # 3. Initial wave directions
        # --------------------------------------------------

        theta0 = torch.as_tensor(theta0, dtype=dtype, device=device)

        if theta0.ndim == 0:
            theta0 = theta0.expand(nb_wave_rays)

        if theta0.shape != (nb_wave_rays,):
            raise ValueError("theta0 must be scalar or have nb_wave_rays values.")

        # --------------------------------------------------
        # 4. Calculate initial wavenumbers
        # --------------------------------------------------

        idt = int(self.velocity_idt[0])
        interpolated = self.interpolate(positions)
        local_U = interpolated[:, 1:3, idt, 0]

        kx, ky = self.wave(T=wave_period, theta=theta0, position=positions, idt=idt)

        # --------------------------------------------------
        # 5. Store initial ray state
        # --------------------------------------------------

        self.initial_rays[:, 2] = kx
        self.initial_rays[:, 3] = ky

        #Check the CFL condition
    
    def initialize_ray_history(self):
        """Allocate histories indexed by timestep and stable ray ID."""
        nt = self.nt
        nr = self.nb_wave_rays

        names = ["ray_x", "ray_y",
                 "ray_k", "ray_kx", "ray_ky",
                 "ray_U", "ray_V",
                 "ray_theta", "ray_cg", "ray_depth",
                 "ray_dudx", "ray_dvdy", "ray_dudy", "ray_dvdx"]

        self.property_history = {name: torch.full((nt, nr), float("nan"), dtype=torch.float32, device=self.device) for name in names}

        # Preserve the initial states in the original tensor format.
        self.rays_history = torch.full((nt, nr, 4), float("nan"), dtype=self.dtype, device=self.device)

        self.rays_history[0] = self.initial_rays.detach()

        # Keep the original public attributes available as lists.
        for name in names:
            setattr(self, name, [])


    def store_ray_properties(self, timestep, ray_ids, rays, properties):
        """
        Store properties by stable ray ID.

        rays:      (N, 4), ordered as x, y, kx, ky
        ray_ids:   (N,), original ray IDs
        properties: properties evaluated at these same ray states
        """

        # All properties must correspond to the same active rays.
        values = {
            "ray_x": rays[:, 0],
            "ray_y": rays[:, 1],
            "ray_k": properties.k_norm,
            "ray_kx": rays[:, 2],
            "ray_ky": rays[:, 3],
            "ray_U": properties.U[:, 0],
            "ray_V": properties.U[:, 1],
            "ray_cg": properties.cg,
            "ray_depth": properties.d,
            "ray_dudx": properties.dU[:, 0],
            "ray_dudy": properties.dU[:, 1],
            "ray_dvdx": properties.dU[:, 2],
            "ray_dvdy": properties.dU[:, 3],
        }

        # Convert stable ray IDs to a tensor.
 #       ray_ids = ray_ids.to(dtype=torch.long)

        # Validate every field before writing.
        n_active = len(ray_ids)

        for name, value in values.items():
            if value.ndim != 1 or len(value) != n_active:
                raise ValueError(f"{name}: expected {n_active} values, got shape {tuple(value.shape)}")

        # Store each property in its original ray-ID column.
        for name, value in values.items():
            self.property_history[name][timestep, ray_ids] = (value.detach().to(dtype=torch.float32))

        # Derive the direction from the stored wavenumber components.
        kx = self.property_history["ray_kx"][timestep, ray_ids]
        ky = self.property_history["ray_ky"][timestep, ray_ids]

        self.property_history["ray_theta"][timestep, ray_ids] = (torch.atan2(ky, kx))


    def solve(self, solver=None):
        """
        Trace rays and store all properties at their corresponding
        timestep and original ray ID.

        State vector: [x, y, kx, ky]
        """
        self.initialize_ray_history()

        nr = self.nb_wave_rays
        nt = self.nt

        scheme = self.derivative_scheme

        active_ids = torch.arange(nr, dtype=torch.long, device=self.device)
        active_rays = self.initial_rays.clone()

        for n in range(nt):
            #check if any rays still exist at all
            if len(active_ids) == 0:
                break

            # identify the relevant time index, if time exceeds defined current field time domain,
            #  revert to last current field snapshot
            idt = int(self.velocity_idt[min(n, len(self.velocity_idt) - 1)])

            # Evaluate the current state and its properties together.
            derivative, properties = scheme.evolution(active_rays, idt)

            # Verify that properties match the active rays.
            count = len(active_ids)
            for name in ("k", "k_norm", "d", "U", "dU", "cg"):
                value = getattr(properties, name)
                if value is None or value.shape[0] != count:
                    raise ValueError(f"Timestep {n}: {name} has shape {None if value is None else tuple(value.shape)}, but there are {count} active rays.")

            # Store properties at the time they were evaluated.
            self.store_ray_properties(timestep=n, ray_ids=active_ids, rays=active_rays, properties=properties)

            if n == nt - 1:
                break

            next_rays = active_rays + self.dt* derivative

            # Check bathymetry at the proposed next position
            next_d, _, _ = scheme.find_U_and_d(next_rays[:, :2], idt)

            # Only retain rays in valid water
            finite = (
                torch.isfinite(next_rays).all(dim=1)
                & torch.isfinite(next_d)
                & (next_d > 0)
            )
            # Check spatial domain
            inside = self.inside_domain(next_rays[:, :2])

            # Retain only valid rays
            valid = inside & finite

            active_ids = active_ids[valid]
            active_rays = next_rays[valid].detach()

            # Store only surviving rays
            if len(active_ids) > 0:
                self.rays_history[n + 1, active_ids] = active_rays
            
            # Remove rays with any non-finite state component.
            #valid = torch.isfinite(next_rays).all(dim=1)

            #active_ids = active_ids[valid]
            #active_rays = next_rays[valid].detach()

            # Store the new positions at the next timestep.
            if len(active_ids) > 0:
                self.rays_history[n + 1, active_ids] = active_rays

        # Synchronize the main state tensor with the stored trajectory.
        self.rays = self.rays_history

        # Expose histories as PyTorch tensors, ordered (ray, time).
        for name, history in self.property_history.items():
            setattr(self, name, history.T)

        logging.info("Wave tracing completed.")


    def to_ds(self, **kwargs):
        """Convert wave ray information to an xarray Dataset."""

        if 'proj4' in kwargs:
            lons, lats = self.to_latlon(kwargs['proj4'])
        else:
            lons = np.full((self.nb_wave_rays, self.nt), np.nan)
            lats = np.full_like(lons, np.nan)

        variables = {
            'ray_k': self.ray_k,
            'ray_kx': self.ray_kx,
            'ray_ky': self.ray_ky,
            'ray_x': self.ray_x,
            'ray_y': self.ray_y,
            'ray_U': self.ray_U,
            'ray_V': self.ray_V,
            'ray_theta': self.ray_theta,
            'ray_cg': self.ray_cg,
            'ray_depth': self.ray_depth,
            'ray_lat': lats,
            'ray_lon': lons,
            'ray_dudx': self.ray_dudx,
            'ray_dvdy': self.ray_dvdy,
            'ray_dudy': self.ray_dudy,
            'ray_dvdx': self.ray_dvdx,
        }

        expected_shape = (self.nb_wave_rays, self.nt)

        for name, value in variables.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            else:
                value = np.asarray(value)

            if value.shape == expected_shape:
                pass
            elif value.shape == expected_shape[::-1]:
                value = value.T
            else:
                raise ValueError(f"{name} has shape {value.shape}; expected {expected_shape} or {expected_shape[::-1]}")

            variables[name] = value

        source = files('ocean_wave_tracing').joinpath('ray_metadata.json')

        with as_file(source) as sfile:
            data = json.loads(sfile.read_text())

        t = np.linspace(0, self.T, self.nt)
        ray_id = np.arange(self.nb_wave_rays)

        vars = [make_xarray_dataArray(var=variables[name], t=t, rays=ray_id, name=name, attribs=data[name]) for name in variables]

        return to_xarray_ds(vars)


    def ray_density(self,x_increment, y_increment, plot=False):
        """ Method computing ray density within boxes. The density of wave rays
        can be used as proxy for wave energy density

        Args:
            x_increment (int): size of box in x direction. Length = x_increment*dx
            y_increment (int): size of box in y direction. Length = y_increment*dy

        Returns:
            xx (2d): x grid
            yy (2d): y grid
            hm (2d): heat map of wave ray density

        #>>> wt = Wave_tracing(U=1,V=1,nx=1,ny=1,nt=1,T=1,dx=1,dy=1,wave_period=1, theta0=1,nb_wave_rays=1,domain_X0=0,domain_XN=0,domain_Y0=0,domain_YN=1,incoming_wave_side='left')
        #>>> wt.solve()
        #>>> wt.ray_density(x_increment=20,y_increment=20)
        """
        xx,yy=np.meshgrid(self.x[::x_increment],self.y[::y_increment])
        hm = np.zeros(xx.shape) # heatmap
        xs = xx[0]
        ys = yy[:,0]

        counter=0
        for i in range(0,self.nb_wave_rays):
            for idx in range(len(xs)-1):
                x0, xn = xs[idx],xs[idx+1]
                for idy in range(len(ys)-1):
                    y0, yn = ys[idy],ys[idy+1]
                    counter+=1
                    valid_x = (self.ray_x[i,:]>x0)*(self.ray_x[i,:]<xn)
                    if (torch.any((self.ray_y[i,:][valid_x]>y0)*(self.ray_y[i,:][valid_x]<yn))):
                        hm[idy,idx]+=1

        if plot:
            plt.pcolormesh(xx,yy,hm)
            plt.colorbar()
            for i in range(0,self.nb_wave_rays):
                plt.plot(self.ray_x[i,:],self.ray_y[i,:],'-r',alpha=0.3)
            plt.scatter(xx,yy);plt.show()

        return xx,yy,hm


    def to_latlon(self, proj4):
        """Reproject ray coordinates to longitude and latitude."""

        ray_x = np.asarray(self.ray_x).T
        ray_y = np.asarray(self.ray_y).T

        # Current layout: (time, ray)
        expected_shape = (self.nt, self.nb_wave_rays)

        if ray_x.shape != expected_shape or ray_y.shape != expected_shape:
            raise ValueError(f"Expected coordinates with shape {expected_shape}; got ray_x={ray_x.shape}, ray_y={ray_y.shape}")

        transformer = pyproj.Transformer.from_crs(proj4, "EPSG:4326", always_xy=True)

        # Transform all rays and timesteps together.
        lons, lats = transformer.transform(ray_x, ray_y)

        # Return (ray, time), matching the original method's output.
        return np.asarray(lons).T, np.asarray(lats).T

    
    def get_ray_curvature(self,decomposed=False):
        """ Compute the approximate analytical ray curvature after Halsne and Li (2025, in rev.)
        """

        # Tangent and normal vector
        ds_ray = self.to_ds()
        eo=2
        xprime=ds_ray.ray_x.differentiate(coord='time',edge_order=eo)
        yprime=ds_ray.ray_y.differentiate(coord='time',edge_order=eo)

        arclength=np.sqrt(xprime**2+yprime**2)
        nx, ny = -yprime/arclength, xprime/arclength

        # intrinsic phase velocity
        non_normalized_ray_curvature_depth = -((nx*self.d_cx) + (ny*self.d_cy)) # NOTE: The negative sign is taken into account here
        
        vorticity = ds_ray.ray_dvdx-ds_ray.ray_dudy #ray curvature currents

        group_velocity = ds_ray.ray_cg 

        # Compute the curvature
        ray_curvature_curr = vorticity/group_velocity
        ray_curvature_depth = non_normalized_ray_curvature_depth/group_velocity 
        ray_curvature_tot = ray_curvature_curr + ray_curvature_depth 

        if decomposed:
            return ray_curvature_tot, ray_curvature_depth, ray_curvature_curr
        else:
            return ray_curvature_tot

    def get_shoaling_coefficient(self):
        """ Compute the shoaling coefficient due to group velocity changes
        """
        ds_ray = self.to_ds()
        sc=np.sqrt(ds_ray.ray_cg[:,0]/ds_ray.ray_cg)
        sc.attrs['units']='-'
        sc.attrs['long_name']='Shoaling coefficient'
        return sc