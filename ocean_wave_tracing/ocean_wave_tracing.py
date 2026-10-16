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

from .util_metric import CartesianMetric
from .util_derivatives import Autoderivative
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
                 temporal_evolution=False, metric=CartesianMetric(),
                 d=None,DEBUG=False,**kwargs):
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

        # make torch tensor of velocity field
        U_t = torch.as_tensor(check_velocity_field(U,temporal_evolution,x=self.x,y=self.y).values, dtype=self.dtype, device=self.device)
        V_t = torch.as_tensor(check_velocity_field(V,temporal_evolution,x=self.x,y=self.y).values, dtype=self.dtype, device=self.device)
        
        # Spatial gradients for every time snapshot
        dU_dy, dU_dx = torch.gradient(U_t, spacing=(self.dy, self.dx), dim=(-2, -1))
        dV_dy, dV_dx = torch.gradient(V_t, spacing=(self.dy, self.dx), dim=(-2, -1))

        # Check the bathymetry and make it into a pytorch compatible tensor
        if d is not None:
            d_t = torch.as_tensor(check_bathymetry(d=d,x=self.x,y=self.y).values, dtype=self.dtype, device=self.device)
        else:
            d_static = 1e5
            logging.info(f'Hardcoding bathymetry to {d_static}m since not given.')
            d_t = torch.as_tensor(check_bathymetry(d=np.ones((ny,nx))*d_static,x=self.x,y=self.y).values, dtype=self.dtype, device=self.device)
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
    

    def wave(self,T,theta,d,U=0,V=0):
        """ Method computing wave number from initial wave period.
        Solving implicitly for the wave number k, with initial guess from the approximate
        wave number according to Eckart (1952)

        Args:
            T (float): Wave period
            theta (float): radians. Wave direction
            d (float): water depth [m]
            U (float): ambient current in x-direction [m/s]
            V (float): ambient current in y-direction [m/s]


        Returns:
            k0 (float): wave number
            ray_kx0 (float): wave number in x-direction
            ray_ky0 (float): wave number in y-direction
        """

        g=self.g

        sigma = (2*np.pi)/T
        k0 = (sigma**2)/g

        # approximation of wave number according to Eckart (1952)
        alpha = k0*d
        k_approx = (alpha/np.sqrt(np.tanh(alpha)))/d

        from scipy.optimize import fsolve

        def k_imp(kk, d=d, g=g,T=T,U=U,V=V):
            kx_ap = kk*np.cos(theta)
            ky_ap = kk*np.sin(theta)
            return (np.sqrt((g*kk * np.tanh(kk*d)))+ kx_ap*U + ky_ap*V ) - (2*np.pi)/T 
        
        
        k = fsolve(k_imp,k_approx)
        k = k.squeeze()

        kx = k*np.cos(theta)
        ky = k*np.sin(theta)
        #logger.info('wave: {}, {},{}, and diff {}'.format(k,kx,ky,np.abs(k_approx)))
        return kx,ky


    def set_initial_condition(self, wave_period, theta0, **kwargs):
        """Set initial positions and wavenumbers for all wave rays."""

        device = self.field.device
        dtype = self.field.dtype
        nb_wave_rays = self.nb_wave_rays

        valid_sides = ["left", "right", "top", "bottom"]

        # 1. Initial ray positions
        if "incoming_wave_side" in kwargs:
            side = kwargs["incoming_wave_side"]

            if side not in valid_sides:
                logger.info("Invalid initial side. Left will be used.")
                side = "left"

            if side in ("left", "right"):
                x_value = (self.domain_X0 if side == "left" else self.domain_XN)
                xs = torch.full((nb_wave_rays,), x_value.item(), dtype=dtype, device=device)
                ys = torch.linspace(self.domain_Y0.item(), self.domain_YN.item(), nb_wave_rays, dtype=dtype, device=device)

            else:
                y_value = (self.domain_YN if side == "top" else self.domain_Y0)
                xs = torch.linspace(self.domain_X0.item(), self.domain_XN.item(), nb_wave_rays, dtype=dtype, device=device)
                ys = torch.full((nb_wave_rays,), y_value.item(), dtype=dtype, device=device)

        else:
            ipx = kwargs.get("ipx")
            ipy = kwargs.get("ipy")

            if ipx is None and ipy is None:
                logger.info("No initial positions given. Using left boundary.")
                xs = np.full(nb_wave_rays, self.domain_X0.item())
                ys = np.linspace(self.domain_Y0.item(), self.domain_YN.item(), nb_wave_rays)
            else:
                if ipx is None or ipy is None:
                    raise ValueError("Provide both ipx and ipy, or neither.")

                xs = np.asarray(ipx)
                ys = np.asarray(ipy)

                if xs.ndim == 0:
                    xs = np.full(nb_wave_rays, xs.item())
                if ys.ndim == 0:
                    ys = np.full(nb_wave_rays, ys.item())

                if len(xs) != nb_wave_rays or len(ys) != nb_wave_rays:
                    raise ValueError("ipx and ipy must each have nb_wave_rays values.")

        # Normalize positions to tensors on the correct device.
        xs = torch.as_tensor(xs, dtype=dtype, device=device).reshape(-1)
        ys = torch.as_tensor(ys, dtype=dtype, device=device).reshape(-1)

        # 2. Store initial positions
        self.initial_rays[:, 0] = xs
        self.initial_rays[:, 1] = ys

        # 3. Initial directions
        if isinstance(theta0, (float, int)):
            theta0 = np.full(nb_wave_rays, theta0)
        elif isinstance(theta0, np.ndarray):
            if len(theta0) != nb_wave_rays:
                raise ValueError("theta0 must have nb_wave_rays values.")
        else:
            raise TypeError("theta0 must be a float, int, or NumPy array.")

        # 4. Read depth and velocity from self.field
        idt = int(self.velocity_idt[0])

        # Convert physical coordinates to grid indices.
        ix = torch.round((xs - self.domain_X0) / (self.domain_XN - self.domain_X0) * (self.nx - 1)).long().clamp(0, self.nx - 1)

        iy = torch.round((ys - self.domain_Y0) / (self.domain_YN - self.domain_Y0) * (self.ny - 1)).long().clamp(0, self.ny - 1)

        # self.field layout: (time, 7, ny, nx)
        depth = self.field[idt, 0, iy, ix]
        U = self.field[idt, 1, iy, ix]
        V = self.field[idt, 2, iy, ix]

        # 5. Calculate initial wavenumbers
        # self.wave() uses NumPy/SciPy, so calculate on the CPU.
        depth_cpu = depth.detach().cpu().numpy()
        U_cpu = U.detach().cpu().numpy()
        V_cpu = V.detach().cpu().numpy()

        kx = np.empty(nb_wave_rays)
        ky = np.empty(nb_wave_rays)

        for i in range(nb_wave_rays):
            kx[i], ky[i] = self.wave(T=wave_period, theta=theta0[i], d=depth_cpu[i], U=U_cpu[i], V=V_cpu[i])

        self.initial_rays[:, 2] = torch.as_tensor(kx, dtype=dtype, device=device)
        self.initial_rays[:, 3] = torch.as_tensor(ky, dtype=dtype, device=device)

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
            "ray_kx": properties.k[:, 0],
            "ray_ky": properties.k[:, 1],
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


    def solve(self, solver=None, derivative_scheme = Autoderivative):
        """
        Trace rays and store all properties at their corresponding
        timestep and original ray ID.

        State vector: [x, y, kx, ky]
        """
        self.initialize_ray_history()

        nr = self.nb_wave_rays
        nt = self.nt

        scheme = derivative_scheme(self)

        active_ids = torch.arange(nr, dtype=torch.long, device=self.device)
        active_rays = active_rays = self.initial_rays.clone()

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

            # Remove rays with any non-finite state component.
            valid = torch.isfinite(next_rays).all(dim=1)

            active_ids = active_ids[valid]
            active_rays = next_rays[valid].detach()

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
                    if (np.any((self.ray_y[i,:][valid_x]>y0)*(self.ray_y[i,:][valid_x]<yn))):
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