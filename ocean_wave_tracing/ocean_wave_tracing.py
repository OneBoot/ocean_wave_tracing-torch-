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

from .util_metric import find_g_im_h, find_gnj, find_J_jm
import torch


logger = logging.getLogger(__name__)
logging.basicConfig(filename='ocean_wave_tracing.log', level=logging.INFO)
logging.info('\nStarted')

#gravitational constant
G=9.81

class Wave_tracing():
    """ Class computing the path of ocean wave rays according to the geometrical
    optics approximation.
    """
    def __init__(self, U, V,  nx, ny, nt, T, dx, dy,
                 nb_wave_rays, domain_X0, domain_XN, domain_Y0, domain_YN,
                 temporal_evolution=False,
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

        #print(torch.__version__)
        #Should be changeable later
        self.device = "cpu"
        self.d_type = torch.float32

        self.g = 9.81
        self.ray_x = np.zeros((nb_wave_rays,nt))
        self.ray_y = np.zeros((nb_wave_rays,nt))
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.dx = dx
        self.dy = dy
        self.nb_wave_rays = nb_wave_rays
        self.C = -1
        assert nb_wave_rays > 0, "Number of wave rays must be larger than zero"


        self.domain_X0 = torch.as_tensor(domain_X0, dtype=self.d_type, device=self.device) # left side
        self.domain_XN = torch.as_tensor(domain_XN, dtype=self.d_type, device=self.device) # right side
        self.domain_Y0 = torch.as_tensor(domain_Y0, dtype=self.d_type, device=self.device) # bottom
        self.domain_YN = torch.as_tensor(domain_YN, dtype=self.d_type, device=self.device) # top
        self.T = T

        self.temporal_evolution = temporal_evolution
        self.debug = DEBUG

        # Setting up X and Y domain
        self.x = np.linspace(domain_X0, domain_XN, nx)
        self.y = np.linspace(domain_Y0, domain_YN, ny)

        # Check the bathymetry
        if d is not None:
            self.d = check_bathymetry(d=d,x=self.x,y=self.y)
        else:
            d_static = 1e5
            logging.info(f'Hardcoding bathymetry to {d_static}m since not given.')
            self.d = check_bathymetry(d=np.ones((ny,nx))*d_static,x=self.x,y=self.y)

        # make xarray DataArray of velocity field
        self.U = check_velocity_field(U,temporal_evolution,x=self.x,y=self.y)
        self.V = check_velocity_field(V,temporal_evolution,x=self.x,y=self.y)

        self.field = torch.from_numpy(
            np.stack([
                self.d.values,
                self.U.values[0],
                self.V.values[0],
            ])
        ).float().unsqueeze(0)

        print(np.isnan(self.U.values).any())

        self.rays = torch.tensor([0.,0.,0.,0.], requires_grad=True).repeat(nt, nb_wave_rays, 1)

        # Time
        self.dt = T/nt
        self.nb_velocity_time_steps = len(self.U.time)
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

    # needs to be rewritten
    def c_intrinsic(self,k,d,group_velocity=False):
        """ Computing the intrinsic wave phase and group velocity according
        to the general dispersion relation

        Args:
            k (float): wave number (numpy array)
            d (float): depth (numpy array)
            group_velocity (bool): returns group velocity (True) or phase
                velocity (False)

        Returns:
            instrinsic velocity (float): group or phase velocity depending on
                flag
        """

        g=self.g
        dw_criteria = k*d>25

        if dw_criteria.all():
            c_in = np.sqrt(g/k)
            n=0.5
        else:
            c_in = np.sqrt((g/k)*np.tanh(k*d)) #intrinsic
            n = 0.5 * (1 + (2*k*d)/np.sinh(2*k*d))

        if group_velocity:
            return c_in*n
        else:
            return c_in

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


    def set_initial_condition(self, wave_period, theta0,**kwargs):
        """ Setting inital conditions for the domain. Support domain side and
            initial x- and y-positions. However, side trumps initial position
            if both are given.

            args:
            wave_period (float): Wave period.
            theta0 (rad, float): Wave initial direction. In radians.
                         (0,.5*pi,pi,1.5*pi) correspond to going
                         (right, up, left, down).

            **kwargs
                incoming_wave_side (str): side for incoming wave direction
                                            [left, right, top, bottom]
                ipx (float, array of floats): initial position x
                ipy (float, array of floats): initial position y
        """

        nb_wave_rays = self.nb_wave_rays

        valid_sides = ['left', 'right','top','bottom']

        if 'incoming_wave_side' in kwargs:
            i_w_side = kwargs['incoming_wave_side']

            if not i_w_side in valid_sides:
                logger.info('Invalid initial side. Left will be used.')
                i_w_side = 'left'

            if i_w_side == 'left':
                xs = np.ones(nb_wave_rays)*self.domain_X0
                ys = np.linspace(self.domain_Y0, self.domain_YN, nb_wave_rays)

            elif i_w_side == 'right':
                xs = np.ones(nb_wave_rays)*self.domain_XN
                ys = np.linspace(self.domain_Y0, self.domain_YN, nb_wave_rays)

            elif i_w_side == 'top':
                xs = np.linspace(self.domain_X0, self.domain_XN, nb_wave_rays)
                ys = np.ones(nb_wave_rays)*self.domain_YN

            elif i_w_side == 'bottom':
                xs = np.linspace(self.domain_X0, self.domain_XN, nb_wave_rays)
                ys = np.ones(nb_wave_rays)*self.domain_Y0

        else:
            logger.info('No initial side given. Try with discrete points')

            ipx = kwargs.get('ipx', None)
            ipy = kwargs.get('ipy', None)

            if ipx is None and ipy is None:
                logger.info('No initial position points given. Left will be used')
                xs = np.ones(nb_wave_rays)*self.domain_X0
                ys = np.linspace(self.domain_Y0, self.domain_YN, nb_wave_rays)

            else:
                # First check initial position x
                if np.isfinite(ipx).all():
                    #Check if it is an array
                    if isinstance(ipx,np.ndarray):
                        assert nb_wave_rays == len(kwargs['ipx']), "Need same dimension on initial x-values"
                        xs=ipx.copy()
                    # if not, use it as single value
                    else:
                        ipx = np.ones(nb_wave_rays)*ipx
                        xs=ipx.copy()

                if np.isfinite(ipy).all():
                    if isinstance(ipy,np.ndarray):
                        assert nb_wave_rays == len(kwargs['ipy']), "Need same dimension on initial x-values"
                        ys=ipy.copy()
                    else:
                        ipy = np.ones(nb_wave_rays)*ipy
                        ys=ipy.copy()

        self.rays[0,:,0] = torch.tensor(xs)
        self.rays[0,:,1] = torch.tensor(ys)

        #Theta0
        if type(theta0) is float or type(theta0) is int:
            theta0 = np.ones(nb_wave_rays)*theta0
        elif isinstance(theta0,np.ndarray):
            assert nb_wave_rays == len(theta0), "Initial values must have same dimension as number of wave rays"
        else:
            logger.error('Theta0 must be either float or numpy array. Terminating.')
            sys.exit()


        # set inital wave properties
        logger.error(f'{self.velocity_idt}, {len(self.velocity_idt)}')
        for i in range(nb_wave_rays):
            ray_kx, ray_ky = self.wave(T=wave_period,
                                        theta=theta0[i],
                                        d=self.d.sel(y=ys[i],x=xs[i],method='nearest').values,
                                        U=self.U.isel(time=self.velocity_idt[0]).sel(y=ys[i],x=xs[i],method='nearest').values,
                                        V=self.V.isel(time=self.velocity_idt[0]).sel(y=ys[i],x=xs[i],method='nearest').values
                                        )
            self.rays[0,i,2:] = torch.tensor([ray_kx, ray_ky])

        #Check the CFL condition

    #-------------------------------------------------------------------------------------------------------------------------------------------------
    # PYTORCH FUNCTIONALITY HERE (WITHOUT DELETING)
    #------------------------------------------------------------------------------------------------------------------------------------------------–

    def find_k(self, Theta, g_metric):
        # Theta: (N, 2)
        # g_metric: (2, 2)

        return torch.einsum("ni,ij->nj", Theta, g_metric)

    def find_sigma(self, k, d):
        # k: (N, 2)
        # d: (N,)
        k_norm = torch.linalg.vector_norm(k, dim=1)

        return torch.sqrt(G * k_norm * torch.tanh(k_norm * d))

    def find_U_and_d(self, positions):
        # positions: (N, 2)

        x = (2*(positions[:, 0] - self.domain_X0) / (self.domain_XN-self.domain_X0)-1)
        y = (2*(positions[:, 1]-self.domain_Y0) / (self.domain_YN-self.domain_Y0)-1)

        # We need to reshape the array, since the interpolation step is batched
        # (N, 2) -> (1, N, 1, 2) 
        grid = torch.stack([x, y], dim=-1).reshape(1, -1, 1, 2)

        interpolated = torch.nn.functional.grid_sample(self.field, grid, mode="bilinear", align_corners=True)

        # We need to reshape the array back, removing redndant layers
        # (1, 3, N, 1) -> (N, 3)
        interpolated = interpolated[0, :, :, 0].T

        local_d = interpolated[:, 0]
        local_U = interpolated[:, 1:]

        return local_d, local_U

    def Omega(self, rays):
        # rays: (N, 4)
        #       x, y, kx, ky

        positions = rays[:, :2]
        Theta = rays[:, 2:]

        d, U = self.find_U_and_d(positions)

        g_im_h = find_g_im_h(positions[0])
        g_nj = find_gnj(positions[0])
        J_jm = find_J_jm(positions[0])

        k = self.find_k(Theta, g_nj)

        sigma = self.find_sigma(k, d)

        current_term = torch.einsum("im,mj,ni,nj->n", g_im_h, J_jm, U, k)

        return sigma + current_term

    def dOmega(self, rays):
        rays = rays.clone().detach().requires_grad_(True)

        omega = self.Omega(rays)

        dOmega = torch.autograd.grad(omega.sum(), rays, create_graph=False, allow_unused=False)[0]

        return dOmega
    
    def solve(self, solver=RungeKutta4):
        """ Solve the geometrical optics equations numerically by means of the
            method of characteristics
        """

        if not callable(solver):
            raise TypeError('f is %s, not a solver' % type(solver))
        
        nt = self.nt
        velocity_idt = self.velocity_idt

        counter=0
        t = np.linspace(0,self.T,nt)
        dt = self.T/nt

        for n in range(0,nt-1):

            #======================================================
            ### numerical integration of the wave ray equations ###
            #======================================================
            
            # Noticed that the original did not make use of the Runge-Kutta functionality, so I skip this here
            dOmega = self.dOmega(self.rays[n, :])

            dOmega = (dOmega[..., [2, 3, 0, 1]] * torch.tensor([1., 1., -1., -1.]))
            self.rays[n+1,:] = self.rays[n,:] + dt*dOmega

        logging.info('Stoppet at time idt: {}'.format(velocity_idt[n]))
        self.ray_x[:,:] = self.rays[:, :, 0].detach().numpy().T
        self.ray_y[:,:] = self.rays[:, :, 1].detach().numpy().T

        print(self.ray_x.shape)

    def to_ds(self,**kwargs):
        """Convert wave ray information to xarray object"""

        if 'proj4' in kwargs:
            lons,lats = self.to_latlon(kwargs['proj4'])
        else:
            lons = np.zeros((self.nb_wave_rays,self.nt))
            lats = lons.copy()

        variables = {'ray_k':self.ray_k,
                    'ray_kx':self.ray_kx,
                    'ray_ky':self.ray_ky,
                    'ray_x':self.ray_x,
                    'ray_y':self.ray_y,
                    'ray_U':self.ray_U,
                    'ray_V':self.ray_V,
                    'ray_theta':self.ray_theta,
                    'ray_cg':self.ray_cg,
                    'ray_depth':self.ray_depth,
                    'ray_lat': lats,
                    'ray_lon':lons,
                    'ray_dudx':self.ray_dudx,
                    'ray_dvdy':self.ray_dvdy,
                    'ray_dudy':self.ray_dudy,
                    'ray_dvdx':self.ray_dvdx
                    }

        source = files('ocean_wave_tracing').joinpath('ray_metadata.json')
        with as_file(source) as sfile:
            logging.info('Loading JSON file')
            data = json.loads(sfile.read_text())
            logging.info('Finished loading JSON file')

        # relative time
        t = np.linspace(0,self.T,self.nt)
        ray_id = np.arange(self.nb_wave_rays)

        vars = [make_xarray_dataArray(var=variables[vname], t=t,rays=ray_id,name=vname,attribs=data[vname]) for vname in list(variables.keys())]

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
        """ Method for reprojecting wave rays to latitude/longitude values

        Args:
            proj4 (str): proj4 string

        Returns:
            lons (2d): wave rays in longitude
            lats (2d): wave rays in latitude
        """
        lats = np.zeros((self.nb_wave_rays,self.nt))
        lons = np.zeros((self.nb_wave_rays,self.nt))
        ##print(pyproj.__dict__.keys())
        for i in range(self.nb_wave_rays):
            lons[i,:],lats[i,:] = pyproj.Transformer.from_proj(proj4,'epsg:4326', always_xy=True).transform(self.ray_x[i,:], self.ray_y[i,:])

        return lons, lats
    
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