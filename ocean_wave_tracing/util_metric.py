import torch as t

# The classes in this document define either the inverse metric tensor for the coordinate system of interest
# or Jacobians for converting between them

#==========================================================================================
#                                    Metric tensors
#==========================================================================================

class CartesianMetric():
# All Metric related stuff for the Cartesian case
    def find_gmj(self, position):
        N = position.shape[0]
        g_11 = t.ones(N, dtype=position.dtype, device=position.device)
        g_12 = t.zeros(N, dtype=position.dtype, device=position.device)
        g_21 = t.zeros(N, dtype=position.dtype, device=position.device)
        g_22 = t.ones(N, dtype=position.dtype, device=position.device)
        return t.stack([
            t.stack([g_11, g_12], dim=-1),
            t.stack([g_21, g_22], dim=-1)
        ], dim=-2)


class SphericalMetric():
# All Metric related stuff for the spherical case
    def __init__(self):
        self.R = 6371e3

    def find_gmj(self, position):
        N = position.shape[0]
        _, lat = position.T
#        lat = t.deg2rad(lat)
        g_11 = 1/(self.R*t.cos(lat))**2
        g_12 = t.zeros(N, dtype=position.dtype, device=position.device)
        g_21 = t.zeros(N, dtype=position.dtype, device=position.device)
        g_22 = t.ones(N, dtype=position.dtype, device=position.device)/(self.R**2)
        return t.stack([
            t.stack([g_11, g_12], dim=-1),
            t.stack([g_21, g_22], dim=-1)
        ], dim=-2)


#==========================================================================================
#                            Projections and Jacobi-Matrices
#==========================================================================================

# Need an own class to facilitate the transformation between two different coordinate systems
class Identity():
    def __init__(self):
        pass

    def a_1(self,x,y):
        return x
    def a_2(self,x,y):
        return y

    def b_1(self,x,y):
        return x
    def b_2(self,x,y):
        return y

    def find_J_ij(self, position):
        N = position.shape[0]
        #Function for computing the jacobi matrix for converting lat, lon coordinates to Cartesian in stereographic projection

        J_11 = t.ones(N, dtype=position.dtype, device=position.device)
        J_12 = t.zeros(N, dtype=position.dtype, device=position.device)
        J_21 = t.zeros(N, dtype=position.dtype, device=position.device)       
        J_22 = t.ones(N, dtype=position.dtype, device=position.device)
        return t.stack([
            t.stack([J_11, J_12], dim=-1),
            t.stack([J_21, J_22], dim=-1)
        ], dim=-2)

    def find_K_ji(self, position):
            N = position.shape[0]
            #Function for computing the jacobi matrix for converting lat, lon coordinates to Cartesian in stereographic projection
    
            J_11 = t.ones(N, dtype=position.dtype, device=position.device)
            J_12 = t.zeros(N, dtype=position.dtype, device=position.device)
            J_21 = t.zeros(N, dtype=position.dtype, device=position.device)       
            J_22 = t.ones(N, dtype=position.dtype, device=position.device)
            return t.stack([
                t.stack([J_11, J_12], dim=-1),
                t.stack([J_21, J_22], dim=-1)
            ], dim=-2)


# The polar case from Snyder, inherits most from the regular ellipsoid projection:
class PolarEllipsoidStereographicProjection():
    def __init__(self, lat_0=t.pi/2, lat_t=t.pi*2/3, lon_0=0, 
                 x_0=0, y_0=0, a=6378137, b=6356752.3142):
        
        self.lat_0 = t.tensor(lat_0)
        self.lon_0 = t.tensor(lon_0)
        self.lat_t = t.tensor(lat_t)

        self.x_0 = t.tensor(x_0)
        self.y_0 = t.tensor(y_0)
        self.r_e = t.tensor(a)

        # calculate the eccentricity e
        assert a > b, "a should be larger than b"
        self.e = t.sqrt(1 - (t.tensor(b)**2) / (self.r_e**2))

        # calculate scale factor for true-scale latitude
        self.c_t = self.c(self.lat_t)
        # calculate helper term
        self.q_t = self.q(self.lat_t)

    def q(self, lat):
        return 1-(self.e*t.sin(lat))**2
    def rho(self, lat):
        return (self.r_e*t.cos(self.lat_t)*self.c(lat))/(t.sqrt(self.q_t)*self.c_t)
    def c(self, lat):
        return t.tan(t.pi/4 - lat/2) / ((1 - self.e*t.sin(lat)) / (1 + self.e*t.sin(lat)))**(self.e/2)

    def dc_dlat(self, lat):
        return self.c(lat) * ((self.e**2 * t.cos(lat) / self.q(lat)) - (1 / t.cos(lat)))
    def drho_dlat(self, lat):
        return (self.r_e*t.cos(self.lat_t)*self.dc_dlat(lat))/(t.sqrt(self.q_t)*self.c_t)

    # Functions that facilitate the direct transformation between lat, lon and x,y
    def a_1(self, lon, lat):
        return self.rho(lat)*t.sin(lon-self.lon_0) + self.x_0
    def a_2(self, lon, lat):
        return -self.rho(lat)*t.cos(lon-self.lon_0) + self.y_0

    # Functions that recover lat, lon from x,y. These are iterative, so use sparingly (here: only for ray initialization)
    def b_1(self, x, y):
        X = x - self.x_0
        Y = y - self.y_0
        return self.lon_0 + t.atan2(X, -Y)
    def b_2(self, x, y, max_iter=20):
        X = x - self.x_0
        Y = y - self.y_0

        rho = t.sqrt(X**2 + Y**2)

        c_target = (
            rho * t.sqrt(self.q_t) * self.c_t
            / (self.r_e * t.cos(self.lat_t))
        )

        # Initial spherical estimate
        lat = t.pi/2 - 2*t.atan(c_target)

        # Iteratively account for ellipsoid eccentricity
        for _ in range(max_iter):
            factor = (
                (1 - self.e*t.sin(lat))
                / (1 + self.e*t.sin(lat))
            )**(self.e/2)

            lat = t.pi/2 - 2*t.atan(c_target * factor)

        return lat
    
    # Functions for the entries of the Jacobi matrix
    def dxdlon(self, rho, deltalon):
        return rho*t.cos(deltalon)
    def dxdlat(self, drho_dlat, deltalon):
        return drho_dlat*t.sin(deltalon)
    def dydlon(self, rho, deltalon):
        return rho*t.sin(deltalon)
    def dydlat(self, drho_dlat, deltalon):
        return -drho_dlat*t.cos(deltalon)
        
    def find_J_ij(self, position):
        #Function for computing the jacobi matrix for converting lat, lon coordinates to Cartesian in stereographic projection
        lon = position[:, 0]
        lat = position[:, 1]

        # precompute some common terms
        drhodlat = self.drho_dlat(lat)
        rho = self.rho(lat)
        deltalon = lon - self.lon_0

        J_11 = self.dxdlon(rho, deltalon)        
        J_12 = self.dxdlat(drhodlat, deltalon)    
        J_21 = self.dydlon(rho, deltalon)
        J_22 = self.dydlat(drhodlat, deltalon)   
        
        return t.stack([
            t.stack([J_11, J_12], dim=-1),
            t.stack([J_21, J_22], dim=-1)
        ], dim=-2)

    def find_K_ji(self, position):

        # function for finding the inverse Jacobi matrix for a given lat, lon coordinate
        return t.linalg.inv(self.find_J_ij(position))
        
