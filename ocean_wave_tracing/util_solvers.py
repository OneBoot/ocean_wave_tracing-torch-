class Advection():
    #def set_variables(self, cg, k, kx, U):
    def __init__(self, cg, k, kx, U):
        #cg, k, kx, U = self.cg, self.k, self.kx, self.U
        self.cg = cg
        self.k = k
        self.kx = kx
        self.U = U

    def __call__(self,u,t):
        cg, k, kx, U = self.cg, self.k, self.kx, self.U

        #f = [cg*(kx/k) + U]
        f = cg*(kx/k) + U
        return f

class WaveNumberEvolution():
    def __init__(self,d_sigma, kx, ky, dUkx, dUky):
        self.d_sigma = d_sigma
        self.kx = kx
        self.ky = ky
        self.dUkx = dUkx
        self.dUky = dUky

    def __call__(self, u, t):
        d_sigma, kx, ky, dUkx, dUky = self.d_sigma, self.kx, self.ky, self.dUkx, self.dUky

        #f = [d_sigma + kx*dUkx + ky*dUky]
        f = -(d_sigma + kx*dUkx + ky*dUky)
        return f


class ODESolver(object):
    def __init__(self, derivative_scheme):
        if not callable(derivative_scheme):
            raise TypeError(f'derivative_scheme is {type(derivative_scheme)}, not callable')
        self.derivative_scheme = derivative_scheme

    def advance(self, u, k, t):
        raise NotImplementedError


class ForwardEuler(ODESolver):
    def advance(self, u, idt, dt):
        du = self.derivative_scheme(u, idt)
        return u + dt * du


class RungeKutta4(ODESolver):
    def advance(self, u, idt, dt):

        K1 = dt * self.derivative_scheme(u, idt)
        K2 = dt * self.derivative_scheme(u + 0.5*K1, idt)
        K3 = dt * self.derivative_scheme(u + 0.5*K2, idt)
        K4 = dt * self.derivative_scheme(u + K3, idt)

        return u + (K1 + 2*K2 + 2*K3 + K4) / 6.0