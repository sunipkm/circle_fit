"""
Source: 
http://www.scipy.org/Cookbook/Least_Squares_Circle
"""
from __future__ import annotations
import numpy as np
from scipy import optimize
from scipy import  odr
from matplotlib import pyplot as p, cm, colors

class CircleFit:
    methods = ['algebraic', 'leastsq', 'leastsq+jacobian', 'odr', 'odr+jacobian']
    def __init__(self, xdata: np.ndarray, ydata: np.ndarray, method: str = 'algebraic'):
        """Fit a circle to data.

        Args:
            xdata (np.ndarray): X coordinates of data
            ydata (np.ndarray): Y coordinates of data
            method (str): Fit method, can be one of 'algebraic', 'leastsq', 'leastsq+jacobian', 'odr' and 'odr+jacobian'. Default is 'algebraic'.

        Raises:
            RuntimeError: Invalid method, fit fails to converge (with ODR methods.)
        """
        if method not in CircleFit.methods:
            raise RuntimeError('Method %s not valid'%(method))
        self._method = method
        self._x = xdata.copy()
        self._y = ydata.copy()

        if method == 'algebraic':
            self._algebraic()
        elif method == 'leastsq':
            self._leastsq()
        elif method == 'leastsq+jacobian':
            self._leastsq_jacobian()
        elif method == 'odr':
            self._odr()
        elif method == 'odr+jacobian':
            self._odr_jacobian()

    def fit_circle(self, method: str = 'algebraic'):
        """Fit circle using the specified method

        Args:
            method (str):  Fit method, can be one of 'algebraic', 'leastsq', 'leastsq+jacobian', 'odr' and 'odr+jacobian'. Default is 'algebraic'.

        Raises:
            RuntimeError: Invalid method, fit fails to converge (with ODR methods.)
        """
        if method not in CircleFit.methods:
            raise RuntimeError('Method %s not valid'%(method))
        self._method = method
        if method == 'algebraic':
            self._algebraic()
        elif method == 'leastsq':
            self._leastsq()
        elif method == 'leastsq+jacobian':
            self._leastsq_jacobian()
        elif method == 'odr':
            self._odr()
        elif method == 'odr+jacobian':
            self._odr_jacobian()

    def _algebraic(self):
        x_m = np.mean(self._x)
        y_m = np.mean(self._y)
        
        u = self._x - x_m
        v = self._y - y_m

        # linear system defining the center in reduced coordinates (uc, vc):
        # Suu * uc +  Suv * vc = (Suuu + Suvv)/2
        # Suv * uc +  Svv * vc = (Suuv + Svvv)/2
        Suv  = np.sum(u*v)
        Suu  = np.sum(u**2)
        Svv  = np.sum(v**2)
        Suuv = np.sum(u**2 * v)
        Suvv = np.sum(u * v**2)
        Suuu = np.sum(u**3)
        Svvv = np.sum(v**3)

        # Solving the linear system
        A = np.array([ [ Suu, Suv ], [Suv, Svv]])
        B = np.array([ Suuu + Suvv, Svvv + Suuv ])/2.0
        uc, vc = np.linalg.solve(A, B)

        xc = x_m + uc
        yc = y_m + vc

        self._center = (xc, yc)
        # Calculation of all distances from the center (xc_1, yc_1)
        Ri      = np.sqrt((self._x-xc)**2 + (self._y-yc)**2)
        self._radius =   R     = np.mean(Ri)
        self._residu  = np.sum((Ri-R)**2)
        self._residu2 = np.sum((Ri**2-R**2)**2)

    @property
    def center(self):
        """Center of the circle.

        Returns:
            tuple(float, float): X and Y coordinates of the center of the circle.
        """
        return self._center
    
    @property
    def radius(self):
        """Radius of the fit circle.

        Returns:
            float: radius of the circle
        """
        return self._radius

    @property
    def rad_std(self):
        """Standard deviation of the fit circle.

        Returns:
            float: Standard deviation of the fit circle.
        """
        return self._rad_std
    
    @property
    def residu(self):
        """Residual of fit.

        Returns:
            float: Residual of fit.
        """
        return self._residu

    @property
    def residu2(self):
        """Residual^2 of fit.

        Returns:
            float: Residual^2 of fit.
        """
        return self._residu2

    @property
    def params(self):
        """Returns a tuple containing the circle fit results.

        Returns:
            tuple: ((x_c, y_c), R, R_std, residu, residu2)
        """
        return (self._center, self._radius, self._rad_std, self._residu, self._residu2)

    def _calc_radius(self, xc, yc):
        return np.sqrt((self._x-xc)**2 + (self._y-yc)**2)

    def _optimize_radius(self, c):
        Ri = self._calc_radius(*c)
        return Ri - np.mean(Ri)
    
    def _leastsq(self):
        x_m = np.mean(self._x)
        y_m = np.mean(self._y)
        center_estimate = x_m, y_m
        center_2, ier = optimize.leastsq(self._optimize_radius, center_estimate)

        xc_2, yc_2 = self._center = center_2
        Ri_2       = self._calc_radius(xc_2, yc_2)
        self._radius = R_2        = Ri_2.mean()
        self._rad_std = Ri_2.std()
        self._residu   = np.sum((Ri_2 - R_2)**2)
        self._residu2  = np.sum((Ri_2**2-R_2**2)**2)

    def _calc_jacobian(self, c):
        """ Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        xc, yc     = c
        df2b_dc    = np.empty((len(c), self._x.size))

        Ri = self._calc_radius(xc, yc)
        df2b_dc[ 0] = (xc - self._x)/Ri                   # dR/dxc
        df2b_dc[ 1] = (yc - self._y)/Ri                   # dR/dyc
        df2b_dc       = df2b_dc - df2b_dc.mean(axis=1)[:, np.newaxis]

        return df2b_dc

    def _leastsq_jacobian(self):
        x_m = np.mean(self._x)
        y_m = np.mean(self._y)
        center_estimate = x_m, y_m
        center_2b, ier = optimize.leastsq(self._optimize_radius, center_estimate, Dfun=self._calc_jacobian, col_deriv=True)

        xc_2b, yc_2b = self._center = center_2b
        Ri_2b        = self._calc_radius(xc_2b, yc_2b)
        R_2b = Ri_2b.mean()
        self._radius = np.mean(Ri_2b)
        self._rad_std = np.std(Ri_2b)
        self._residu    = np.sum((Ri_2b - R_2b)**2)
        self._residu2   = np.sum((Ri_2b**2-R_2b**2)**2)

    @staticmethod
    def _calc_radius_implicit(beta, x):
        return (x[0]-beta[0])**2 + (x[1]-beta[1])**2 -beta[2]**2

    def _odr(self):
        x_m = np.mean(self._x)
        y_m = np.mean(self._y)
        # initial guess for parameters
        R_m = self._calc_radius(x_m, y_m).mean()
        beta0 = [ x_m, y_m, R_m]

        # for implicit function :
        #       data.x contains both coordinates of the points
        #       data.y is the dimensionality of the response
        lsc_data   = odr.Data(np.row_stack([self._x, self._y]), y=1)
        lsc_model  = odr.Model(CircleFit._calc_radius_implicit, implicit=True)
        lsc_odr    = odr.ODR(lsc_data, lsc_model, beta0)
        lsc_out    = lsc_odr.run()

        xc_3, yc_3, R_3 = lsc_out.beta
        self._center = (xc_3, yc_3)
        Ri_3       = self._calc_radius(xc_3, yc_3)
        self._radius = np.mean(Ri_3)
        self._rad_std = np.std(Ri_3)
        self._residu   = np.sum((Ri_3 - R_3)**2)
        self._residu2  = np.sum((Ri_3**2-R_3**2)**2)

    @staticmethod
    def _odr_jacb(beta, x):
        """ Jacobian function with respect to the parameters beta.
        return df_3b/dbeta
        """
        xc, yc, r = beta
        xi, yi    = x

        df_db    = np.empty((beta.size, x.shape[1]))
        df_db[0] =  2*(xc-xi)                     # d_f/dxc
        df_db[1] =  2*(yc-yi)                     # d_f/dyc
        df_db[2] = -2*r                           # d_f/dr

        return df_db

    @staticmethod
    def _odr_jacd(beta, x):
        """ Jacobian function with respect to the input x.
        return df_3b/dx
        """
        xc, yc, r = beta
        xi, yi    = x

        df_dx    = np.empty_like(x)
        df_dx[0] =  2*(xi-xc)                     # d_f/dxi
        df_dx[1] =  2*(yi-yc)                     # d_f/dyi

        return df_dx

    @staticmethod
    def _odr_calc_estimate(data):
        """ Return a first estimation on the parameter from the data  """
        xc0, yc0 = data.x.mean(axis=1)
        r0 = np.sqrt((data.x[0]-xc0)**2 +(data.x[1] -yc0)**2).mean()
        return xc0, yc0, r0

# for implicit function :
#       data.x contains both coordinates of the points
#       data.y is the dimensionality of the response
    def _odr_jacobian(self):
        lsc_data  = odr.Data(np.row_stack([self._x, self._y]), y=1)
        lsc_model = odr.Model(CircleFit._calc_radius_implicit, implicit=True, estimate=CircleFit._odr_calc_estimate, fjacd=CircleFit._odr_jacd, fjacb=CircleFit._odr_jacb)
        lsc_odr   = odr.ODR(lsc_data, lsc_model)    # beta0 has been replaced by an estimate function
        lsc_odr.set_job(deriv=3)                    # use user derivatives function without checking
        lsc_odr.set_iprint(iter=1, iter_step=1)     # print details for each iteration
        lsc_out   = lsc_odr.run()

        xc_3b, yc_3b, R_3b = lsc_out.beta
        self._center = (xc_3b, yc_3b)
        Ri_3b       = self._calc_radius(xc_3b, yc_3b)
        self._rad_std = np.std(Ri_3b)
        self._radius = np.mean(Ri_3b)
        self._residu   = np.sum((Ri_3b - R_3b)**2)
        self._residu2  = np.sum((Ri_3b**2-R_3b**2)**2)

    def __repr__(self):
        fmt = '%-22s %10.5f %10.5f %10.5f %10.6f %10.6f %10.2f'
        l1 = ('%-22s' +' %10s'*6) % tuple('METHOD Xc Yc Rc std(Ri) residu residu2'.split())
        l2 = '\n' + '-'*(22 +7*(10+1)) + '\n'
        l3 = fmt % (self._method, self._center[0], self._center[1], self._radius, self._rad_std, self._residu, self._residu2)
        return l1 + l2 + l3 + '\n\n'

    def plot(self, residu2: bool = False, first_plot: bool = True, last_plot: bool = True, force_plot_data: bool = False, figsize: tuple(float, float) | None = (10, 10), dpi: int | None = 300, facecolor: p._ColorLike | None = 'white', edgecolor: p._ColorLike | None = 'black'):
        """Plot the results of circle fit.

        Args:
            residu2 (bool, optional): Plots residual^2 if set, otherwise plots residual. Defaults to False.
            first_plot (bool, optional): Turn this on for the first plot in a series comparing different algorithms. Defaults to True.
            last_plot (bool, optional): Turn this on for the last plot in a series to compare different algorithms. Defaults to True.
            force_plot_data (bool, optional): Turn this on to compare fit with two different data sets. Defaults to False.
        """
        if first_plot:
            p.close('all')
            f = p.figure(None, figsize, dpi, facecolor, edgecolor)
        p.axis('equal')

        theta_fit = np.linspace(-np.pi, np.pi, 180)

        x_fit1 = self._center[0] + self._radius*np.cos(theta_fit)
        y_fit1 = self._center[1] + self._radius*np.sin(theta_fit)
        p.plot(x_fit1, y_fit1, 'b-' , label=self._method, lw=2)

        p.plot([self._center[0]], [self._center[1]], 'bD', mec='y', mew=1)

        # draw
        p.xlabel('x')
        p.ylabel('y')

        # plot the residu fields
        nb_pts = 100

        p.draw()
        xmin, xmax = p.xlim()
        ymin, ymax = p.ylim()

        vmin = min(xmin, ymin)
        vmax = max(xmax, ymax)

        xg, yg = np.ogrid[vmin:vmax:nb_pts*1j, vmin:vmax:nb_pts*1j]
        xg = xg[..., np.newaxis]
        yg = yg[..., np.newaxis]

        Rig    = np.sqrt( (xg - self._x)**2 + (yg - self._y)**2 )
        Rig_m  = Rig.mean(axis=2)[..., np.newaxis]

        if residu2 : residu = np.sum( (Rig**2 - Rig_m**2)**2 ,axis=2)
        else       : residu = np.sum( (Rig-Rig_m)**2 ,axis=2)

        lvl = np.exp(np.linspace(np.log(residu.min()), np.log(residu.max()), 15))

        p.contourf(xg.flat, yg.flat, residu.T, lvl, alpha=0.4, cmap=cm.Purples_r) # , norm=colors.LogNorm())
        cbar = p.colorbar(fraction=0.175, format='%.f')
        p.contour (xg.flat, yg.flat, residu.T, lvl, alpha=0.8, colors="lightblue")

        if residu2 : cbar.set_label('Residu_2 - algebraic approximation')
        else       : cbar.set_label('Residu')

        # plot data
        if first_plot or force_plot_data:
            p.plot(self._x, self._y, 'rx', label='data', ms=8, mec='b', mew=1)
        p.legend(loc='best',labelspacing=0.1 )

        p.xlim(xmin=vmin, xmax=vmax)
        p.ylim(ymin=vmin, ymax=vmax)

        p.grid()
        p.title('Least Squares Circle')
        if last_plot:
            p.show()
