import numpy as np
from scipy import interpolate
from scipy.integrate import quad
import camb
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline

# correlation function and its derivative given kmin, kmax and P(k)

# np.exp(-k**4) : smoothing kernel to regularize function for the transition k > 1

# correlation function


def xi(r, kmin, kmax, pk):
    """Two point correlation function for r.
    - pars
    r: float, correlation scale in real space
    kmin, kmax: floats, min and max of k range
    pk: function object, power spectrum as function of k, P(k)
    - returns
    xi(r): float, value of correlation func at r"""
    # sin is omitted in the integrand because its being weighted in quad
    def int_(k): return np.exp(-k**4) * k**2 * pk(k) / (2 * (np.pi**2) * k * r)
    xir = quad(int_, kmin, kmax, weight='sin', wvar=r)
    return xir[0]

# first derivative of correlation function


def xi_r(r, kmin, kmax, pk, a=1., n=4):
    """First derivative of two point correlation function.
    - pars
    r: float, correlation scale in real space
    kmin, kmax: floats, min and max of k range
    pk: function object, power spectrum as function of k, P(k)
    a, n: parameters for filter exp(-(k/a)^n)
    - returns
    xi_r(r): float, value of first derivative of correlation func at r"""
    # sin and cos are omitted in the integrand because they're being weighted in quad
    def int_a(k): return k**2 * np.exp(-(k / a)**n) * \
        pk(k) / (2 * (np.pi**2) * r)

    def int_b(k): return -1 * k**2 * np.exp(-(k / a)**n) * \
        pk(k) / (2 * (np.pi**2) * k * r**2)
    xi_r_a = quad(int_a, kmin, kmax, weight='cos', wvar=r)
    xi_r_b = quad(int_b, kmin, kmax, weight='sin', wvar=r)
    return xi_r_a[0] + xi_r_b[0]

# Dip and peak positions given first derivative of correlation function


def get_lp(xi_r, rmin=60., rmax=150., rsamples=8, root_dr=3.):
    """Given first derivative of matter correlation function, get position of
    dip and peak.
    - pars
    xi_r: function object, first derivative of xi(r)
    rmin, rmax: floats, range to which look for roots
    rsamples: int, derivative is sampled evenly in rsamples parts along specified range
    root_dr: float, interval around each root to do a finer search with brentq
    - returns
    dip, peak: float, position of dip and peak of correlation function"""
    # sample derivative in rsamples points and interpolate
    r_list = np.linspace(rmin, rmax, rsamples)
    xi_r_list = [xi_r(r) for r in r_list]
    xi_r_interp = CubicSpline(r_list, xi_r_list, extrapolate=False)
    # find roots of interpolated function
    roots = xi_r_interp.roots(extrapolate=False)
    # defined region to look for each root in derivative
    ra = (roots[0] - root_dr, roots[0] + root_dr)
    rb = (roots[1] - root_dr, roots[1] + root_dr)
    # find each root and return
    dip = brentq(xi_r, *ra)
    peak = brentq(xi_r, *rb)
    return dip, peak

# P(k) function object from CAMB's results object


def get_pk_func(results, kmin, kmax):
    """Given cambs results object, calculate power spectrum as function of k.
    - pars
    results: camb's results object
    kmin, kmax: floats, limits in k space"""
    kh, z, [pk] = results.get_matter_power_spectrum(minkh=kmin * 0.99,
                                                    maxkh=kmax * 1.001,
                                                    npoints=500)
    return interpolate.interp1d(kh, pk, kind='cubic')

# Linear point position from results object


def lp_from_cosmo(results, kmin=0.001, kmax=10., a=1., n=4):
    """Given cambs results object, calculate position of linear point.
    - pars
    results: camb's results object
    kmin, kmax: floats, limits in k space (should be consistent with limis in results)
    a, n: parameters for filter exp(-(k/a)^n)"""
    pk_func = get_pk_func(results, kmin, kmax)

    def dxi_dr(r): return xi_r(r, kmin, kmax, pk_func, a, n)
    lp = get_lp(dxi_dr)
    return lp
