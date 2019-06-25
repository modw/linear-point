"""Collection of functions to get linear point from camb results. It follows the convention of camb
when getting the matter power spectrum P(kh). To get the linear point in Mpc h, use lp_from_cosmo,
to get it in Mpc, use lp_from_cosmo_mpc.
"""

import numpy as np
from scipy import interpolate
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline


# correlation function
# np.exp(-(k/a)**n) : smoothing kernel to regularize function for the transition k > 1

def xi(r, pk, a=1., n=4):
    """Two point correlation function for r.
    - pars
    r: float, correlation scale in real space
    pk: interp function object, power spectrum as function of k, P(k),
    limits of integration are given by this object through the x attribute 
    - returns
    xi(r): float, value of correlation func at r"""
    # sin is omitted in the integrand because its being weighted in quad
    def int_(k): return np.exp(-(k / a)**n) * \
        k**2 * pk(k) / (2 * (np.pi**2) * k * r)
    xir = quad(int_, pk.x[0]*(1+1e-8), pk.x[-1]*(1-1e-8), weight='sin', wvar=r)
    return xir[0]

# first derivative of correlation function


def xi_r(r, pk, a=1., n=4):
    """First derivative of two point correlation function.
    - pars
    r: float, correlation scale in real space
    pk: interp function object, power spectrum as function of k, P(k),
    limits of integration are given by this object through the x attribute 
    a, n: parameters for filter exp(-(k/a)^n)
    - returns
    xi_r(r): float, value of first derivative of correlation func at r"""
    # sin and cos are omitted in the integrand because they're being weighted in quad
    def int_a(k): return k**2 * np.exp(-(k / a)**n) * \
        pk(k) / (2 * (np.pi**2) * r)

    def int_b(k): return -1 * k**2 * np.exp(-(k / a)**n) * \
        pk(k) / (2 * (np.pi**2) * k * r**2)
    # small shift on integration limitis to stay within
    # interpolation bounds
    xi_r_a = quad(int_a, pk.x[0]*(1+1e-8), pk.x[-1]
                  * (1-1e-8), weight='cos', wvar=r)
    xi_r_b = quad(int_b, pk.x[0]*(1+1e-8), pk.x[-1]
                  * (1-1e-8), weight='sin', wvar=r)
    return xi_r_a[0] + xi_r_b[0]

# Dip and peak positions given first derivative of correlation function


def get_lp(xi_r, rmin=60., rmax=130., rsamples=10, root_dr=5.):
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


def get_pk_func(results, khmin, khmax, k_hunit=True):
    """Given cambs results object, calculate power spectrum as function of k.
    - pars
    results: camb's results object
    kmin, kmax: floats, limits in k space"""
    kh, z, [pk] = results.get_matter_power_spectrum(minkh=khmin,
                                                    maxkh=khmax,
                                                    npoints=500)
    if k_hunit:
        return CubicSpline(kh, pk)
    else:
        h = results.hubble_parameter(0)/100
        return CubicSpline(kh*h, pk)

# Linear point position from results object


def lp_from_cosmo(results, khmin=0.001, khmax=10., a=1., n=4, rmin=60., rmax=130., rsamples=10, root_dr=5.):
    """Given cambs results object, calculate positions of dip and peak in Mpc h.
    - pars
    results: camb's results object
    khmin, khmax: floats, limits in kh space (should be consistent with limis in results)
    a, n: parameters for filter exp(-(k/a)^n)
    rmin, rmax, rsamples, root_dr: parameters for root finding of peak and dip in xi(r)
    - returns
    dip, peak: set, floats"""
    pk_func = get_pk_func(results, khmin, khmax)

    def dxi_dr(r): return xi_r(r, pk_func, a, n)
    lp = get_lp(dxi_dr, rmin, rmax, rsamples, root_dr)
    return lp


def lp_from_cosmo_mpc(results, khmin=0.001, khmax=10., a=1., n=4, rmin=115., rmax=160., rsamples=14, root_dr=5.):
    """Given cambs results object, calculate positions of dip and peak in Mpc.
    - pars
    results: camb's results object
    kmin, kmax: floats, limits in k space (should be consistent with limis in results)
    a, n: parameters for filter exp(-(k/a)^n)
    rmin, rmax, rsamples, root_dr: parameters for root finding of peak and dip in xi(r)
    - returns
    dip, peak: set, floats"""
    pk_func = get_pk_func(results, khmin, khmax, k_hunit=False)

    def dxi_dr(r): return xi_r(r, pk_func, a, n)
    lp = get_lp(dxi_dr, rmin, rmax, rsamples, root_dr)
    return lp
