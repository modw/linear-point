import numpy as np
from scipy import interpolate
from scipy.integrate import quad
import camb
from scipy.optimize import brentq

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


def xi_r(r, kmin, kmax, pk):
    """First derivative of two point correlation function.
    - pars
    r: float, correlation scale in real space
    kmin, kmax: floats, min and max of k range
    pk: function object, power spectrum as function of k, P(k)
    - returns
    xi_r(r): float, value of first derivative of correlation func at r"""
    # sin and cos are omitted in the integrand because they're being weighted in quad
    def int_a(k): return k**2 * np.exp(-k**4) * pk(k) / (2 * (np.pi**2) * r)

    def int_b(k): return -1 * k**2 * np.exp(-k**4) * \
        pk(k) / (2 * (np.pi**2) * k * r**2)
    xi_r_a = quad(int_a, kmin, kmax, weight='cos', wvar=r)
    xi_r_b = quad(int_b, kmin, kmax, weight='sin', wvar=r)
    return xi_r_a[0] + xi_r_b[0]

# Linear point position given first derivative of correlation function


def get_lp(xi_der, ra=(82., 90.), rb=(97., 105.)):
    """Given first derivative of matter correlation function, get position of
    linear point.
    - pars
    xi_r: function object, first derivative of xi(r)
    ra, rb: sets of floats, range of to look for dip and peak, respectively
    - returns
    lp: float, position of linear point"""
    dip = brentq(xi_der, *ra)
    peak = brentq(xi_der, *rb)
    return (dip + peak) / 2

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


def lp_from_pk(results, kmin=0.001, kmax=100.):
    """Given cambs results object, calculate position of linear point.
    - pars
    results: camb's results object
    kmin, kmax: floats, limits in k space"""
    pk_func = get_pk_func(results, kmin, kmax)

    def dxi_dr(r): return xi_r(r, kmin, kmax, pk_func)
    lp = get_lp(dxi_dr)
    return lp
