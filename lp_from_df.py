"""
Takes Planck chain in pandas DataFrame format from .pkl file and calculate 
the positions of the peak, dip and the linear point from matter 2pt correlation
funtion.

Script should be called like this:
python lp_from_df.py [input df] [output directory - defaults to cwd]
"""
#%% importing modules
import pandas as pd
import linear_point as lp
import camb
import sys

#%% reading terminal input
in_file = sys.argv[1]
fname = in_file.split('.')[-2].split('/')[-1]

if len(sys.argv) > 2:
    out_dir = sys.argv[2]
else:
    out_dir = './'

# formatting out_dir
if out_dir[-1] != '/':
    out_dir += '/'

#%% starting camb pars and setting krange
# creating instance of CAMBparams object
pars = camb.CAMBparams()
# krange
kmin = 0.001
kmax = 10.

#%%
# function to get camb results from params


def get_results(pars, z, ombh2, omch2, cosmomc_theta, tau, As, ns):
    pars.set_cosmology(ombh2=ombh2,
                       omch2=omch2,
                       cosmomc_theta=cosmomc_theta,
                       H0=None,
                       tau=tau)
    pars.InitPower.set_params(ns=ns,
                              As=As)
    pars.set_matter_power(
        redshifts=[z], kmax=kmax, nonlinear=False, k_per_logint=0)
    return camb.get_results(pars)


#%% importing chain and preparing out chain
df_in = pd.read_pickle(in_file).iloc[:10]
# output chain
df_out = df_in.copy()
df_out['cf_peak'] = 0
df_out['cf_dip'] = 0

#%% looping over input chain and adding results to output chain
# param names to input in result
param_names = ['omegabh2', 'omegach2', 'theta', 'tau', 'A*', 'ns']

for i in df_out.index:
    ombh2, omch2, cosmomc_theta, tau, As, ns = df_out.loc[i, param_names]
    cosmomc_theta /= 100
    As *= 1e-9
    results = get_results(pars, 0.0,
                          ombh2, omch2, cosmomc_theta, tau, As, ns)
    dip, peak = lp.lp_from_cosmo(results, kmin, kmax)
    df_out.loc[i, ['cf_peak', 'cf_dip']] = peak, dip

df_out['lp'] = (df_out['cf_peak'] + df_out['cf_dip']) / 2

#%% writing out

df_out.to_pickle(out_dir + '{}_lp.pkl'.format(fname))
