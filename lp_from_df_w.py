# to do:
# figure out differences between using cosmomctheta and H0
# when using theta, H0 is calculated in a specified range. We might need
# to increase this range. See param `theta_H0_range` in CAMBparams()
# suppress dark energy warning
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
import numpy as np
import camb
import sys
# for timing
import time
t0 = time.time()

#%% reading terminal input
in_file = sys.argv[1]
fname = in_file.split('.')[-2].split('/')[-1]
print("Working on file: {}".format(fname))

# formatting out_dir
if len(sys.argv) > 2:
    out_dir = sys.argv[2]
    if out_dir[-1] != '/':
        out_dir += '/'
else:
    out_dir = './'


#%% starting camb pars and setting krange
# creating instance of CAMBparams object
pars = camb.CAMBparams()
# krange
kmin = 0.001
kmax = 20.

#%%
# function to get camb results from params


def get_results(pars, z, ombh2, omch2, cosmomc_theta, tau, w, As, ns):
    pars.set_dark_energy(w=w)
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
df_in = pd.read_pickle(in_file)
# output chain
df_out = df_in.copy()
df_out['cf_peak'] = 0
df_out['cf_dip'] = 0

#%% looping over input chain and adding results to output chain
# param names to input in result
param_names = ['omegabh2', 'omegach2', 'theta', 'tau', 'w', 'A*', 'ns']

for i in df_out.index:
    ombh2, omch2, cosmomc_theta, tau, w, As, ns = df_out.loc[i, param_names]
    cosmomc_theta /= 100
    As *= 1e-9
    results = get_results(pars, 0.0,
                          ombh2, omch2, cosmomc_theta, tau, w, As, ns)
    try:
        dip, peak = lp.lp_from_cosmo_mpc(results, kmin, kmax)
        df_out.loc[i, ['cf_peak', 'cf_dip']] = peak, dip
    except:
        print('ERROR Program failed at line {} of file {}'.format(i, fname))
        df_out.loc[i, ['cf_peak', 'cf_dip']] = np.nan, np.nan

    if i//100 == 0:
        print(i + " lines done.")
    

df_out['lp'] = (df_out['cf_peak'] + df_out['cf_dip']) / 2

#%% writing out

df_out.to_pickle(out_dir + '{}_lp.pkl'.format(fname))

tf = time.time() - t0
tf /= 60

print("It took {} hours to go over {} lines. That's {} min/1000 lines.".format(tf /
                                                                               60, len(df_in), tf*1000/len(df_in)))
