
"""Quick script that takes .pkl pandas DataFrame as input, splits it
and writes it in several different files. Written with the intention
of trivially parallelizing in the cluster.

Script should be called like this:
python df_splitter.py [input df] [output directory] [number of files]
"""

#%%
import pandas as pd
import numpy as np
import sys

#%%

in_df = sys.argv[1]
out_dir = sys.argv[2]
n_files = int(sys.argv[3])

fname = in_df.split('.')[-2].split('/')[-1]
#%%
df = pd.read_pickle(in_df)

for i, dfi in enumerate(np.array_split(df, n_files)):
    dfi.to_pickle(out_dir + '/{}_split_{}.pkl'.format(fname, i + 1))
    print("Exported file {}/{}".format(i + 1, n_files))
