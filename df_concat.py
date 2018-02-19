"""Quick script that iterates over pandas DataFrame files in .pkl
format and concatenates them into an output file.

Script should be called like this:
python df_concat.py [input folder] [output directory/name] 
"""
# %% importing modules

import os
import pandas as pd
import sys

# %% params

in_dir = sys.argv[1]
if in_dir[-1] != "/":
    in_dir += "/"

file_out = sys.argv[2]
# %%

for i, file in enumerate(os.listdir(in_dir)):
    if i == 0:
        df = pd.read_pickle(in_dir + file)
    else:
        df = pd.concat([df, pd.read_pickle(in_dir + file)])

#%% write out

df.to_pickle(file_out)
