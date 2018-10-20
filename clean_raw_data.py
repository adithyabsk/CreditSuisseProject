#!/usr/bin/env python

from functools import partial
import pprint

import numpy as np
import pandas as pd
from tqdm import tqdm
from fuzzywuzzy import process


df = pd.read_csv('coe_software_usage.csv')
df = df[(df['count'] != 0) & ~np.isnan(df['count'].values)]
df[['Major', 'Unknown1', 'Provider', 'AppName', 'Version', 'Unknown2']] = df.software.str.split('-', 5, expand=True)
df = df.drop(columns=['software', 'Unknown1', 'Version', 'Unknown2'])
df = df.groupby(['Major', 'Provider', 'AppName'])['count'].sum().reset_index()

temp_gbp = df[['Provider', 'AppName']].drop_duplicates().groupby(['Provider'])
merge_dict = {}
for k, t_df in temp_gbp:
    apps = t_df['AppName'].values.tolist()
    skip = []
    for a in apps:
        if a in skip: continue
        merge_values = [app for app, v in process.extract(a, apps) if v>90]
        if len(merge_values) == 1: continue
        merge_dict[a] = merge_values
        skip.extend(merge_values)

print('Fuzzy cleaning duplicate names')
def merger(grouped_df, values):
    p1 = grouped_df[grouped_df['AppName'].isin(values)]
    if p1.empty: return grouped_df
    grouped_df = grouped_df.copy(deep=True)
    indices = p1.index.values.tolist()
    grouped_df.loc[indices[0], ['AppName', 'count']] = [values[0], p1['count'].sum()]
    if len(indices) > 1:
        grouped_df = grouped_df.drop(index=indices[1:])
    return  grouped_df.reset_index(drop=True)

for k, v in tqdm(merge_dict.items()):
    merge_gbp = df.groupby(['Major', 'Provider'])
    p_merger = partial(merger, values=v)
    df = merge_gbp.apply(p_merger).reset_index(drop=True)

print(df.Major.value_counts())
print(df[df['Major'] == 'ECE'])
print(df)
