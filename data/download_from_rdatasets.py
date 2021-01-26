#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:57:47 2021

@author: rindt
"""

import pandas as pd

def download_from_rdatasets(package, name):
    datasets = (pd.read_csv("http://vincentarelbundock.github.com/Rdatasets/datasets.csv")
                .loc[lambda x: x['Package'] == package].set_index('Item'))
    if not name in datasets.index:
        raise ValueError(f"Dataset {name} not found.")
    info = datasets.loc[name]
    url = info.CSV 
    return pd.read_csv(url), info

a = download_from_rdatasets('survival', 'gbsg')

print(len(a))
print(type(a[0]))