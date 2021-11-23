#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improve CNV so that only genes with CNV are produced

Developed by: Harry Chown

Date: 28/10/21
"""
import pandas as pd
raw_data = pd.read_csv("/home/harry/Documents/partition_scripts/bacteria_pangenome/ML_indata/cnv_matrix.csv", 
                       index_col=0)

t_data = raw_data.transpose()

dropped_t = t_data.loc[((t_data>1)).any(1)]

updated_data = dropped_t.transpose()
 
updated_data.to_csv("/home/harry/Documents/partition_scripts/bacteria_pangenome/ML_indata/updated_cnv_matrix.csv")
