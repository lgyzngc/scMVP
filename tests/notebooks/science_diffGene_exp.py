import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import scanpy as sc
import seaborn as sns
import anndata
import scipy.io as sp_io
import shutil
from scipy.sparse import csr_matrix, issparse


#save_path: str = "E:/data/qiliu/single-cell program/ATAC/science data/"
save_path: str = "E:/data/qiliu/single-cell program/ATAC/snare data/"


#gene_exp = pd.read_csv(save_path+'gene_exp.csv', header=0, index_col=0)
#gene_exp = pd.read_csv(save_path+'gene_scvi_imputation.csv', header=0, index_col=0)
gene_exp = pd.read_csv(save_path+'gene_multivae_imputation.csv', header=0, index_col=0)
#gene_exp = pd.read_csv(save_path+'gene_imputation.csv', header=0, index_col=0)
diff_gene = pd.read_csv(save_path+'gene_diff_matrix_adj.csv', header=0, index_col=0)
#cell_clu = pd.read_csv(save_path+'multivae_umap.csv', header=0, index_col=0)
cell_clu = pd.read_csv(save_path+'multivae_umap_imputation.csv', header=0, index_col=0)
#cell_clu_index = cell_clu['labels']
cell_clu_index = cell_clu['atac_cluster']
cell_clu_sort = None
for i in np.unique(cell_clu_index):
    if cell_clu_sort is None:
        cell_clu_sort = np.where(cell_clu_index == i)[0]
    else:
        cell_clu_sort = np.append(cell_clu_sort, np.where(cell_clu_index == i)[0])


diff_gene_exp = None
for i in range(len(diff_gene.index.values)):
    temp = gene_exp.values[:,cell_clu_sort]
    temp = temp[diff_gene.values[i,:],:]
    if diff_gene_exp is None:
        diff_gene_exp = temp
    else:
        diff_gene_exp = np.vstack((diff_gene_exp,temp))
diff_gene_exp = np.log1p(diff_gene_exp)
diff_gene_exp = np.log1p(diff_gene_exp)
diff_gene_exp = np.log1p(diff_gene_exp)

ax = sns.heatmap(diff_gene_exp,cmap='rainbow')
plt.show()
ax = sns.clustermap(diff_gene_exp, metric="correlation",cmap='rainbow')
plt.show()
ax = sns.clustermap(diff_gene_exp, cmap='rainbow')
plt.show()

gene_exp_sort = gene_exp.values[:,cell_clu_sort]
gene_exp_sort = np.log1p(gene_exp_sort)
gene_exp_sort = np.log1p(gene_exp_sort)
gene_exp_sort = np.log1p(gene_exp_sort)
ax = sns.heatmap(gene_exp_sort,cmap='rainbow')
plt.show()
ax = sns.clustermap(gene_exp_sort, cmap='rainbow')
plt.show()
print()
