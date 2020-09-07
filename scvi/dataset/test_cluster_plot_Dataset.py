import scanpy as sc
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
adata = sc.datasets.pbmc68k_reduced()
markers = ['C1QA', 'PSAP', 'CD79A', 'CD79B', 'CST3', 'LYZ']
sc.pl.stacked_violin(adata, markers, groupby='bulk_labels', dendrogram=True)

markers = {'T-cell': 'CD3D', 'B-cell': 'CD79A', 'myeloid': 'CST3'}
sc.pl.stacked_violin(adata, markers, groupby='bulk_labels', dendrogram=True)
print()
