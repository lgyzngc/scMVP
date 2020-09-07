# 加载所需的python包
import palantir
# Plotting and miscellaneous imports
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import pandas as pd

palantir_dir = os.path.expanduser('E:/data/qiliu/single-cell program/ATAC/result/whole figure/top 5000 knn 3 pvalue 1/data/')
#tissue_data = palantir.io.from_csv(palantir_dir + 'gene_imputation.csv')
#tissue_data_norm = pd.DataFrame(data=tissue_data.values.T,index = tissue_data.columns,columns = tissue_data.index)
tissue_data_norm = pd.read_csv(palantir_dir + 'multivae_latent_imputation.csv', header=0, index_col=0)
#tissue_data_norm = palantir.io.from_csv(palantir_dir + 'multivae_latent_imputation.csv')


# PCA reduction
pca_projections, _ = palantir.utils.run_pca(tissue_data_norm, n_components=5)
# Run diffusion maps
dm_res = palantir.utils.run_diffusion_maps(pca_projections, n_components=5)
dm_res = palantir.utils.run_diffusion_maps(tissue_data_norm, n_components=10)
ms_data = palantir.utils.determine_multiscale_space(dm_res)

# tSNE visualization
tsne = palantir.utils.run_tsne(ms_data)
fig, ax = palantir.plot.plot_tsne(tsne)
plt.show()

# 数据聚类
#clusters = palantir.utils.determine_cell_clusters(tissue_data_norm)
# 聚类结果可视化
#palantir.plot.plot_cell_clusters(tsne, clusters )

# 运行Palantir
start_cell = 'AAACTTAGCCAC'
pr_res = palantir.core.run_palantir(ms_data, start_cell, num_waypoints=1000)

pr_res.branch_probs.columns
palantir.plot.plot_palantir_results(pr_res, tsne)
plt.show()

df = pd.DataFrame(data=pr_res.pseudotime.values, index= pr_res.pseudotime.index)
df.to_csv(os.path.join(palantir_dir,"psudotime_palantir.csv"))

df = pr_res.branch_probs
df.to_csv(os.path.join(palantir_dir,"psudotime_branch_probs_palantir.csv"))

tsne.to_csv(os.path.join(palantir_dir,"psudotime_palantir_tsne.csv"))


print()
