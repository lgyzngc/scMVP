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

save_path: str = "E:/data/qiliu/single-cell program/ATAC/result/whole figure/top 5000 knn 3 pvalue 1/data/"
gene_exp = pd.read_csv(save_path+'gene_imputation.csv', header=0, index_col=0)
atac_exp = pd.read_csv(save_path+'atac_aggregation.csv', header=0, index_col=0)
promoter_location = pd.read_csv(save_path+'promoter/promoter_ucsc.txt', sep='\t', header=0, index_col=0)
diff_gene_set = pd.read_csv(save_path+'gene_diff_matrix_adj.csv', header=0, index_col=0)
diff_atac_set = pd.read_csv(save_path+'atac_diff_matrix.csv', header=0, index_col=0)

the_geneclu_0 = pd.read_csv(save_path+'the0gene_pls_cluster_names.csv', header=0, index_col=0)
the_geneclu_2 = pd.read_csv(save_path+'the2gene_pls_cluster_names.csv', header=0, index_col=0)
the_geneclu_4 = pd.read_csv(save_path+'the4gene_pls_cluster_names.csv', header=0, index_col=0)
the_geneclu_5 = pd.read_csv(save_path+'the5gene_pls_cluster_names.csv', header=0, index_col=0)
the_geneclu_6 = pd.read_csv(save_path+'the6gene_pls_cluster_names.csv', header=0, index_col=0)
the_geneclu_9 = pd.read_csv(save_path+'the9gene_pls_cluster_names.csv', header=0, index_col=0)
the_geneclu_13 = pd.read_csv(save_path+'the13gene_pls_cluster_names.csv', header=0, index_col=0)
the_geneclu_15 = pd.read_csv(save_path+'the15gene_pls_cluster_names.csv', header=0, index_col=0)
the_geneclu_16 = pd.read_csv(save_path+'the16gene_pls_cluster_names.csv', header=0, index_col=0)
the_geneclu_19 = pd.read_csv(save_path+'the19gene_pls_cluster_names.csv', header=0, index_col=0)
the_geneclu_20 = pd.read_csv(save_path+'the20gene_pls_cluster_names.csv', header=0, index_col=0)

joint_geneclu_index = np.append(the_geneclu_0.values[:, 0],the_geneclu_2.values[:, 0])
#joint_geneclu_index = None
joint_geneclu_index = np.append(joint_geneclu_index,the_geneclu_4.values[:, 0])
joint_geneclu_index = np.append(joint_geneclu_index,the_geneclu_5.values[:, 0])
joint_geneclu_index = np.append(joint_geneclu_index,the_geneclu_6.values[:, 0])
joint_geneclu_index = np.append(joint_geneclu_index,the_geneclu_9.values[:, 0])
joint_geneclu_index = np.append(joint_geneclu_index,the_geneclu_13.values[:, 0])
joint_geneclu_index = np.append(joint_geneclu_index,the_geneclu_15.values[:, 0])
joint_geneclu_index = np.append(joint_geneclu_index,the_geneclu_16.values[:, 0])
joint_geneclu_index = np.append(joint_geneclu_index,the_geneclu_19.values[:, 0])
joint_geneclu_index = np.append(joint_geneclu_index,the_geneclu_20.values[:, 0])

diff_gene_index = np.unique(diff_gene_set.values.flatten())
joint_geneclu_dff_index = np.array([i for i in joint_geneclu_index if i in diff_gene_index])

# atac regress clu
the_atacclu_0 = pd.read_csv(save_path+'the0atac_pls_cluster_names.csv', header=0, index_col=0)
the_atacclu_2 = pd.read_csv(save_path+'the2atac_pls_cluster_names.csv', header=0, index_col=0)
the_atacclu_4 = pd.read_csv(save_path+'the4atac_pls_cluster_names.csv', header=0, index_col=0)
the_atacclu_5 = pd.read_csv(save_path+'the5atac_pls_cluster_names.csv', header=0, index_col=0)
the_atacclu_6 = pd.read_csv(save_path+'the6atac_pls_cluster_names.csv', header=0, index_col=0)
the_atacclu_9 = pd.read_csv(save_path+'the9atac_pls_cluster_names.csv', header=0, index_col=0)
the_atacclu_13 = pd.read_csv(save_path+'the13atac_pls_cluster_names.csv', header=0, index_col=0)
the_atacclu_15 = pd.read_csv(save_path+'the15atac_pls_cluster_names.csv', header=0, index_col=0)
the_atacclu_16 = pd.read_csv(save_path+'the16atac_pls_cluster_names.csv', header=0, index_col=0)
the_atacclu_19 = pd.read_csv(save_path+'the19atac_pls_cluster_names.csv', header=0, index_col=0)
the_atacclu_20 = pd.read_csv(save_path+'the20atac_pls_cluster_names.csv', header=0, index_col=0)
the_atacclu_0_unique = np.unique(the_atacclu_0.index.values)
the_atacclu_2_unique = np.unique(the_atacclu_2.index.values)
the_atacclu_4_unique = np.unique(the_atacclu_4.index.values)
the_atacclu_5_unique = np.unique(the_atacclu_5.index.values)
the_atacclu_6_unique = np.unique(the_atacclu_6.index.values)
the_atacclu_9_unique = np.unique(the_atacclu_9.index.values)
the_atacclu_13_unique = np.unique(the_atacclu_13.index.values)
the_atacclu_15_unique = np.unique(the_atacclu_15.index.values)
the_atacclu_16_unique = np.unique(the_atacclu_16.index.values)
the_atacclu_19_unique = np.unique(the_atacclu_19.index.values)
the_atacclu_20_unique = np.unique(the_atacclu_20.index.values)
# common atac peaks analysis
'''
atac_clu_str = np.array([0,2,4,5,6,9,13,15,16,19,20])
atac_common_regu_array = None
for i in atac_clu_str:
    temp_atac_count = np.array([])
    atac_unique = eval("the_atacclu_"+str(i)+"_unique")
    atac_set = eval("the_atacclu_"+str(i))
    atac_set = atac_set.index.values
    for j in atac_unique:
        temp_atac_count = np.append(temp_atac_count, len(np.where(atac_set == j)[0]))
    temp_atac_count = temp_atac_count/np.max(temp_atac_count)
    temp_atac_comm_regu = np.zeros((12,2))
    temp_atac_comm_regu[0, 0] = 1
    temp_atac_comm_regu[0, 1] = len(np.where(temp_atac_count == 1/np.max(temp_atac_count))[0])/len(temp_atac_count)
    temp_atac_comm_regu[11, 0] = 101
    temp_atac_comm_regu[11, 1] = 1
    for j in range(10):
        temp_atac_comm_regu[j+1,0] = (j+1)*10
        temp_atac_comm_regu[j+1,1] = len(np.where(temp_atac_count < (j+1)/10.0)[0])/len(temp_atac_count)
    if atac_common_regu_array is None:
        atac_common_regu_array = temp_atac_comm_regu
    else:
        atac_common_regu_array = np.vstack((atac_common_regu_array,temp_atac_comm_regu))

atac_common_regu_array_list = atac_common_regu_array[:,0].tolist()
for i in range(len(atac_common_regu_array_list)):
    if atac_common_regu_array_list[i] == 1:
        continue
    elif atac_common_regu_array_list[i] == 101:
        atac_common_regu_array_list[i] = '100%'
    else:
        atac_common_regu_array_list[i] = '<'+str(atac_common_regu_array_list[i]) + '%'

atac_ratio_df = pd.DataFrame({'accumulated distribution of common peaks':atac_common_regu_array[:,1],'common peaks percentage':atac_common_regu_array_list})
atac_ratio_df.to_csv(os.path.join(save_path,"boxplot_common_peaks.csv"))
'''

atac_ratio_df = pd.read_csv(save_path+'boxplot_common_peaks.csv', header=0, index_col=0)
ax = plt.subplot(111)
sns.boxplot(x = 'common peaks percentage', y = 'accumulated distribution of common peaks', data = atac_ratio_df,color='b')
ax.spines['left'].set_position(('outward', 10))  # outward by 10 points
ax.spines['right'].set_visible(False)  # outward by 10 points
ax.spines['top'].set_visible(False)  # outward by 10 points
ax.spines['bottom'].set_position(('outward', 10))   # outward by 10 points
plt.xticks(fontsize=20,rotation = 45)
plt.yticks(fontsize=20)
# 设置坐标标签字体大小
ax.set_xlabel('common peaks percentage', fontsize=20)
ax.set_ylabel('accumulated distribution of common peaks', fontsize=20)
plt.show()
# atac regression indicate TF according all the diff gene clu

atac_allTF_feature_set ={}
for i in (the_atacclu_0.index.values):
    if i in atac_allTF_feature_set.keys():
        atac_allTF_feature_set[i] = atac_allTF_feature_set[i]+1
    else:
        atac_allTF_feature_set[i] = 1
feature_filter_rate = len(the_atacclu_0.index.values)/len(the_atacclu_0_unique)
feature_filter_rate = feature_filter_rate*0.8
feature_filter_allTF_result = None
for key,value in atac_allTF_feature_set.items():
    if value < feature_filter_rate:
        continue
    key_index = np.where(atac_exp.index.values == key)[0]
    if feature_filter_allTF_result is None:
        feature_filter_allTF_result = key_index
    else:
        feature_filter_allTF_result = np.append(feature_filter_allTF_result,key_index)
feature_filter_allTF_intersect = np.array([j for j in diff_atac_set.values[2,:] if j in feature_filter_allTF_result])
temp_intersect = np.array([j for j in diff_atac_set.values[7,:] if j in feature_filter_allTF_result])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)

temp_intersect = np.array([j for j in diff_atac_set.values[9,:] if j in feature_filter_allTF_result])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[3,:] if j in feature_filter_allTF_result])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[0,:] if j in feature_filter_allTF_result])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[5,:] if j in feature_filter_allTF_result])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)

atac_allTF_feature_set ={}
temp_len = len(feature_filter_allTF_result)
for i in (the_atacclu_2.index.values):
    if i in atac_allTF_feature_set.keys():
        atac_allTF_feature_set[i] = atac_allTF_feature_set[i]+1
    else:
        atac_allTF_feature_set[i] = 1
feature_filter_rate = len(the_atacclu_2.index.values)/len(the_atacclu_2_unique)
feature_filter_rate = feature_filter_rate*0.8
for key,value in atac_allTF_feature_set.items():
    if value < feature_filter_rate:
        continue
    key_index = np.where(atac_exp.index.values == key)[0]
    feature_filter_allTF_result = np.append(feature_filter_allTF_result,key_index)
temp_intersect = np.array([j for j in diff_atac_set.values[0,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)

temp_intersect = np.array([j for j in diff_atac_set.values[3,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[9,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[5,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_len = len(feature_filter_allTF_result)

'''
atac_allTF_feature_set ={}
for i in (the_atacclu_4.index.values):
    if i in atac_allTF_feature_set.keys():
        atac_allTF_feature_set[i] = atac_allTF_feature_set[i]+1
    else:
        atac_allTF_feature_set[i] = 1
feature_filter_rate = len(the_atacclu_4.index.values)/len(the_atacclu_4_unique)
feature_filter_rate = feature_filter_rate*0.8
for key,value in atac_allTF_feature_set.items():
    if value < feature_filter_rate:
        continue
    key_index = np.where(atac_exp.index.values == key)[0]
    feature_filter_allTF_result = np.append(feature_filter_allTF_result,key_index)


atac_allTF_feature_set ={}
for i in (the_atacclu_5.index.values):
    if i in atac_allTF_feature_set.keys():
        atac_allTF_feature_set[i] = atac_allTF_feature_set[i]+1
    else:
        atac_allTF_feature_set[i] = 1
feature_filter_rate = len(the_atacclu_5.index.values)/len(the_atacclu_5_unique)
feature_filter_rate = feature_filter_rate*0.8
for key,value in atac_allTF_feature_set.items():
    if value < feature_filter_rate:
        continue
    key_index = np.where(atac_exp.index.values == key)[0]
    feature_filter_allTF_result = np.append(feature_filter_allTF_result,key_index)
temp_intersect = np.array([j for j in diff_atac_set.values[4,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[6,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[2,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[5,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[8,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_len = len(feature_filter_allTF_result)

atac_allTF_feature_set ={}
for i in (the_atacclu_6.index.values):
    if i in atac_allTF_feature_set.keys():
        atac_allTF_feature_set[i] = atac_allTF_feature_set[i]+1
    else:
        atac_allTF_feature_set[i] = 1
feature_filter_rate = len(the_atacclu_6.index.values)/len(the_atacclu_6_unique)
feature_filter_rate = feature_filter_rate*0.8
for key,value in atac_allTF_feature_set.items():
    if value < feature_filter_rate:
        continue
    key_index = np.where(atac_exp.index.values == key)[0]
    feature_filter_allTF_result = np.append(feature_filter_allTF_result,key_index)


atac_allTF_feature_set ={}
for i in (the_atacclu_9.index.values):
    if i in atac_allTF_feature_set.keys():
        atac_allTF_feature_set[i] = atac_allTF_feature_set[i]+1
    else:
        atac_allTF_feature_set[i] = 1
feature_filter_rate = len(the_atacclu_9.index.values)/len(the_atacclu_9_unique)
feature_filter_rate = feature_filter_rate*0.8
for key,value in atac_allTF_feature_set.items():
    if value < feature_filter_rate:
        continue
    key_index = np.where(atac_exp.index.values == key)[0]
    feature_filter_allTF_result = np.append(feature_filter_allTF_result,key_index)
temp_intersect = np.array([j for j in diff_atac_set.values[1,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[12,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[7,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[9,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[16,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[17,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[15,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[14,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[10,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_len = len(feature_filter_allTF_result)

atac_allTF_feature_set ={}
for i in (the_atacclu_13.index.values):
    if i in atac_allTF_feature_set.keys():
        atac_allTF_feature_set[i] = atac_allTF_feature_set[i]+1
    else:
        atac_allTF_feature_set[i] = 1
feature_filter_rate = len(the_atacclu_13.index.values)/len(the_atacclu_13_unique)
feature_filter_rate = feature_filter_rate*0.8
for key,value in atac_allTF_feature_set.items():
    if value < feature_filter_rate:
        continue
    key_index = np.where(atac_exp.index.values == key)[0]
    feature_filter_allTF_result = np.append(feature_filter_allTF_result,key_index)
temp_intersect = np.array([j for j in diff_atac_set.values[12,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[9,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[1,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[14,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_len = len(feature_filter_allTF_result)

atac_allTF_feature_set ={}
for i in (the_atacclu_15.index.values):
    if i in atac_allTF_feature_set.keys():
        atac_allTF_feature_set[i] = atac_allTF_feature_set[i]+1
    else:
        atac_allTF_feature_set[i] = 1
feature_filter_rate = len(the_atacclu_15.index.values)/len(the_atacclu_15_unique)
feature_filter_rate = feature_filter_rate*0.8
for key,value in atac_allTF_feature_set.items():
    if value < feature_filter_rate:
        continue
    key_index = np.where(atac_exp.index.values == key)[0]
    feature_filter_allTF_result = np.append(feature_filter_allTF_result,key_index)
temp_intersect = np.array([j for j in diff_atac_set.values[7,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[9,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[16,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[17,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_len = len(feature_filter_allTF_result)

atac_allTF_feature_set ={}
for i in (the_atacclu_16.index.values):
    if i in atac_allTF_feature_set.keys():
        atac_allTF_feature_set[i] = atac_allTF_feature_set[i]+1
    else:
        atac_allTF_feature_set[i] = 1
feature_filter_rate = len(the_atacclu_16.index.values)/len(the_atacclu_16_unique)
feature_filter_rate = feature_filter_rate*0.8
for key,value in atac_allTF_feature_set.items():
    if value < feature_filter_rate:
        continue
    key_index = np.where(atac_exp.index.values == key)[0]
    feature_filter_allTF_result = np.append(feature_filter_allTF_result,key_index)
temp_intersect = np.array([j for j in diff_atac_set.values[1,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[4,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[5,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[7,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[8,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[2,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[10,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[15,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[4,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[12,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_len = len(feature_filter_allTF_result)
'''

atac_allTF_feature_set ={}
for i in (the_atacclu_19.index.values):
    if i in atac_allTF_feature_set.keys():
        atac_allTF_feature_set[i] = atac_allTF_feature_set[i]+1
    else:
        atac_allTF_feature_set[i] = 1
feature_filter_rate = len(the_atacclu_19.index.values)/len(the_atacclu_19_unique)
feature_filter_rate = feature_filter_rate*0.8
for key,value in atac_allTF_feature_set.items():
    if value < feature_filter_rate:
        continue
    key_index = np.where(atac_exp.index.values == key)[0]
    feature_filter_allTF_result = np.append(feature_filter_allTF_result,key_index)
temp_intersect = np.array([j for j in diff_atac_set.values[4,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[7,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[17,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[8:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[10,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)

temp_len = len(feature_filter_allTF_result)
'''
atac_allTF_feature_set ={}
for i in (the_atacclu_20.index.values):
    if i in atac_allTF_feature_set.keys():
        atac_allTF_feature_set[i] = atac_allTF_feature_set[i]+1
    else:
        atac_allTF_feature_set[i] = 1
feature_filter_rate = len(the_atacclu_20.index.values)/len(the_atacclu_20_unique)
feature_filter_rate = feature_filter_rate*0.8
for key,value in atac_allTF_feature_set.items():
    if value < feature_filter_rate:
        continue
    key_index = np.where(atac_exp.index.values == key)[0]
    feature_filter_allTF_result = np.append(feature_filter_allTF_result,key_index)
temp_intersect = np.array([j for j in diff_atac_set.values[2:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[5,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[4:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[14,:] if j in feature_filter_allTF_result[temp_len:-1]])
feature_filter_allTF_intersect = np.append(feature_filter_allTF_intersect,temp_intersect)
temp_len = len(feature_filter_allTF_result)
'''
# atac regression spliting by gene, the atac features of each gene is 5000
atac_feature_set = the_atacclu_20.index.values
#atac_feature_set = atac_feature_set.reshape(-1,5000)
atac_feature_dict = {}
for i in (atac_feature_set):
    if i in atac_feature_dict.keys():
        atac_feature_dict[i] = atac_feature_dict[i]+1
    else:
        atac_feature_dict[i] = 1
feature_filter_rate = len(atac_feature_set)/len(the_atacclu_20_unique)
feature_filter_rate = feature_filter_rate*0.8
feature_filter_result = None
for key,value in atac_feature_dict.items():
    if value < feature_filter_rate:
        continue
    key_index = np.where(atac_exp.index.values == key)[0]
    if feature_filter_result is None:
        feature_filter_result = key_index
    else:
        feature_filter_result = np.append(feature_filter_result,key_index)
cluster_spectific_atac_regulationSite = {}
for i in range(diff_atac_set.values.shape[0]):
    temp_intersection = np.array([j for j in diff_atac_set.values[i,:] if j in feature_filter_result])
    cluster_spectific_atac_regulationSite[i] = temp_intersection

cluster_spectific_atac_regulationSit_index = None
for key,value in cluster_spectific_atac_regulationSite.items():
    if cluster_spectific_atac_regulationSit_index is None:
        cluster_spectific_atac_regulationSit_index = value
    else:
        cluster_spectific_atac_regulationSit_index = np.append(cluster_spectific_atac_regulationSit_index,value)
cluster_0_specfici_atac_regulationSit_index = np.append(cluster_spectific_atac_regulationSite[2],cluster_spectific_atac_regulationSite[7])
# atac clu
the_atacclu_set = pd.read_csv(save_path+'atac_cluster_umap.csv', header=0, index_col=0)
the_atacclu = the_atacclu_set['atac_cluster']
the_atac_20_clu = the_atacclu_set.index.values[the_atacclu == 2]
the_atac_22_clu = the_atacclu_set.index.values[the_atacclu == 4]
the_atac_24_clu = the_atacclu_set.index.values[the_atacclu == 6]
the_atac_25_clu = the_atacclu_set.index.values[the_atacclu == 7]

cell_clu = pd.read_csv(save_path+'multivae_umap_imputation.csv', header=0, index_col=0)
cell_clu_index = cell_clu['atac_cluster'].values

gene_names = gene_exp.index.values
atac_names = atac_exp.index.values
unique_gene_index = np.unique(diff_gene_set.values.flatten())
unique_atac_index = np.unique(diff_atac_set.values.flatten())
diff_gene_names = gene_names[unique_gene_index]
diff_atac_names = atac_names[unique_atac_index]

the_atacclu_20_index = None
for i in the_atacclu_20_unique:
    temp = np.where(atac_names == i)[0]
    if len(temp) == 0:
        continue
    if the_atacclu_20_index is None:
        the_atacclu_20_index = temp
    else:
        the_atacclu_20_index = np.append(the_atacclu_20_index,temp)

the_atacclu_20_index = None
for i in the_atacclu_20_unique:
    temp = np.where(atac_names == i)[0]
    if len(temp) == 0:
        continue
    if the_atacclu_20_index is None:
        the_atacclu_20_index = temp
    else:
        the_atacclu_20_index = np.append(the_atacclu_20_index,temp)

the_atac_20_clu_index = None
for i in the_atac_20_clu:
    temp = np.where(atac_names == i)[0]
    if len(temp) == 0:
        continue
    if the_atac_20_clu_index is None:
        the_atac_20_clu_index = temp
    else:
        the_atac_20_clu_index = np.append(the_atac_20_clu_index,temp)
temp_intersection_20 = np.array([i for i in the_atac_20_clu_index if i in feature_filter_result])
the_atac_20_clu_with_geneClu_0_2_19 = np.array([i for i in the_atac_20_clu_index if i in feature_filter_allTF_result])
#cluster_spectific_atac_regulationSit_index = np.append(cluster_spectific_atac_regulationSit_index,temp_intersection_20)


the_atac_22_clu_index = None
for i in the_atac_22_clu:
    temp = np.where(atac_names == i)[0]
    if len(temp) == 0:
        continue
    if the_atac_22_clu_index is None:
        the_atac_22_clu_index = temp
    else:
        the_atac_22_clu_index = np.append(the_atac_22_clu_index,temp)
temp_intersection_22 = np.array([i for i in the_atac_22_clu_index if i in feature_filter_result])
the_atac_22_clu_with_geneClu_0_2_19 = np.array([i for i in the_atac_22_clu_index if i in feature_filter_allTF_result])

#cluster_spectific_atac_regulationSit_index = np.append(cluster_spectific_atac_regulationSit_index,temp_intersection_22)

the_atac_24_clu_index = None
for i in the_atac_24_clu:
    temp = np.where(atac_names == i)[0]
    if len(temp) == 0:
        continue
    if the_atac_24_clu_index is None:
        the_atac_24_clu_index = temp
    else:
        the_atac_24_clu_index = np.append(the_atac_24_clu_index,temp)
the_atac_24_clu_with_geneClu_0_2_19 = np.array([i for i in the_atac_24_clu_index if i in feature_filter_allTF_result])

the_atac_25_clu_index = None
for i in the_atac_25_clu:
    temp = np.where(atac_names == i)[0]
    if len(temp) == 0:
        continue
    if the_atac_25_clu_index is None:
        the_atac_25_clu_index = temp
    else:
        the_atac_25_clu_index = np.append(the_atac_25_clu_index,temp)
the_atac_25_clu_with_geneClu_0_2_19 = np.array([i for i in the_atac_25_clu_index if i in feature_filter_allTF_result])

# TF genes exp
TF_gene_names = pd.read_csv(save_path+'geneClu_TF_map.csv', header=0, index_col=0)
TF_gene_index = None
for i in TF_gene_names.index.values:
    temp = np.where(gene_exp.index.values == "('"+i+"',)")[0]
    if TF_gene_index is None:
        TF_gene_index = temp
    else:
        TF_gene_index = np.append(TF_gene_index,temp)
TF_gene_intersection = np.array([i for i in TF_gene_index if i in joint_geneclu_index])


plot_exp_data = None
plot_atac_data = None
plot_exp_rate = None
for cell_clu in np.unique(cell_clu_index):
    temp_exp = gene_exp.values[:,cell_clu_index == cell_clu]
    #temp_exp = temp_exp[the_geneclu_20.values[:,0],:]
    #temp_exp = temp_exp[np.append(the_geneclu_0.values[:, 0],the_geneclu_20.values[:, 0]), :]
    #temp_exp = temp_exp[joint_geneclu_index,:]

    temp_exp = temp_exp[TF_gene_index, :]
    #temp_exp = temp_exp[joint_geneclu_dff_index, :]
    #temp_exp = np.log1p(temp_exp)
    temp_atac = atac_exp.values[:, cell_clu_index == cell_clu]
    #temp_atac = temp_atac[the_atac_25_clu_with_geneClu_0_2_19,:]
    #temp_atac = temp_atac[temp_intersection_22,:]
    temp_atac = temp_atac[feature_filter_allTF_result, :]
    #temp_atac = temp_atac[cluster_spectific_atac_regulationSit_index,:]
    #temp_atac = temp_atac[np.append(the_atac_24_clu_index,the_atac_25_clu_index), :]
    #temp_atac = temp_atac[np.append(the_atacclu_0_index,the_atacclu_20_index), :]

    temp_exp_ave = np.ones(temp_exp.shape[0])
    temp_exp_rate = np.ones(temp_exp.shape[0])
    for i in range(temp_exp.shape[0]):
        temp = temp_exp[i,:]
        temp_exp_ave[i] = np.average(temp[temp > 0])
        if np.isnan(temp_exp_ave[i]):
            temp_exp_ave[i] = 0
        temp_exp_rate[i] = 1-len(np.where(temp == 0)[0])/len(temp)
    #temp_exp = temp_exp_ave
    temp_exp = np.average(temp_exp,axis=1)
    #temp_exp = temp_exp/np.max(temp_exp)
    temp_atac = np.sum(temp_atac,axis=1)/len(np.where(cell_clu_index == cell_clu)[0])
    if plot_exp_data is None:
        plot_exp_data = temp_exp
        plot_atac_data = temp_atac
        plot_exp_rate = temp_exp_rate
    else:
        plot_exp_data = np.vstack((plot_exp_data,temp_exp))
        plot_atac_data = np.vstack((plot_atac_data,temp_atac))
        plot_exp_rate = np.vstack((plot_exp_rate,temp_exp_rate))
for i in range(plot_exp_data.shape[1]):
    #plot_exp_data[:,i] = plot_exp_data[:,i]/np.max(plot_exp_data[:,i])
    plot_exp_data[:, i] = (plot_exp_data[:, i]-np.min(plot_exp_data[:, i])) / (np.max(plot_exp_data[:, i]) - np.min(plot_exp_data[:, i]))+0.000001

for i in range(plot_atac_data.shape[1]):
#    plot_atac_data[:,i] = plot_atac_data[:,i]/np.max(plot_atac_data[:,i])
    plot_atac_data[:, i] = (plot_atac_data[:, i]-np.min(plot_atac_data[:, i])) / (np.max(plot_atac_data[:, i]) - np.min(plot_atac_data[:, i]))+0.000001
#ax = sns.heatmap(plot_exp_data.T,yticklabels=gene_names[np.append(the_geneclu_0.values[:, 0],the_geneclu_20.values[:, 0])],cmap='rainbow')
filter_0_2_19_clu_atac = None
for i in range(plot_atac_data.shape[1]):
    if plot_atac_data[0, i] <= 0.4:
        continue
    if plot_atac_data[3, i] <= 0.4:
        continue
    if plot_atac_data[5, i] <= 0.4:
        continue
    if plot_atac_data[9, i] <= 0.4:
        continue
    if filter_0_2_19_clu_atac is None:
        filter_0_2_19_clu_atac = plot_atac_data[:,i]
    else:
        filter_0_2_19_clu_atac = np.vstack((filter_0_2_19_clu_atac,plot_atac_data[:,i]))
ax = sns.heatmap(plot_exp_data.T,cmap='rainbow')
plt.show()
#ax = sns.clustermap(plot_exp_data.T,yticklabels=gene_names[np.append(the_geneclu_0.values[:, 0],the_geneclu_20.values[:, 0])], metric="correlation",cmap='rainbow')
ax = sns.clustermap(plot_exp_data.T, metric="correlation",cmap='rainbow')
plt.show()
ax = sns.clustermap(plot_exp_data.T, cmap='rainbow')
plt.show()
ax = sns.heatmap(plot_atac_data.T,cmap='rainbow')
plt.show()
ax = sns.clustermap(plot_atac_data.T, metric="correlation",cmap='rainbow')
plt.show()
ax = sns.clustermap(plot_atac_data.T, cmap='rainbow')
plt.show()

ax = sns.heatmap(filter_0_2_19_clu_atac,cmap='rainbow')
plt.show()
ax = sns.clustermap(filter_0_2_19_clu_atac, metric="correlation",cmap='rainbow')
plt.show()
ax = sns.clustermap(filter_0_2_19_clu_atac, cmap='rainbow')
plt.show()

print("ok")
