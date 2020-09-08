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
from scipy.stats import pearsonr
import re
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
import random


save_path: str = "E:/data/qiliu/single-cell program/ATAC/result/whole figure/top 5000 knn 3 pvalue 1/data/"
gene_exp = pd.read_csv(save_path+'gene_imputation.csv', header=0, index_col=0)
atac_exp = pd.read_csv(save_path+'atac_aggregation.csv', header=0, index_col=0)
promoter_location = pd.read_csv(save_path+'promoter/promoter_ucsc.txt', sep='\t', header=0, index_col=0)
diff_gene_set = pd.read_csv(save_path+'gene_diff_matrix_adj.csv', header=0, index_col=0)
diff_atac_set = pd.read_csv(save_path+'atac_diff_matrix.csv', header=0, index_col=0)
cell_clu_set = pd.read_csv(save_path+'multivae_umap_imputation.csv', header=0, index_col=0)
cell_clu = np.array(cell_clu_set['atac_cluster'].tolist())
gene_names = gene_exp.index.values
gene_names_index_dict = {}
for i in range(len(gene_names)):
    gene_names_index_dict[gene_names[i]] = i

atac_names = atac_exp.index.values
atac_names_index_dict = {}
for i in range(len(atac_names)):
    atac_names_index_dict[atac_names[i]] = i

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

# atac clu
the_atacclu_set = pd.read_csv(save_path+'atac_cluster_umap.csv', header=0, index_col=0)
the_atacclu = the_atacclu_set['atac_cluster']
the_atac_20_clu = the_atacclu_set.index.values[the_atacclu == 2]
the_atac_22_clu = the_atacclu_set.index.values[the_atacclu == 4]
the_atac_24_clu = the_atacclu_set.index.values[the_atacclu == 6]
the_atac_25_clu = the_atacclu_set.index.values[the_atacclu == 7]

#neighborhood relationship
clu_neighbor_dict = {0:[0,3,9,14], 1:[1,11,13,15], 2:[2,8,1,15], 3:[3,0,9,14], 4:[4,7,12,14], 5:[5,7,13,14], 6:[6,2,8,15], 7:[7,4,12,14], 8:[8,2,6,15],
                     9:[9,0,3], 10:[10,11,12,17], 11:[11,2,10,17], 12:[12,4,10,7], 13:[13,1,2,15], 14:[14,3,9,13], 15:[15,1,2,13], 16:[16], 17:[17,2,11]}
clu_gene_coexp_dict = {"gene clu 0":["gene clu 0","gene clu 2","gene clu 19"], "gene clu 2":["gene clu 0","gene clu 2","gene clu 19"],"gene clu 4":["gene clu 4","gene clu 5","gene clu 6","gene clu 9","gene clu 13","gene clu 15","gene clu 16","gene clu 20"],
                       "gene clu 5":["gene clu 4","gene clu 5","gene clu 6","gene clu 9","gene clu 13","gene clu 15","gene clu 16","gene clu 20"],"gene clu 6":["gene clu 4","gene clu 5","gene clu 6","gene clu 9","gene clu 13","gene clu 15","gene clu 16","gene clu 20"],
                       "gene clu 9":["gene clu 4","gene clu 5","gene clu 6","gene clu 9","gene clu 13","gene clu 15","gene clu 16","gene clu 20"],"gene clu 13":["gene clu 4","gene clu 5","gene clu 6","gene clu 9","gene clu 13","gene clu 15","gene clu 16","gene clu 20"],
                       "gene clu 15":["gene clu 4","gene clu 5","gene clu 6","gene clu 9","gene clu 13","gene clu 15","gene clu 16","gene clu 20"],"gene clu 16":["gene clu 4","gene clu 5","gene clu 6","gene clu 9","gene clu 13","gene clu 15","gene clu 16","gene clu 20"],
                       "gene clu 19":["gene clu 0","gene clu 2","gene clu 19"], "gene clu 20":["gene clu 4","gene clu 5","gene clu 6","gene clu 9","gene clu 13","gene clu 15","gene clu 16","gene clu 20"]}
# enhancer-gene exp correlation
predict_gene_exp_corr = np.array([])
random_predict_gene_exp_corr = np.array([])
predict_gene_exp_corr_index = np.array([])
random_predict_gene_exp_corr_index = np.array([])
PLS_predict_gene_exp_corr = np.array([])
PLS_predict_gene_exp_corr_index = np.array([])

temp_geneclu_index = np.array([0,2,4,5,6,9,13,15,16,19,20])
for i in temp_geneclu_index:
    sub_gene_set = eval("the_geneclu_"+str(i))
    sub_gene_set = sub_gene_set.values[:,0]
    x_train, x_test, y_train, y_test = train_test_split(atac_exp.values.T,
                                                        gene_exp.values[sub_gene_set, :].T,
                                                        test_size=0.1,
                                                        random_state=33)
    pls = PLSRegression(n_components=40, scale=False)
    pls.fit(x_train, y_train)  # fit也是一个函数，，，两个参数，第一个参数是训练集，第二个参数是目标。
    y_pls_predict = pls.predict(x_test)
    test_pls_error = np.mean((y_pls_predict - y_test) ** 2, axis=0)
    test_pls_R2 = np.zeros(y_test.shape[0])
    for j in range(y_test.shape[0]):
        test_pls_R2[j] = np.corrcoef(y_pls_predict[j,:], y_test[j,:])[0,1]
    #test_pls_R2 = np.diagonal(np.corrcoef(y_pls_predict, y_test)).flatten()
    PLS_predict_gene_exp_corr = np.append(PLS_predict_gene_exp_corr,test_pls_R2)
    PLS_predict_gene_exp_corr_index = np.append(PLS_predict_gene_exp_corr_index,i*np.ones(len(test_pls_R2)).astype(np.int))
    print(i)
    indices_pls = np.argsort(np.abs(pls.coef_).T)[:, ::-1].T
    selected_pls_feature_index = indices_pls[0:5000, :]
    selected_pls_feature_weight = None
    for j in range(selected_pls_feature_index.shape[1]):
        if selected_pls_feature_weight is None:
            selected_pls_feature_weight = pls.coef_[selected_pls_feature_index[:, j], j]
        else:
            selected_pls_feature_weight = np.vstack((selected_pls_feature_weight, pls.coef_[selected_pls_feature_index[:, j], j]))
    df = pd.DataFrame(
        data=selected_pls_feature_weight,
        index=gene_exp.index.values[sub_gene_set])
    df.to_csv(os.path.join(save_path, "the" + str(i) + "atac_pls_cluster_weight_rerun.csv"))
    df = pd.DataFrame(
        data=selected_pls_feature_index.T,
        index=gene_exp.index.values[sub_gene_set])
    df.to_csv(os.path.join(save_path, "the" + str(i) + "atac_pls_cluster_index_rerun.csv"))


for i in temp_geneclu_index:
    sub_gene_set = eval("the_geneclu_" + str(i))
    sub_gene_set = sub_gene_set.values[:, 0]
    sub_atac_set = eval("the_atacclu_" + str(i))
    sub_atac_weight = sub_atac_set.values[:,0]
    sub_atac_index = np.zeros(len(sub_atac_weight)).astype(np.int)
    for j in range(len(sub_atac_index)):
        sub_atac_index[j] = atac_names_index_dict[sub_atac_set.index.values[j]]
    #sub_atac_index_unique = np.unique(sub_atac_index)
    sub_atac_index_unique = np.array([n for n in range(len(atac_names))])
    sub_atac_index_temp = np.zeros((len(sub_gene_set),1500)).astype(np.int)
    sub_atac_weight_temp = np.zeros((len(sub_gene_set),1500))
    for j in range(sub_atac_index_temp.shape[1]):
        sub_atac_index_temp[:,j] = sub_atac_index[j*len(sub_gene_set):(j+1)*len(sub_gene_set)]
        sub_atac_weight_temp[:,j] = sub_atac_weight[j*len(sub_gene_set):(j+1)*len(sub_gene_set)]
    sub_atac_index = sub_atac_index_temp
    sub_atac_weight = sub_atac_weight_temp
    #sub_atac_index = sub_atac_index.reshape((len(sub_gene_set),-1))
    #sub_atac_weight = sub_atac_weight.reshape((len(sub_gene_set),-1))

    sub_gene_exp = gene_exp.values[sub_gene_set,:]
    sub_atac_predict_exp = np.zeros((sub_gene_exp.shape[0],sub_gene_exp.shape[1]))
    sub_atac_random_predict_exp = np.zeros((sub_gene_exp.shape[0],sub_gene_exp.shape[1]))
    for j in range(sub_atac_index.shape[0]):
        sub_atac_exp = atac_exp.values[sub_atac_index[j,:],:]
        sub_atac_predict_exp[j,:] = np.dot(sub_atac_weight[j,:], sub_atac_exp)

        random.shuffle(sub_atac_index_unique)
        sub_atac_exp = atac_exp.values[sub_atac_index_unique[0:sub_atac_index.shape[1]], :]
        sub_atac_weight_shuffle = np.array([n for n in range(sub_atac_weight.shape[1])])
        sub_atac_weight_shuffle = random.shuffle(sub_atac_weight_shuffle)
        sub_atac_random_predict_exp[j, :] = np.dot((sub_atac_weight[j, :])[sub_atac_weight_shuffle], sub_atac_exp)

    sub_gene_pred_corr = np.zeros((2,sub_gene_exp.shape[0]))
    for j in range(sub_gene_exp.shape[0]):
        sub_gene_pred_corr[0,j] = np.corrcoef(sub_gene_exp[j,:],sub_atac_predict_exp[j,:])[0,1]
        sub_gene_pred_corr[1, j] = np.corrcoef(sub_gene_exp[j, :], sub_atac_random_predict_exp[j, :])[0,1]
    predict_gene_exp_corr = np.append(predict_gene_exp_corr,sub_gene_pred_corr[0,:])
    random_predict_gene_exp_corr = np.append(random_predict_gene_exp_corr, sub_gene_pred_corr[1, :])
    predict_gene_exp_corr_index = np.append(predict_gene_exp_corr_index,i*np.ones(sub_gene_exp.shape[0]).astype(np.int))
    random_predict_gene_exp_corr_index = np.append(random_predict_gene_exp_corr_index, i*np.ones(sub_gene_exp.shape[0]).astype(np.int))
joint_predict_gene_exp_corr = np.append(predict_gene_exp_corr,random_predict_gene_exp_corr)
joint_predict_gene_exp_corr_1 = np.append(np.abs(predict_gene_exp_corr),np.abs(random_predict_gene_exp_corr))
joint_predict_gene_exp_corr_index = np.append(predict_gene_exp_corr_index,random_predict_gene_exp_corr_index)
predict_class = np.append(np.zeros(len(predict_gene_exp_corr)).astype(np.int),np.ones(len(random_predict_gene_exp_corr)).astype(np.int))
joint_predict_matrix = np.vstack((joint_predict_gene_exp_corr,joint_predict_gene_exp_corr_index))
joint_predict_matrix = np.vstack((joint_predict_matrix,predict_class))



R2_df = pd.DataFrame(joint_predict_matrix.T,columns=['R2', 'gene_clusters','random'])
sns.violinplot(x="gene_clusters", y="R2", data=R2_df,
            hue='random',
            split=True,
            linewidth = 2,   # 线宽
            width = 0.8,     # 箱之间的间隔比例
            palette = 'muted', # 设置调色板
            scale = 'count',  # 测度小提琴图的宽度：area-面积相同，count-按照样本数量决定宽度，width-宽度一样
            gridsize = 50,   # 设置小提琴图边线的平滑度，越高越平滑
            inner = 'box',   # 设置内部显示类型 → “box”, “quartile”, “point”, “stick”, None
            #bw = 0.8        # 控制拟合程度，一般可以不设置
           )
plt.show()
sns.boxplot(x="gene_clusters",y="R2",data=R2_df,hue="random",palette="hls")
plt.show()

joint_predict_gene_exp_corr_PLS = np.append(PLS_predict_gene_exp_corr,random_predict_gene_exp_corr)
joint_predict_gene_exp_corr_PLS_index = np.append(PLS_predict_gene_exp_corr_index,random_predict_gene_exp_corr_index)
predict_class_PLS = np.append(np.zeros(len(PLS_predict_gene_exp_corr)).astype(np.int),np.ones(len(random_predict_gene_exp_corr)).astype(np.int))


joint_predict_matrix_PLS = np.vstack((joint_predict_gene_exp_corr_PLS,joint_predict_gene_exp_corr_PLS_index))
joint_predict_matrix_PLS = np.vstack((joint_predict_matrix_PLS,predict_class_PLS))
R2_df_PLS = pd.DataFrame(joint_predict_matrix_PLS.T,columns=['R2', 'gene_clusters','random'])
sns.violinplot(x="gene_clusters", y="R2", data=R2_df_PLS,
            hue='random',
            split=True,
            linewidth = 2,   # 线宽
            width = 0.8,     # 箱之间的间隔比例
            palette = 'muted', # 设置调色板
            scale = 'count',  # 测度小提琴图的宽度：area-面积相同，count-按照样本数量决定宽度，width-宽度一样
            gridsize = 50,   # 设置小提琴图边线的平滑度，越高越平滑
            inner = 'box',   # 设置内部显示类型 → “box”, “quartile”, “point”, “stick”, None
            #bw = 0.8        # 控制拟合程度，一般可以不设置
           )
plt.show()
sns.boxplot(x="gene_clusters",y="R2",data=R2_df_PLS,hue="random",palette="hls")
plt.show()
R2_df_PLS.to_csv(os.path.join(save_path, "PLS_prediction_corr_R2.csv"))

joint_predict_matrix_1 = np.vstack((joint_predict_gene_exp_corr_1,joint_predict_gene_exp_corr_index))
joint_predict_matrix_1 = np.vstack((joint_predict_matrix_1,predict_class))
R2_df_1 = pd.DataFrame(joint_predict_matrix_1.T,columns=['R2', 'gene_clusters','random'])
sns.violinplot(x="gene_clusters", y="R2", data=R2_df_1,
            hue='random',
            split=True,
            linewidth = 2,   # 线宽
            width = 0.8,     # 箱之间的间隔比例
            palette = 'muted', # 设置调色板
            scale = 'count',  # 测度小提琴图的宽度：area-面积相同，count-按照样本数量决定宽度，width-宽度一样
            gridsize = 50,   # 设置小提琴图边线的平滑度，越高越平滑
            inner = 'box',   # 设置内部显示类型 → “box”, “quartile”, “point”, “stick”, None
            #bw = 0.8        # 控制拟合程度，一般可以不设置
           )
plt.show()
sns.boxplot(x="gene_clusters",y="R2",data=R2_df_1,hue="random",palette="hls")
plt.show()
# TF of each atac clu
the_TF_clu_set = pd.read_csv(save_path+'TF enrichment.csv', header=0)

ori_TF = the_TF_clu_set['Best Match'].tolist()
TF_names = None
index = 1
for i in ori_TF:
    temp_TF = i.split("/")
    if len(temp_TF[0].split("_")) > 1:
        temp_TF[0] = (temp_TF[0].split("_"))[1]
    elif len(temp_TF[0].split("(")) > 1:
        temp_TF[0] = (temp_TF[0].split("("))[0]
    # combine the sub-TF of TF family
    reg = re.compile(r'[a-zA-Z]+')
    res = reg.findall(temp_TF[0])
    #if len(res) > 0:  # identify the TF family
    #    temp_TF[0] = res[0]

    if TF_names is None:
        TF_names = np.array(temp_TF[0])
    else:
        TF_names = np.append(TF_names,temp_TF[0])
peak_clu_row = np.unique(np.array(the_TF_clu_set['gene clu ID'].tolist()))
TF_name_column = np.unique(TF_names)
peak_clu_TF_exp = np.zeros((len(peak_clu_row),len(TF_name_column)))

peak_clu_row_dict = {}
for i in range(len(peak_clu_row)):
    peak_clu_row_dict[peak_clu_row[i]] = i
TF_name_column_dict = {}
for i in range(len(TF_name_column)):
    TF_name_column_dict[TF_name_column[i]] = i
general_TF_dict = {}
specific_TF_dict = {}
for i in the_TF_clu_set.index.values:
    temp_data = the_TF_clu_set.loc[i]
    temp_TF = (temp_data[3].split("/"))[0]
    if len(temp_TF.split("_")) > 1:
        temp_TF = (temp_TF.split("_"))[1]
    elif len(temp_TF.split("(")) > 1:
        temp_TF = (temp_TF.split("("))[0]
    # combine the sub-TF of TF family
    reg = re.compile(r'[a-zA-Z]+')
    res = reg.findall(temp_TF)
    #if len(res) > 0:     # identify the TF family
    #    temp_TF = res[0]

    temp_value = -np.log(temp_data[2])
    if np.isinf(temp_value):
        temp_value = 690
    if peak_clu_TF_exp[peak_clu_row_dict[temp_data[0]],TF_name_column_dict[temp_TF]] > 0:
        temp_value = np.max([peak_clu_TF_exp[peak_clu_row_dict[temp_data[0]],TF_name_column_dict[temp_TF]],temp_value])
    peak_clu_TF_exp[peak_clu_row_dict[temp_data[0]],TF_name_column_dict[temp_TF]] = np.log1p(temp_value)
    if temp_data[0] in specific_TF_dict.keys():
        specific_TF_dict[temp_data[0]] = np.append(specific_TF_dict[temp_data[0]], temp_TF)
    else:
        specific_TF_dict[temp_data[0]] = temp_TF

    if temp_data[0] == "peak clu 20" or temp_data[0] == "peak clu 22":
        if temp_data[0] in general_TF_dict.keys():
            general_TF_dict[temp_data[0]] = np.append(general_TF_dict[temp_data[0]], temp_TF)
        else:
            general_TF_dict[temp_data[0]] = temp_TF


for i in range(peak_clu_TF_exp.shape[1]):
    peak_clu_TF_exp[:,i] = peak_clu_TF_exp[:,i]/np.max(peak_clu_TF_exp[:,i])

peak_clu_TF_exp_adata = pd.DataFrame(peak_clu_TF_exp.T,index=TF_name_column,columns=peak_clu_row)
ax = sns.heatmap(peak_clu_TF_exp_adata,cmap='RdYlBu_r')
plt.show()
ax = sns.clustermap(peak_clu_TF_exp_adata, metric="correlation",cmap='RdYlBu_r')
plt.show()
rearranged_row = ax.dendrogram_row.reordered_ind
rearranged_col = ax.dendrogram_col.reordered_ind
ax = sns.clustermap(peak_clu_TF_exp_adata , cmap='RdYlBu_r')
plt.show()
ax = sns.clustermap(peak_clu_TF_exp_adata , row_cluster=False, col_cluster=True, cmap='RdYlBu_r')
plt.show()

peak_clu_TF_exp = peak_clu_TF_exp.T[rearranged_row,:]
peak_clu_TF_exp = peak_clu_TF_exp[:,rearranged_col]
peak_clu_TF_exp_adata = pd.DataFrame(peak_clu_TF_exp,index=TF_name_column[rearranged_row],columns=peak_clu_row[rearranged_col])
peak_clu_TF_exp_adata.to_csv(os.path.join(save_path,"diff_peak_clu_TF.csv"))


# gene regulation associatited  TF
the_geneTF_clu_set = pd.read_csv(save_path+'TF regulator enrichment.csv', header=0)
gene_clu_spectific_TF_dict ={}
unique_TFs = np.array([])
for i in the_geneTF_clu_set.index.values:
    temp_data = the_geneTF_clu_set.loc[i]
    temp_TF = (temp_data[3].split("/"))[0]
    if len(temp_TF.split("_")) > 1:
        temp_TF = (temp_TF.split("_"))[1]
    elif len(temp_TF.split("(")) > 1:
        temp_TF = (temp_TF.split("("))[0]
    # combine the sub-TF of TF family
    reg = re.compile(r'[a-zA-Z]+')
    res = reg.findall(temp_TF)
    #if len(res) > 0: # identify the TF family
    #    temp_TF = res[0]

    unique_TFs = np.append(unique_TFs,temp_TF)
    if np.any(TF_name_column == temp_TF):
        if temp_data[0] in gene_clu_spectific_TF_dict.keys():
            gene_clu_spectific_TF_dict[temp_data[0]] = np.append(gene_clu_spectific_TF_dict[temp_data[0]],temp_TF)
        else:
            gene_clu_spectific_TF_dict[temp_data[0]] = np.array(temp_TF)

unique_TFs_dict = {}
index = 0
for i in np.unique(unique_TFs):
    unique_TFs_dict[i] = index
    index = index +1
reg_peak_clu_dict = {}
index = 0
for i in gene_clu_spectific_TF_dict.keys():
    reg_peak_clu_dict[i] = index
    index = index+1
gene_clu_spectific_TF_exp = np.zeros((len(np.unique(unique_TFs)),len(gene_clu_spectific_TF_dict.keys())))
for i in the_geneTF_clu_set.index.values:
    temp_data = the_geneTF_clu_set.loc[i]
    temp_TF = (temp_data[3].split("/"))[0]
    if len(temp_TF.split("_")) > 1:
        temp_TF = (temp_TF.split("_"))[1]
    elif len(temp_TF.split("(")) > 1:
        temp_TF = (temp_TF.split("("))[0]
    # combine the sub-TF of TF family
    reg = re.compile(r'[a-zA-Z]+')
    res = reg.findall(temp_TF)
    #if len(res) > 0: # identify the TF family
    #    temp_TF = res[0]

    #if np.any(TF_name_column == temp_TF):
    if index >= 0:
        temp_value = -np.log(temp_data[2])
        if np.isinf(temp_value):
            temp_value = 690
        if gene_clu_spectific_TF_exp[unique_TFs_dict[temp_TF],reg_peak_clu_dict[temp_data[0]]] > 0:
            temp_value = np.max(
                [gene_clu_spectific_TF_exp[unique_TFs_dict[temp_TF],reg_peak_clu_dict[temp_data[0]]], temp_value])
        gene_clu_spectific_TF_exp[unique_TFs_dict[temp_TF],reg_peak_clu_dict[temp_data[0]]] = np.log1p(temp_value)

for i in range(gene_clu_spectific_TF_exp.shape[1]):
    gene_clu_spectific_TF_exp[:,i] = gene_clu_spectific_TF_exp[:,i]/np.max(gene_clu_spectific_TF_exp[:,i])
gene_clu_spectific_TF_exp_atata = pd.DataFrame(gene_clu_spectific_TF_exp,index=np.unique(unique_TFs),columns=reg_peak_clu_dict.keys())
ax = sns.heatmap(gene_clu_spectific_TF_exp_atata,cmap='RdYlBu_r')
plt.show()
ax = sns.clustermap(gene_clu_spectific_TF_exp_atata, metric="correlation",cmap='RdYlBu_r')
plt.show()
rearranged_row = ax.dendrogram_row.reordered_ind
rearranged_col = ax.dendrogram_col.reordered_ind
ax = sns.clustermap(gene_clu_spectific_TF_exp_atata + 0.05*np.random.rand(gene_clu_spectific_TF_exp.shape[0],gene_clu_spectific_TF_exp.shape[1]), cmap='RdYlBu_r')
plt.show()
ax = sns.clustermap(gene_clu_spectific_TF_exp_atata + 0.05*np.random.rand(gene_clu_spectific_TF_exp.shape[0],gene_clu_spectific_TF_exp.shape[1]), row_cluster=False, col_cluster=True, cmap='RdYlBu_r')
plt.show()
gene_clu_spectific_TF_exp = gene_clu_spectific_TF_exp[rearranged_row,:]
gene_clu_spectific_TF_exp = gene_clu_spectific_TF_exp[:,rearranged_col]
gene_clu_spectific_TF_exp_atata = pd.DataFrame(gene_clu_spectific_TF_exp,index=(np.unique(unique_TFs))[rearranged_row],columns=np.array(list(reg_peak_clu_dict.keys()))[rearranged_col])
gene_clu_spectific_TF_exp_atata.to_csv(os.path.join(save_path,"regression_peak_clu_TF.csv"))
# gene clu associatited  TF
the_geneTF_clu_set = pd.read_csv(save_path+'TF gene enrichment.csv', header=0)
unique_TFs = np.array([])
geneEnrich_clu_spectific_TF_dict ={}
for i in the_geneTF_clu_set.index.values:
    temp_data = the_geneTF_clu_set.loc[i]
    temp_TF = (temp_data[3].split("/"))[0]
    if len(temp_TF.split("_")) > 1:
        temp_TF = (temp_TF.split("_"))[1]
    elif len(temp_TF.split("(")) > 1:
        temp_TF = (temp_TF.split("("))[0]
    # combine the sub-TF of TF family
    reg = re.compile(r'[a-zA-Z]+')
    res = reg.findall(temp_TF)
    #if len(res) > 0: # identify the TF family
    #    temp_TF = res[0]

    unique_TFs = np.append(unique_TFs, temp_TF)
    if np.any(TF_name_column == temp_TF):
        if temp_data[0] in geneEnrich_clu_spectific_TF_dict.keys():
            geneEnrich_clu_spectific_TF_dict[temp_data[0]] = np.append(geneEnrich_clu_spectific_TF_dict[temp_data[0]],temp_TF)
        else:
            geneEnrich_clu_spectific_TF_dict[temp_data[0]] = np.array(temp_TF)
unique_TFs_dict = {}
index = 0
for i in np.unique(unique_TFs):
    unique_TFs_dict[i] = index
    index = index +1
reg_peak_clu_dict = {}
index = 0
for i in geneEnrich_clu_spectific_TF_dict.keys():
    reg_peak_clu_dict[i] = index
    index = index+1
gene_clu_spectific_TF_exp = np.zeros((len(np.unique(unique_TFs)),len(geneEnrich_clu_spectific_TF_dict.keys())))
for i in the_geneTF_clu_set.index.values:
    temp_data = the_geneTF_clu_set.loc[i]
    temp_TF = (temp_data[3].split("/"))[0]
    if len(temp_TF.split("_")) > 1:
        temp_TF = (temp_TF.split("_"))[1]
    elif len(temp_TF.split("(")) > 1:
        temp_TF = (temp_TF.split("("))[0]
    # combine the sub-TF of TF family
    reg = re.compile(r'[a-zA-Z]+')
    res = reg.findall(temp_TF)
    #if len(res) > 0: # identify the TF family
    #    temp_TF = res[0]

    #if np.any(TF_name_column == temp_TF):
    if index >= 0:
        temp_value = -np.log(float(temp_data[2]))
        if np.isinf(temp_value):
            temp_value = 690
        if gene_clu_spectific_TF_exp[unique_TFs_dict[temp_TF],reg_peak_clu_dict[temp_data[0]]] > 0:
            temp_value = np.max(
                [gene_clu_spectific_TF_exp[unique_TFs_dict[temp_TF],reg_peak_clu_dict[temp_data[0]]], temp_value])
        gene_clu_spectific_TF_exp[unique_TFs_dict[temp_TF],reg_peak_clu_dict[temp_data[0]]] = np.log1p(temp_value)
for i in range(gene_clu_spectific_TF_exp.shape[1]):
    gene_clu_spectific_TF_exp[:,i] = gene_clu_spectific_TF_exp[:,i]/np.max(gene_clu_spectific_TF_exp[:,i])
gene_clu_spectific_TF_exp_atata = pd.DataFrame(gene_clu_spectific_TF_exp,index=np.unique(np.unique(unique_TFs)),columns=reg_peak_clu_dict.keys())
ax = sns.heatmap(gene_clu_spectific_TF_exp_atata,cmap='RdYlBu_r')
plt.show()
ax = sns.clustermap(gene_clu_spectific_TF_exp_atata, metric="correlation",cmap='RdYlBu_r')
plt.show()
rearranged_row = ax.dendrogram_row.reordered_ind
rearranged_col = ax.dendrogram_col.reordered_ind
ax = sns.clustermap(gene_clu_spectific_TF_exp_atata , cmap='RdYlBu_r')
plt.show()
ax = sns.clustermap(gene_clu_spectific_TF_exp_atata + 0.05*np.random.rand(gene_clu_spectific_TF_exp.shape[0],gene_clu_spectific_TF_exp.shape[1]), row_cluster=False, col_cluster=True, cmap='RdYlBu_r')
plt.show()
gene_clu_spectific_TF_exp = gene_clu_spectific_TF_exp[rearranged_row,:]
gene_clu_spectific_TF_exp = gene_clu_spectific_TF_exp[:,rearranged_col]
gene_clu_spectific_TF_exp_atata = pd.DataFrame(gene_clu_spectific_TF_exp,index=(np.unique(unique_TFs))[rearranged_row],columns=np.array(list(reg_peak_clu_dict.keys()))[rearranged_col])
gene_clu_spectific_TF_exp_atata.to_csv(os.path.join(save_path,"regression_gene_clu_TF.csv"))





the_atac_20_clu_index = np.array([])
for i in the_atac_20_clu:
    the_atac_20_clu_index = np.append(the_atac_20_clu_index,atac_names_index_dict[i])
the_atac_22_clu_index = np.array([])
for i in the_atac_22_clu:
    the_atac_22_clu_index = np.append(the_atac_22_clu_index,atac_names_index_dict[i])

# psudotime
the_psudotime_set = pd.read_csv(save_path+'psudotime_palantir.csv', header=0, index_col=0)
the_psudotime_tsne_set = pd.read_csv(save_path+'psudotime_palantir_tsne.csv', header=0, index_col=0)
the_psudotime = the_psudotime_set.values

the_psudotime_cluster = np.zeros(the_psudotime_tsne_set.values.shape[0])
for i in range(the_psudotime_tsne_set.values.shape[0]):
    if the_psudotime_tsne_set.values[i,1] > 12:
        the_psudotime_cluster[i] = 1





genepeak_clu_str = ['the_atacclu_0_unique','the_atacclu_2_unique','the_atacclu_4_unique','the_atacclu_5_unique','the_atacclu_6_unique',
                    'the_atacclu_9_unique','the_atacclu_13_unique','the_atacclu_15_unique','the_atacclu_16_unique','the_atacclu_19_unique',
                    'the_atacclu_20_unique']
genepeak_clu_atac_index_set = {}
for i in genepeak_clu_str:
    for j in eval(i):
        if i in genepeak_clu_atac_index_set.keys():
            genepeak_clu_atac_index_set[i] = np.append(genepeak_clu_atac_index_set[i],atac_names_index_dict[j])
        else:
            genepeak_clu_atac_index_set[i] = atac_names_index_dict[j]

geneEnrich_clu_str = ['the_geneclu_0','the_geneclu_2','the_geneclu_4','the_geneclu_5','the_geneclu_6','the_geneclu_9',
                      'the_geneclu_13','the_geneclu_15','the_geneclu_16','the_geneclu_19','the_geneclu_20']
geneEnrich_clu_atac_index_set = {}
for i in geneEnrich_clu_str:
    for j in (eval(i)).index.values:
        if i in geneEnrich_clu_atac_index_set.keys():
            geneEnrich_clu_atac_index_set[i] = np.append(geneEnrich_clu_atac_index_set[i],gene_names_index_dict[j])
        else:
            geneEnrich_clu_atac_index_set[i] = gene_names_index_dict[j]

diff_atac_index_set = diff_atac_set.values

gene_clu_spectific_TF_exp_dict = {}
gene_clu_spectific_TF_index_dict = {}
gene_clu_spectific_TF_name_dict = {}
plot_index = 1
for key in gene_clu_spectific_TF_dict.keys():
    temp_tfs = gene_clu_spectific_TF_dict[key]
    sub_clu_TF_enrichment = None
    sub_genepeak_diffpeak_intersect = None
    sub_TFs = None
    print(key)
    for i in temp_tfs:
        sub_clu_TF_exp = None
        if sub_clu_TF_enrichment is None:
            sub_clu_TF_enrichment = peak_clu_TF_exp[:, TF_name_column_dict[i]]
            var_str = genepeak_clu_atac_index_set["the_atacclu_" + (key.split())[-1] + "_unique"]
            #temp_intersect = np.array([])
            for j in range(diff_atac_index_set.shape[0]):
                #temp_intersect = np.append(temp_intersect, [a for a in var_str if a in diff_atac_index_set[j, :]])
                #if j not in clu_neighbor_dict[j]:
                #    sub_clu_TF_enrichment[j] = 0.05
                #    continue
                temp_intersect = [a for a in var_str if a in diff_atac_index_set[j,:]]
                #temp_intersect_generalAtac_20 = [a for a in var_str if a in the_atac_20_clu_index]
                #temp_intersect_generalAtac_22 = [a for a in var_str if a in the_atac_22_clu_index]

                temp_specific_TF = None
                for temp_j in clu_neighbor_dict[j]:
                    if temp_j != j:
                        continue
                    var_str_TF = specific_TF_dict["peak clu " + str(temp_j)]
                    if temp_specific_TF is None:
                        temp_specific_TF = np.where(var_str_TF == i)[0]
                    else:
                        temp_specific_TF = np.append(temp_specific_TF, np.where(var_str_TF == i)[0])

                #var_str_TF = geneEnrich_clu_spectific_TF_dict["gene clu " + (key.split())[-1]]
                #temp_intersect_TF =np.where(var_str_TF == i)[0]



                temp_intersect_TF = None
                for sub_key in gene_clu_spectific_TF_dict.keys():
                    temp_str = "gene clu " + (key.split())[-1]
                    if temp_str not in clu_gene_coexp_dict[temp_str]:
                        continue
                    var_str_TF = geneEnrich_clu_spectific_TF_dict["gene clu " + (sub_key.split())[-1]]
                    if temp_intersect_TF is None or len(temp_intersect_TF) == 0:
                        temp_intersect_TF = np.where(var_str_TF == i)[0]
                    else:
                        temp_intersect_TF = np.append(temp_intersect_TF,np.where(var_str_TF == i)[0])


                var_str_TF_20 = general_TF_dict["peak clu 20"]
                temp_intersect_TF_20 = np.where(var_str_TF_20 == i)[0]
                var_str_TF_22 = general_TF_dict["peak clu 22"]
                temp_intersect_TF_22 = np.where(var_str_TF_22 == i)[0]

                if len(temp_intersect_TF_20) == 0:
                    sub_clu_TF_enrichment[-2] = 0.05
                if len(temp_intersect_TF_22) == 0:
                    sub_clu_TF_enrichment[-1] = 0.05

                if  len(temp_intersect_TF) == 0 or len(temp_intersect) == 0 or len(temp_specific_TF) == 0:
                    sub_clu_TF_enrichment[j] = 0.05
                else:

                    if sub_genepeak_diffpeak_intersect is None:
                        sub_genepeak_diffpeak_intersect = temp_intersect
                    else:
                        sub_genepeak_diffpeak_intersect = np.append(sub_genepeak_diffpeak_intersect, temp_intersect)

                    if sub_TFs is None:
                        sub_TFs = i
                    else:
                        sub_TFs = np.append(sub_TFs,i)

                    if sub_clu_TF_exp is None:
                        sub_clu_TF_exp = atac_exp.values[temp_intersect, :]
                    else:
                        sub_clu_TF_exp = np.vstack((sub_clu_TF_exp,atac_exp.values[temp_intersect, :]))

        else:
            temp_value = peak_clu_TF_exp[:, TF_name_column_dict[i]]
            temp_intersect = np.array([])
            #var_str = genepeak_clu_atac_index_set["the_atacclu_" + (key.split())[-1] + "_unique"]
            for j in range(diff_atac_index_set.shape[0]):
                #if j not in clu_neighbor_dict[j]:
                #    temp_value[j] = 0.05
                #    continue
                temp_intersect = [a for a in var_str if a in diff_atac_index_set[j,:]]
                #temp_intersect = np.append(temp_intersect,[a for a in var_str if a in diff_atac_index_set[j,:]])
                #temp_intersect_generalAtac_20 = [a for a in var_str if a in the_atac_20_clu_index]
                #temp_intersect_generalAtac_22 = [a for a in var_str if a in the_atac_22_clu_index]

                temp_specific_TF = None
                for temp_j in clu_neighbor_dict[j]:
                    if temp_j != j:
                        continue
                    var_str_TF = specific_TF_dict["peak clu " + str(temp_j)]
                    if temp_specific_TF is None:
                        temp_specific_TF = np.where(var_str_TF == i)[0]
                    else:
                        temp_specific_TF = np.append(temp_specific_TF, np.where(var_str_TF == i)[0])

                #var_str = geneEnrich_clu_atac_index_set["the_geneclu_" + (key.split())[-1]]
                #temp_intersect = [a for a in var_str if a in temp_intersect]

                #var_str_TF = geneEnrich_clu_spectific_TF_dict["gene clu " + (key.split())[-1]]
                #temp_intersect_TF = np.where(var_str_TF == i)[0]

                temp_intersect_TF = None
                for sub_key in gene_clu_spectific_TF_dict.keys():
                    temp_str = "gene clu " + (key.split())[-1]
                    if temp_str not in clu_gene_coexp_dict[temp_str]:
                        continue
                    var_str_TF = geneEnrich_clu_spectific_TF_dict["gene clu " + (sub_key.split())[-1]]
                    if temp_intersect_TF is None or len(temp_intersect_TF) == 0:
                        temp_intersect_TF = np.where(var_str_TF == i)[0]
                    else:
                        temp_intersect_TF = np.append(temp_intersect_TF, np.where(var_str_TF == i)[0])


                var_str_TF_20 = general_TF_dict["peak clu 20"]
                temp_intersect_TF_20 = np.where(var_str_TF_20 == i)[0]
                var_str_TF_22 = general_TF_dict["peak clu 22"]
                temp_intersect_TF_22 = np.where(var_str_TF_22 == i)[0]

                if len(temp_intersect_TF_20) == 0:
                    temp_value[-2] = 0.05
                if len(temp_intersect_TF_22) == 0:
                    temp_value[-1] = 0.05

                if len(temp_intersect) == 0 or len(temp_intersect_TF) == 0 or len(temp_specific_TF) == 0:
                    temp_value[j] = 0.05
                else:

                    if sub_genepeak_diffpeak_intersect is None:
                        sub_genepeak_diffpeak_intersect = temp_intersect
                    else:
                        sub_genepeak_diffpeak_intersect = np.append(sub_genepeak_diffpeak_intersect, temp_intersect)

                    if sub_TFs is None:
                        sub_TFs = i
                    else:
                        sub_TFs = np.append(sub_TFs,i)

                    if sub_clu_TF_exp is None:
                        sub_clu_TF_exp = atac_exp.values[temp_intersect, :]
                    else:
                        sub_clu_TF_exp = np.vstack((sub_clu_TF_exp,atac_exp.values[temp_intersect, :]))

            sub_clu_TF_enrichment = np.vstack((sub_clu_TF_enrichment,temp_value))

            if sub_clu_TF_exp is None:
                continue
            sub_peak_exp_agg = np.zeros((sub_clu_TF_exp.shape[0], len(np.unique(cell_clu))))
            for cell_index in np.unique(cell_clu):
                sub_peak_exp_agg[:, cell_index] = np.sum(sub_clu_TF_exp[:, cell_clu == cell_index], axis=1) / len(
                    np.where(cell_clu == cell_index)[0])
            for clu_index in range(sub_peak_exp_agg.shape[0]):
                sub_peak_exp_agg[clu_index, :] = sub_peak_exp_agg[clu_index, :] / np.max(sub_peak_exp_agg[clu_index, :])
            #ax = sns.clustermap(sub_peak_exp_agg, cmap='rainbow')
            #plt.ylabel(i)
            #plt.show()

            sub_peak_psudotime = np.array([])
            for cell_index in range(sub_clu_TF_exp.shape[1]):
                temp = np.sum(sub_clu_TF_exp[:,cell_index]) / sub_clu_TF_exp.shape[0]
                sub_peak_psudotime = np.append(sub_peak_psudotime,temp)
            temp_gene_index = (eval("the_geneclu_"+(key.split())[-1])).values[:, 0]
            sub_clu_gene_exp = gene_exp.values[temp_gene_index,:]
            for cell_index in range(sub_clu_gene_exp.shape[0]):
                sub_clu_gene_exp[cell_index,:] = sub_clu_gene_exp[cell_index,:] / np.max(sub_clu_gene_exp[cell_index,:])
            sub_gene_psudotime = np.array([])
            for cell_index in range(sub_clu_gene_exp.shape[1]):
                temp = np.sum(sub_clu_gene_exp[:, cell_index])/ sub_clu_gene_exp.shape[0]
                sub_gene_psudotime = np.append(sub_gene_psudotime, temp)
            '''
            sub_gene_psudotime_nureo = sub_gene_psudotime[np.where(the_psudotime_cluster == 0)[0]]
            sub_peak_psudotime_nureo = sub_peak_psudotime[np.where(the_psudotime_cluster == 0)[0]]
            sub_psudotime_nureo = the_psudotime[np.where(the_psudotime_cluster == 0)[0],:]
            gp = GaussianProcessRegressor(n_restarts_optimizer=9)
            gp.fit(sub_psudotime_nureo, sub_gene_psudotime_nureo)
            means, sigmas = gp.predict(sub_psudotime_nureo, return_std=True)
            plt.figure(4)
            plt.subplot(221)
            plt.scatter(x=sub_psudotime_nureo, y=means, c='darkorange')
            plt.fill_between(sub_psudotime_nureo[:, 0], means - sigmas, means + sigmas, color='turquoise', alpha=0.2)
            plt.ylabel("gene_nureo")

            gp = GaussianProcessRegressor(n_restarts_optimizer=9)
            gp.fit(sub_psudotime_nureo, sub_peak_psudotime_nureo)
            means, sigmas = gp.predict(sub_psudotime_nureo, return_std=True)
            plt.subplot(222)
            plt.scatter(x=sub_psudotime_nureo, y=means, c='darkorange')
            plt.fill_between(sub_psudotime_nureo[:, 0], means - sigmas, means + sigmas, color='turquoise', alpha=0.2)
            plt.ylabel("peak_nureo")

            sub_gene_psudotime_nonnureo = sub_gene_psudotime[np.where(the_psudotime_cluster == 1)[0]]
            sub_peak_psudotime_nonnureo = sub_peak_psudotime[np.where(the_psudotime_cluster == 1)[0]]
            sub_psudotime_nonnureo = the_psudotime[np.where(the_psudotime_cluster == 1)[0],:]
            gp = GaussianProcessRegressor(n_restarts_optimizer=9)
            gp.fit(sub_psudotime_nonnureo, sub_gene_psudotime_nonnureo)
            means, sigmas = gp.predict(sub_psudotime_nonnureo, return_std=True)
            plt.subplot(223)
            plt.scatter(x=sub_psudotime_nonnureo, y=means, c='darkorange')
            plt.fill_between(sub_psudotime_nonnureo[:, 0], means - sigmas, means + sigmas, color='turquoise', alpha=0.2)
            plt.ylabel("gene_nonnureo")

            gp = GaussianProcessRegressor(n_restarts_optimizer=9)
            gp.fit(sub_psudotime_nonnureo, sub_peak_psudotime_nonnureo)
            means, sigmas = gp.predict(sub_psudotime_nonnureo, return_std=True)
            plt.subplot(224)
            plt.scatter(x=sub_psudotime_nonnureo, y=means, c='darkorange')
            plt.fill_between(sub_psudotime_nonnureo[:, 0], means - sigmas, means + sigmas, color='turquoise', alpha=0.2)
            plt.ylabel("peak_nonnureo")

            gp = GaussianProcessRegressor( n_restarts_optimizer=9)
            gp.fit(the_psudotime, sub_gene_psudotime)
            means, sigmas = gp.predict(the_psudotime, return_std=True)
            plt.figure(2)
            plt.subplot(211)
            #plt.errorbar(the_psudotime, means, yerr=sigmas, alpha=0.5)
            #plt.plot(the_psudotime, means, 'darkorange')
            plt.scatter(x=the_psudotime, y=means, c='darkorange')
            plt.fill_between(the_psudotime[:, 0], means - sigmas, means + sigmas, color='turquoise',alpha=0.2)
            #plt.show()

            plt.subplot(212)
            gp = GaussianProcessRegressor(n_restarts_optimizer=9)
            gp.fit(the_psudotime, sub_peak_psudotime)
            means, sigmas = gp.predict(the_psudotime, return_std=True)
            #plt.errorbar(the_psudotime, means, yerr=sigmas, alpha=0.5)
            #plt.plot(the_psudotime, means, 'darkorange')
            plt.scatter(x=the_psudotime, y=means, c='darkorange')
            plt.fill_between(the_psudotime[:, 0], means - sigmas, means + sigmas, color='turquoise',alpha=0.2)
            #plt.show()

            plt.figure(3)
            plt.subplot(211)
            plt.scatter(x=the_psudotime[:,0], y = sub_gene_psudotime, s=1)
            #plt.scatter(x=the_psudotime[:,0], y=the_psudotime[:,1],c = sub_gene_psudotime, s=1)
            plt.ylabel("the_geneclu_"+(key.split())[-1])
            plt.subplot(212)
            plt.scatter(x=the_psudotime[:, 0], y = sub_peak_psudotime, s=1)
            #plt.scatter(x=the_psudotime[:, 0], y= the_psudotime[:,1],c = sub_peak_psudotime, s=1)
            plt.ylabel("the_peakclu_" + (key.split())[-1])
            '''

    if sub_genepeak_diffpeak_intersect is None:
        continue


    gene_clu_spectific_TF_exp_dict[key] = sub_clu_TF_enrichment
    gene_clu_spectific_TF_index_dict[key] = np.unique(sub_genepeak_diffpeak_intersect).astype(np.int32)
    gene_clu_spectific_TF_name_dict[key] = np.unique(sub_TFs)
    sub_peak_exp = atac_exp.values[gene_clu_spectific_TF_index_dict[key],:]
    sub_peak_exp_agg = np.zeros((sub_peak_exp.shape[0],len(np.unique(cell_clu))))
    for cell_index in np.unique(cell_clu):
        sub_peak_exp_agg[:,cell_index] = np.sum(sub_peak_exp[:, cell_clu == cell_index],axis=1)/len(np.where(cell_clu == cell_index)[0])
    for clu_index in range(sub_peak_exp_agg.shape[0]):
        sub_peak_exp_agg[clu_index,:] = sub_peak_exp_agg[clu_index,:]/np.max(sub_peak_exp_agg[clu_index,:])
    #plot_str = "4"+"3"+str(plot_index)
    #plot_index = plot_index +1
    #plt.subplot(int(plot_str))
    pd_array = sub_clu_TF_enrichment+ 0.05*np.random.rand(sub_clu_TF_enrichment.shape[0],sub_clu_TF_enrichment.shape[1])
    pd_data = pd.DataFrame(pd_array,index=temp_tfs)
    ax = sns.clustermap(pd_data, metric="correlation", cmap='rainbow')
    plt.show()
    #ax = sns.clustermap(sub_peak_exp_agg,metric="correlation", cmap='rainbow')
    #plt.show()

gene_clu_spectific_TF_exp = None
gene_clu_spectific_TF_names = None
for i in gene_clu_spectific_TF_exp_dict.keys():
    if gene_clu_spectific_TF_exp is None:
        gene_clu_spectific_TF_exp = gene_clu_spectific_TF_exp_dict[i]
        gene_clu_spectific_TF_names = gene_clu_spectific_TF_dict[i]
    else:
        gene_clu_spectific_TF_exp = np.vstack((gene_clu_spectific_TF_exp, gene_clu_spectific_TF_exp_dict[i]))
        gene_clu_spectific_TF_names = np.append(gene_clu_spectific_TF_names,gene_clu_spectific_TF_dict[i])
pd_array = gene_clu_spectific_TF_exp+ 0.05*np.random.rand(gene_clu_spectific_TF_exp.shape[0],gene_clu_spectific_TF_exp.shape[1])
pd_data = pd.DataFrame(pd_array, index= gene_clu_spectific_TF_names)
ax = sns.clustermap(pd_data, metric="correlation", cmap='rainbow')
plt.show()

ax = sns.clustermap(pd_data, cmap='rainbow')
plt.show()

rearranged_row = ax.dendrogram_row.reordered_ind
rearranged_col = np.array([11,17,1,15,2,6,8,5,13,14,3,9,10,0,16,12,4,7,18,19])
pd_array = pd_array[rearranged_row,:]
pd_array = pd_array[:,rearranged_col]
pd_data = pd.DataFrame(pd_array, index= gene_clu_spectific_TF_names[rearranged_row])
ax = sns.heatmap(pd_data, cmap='RdYlBu_r')
plt.show()


#promoter
promoter_gene_dic = {}
promoter_chr_dic = {}
for i in range(len(promoter_location.index.values)):
    temp = promoter_location.index.values[i].split('_')
    if len(temp) != 2:
        continue
    temp = temp[0]
    if not (temp in promoter_gene_dic):
        promoter_gene_dic[temp] = np.array([promoter_location['location'][i].astype(np.int)])
        promoter_chr_dic[temp] = np.array([promoter_location['chromsome'][i]])
    else:
        promoter_gene_dic[temp] = np.append(promoter_gene_dic[temp],(promoter_location['location'][i].astype(np.int)))
        promoter_chr_dic[temp] = np.append(promoter_chr_dic[temp],([promoter_location['chromsome'][i]]))

gene_names = gene_exp.index.values
atac_names = atac_exp.index.values
atac_chr_set = None
atac_loc_set = None
for atac_key in (atac_names):
    atac_key = atac_key[2:-3]
    temp = atac_key.split(':')
    if len(temp) != 2:
        continue
    temp_loc = temp[1].split('-')
    if len(temp_loc) != 2:
        continue
    if atac_chr_set is None:
        atac_chr_set = np.array(temp[0])
        atac_loc_set = np.array([temp_loc[0],temp_loc[1]])
    else:
        atac_chr_set = np.append(atac_chr_set,(temp[0]))
        atac_loc_set = np.vstack((atac_loc_set,np.array([temp_loc[0],temp_loc[1]])))
atac_loc_set = atac_loc_set.astype(np.int)

# intersecting promoter gene and atac
joint_gene_promoter_map = {}
joint_promoter_gene_map = {}
for i in gene_names:
    temp_gene = i
    temp_gene = temp_gene[2:-3]
    if temp_gene in promoter_gene_dic:
        temp_promoters_loc = promoter_gene_dic[temp_gene]
        temp_promoter_chr = promoter_chr_dic[temp_gene]
        temp_atac_index = np.where(atac_chr_set == temp_promoter_chr[0])
        temp_atac_index = temp_atac_index[0]
        flag = False
        for j in range(len(temp_atac_index)):
            atac_loc = atac_loc_set[temp_atac_index[j], :]
            for temp_promoter_loc in temp_promoters_loc:
                if temp_promoter_loc >= atac_loc[0] and temp_promoter_loc <= atac_loc[1]:
                    joint_gene_promoter_map[temp_gene] = temp_atac_index[j]
                    joint_promoter_gene_map[temp_atac_index[j]] = temp_gene
                    flag = True
                    break
            if flag:
                break
# intersecting gene cluster 9 with promoter
geneclu_temp = None
for i in range(len(the_geneclu_2.index.values)): # the specific clu control
    if geneclu_temp is None:
        geneclu_temp =  the_geneclu_2.index.values[i][2:-3]  # the specific clu control
    else:
        geneclu_temp = np.append(geneclu_temp,the_geneclu_2.index.values[i][2:-3]) # the specific clu control
geneClu_promoter_atac_gene_map = {}
geneClu_promoter_atac_gene_map[20] = [i for i in geneclu_temp if i in joint_gene_promoter_map.keys()]
geneClu_promoter_atac_index = None
geneClu_promoter_gene_index = None
for i in geneClu_promoter_atac_gene_map.keys():
    for j in geneClu_promoter_atac_gene_map[i]:
        if geneClu_promoter_atac_index is None:
            geneClu_promoter_atac_index = joint_gene_promoter_map[j]
            geneClu_promoter_gene_index = np.where(gene_names == "('"+j+"',)")[0]
        else:
            geneClu_promoter_atac_index = np.append(geneClu_promoter_atac_index,joint_gene_promoter_map[j])
            geneClu_promoter_gene_index = np.append(geneClu_promoter_gene_index,np.where(gene_names == "('"+j+"',)")[0])

#read time series
#the_psudotime = pd.read_csv('E:/data/qiliu/single-cell program/ATAC/result/whole figure/top 5000 knn 3 pvalue 1/20200427/pseudotime_update.txt', sep='\t', index_col=0)
#the_psudotime = pd.read_csv(save_path+'pseudotime.csv', header=0, index_col=0)
the_psudotime = pd.read_csv(save_path+'psudotime_palantir.csv', header=0, index_col=0)
the_psudotime_branch = pd.read_csv(save_path+'psudotime_branch_probs_palantir.csv', header=0, index_col=0)
#the_psudotime_tsne = pd.read_csv(save_path+'psudotime_palantir_tsne.csv', header=0, index_col=0)
the_psudotime_tsne = pd.read_csv(save_path+'umap.csv', header=0, index_col=0)


atac_TF_feature_cluSpec_set = {}
for i in (the_atacclu_2.index.values): # the specific clu control
    if i in atac_TF_feature_cluSpec_set.keys():
        atac_TF_feature_cluSpec_set[i] = atac_TF_feature_cluSpec_set[i]+1
    else:
        atac_TF_feature_cluSpec_set[i] = 1
feature_filter_rate = len(the_atacclu_2.index.values)/len(the_atacclu_2_unique) # the specific clu control
feature_filter_rate = feature_filter_rate*0.8
atac_TF_feature_cluSpec_result = None
for key,value in atac_TF_feature_cluSpec_set.items():
    if value < feature_filter_rate:
        continue
    key_index = np.where(atac_exp.index.values == key)[0]
    if atac_TF_feature_cluSpec_result is None:
        atac_TF_feature_cluSpec_result = key_index
    else:
        atac_TF_feature_cluSpec_result = np.append(atac_TF_feature_cluSpec_result,key_index)

# the specific clu control
'''
# for gene cluster 0
feature_filter_cluSpec_intersect = np.array([j for j in diff_atac_set.values[2,:] if j in atac_TF_feature_cluSpec_result])
temp_intersect = np.array([j for j in diff_atac_set.values[4,:] if j in atac_TF_feature_cluSpec_result])
feature_filter_cluSpec_intersect = np.append(feature_filter_cluSpec_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[5,:] if j in atac_TF_feature_cluSpec_result])
feature_filter_cluSpec_intersect = np.append(feature_filter_cluSpec_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[14,:] if j in atac_TF_feature_cluSpec_result])
feature_filter_cluSpec_intersect = np.append(feature_filter_cluSpec_intersect,temp_intersect)
'''

# for gene cluster 2
feature_filter_cluSpec_intersect = np.array([j for j in diff_atac_set.values[0,:] if j in atac_TF_feature_cluSpec_result])

temp_intersect = np.array([j for j in diff_atac_set.values[3,:] if j in atac_TF_feature_cluSpec_result])
feature_filter_cluSpec_intersect = np.append(feature_filter_cluSpec_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[9,:] if j in atac_TF_feature_cluSpec_result])
feature_filter_cluSpec_intersect = np.append(feature_filter_cluSpec_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[5,:] if j in atac_TF_feature_cluSpec_result])
feature_filter_cluSpec_intersect = np.append(feature_filter_cluSpec_intersect,temp_intersect)

'''
# for gene cluster 20
feature_filter_cluSpec_intersect = np.array([j for j in diff_atac_set.values[0,:] if j in atac_TF_feature_cluSpec_result])
#feature_filter_cluSpec_intersect = diff_atac_set.values[0,:] # the specific clu control
temp_intersect = np.array([j for j in diff_atac_set.values[2,:] if j in atac_TF_feature_cluSpec_result])
feature_filter_cluSpec_intersect = np.append(feature_filter_cluSpec_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[7,:] if j in atac_TF_feature_cluSpec_result])
feature_filter_cluSpec_intersect = np.append(feature_filter_cluSpec_intersect,temp_intersect)

temp_intersect = np.array([j for j in diff_atac_set.values[9,:] if j in atac_TF_feature_cluSpec_result])
feature_filter_cluSpec_intersect = np.append(feature_filter_cluSpec_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[3,:] if j in atac_TF_feature_cluSpec_result])
feature_filter_cluSpec_intersect = np.append(feature_filter_cluSpec_intersect,temp_intersect)
temp_intersect = np.array([j for j in diff_atac_set.values[5,:] if j in atac_TF_feature_cluSpec_result])
feature_filter_cluSpec_intersect = np.append(feature_filter_cluSpec_intersect,temp_intersect)
'''

the_atacclu_20_index = None
for i in np.unique(the_atac_20_clu):
    temp = np.where(atac_names == i)[0]
    if len(temp) == 0:
        continue
    if the_atacclu_20_index is None:
        the_atacclu_20_index = temp
    else:
        the_atacclu_20_index = np.append(the_atacclu_20_index,temp)

the_atacclu_22_index = None
for i in np.unique(the_atac_22_clu):
    temp = np.where(atac_names == i)[0]
    if len(temp) == 0:
        continue
    if the_atacclu_22_index is None:
        the_atacclu_22_index = temp
    else:
        the_atacclu_22_index = np.append(the_atacclu_22_index,temp)

feature_filter_cluGeneral20_intersect = np.array([j for j in the_atacclu_20_index if j in atac_TF_feature_cluSpec_result])
feature_filter_cluGeneral22_intersect = np.array([j for j in the_atacclu_22_index if j in atac_TF_feature_cluSpec_result])

gene_index_cluSpec = np.array(the_geneclu_2.values[:, 0]) # the specific clu control

# expression renormalizing:
cell_name_sort = None
for cell_name in gene_exp.columns.values:
    cell_name = cell_name[2:-3]
    temp = the_psudotime.loc[cell_name][0]
    if temp < 0.6:
        continue
    #temp = the_psudotime_branch.loc[cell_name][0]
    #if temp > 0.7:
    #    continue
    temp_index = np.where(the_psudotime.index.values == cell_name)[0]
    if cell_name_sort is None:
        cell_name_sort = temp_index
    else:
        cell_name_sort = np.append(cell_name_sort,temp_index)

gene_Exp_clu = gene_exp.values[gene_index_cluSpec,:]
gene_Exp_clu = gene_Exp_clu[:,cell_name_sort]
for i in range(gene_Exp_clu.shape[0]):
    gene_Exp_clu[i,:] = gene_Exp_clu[i,:]/np.max(gene_Exp_clu[i,:])
gene_Exp_clu = np.sum(gene_Exp_clu,axis=0)
gene_Exp_clu = gene_Exp_clu/np.max(gene_Exp_clu)
#gene_Exp_clu = (gene_Exp_clu - np.min(gene_Exp_clu))/(np.max(gene_Exp_clu) - np.min(gene_Exp_clu))

atac_PromoterExp_clu = atac_exp.values[geneClu_promoter_atac_index,:]
atac_PromoterExp_clu = atac_PromoterExp_clu[:,cell_name_sort]
atac_PromoterExp_clu = np.sum(atac_PromoterExp_clu,axis=0)
atac_PromoterExp_clu = atac_PromoterExp_clu/np.max(atac_PromoterExp_clu)
corr_promoter_gene = pd.Series(gene_Exp_clu).corr(pd.Series(atac_PromoterExp_clu),method='pearson')

atac_SepcExp_clu = atac_exp.values[feature_filter_cluSpec_intersect,:]
atac_SepcExp_clu = atac_SepcExp_clu[:,cell_name_sort]
atac_SepcExp_clu = np.sum(atac_SepcExp_clu,axis=0)
atac_SepcExp_clu = atac_SepcExp_clu/np.max(atac_SepcExp_clu)
corr_specficTF_gene = pd.Series(gene_Exp_clu).corr(pd.Series(atac_SepcExp_clu),method='pearson')


atac_generalExp20_clu = atac_exp.values[feature_filter_cluGeneral20_intersect,:]
atac_generalExp20_clu = atac_generalExp20_clu[:,cell_name_sort]
atac_generalExp20_clu = np.sum(atac_generalExp20_clu,axis=0)
atac_generalExp20_clu = atac_generalExp20_clu/np.max(atac_generalExp20_clu)
corr_general20_gene = pd.Series(gene_Exp_clu).corr(pd.Series(atac_generalExp20_clu),method='pearson')


atac_generalExp22_clu = atac_exp.values[feature_filter_cluGeneral22_intersect,:]
atac_generalExp22_clu = atac_generalExp22_clu[:,cell_name_sort]
atac_generalExp22_clu = np.sum(atac_generalExp22_clu,axis=0)
atac_generalExp22_clu = atac_generalExp22_clu/np.max(atac_generalExp22_clu)
corr_general22_gene = pd.Series(gene_Exp_clu).corr(pd.Series(atac_generalExp22_clu),method='pearson')




#pusdo_time = the_psudotime.values[cell_name_sort]
pusdo_time = np.array(the_psudotime['pseudotime'].tolist())
pusdo_time = pusdo_time[cell_name_sort]


fig1 = plt.figure(num=1,figsize=(10, 2))
plt.scatter(pusdo_time, gene_Exp_clu)
plt.show()

fig2 = plt.figure(num=2,figsize=(10, 2))
plt.scatter(pusdo_time, atac_PromoterExp_clu)
plt.show()

fig3 = plt.figure(num=3,figsize=(10, 2))
plt.scatter(pusdo_time, atac_SepcExp_clu)
plt.show()

fig4 = plt.figure(num=4,figsize=(10, 2))
plt.scatter(pusdo_time, atac_generalExp20_clu)
plt.show()

fig5 = plt.figure(num=5,figsize=(10, 2))
plt.scatter(pusdo_time, atac_generalExp22_clu)
plt.show()

cell_time_df = pd.DataFrame({'cell_time':np.round(pusdo_time*100),'gene_exp':gene_Exp_clu,'promoter_exp':atac_PromoterExp_clu,

                            'specific_TF':atac_SepcExp_clu,'general20_TF':atac_generalExp20_clu,'general22_TF':atac_generalExp22_clu})
psudo_label = np.array(cell_time_df['cell_time'].tolist())
corr_ave_psudotime = np.zeros([5,101])
for i in np.unique(psudo_label):
    temp_time = np.where(psudo_label == i)[0]

    temp = np.array(cell_time_df['gene_exp'].tolist())
    corr_ave_psudotime[0,int(i)] = np.average(temp[temp_time])

    temp = np.array(cell_time_df['promoter_exp'].tolist())
    corr_ave_psudotime[1, int(i)] = np.average(temp[temp_time])

    temp = np.array(cell_time_df['specific_TF'].tolist())
    corr_ave_psudotime[2, int(i)] = np.average(temp[temp_time])

    temp = np.array(cell_time_df['general20_TF'].tolist())
    corr_ave_psudotime[3, int(i)] = np.average(temp[temp_time])

    temp = np.array(cell_time_df['general22_TF'].tolist())
    corr_ave_psudotime[4, int(i)] = np.average(temp[temp_time])

psudodata = pd.DataFrame(corr_ave_psudotime.T)
print(pearsonr(corr_ave_psudotime[0,:],corr_ave_psudotime[1,:]))
print(pearsonr(corr_ave_psudotime[0,:],corr_ave_psudotime[2,:]))
print(pearsonr(corr_ave_psudotime[0,:],corr_ave_psudotime[3,:]))
print(pearsonr(corr_ave_psudotime[0,:],corr_ave_psudotime[4,:]))
corr_pearson = psudodata.corr('pearson')


plt.figure(1)
plt.subplot(511)
sns.boxplot(x = 'cell_time', y = 'gene_exp', data = cell_time_df)
plt.xticks([])
plt.subplot(512)
sns.boxplot(x = 'cell_time', y = 'promoter_exp', data = cell_time_df)
plt.xticks([])
plt.subplot(513)
sns.boxplot(x = 'cell_time', y = 'specific_TF', data = cell_time_df)
plt.xticks([])
plt.subplot(514)
sns.boxplot(x = 'cell_time', y = 'general20_TF', data = cell_time_df)
plt.xticks([])
plt.subplot(515)
sns.boxplot(x = 'cell_time', y = 'general22_TF', data = cell_time_df)
plt.xticks(rotation=60)
plt.show()

plt.figure(2)
x_data = np.array(the_psudotime_tsne['x'].tolist())
x_data = x_data[cell_name_sort]
y_data = np.array(the_psudotime_tsne['y'].tolist())
y_data = y_data[cell_name_sort]
plt.subplot(231)
c_data = np.array(cell_time_df['cell_time'].tolist())
c_data = (c_data - np.min(c_data))/(max(c_data)-min(c_data))
plt.scatter(x = x_data, y = y_data, c = c_data, s = 1)
plt.subplot(232)
c_data = np.array(cell_time_df['gene_exp'].tolist())
c_data = (c_data - np.min(c_data))/(max(c_data)-min(c_data))
plt.scatter(x = x_data, y = y_data, c = c_data, s = 1)
plt.subplot(233)
c_data = np.array(cell_time_df['promoter_exp'].tolist())
c_data = (c_data - np.min(c_data))/(max(c_data)-min(c_data))
plt.scatter(x = x_data, y = y_data, c = c_data, s = 1)
plt.subplot(234)
c_data = np.array(cell_time_df['specific_TF'].tolist())
c_data = (c_data - np.min(c_data))/(max(c_data)-min(c_data))
plt.scatter(x = x_data, y = y_data, c = c_data, s = 1)
plt.subplot(235)
c_data = np.array(cell_time_df['general20_TF'].tolist())
c_data = (c_data - np.min(c_data))/(max(c_data)-min(c_data))
plt.scatter(x = x_data, y = y_data, c = c_data, s = 1)
plt.subplot(236)
c_data = np.array(cell_time_df['general22_TF'].tolist())
c_data = (c_data - np.min(c_data))/(max(c_data)-min(c_data))
plt.scatter(x = x_data, y = y_data, c = c_data, s = 1)
plt.show()


cell_clu_table = pd.read_csv(save_path+'multivae_umap_imputation.csv', header=0, index_col=0)
cell_clu = np.array(cell_clu_table['atac_cluster'].tolist())
cell_clu = cell_clu[cell_name_sort]


cell_clu_df = pd.DataFrame({'cell_clu':cell_clu,'gene_exp':gene_Exp_clu,'promoter_exp':atac_PromoterExp_clu,
                            'specific_TF':atac_SepcExp_clu,'general20_TF':atac_generalExp20_clu,'general22_TF':atac_generalExp22_clu})
plt.figure(2)
plt.scatter(x = x_data, y = y_data, c = np.array(cell_clu_df['cell_clu'].tolist()), s = 1)
plt.show()
plt.figure(3)
x_data = np.array(the_psudotime_tsne['x'].tolist())
x_data = x_data[cell_name_sort]
y_data = np.array(the_psudotime_tsne['y'].tolist())
y_data = y_data[cell_name_sort]
plt.subplot(231)
c_data = np.array(cell_clu_df['cell_clu'].tolist())
c_data = (c_data - np.min(c_data))/(max(c_data)-min(c_data))
plt.scatter(x = x_data, y = y_data, c = c_data, s = 1)
plt.subplot(232)
c_data = np.array(cell_clu_df['gene_exp'].tolist())
c_data = (c_data - np.min(c_data))/(max(c_data)-min(c_data))
plt.scatter(x = x_data, y = y_data, c = c_data, s = 1)
plt.subplot(233)
c_data = np.array(cell_clu_df['promoter_exp'].tolist())
c_data = (c_data - np.min(c_data))/(max(c_data)-min(c_data))
plt.scatter(x = x_data, y = y_data, c = c_data, s = 1)
plt.subplot(234)
c_data = np.array(cell_clu_df['specific_TF'].tolist())
c_data = (c_data - np.min(c_data))/(max(c_data)-min(c_data))
plt.scatter(x = x_data, y = y_data, c = c_data, s = 1)
plt.subplot(235)
c_data = np.array(cell_clu_df['general20_TF'].tolist())
c_data = (c_data - np.min(c_data))/(max(c_data)-min(c_data))
plt.scatter(x = x_data, y = y_data, c = c_data, s = 1)
plt.subplot(236)
c_data = np.array(cell_clu_df['general22_TF'].tolist())
c_data = (c_data - np.min(c_data))/(max(c_data)-min(c_data))
plt.scatter(x = x_data, y = y_data, c = c_data, s = 1)
plt.show()

plt.figure(2)
plt.subplot(511)
sns.boxplot(x = 'cell_clu', y = 'gene_exp', data = cell_clu_df)
plt.subplot(512)
sns.boxplot(x = 'cell_clu', y = 'promoter_exp', data = cell_clu_df)
plt.subplot(513)
sns.boxplot(x = 'cell_clu', y = 'specific_TF', data = cell_clu_df)
plt.subplot(514)
sns.boxplot(x = 'cell_clu', y = 'general20_TF', data = cell_clu_df)
plt.subplot(515)
sns.boxplot(x = 'cell_clu', y = 'general22_TF', data = cell_clu_df)
plt.show()

#pusdo_time_discrete = np.round(pusdo_time*100)
#for i in range(len(pusdo_time_discrete)):
#    pusdo_time_discrete[i] = pusdo_time_discrete[i,0]

cell_clu_dict = {}
for i in np.unique(cell_clu):
    cell_clu_dict[i] = cell_name_sort[np.where(cell_clu == i)[0]]
    psudotime_temp = pusdo_time[cell_clu_dict[i]]
    psudotime_temp = (psudotime_temp - np.min(psudotime_temp))/(np.max(psudotime_temp) - np.min(psudotime_temp))
    psudotime_temp = np.round(psudotime_temp*30)
    cell_clu_psudotime_df = pd.DataFrame({'psudotime': psudotime_temp, 'gene_exp': gene_Exp_clu[cell_clu_dict[i]], 'promoter_exp': atac_PromoterExp_clu[cell_clu_dict[i]],
                                'specific_TF': atac_SepcExp_clu[cell_clu_dict[i]], 'general20_TF': atac_generalExp20_clu[cell_clu_dict[i]],
                                'general22_TF': atac_generalExp22_clu[cell_clu_dict[i]]})

    plt.figure(3)
    plt.subplot(511)
    sns.boxplot(x='psudotime', y='gene_exp', data=cell_clu_psudotime_df)
    plt.subplot(512)
    sns.boxplot(x='psudotime', y='promoter_exp', data=cell_clu_psudotime_df)
    plt.subplot(513)
    sns.boxplot(x='psudotime', y='specific_TF', data=cell_clu_psudotime_df)
    plt.subplot(514)
    sns.boxplot(x='psudotime', y='general20_TF', data=cell_clu_psudotime_df)
    plt.subplot(515)
    sns.boxplot(x='psudotime', y='general22_TF', data=cell_clu_psudotime_df)
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
