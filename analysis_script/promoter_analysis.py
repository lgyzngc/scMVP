import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


import scanpy as sc
import seaborn as sns
import anndata
import scipy.io as sp_io
import shutil
from scipy.sparse import csr_matrix, issparse

save_path: str = "E:/data/qiliu/single-cell program/ATAC/result/whole figure/top 5000 knn 3 pvalue 1/data/"
gene_exp = pd.read_csv(save_path+'gene_imputation.csv', header=0, index_col=0)
atac_exp = pd.read_csv(save_path+'atac_aggregation.csv', header=0, index_col=0)
#gene_exp = pd.read_csv(save_path+'gene_exp.csv', header=0, index_col=0)
#atac_exp = pd.read_csv(save_path+'atac_exp.csv', header=0, index_col=0)
promoter_location = pd.read_csv(save_path+'promoter/promoter_ucsc.txt', sep='\t', header=0, index_col=0)
diff_gene_set = pd.read_csv(save_path+'gene_diff_matrix_adj.csv', header=0, index_col=0)
diff_atac_set = pd.read_csv(save_path+'atac_diff_matrix.csv', header=0, index_col=0)
marker_gene_top50 = pd.read_csv(save_path+'marker_gene_top50.csv', header=0)
marker_snare_gene = pd.read_csv(save_path+'snare_clu_marker.csv', header=0)

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

marker_snare_gene_clu = {}
for i in range(len(marker_snare_gene['Cluster'])):
    if marker_snare_gene['Cluster'][i] in marker_snare_gene_clu:
        temp = marker_snare_gene_clu[marker_snare_gene['Cluster'][i]]
        marker_snare_gene_clu[marker_snare_gene['Cluster'][i]] = np.append(temp,marker_snare_gene['Gene'][i])
    else:
        marker_snare_gene_clu[marker_snare_gene['Cluster'][i]] = np.array(marker_snare_gene['Gene'][i])


gene_names = gene_exp.index.values
atac_names = atac_exp.index.values
unique_gene_index = np.unique(diff_gene_set.values.flatten())
unique_atac_index = np.unique(diff_atac_set.values.flatten())
diff_gene_names = gene_names[unique_gene_index]
diff_atac_names = atac_names[unique_atac_index]

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
# intersecting gene cluster 0,2,19 with promoter
for i in range(len(the_geneclu_0.index.values)):
    the_geneclu_0.index.values[i] = the_geneclu_0.index.values[i][2:-3]
for i in range(len(the_geneclu_2.index.values)):
    the_geneclu_2.index.values[i] = the_geneclu_2.index.values[i][2:-3]
for i in range(len(the_geneclu_19.index.values)):
    the_geneclu_19.index.values[i] = the_geneclu_19.index.values[i][2:-3]
geneClu_promoter_atac_gene_map = {}
geneClu_promoter_atac_gene_map[0] = [i for i in the_geneclu_0.index.values if i in joint_gene_promoter_map.keys()]
geneClu_promoter_atac_gene_map[2] = [i for i in the_geneclu_2.index.values if i in joint_gene_promoter_map.keys()]
geneClu_promoter_atac_gene_map[19] = [i for i in the_geneclu_19.index.values if i in joint_gene_promoter_map.keys()]
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
# intersecting diff atac and gene-promoter 221
diff_promoter_atac_gene_map = {}
diff_promoter_clu_map = {}

for i in range(diff_atac_set.values.shape[0]):
    promoter_index = np.array(list(joint_promoter_gene_map.keys()))
    temp_atac = np.array([j for j in diff_atac_set.values[i,:] if j in promoter_index])
    if len(temp_atac) > 1:
        for j in temp_atac:
            diff_promoter_atac_gene_map[j] = joint_promoter_gene_map[j]
            diff_promoter_clu_map[j] = i

df = pd.DataFrame(data=list(diff_promoter_atac_gene_map.values()), index= list(diff_promoter_clu_map.values()))
df.to_csv(os.path.join(save_path,"atac_promoter_genes.csv"))

# intersecting diff gene, atac and promoter 299
joint_diff_gene_promoter_map = {}
joint_diff_gene_index_promoter_map = {}
for i in range(len(diff_gene_names)):
    temp_gene = diff_gene_names[i]
    temp_gene = temp_gene[2:-3]
    if temp_gene in promoter_gene_dic:
        temp_promoters_loc = promoter_gene_dic[temp_gene]
        temp_promoter_chr = promoter_chr_dic[temp_gene]
        temp_atac_index = np.where(atac_chr_set == temp_promoter_chr[0])
        temp_atac_index = temp_atac_index[0]
        flag = False
        for j in range(len(temp_atac_index)):
            atac_loc = atac_loc_set[temp_atac_index[j],:]
            for temp_promoter_loc in temp_promoters_loc:
                if temp_promoter_loc >= (atac_loc[0] - 3000) and temp_promoter_loc <= (atac_loc[1] + 3000):
                    joint_diff_gene_promoter_map[temp_gene] = temp_atac_index[j]
                    joint_diff_gene_index_promoter_map[(np.where(gene_names == diff_gene_names[i])[0])[0]] = temp_atac_index[j]
                    flag = True
                    break
            if flag:
                break
    else:
        print(temp_gene+'\n')


diff_gene_matrix = diff_gene_set.values
gene_promoter_matrix = {}
gene_index_matrix = {}
for i in range(diff_gene_matrix.shape[0]):
    diff_genes = gene_names[diff_gene_matrix[i,:]]
    for diff_gene, j in zip(diff_genes, range(len(diff_genes))):
        diff_gene = diff_gene[2:-3]
        if not (diff_gene in joint_diff_gene_promoter_map):
            continue
        if i in gene_promoter_matrix:
            gene_promoter_matrix[i] = np.append(gene_promoter_matrix[i],joint_diff_gene_promoter_map[diff_gene])
            gene_index_matrix[i] = np.append(gene_index_matrix[i],diff_gene_matrix[i,j])
        else:
            gene_promoter_matrix[i] = joint_diff_gene_promoter_map[diff_gene]
            gene_index_matrix[i] = diff_gene_matrix[i,j]

# cluster specific diff gene , diff atac and promoter mapping
intersect_promoter_diffAtac = {}
intersect_gene_diffAtac = {}
diff_atac_matrix = diff_atac_set.values
for i in range(len(diff_atac_set.index)):
    #intersect_promoter_diffAtac[i] = [val for val in gene_promoter_matrix[i] if val in diff_atac_matrix[i,:]]
    intersect_promoter_diffAtac[i] = [val for val in gene_promoter_matrix[i] if val in diff_atac_matrix]
    temp = None
    for j in intersect_promoter_diffAtac[i]:
        if temp is None:
            temp = np.array(gene_index_matrix[i][gene_promoter_matrix[i] == j])
        else:
            temp = np.append(temp, gene_index_matrix[i][gene_promoter_matrix[i] == j])
    intersect_gene_diffAtac[i] = temp

unique_atac_gene_diff_map = None
unique_gene_index_diff_map = None
for i in intersect_promoter_diffAtac:
    if unique_atac_gene_diff_map is None:
        unique_atac_gene_diff_map = intersect_promoter_diffAtac[i]
        unique_gene_index_diff_map = intersect_gene_diffAtac[i]
    else:
        unique_atac_gene_diff_map = np.append(unique_atac_gene_diff_map,intersect_promoter_diffAtac[i])
        unique_gene_index_diff_map = np.append(unique_gene_index_diff_map,intersect_gene_diffAtac[i])

# intersecting diff gene, diff atac and promoter 36
unique_atac_gene_diff = np.unique(unique_atac_gene_diff_map)
unique_gene_index_diff = None
for atac_index in unique_atac_gene_diff:
    temp = np.unique(unique_gene_index_diff_map[unique_atac_gene_diff_map == atac_index])
    if unique_gene_index_diff is None:
        unique_gene_index_diff = temp
    else:
        unique_gene_index_diff = np.append(unique_gene_index_diff,temp)

#unique_gene_index_diff = np.unique(unique_gene_index_diff_map)

#cell_clu = pd.read_csv(save_path+'multivae_umap_louvain.csv', header=0, index_col=0)
cell_clu = pd.read_csv(save_path+'multivae_umap_imputation.csv', header=0, index_col=0)
cell_clu_index = cell_clu['atac_cluster'].values
plot_exp_data = None
plot_atac_data = None
plot_exp_rate = None
for cell_clu in np.unique(cell_clu_index):
    temp_exp = gene_exp.values[:,cell_clu_index == cell_clu]
    temp_exp = temp_exp[unique_gene_index_diff,:]
    #temp_exp = temp_exp[geneClu_promoter_gene_index, :]
    #temp_exp = np.log1p(temp_exp)
    temp_atac = atac_exp.values[:, cell_clu_index == cell_clu]
    temp_atac = temp_atac[unique_atac_gene_diff,:]
    #temp_atac = temp_atac[geneClu_promoter_atac_index, :]


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
    plot_exp_data[:, i] = (plot_exp_data[:, i]-np.min(plot_exp_data[:, i])) / (np.max(plot_exp_data[:, i]) - np.min(plot_exp_data[:, i]))

for i in range(plot_atac_data.shape[1]):
#    plot_atac_data[:,i] = plot_atac_data[:,i]/np.max(plot_atac_data[:,i])
    plot_atac_data[:, i] = (plot_atac_data[:, i]-np.min(plot_atac_data[:, i])) / (np.max(plot_atac_data[:, i]) - np.min(plot_atac_data[:, i]))

#ax = sns.heatmap(plot_exp_data.T,yticklabels=gene_names[unique_gene_index_diff],cmap='rainbow')
ax = sns.heatmap(plot_exp_data.T,cmap='rainbow')
plt.show()
#ax = sns.clustermap(plot_exp_data.T,yticklabels=gene_names[unique_gene_index_diff], metric="correlation",cmap='rainbow')
ax = sns.clustermap(plot_exp_data.T,metric="correlation",cmap='rainbow')

plt.show()
#ax = sns.heatmap(plot_atac_data.T,yticklabels=gene_names[unique_gene_index_diff],cmap='rainbow')
ax = sns.heatmap(plot_atac_data.T,cmap='rainbow')
plt.show()
#ax = sns.clustermap(plot_atac_data.T,yticklabels=gene_names[unique_gene_index_diff], metric="correlation",cmap='rainbow')
ax = sns.clustermap(plot_atac_data.T, cmap='rainbow')
plt.show()

df = pd.DataFrame(data=plot_exp_data.T, index= gene_names[unique_gene_index_diff])
df.to_csv(os.path.join(save_path,"gene_clu_exp_avg.csv"))
df = pd.DataFrame(data=plot_atac_data.T, index= atac_names[unique_atac_gene_diff])
df.to_csv(os.path.join(save_path,"atac_clu_exp_avg.csv"))

gene_order_index = np.array([11,17,1,15,2,6,8,5,13,14,3,9,10,0,16,12,4,7])
column_1 = None
column_2 = None
column_3 = None
column_4 = None
column_5 = None
for i in range(plot_exp_data.shape[1]):
    for j in range(plot_exp_data.shape[0]):
        if column_1 is None:
            column_1 = gene_names[unique_gene_index_diff[i]]
            column_2 = gene_order_index[j]
            column_3 = plot_exp_data[j,i]
            column_4 = plot_atac_data[j,i]
            column_5 = plot_exp_rate[j,i]
        else:
            column_1 = np.append(column_1,gene_names[unique_gene_index_diff[i]])
            column_2 = np.append(column_2,gene_order_index[j])
            column_3 = np.append(column_3,plot_exp_data[j, i])
            column_4 = np.append(column_4,plot_atac_data[j, i])
            column_5 = np.append(column_5,plot_exp_rate[j,i])

cm = plt.cm.get_cmap('Blues')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xticklabels(labels=gene_names[unique_gene_index_diff],rotation=45)
plt.scatter(column_1,column_2,c=column_3,s=column_4*200,cmap=cm,alpha=0.6)
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xticklabels(labels=gene_names[unique_gene_index_diff],rotation=45)
plt.scatter(column_1,column_2,c=column_3,s=column_5*500,cmap=cm,alpha=0.6)
plt.show()

plot_exp_data = None
plot_atac_data = None
plot_exp_rate = None
plot_atac_diff_data = None
for cell_clu in np.unique(cell_clu_index):
    temp_exp = gene_exp.values[:,cell_clu_index == cell_clu]
    temp_exp = temp_exp[list(joint_diff_gene_index_promoter_map.keys()),:]
    #temp_exp = np.log1p(temp_exp)
    temp_atac = atac_exp.values[:, cell_clu_index == cell_clu]
    temp_atac = temp_atac[list(joint_diff_gene_index_promoter_map.values()),:]

    temp_atac_diff = atac_exp.values[:, cell_clu_index == cell_clu]
    temp_atac_diff = temp_atac_diff[diff_atac_set.values.flatten(),:]

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
    temp_atac_diff = np.sum(temp_atac_diff,axis=1)/len(np.where(cell_clu_index == cell_clu)[0])
    if plot_exp_data is None:
        plot_exp_data = temp_exp
        plot_atac_data = temp_atac
        plot_exp_rate = temp_exp_rate
        plot_atac_diff_data = temp_atac_diff
    else:
        plot_exp_data = np.vstack((plot_exp_data,temp_exp))
        plot_atac_data = np.vstack((plot_atac_data,temp_atac))
        plot_exp_rate = np.vstack((plot_exp_rate,temp_exp_rate))
        plot_atac_diff_data = np.vstack((plot_atac_diff_data,temp_atac_diff))
for i in range(plot_exp_data.shape[1]):
    #plot_exp_data[:,i] = plot_exp_data[:,i]/np.max(plot_exp_data[:,i])
    plot_exp_data[:, i] = (plot_exp_data[:, i]-np.min(plot_exp_data[:, i])) / (np.max(plot_exp_data[:, i]) - np.min(plot_exp_data[:, i]))

for i in range(plot_atac_data.shape[1]):
#    plot_atac_data[:,i] = plot_atac_data[:,i]/np.max(plot_atac_data[:,i])
    plot_atac_data[:, i] = (plot_atac_data[:, i]-np.min(plot_atac_data[:, i])) / (np.max(plot_atac_data[:, i]) - np.min(plot_atac_data[:, i]))
for i in range(plot_atac_diff_data.shape[1]):
    plot_atac_diff_data[:, i] = (plot_atac_diff_data[:, i]-np.min(plot_atac_diff_data[:, i])) / (np.max(plot_atac_diff_data[:, i]) - np.min(plot_atac_diff_data[:, i]))
cell_type_annotation = np.array(['CTX PyrL4/L5','CB Int Golgi/Stellate/Basket','Astro Prdm16','HIPP Pyr Precursor','CTX PyrL5/L6 Npr3',
                                 'Astro Gfap','OEC','CTX PyrL5 Fezf2','CB Int Progenitor','Migrating Int Trdn','CTX PyrL4 Rorb',
                                 'CTX PyrL2/L3/L4 Mef2c','CTX PyrL5/L6 Sulf1','Astro Slc7a10','THAL Glut','CB Granule Precursor',
                                 'CTX PyrL6','HIPP Granule Mki67'])
gene_order_index = np.array([11,17,1,15,2,6,8,5,13,14,3,9,10,0,16,12,4,7])
ax = sns.clustermap(plot_exp_data.T,xticklabels=cell_type_annotation,yticklabels=False,cmap='RdYlBu')
plt.xticks(rotation=90)
plt.show()
ax = sns.clustermap(1 - plot_exp_data.T, metric="correlation",xticklabels=cell_type_annotation,yticklabels=False,cmap='RdYlBu')
plt.xticks(rotation=90)
plt.savefig(save_path+'diff_gene_ordered.png')
plt.savefig(save_path+'diff_gene_ordered.pdf')
plt.show()
rearranged_row = ax.dendrogram_row.reordered_ind
rearranged_col = ax.dendrogram_col.reordered_ind
pdf = PdfPages(save_path+'diff_gene_ordered1.pdf')
ax = sns.clustermap(1 - plot_exp_data.T, metric="correlation",xticklabels=cell_type_annotation,yticklabels=False,cmap='RdYlBu')
plt.xticks(rotation=90)
pdf.savefig()                            #将图片保存在pdf文件中
pdf.close()

ax = sns.clustermap(1 - plot_atac_data.T,yticklabels=False,cmap='RdYlBu')
plt.show()
ax = sns.clustermap(1 - plot_atac_data.T, metric="correlation",yticklabels=False,cmap='RdYlBu')
plt.show()
ax = sns.heatmap(1 - plot_atac_data[gene_order_index,:].T, xticklabels=cell_type_annotation[gene_order_index],yticklabels=False,cmap='RdYlBu')
plt.show()
ax = sns.clustermap(np.vstack((plot_exp_data.T, plot_atac_data.T)), metric="correlation",yticklabels=False,cmap='rainbow')
plt.show()
ax = sns.clustermap(np.vstack((plot_exp_data.T, plot_atac_data.T)),yticklabels=False,cmap='rainbow')
plt.show()
temp_atac_exp_plot = None
for i in range(plot_atac_diff_data.shape[0]):
    if temp_atac_exp_plot is None:
        temp_atac_exp_plot = plot_atac_diff_data[:,gene_order_index[i]*1000:(gene_order_index[i]+1)*1000].T
    else:
        temp_atac_exp_plot = np.vstack((temp_atac_exp_plot,plot_atac_diff_data[:,gene_order_index[i]*1000:(gene_order_index[i]+1)*1000].T))
ax = sns.heatmap( 1- plot_atac_diff_data.T,xticklabels=cell_type_annotation,yticklabels=False,cmap='RdYlBu')
plt.xticks(rotation=90)
plt.show()
ax = sns.heatmap( 1 - temp_atac_exp_plot[:,gene_order_index],xticklabels=cell_type_annotation[gene_order_index],yticklabels=False,cmap='RdYlBu')
plt.xticks(rotation=90)
hist_fig = ax.get_figure()
hist_fig.savefig(save_path+'diff_atac_ordered1.png')
hist_fig.savefig(save_path+'diff_atac_ordered1.pdf')
plt.savefig(save_path+'diff_atac_ordered.png')
plt.savefig(save_path+'diff_atac_ordered.pdf')
plt.show()
ax = sns.clustermap( 1 - temp_atac_exp_plot[:,gene_order_index],xticklabels=cell_type_annotation[gene_order_index],yticklabels=False,cmap='RdYlBu')
plt.xticks(rotation=90)
plt.show()

df = pd.DataFrame(data=plot_exp_data.T, index= gene_names[list(joint_diff_gene_index_promoter_map.keys())])
df.to_csv(os.path.join(save_path,"diffgene_clu_exp_avg.csv"))
df = pd.DataFrame(data=plot_atac_diff_data.T, index= atac_names[diff_atac_set.values.flatten()])
df.to_csv(os.path.join(save_path,"diffatac_clu_exp_avg.csv"))
df = pd.DataFrame(data=temp_atac_exp_plot[:,gene_order_index],columns=gene_order_index)
df.to_csv(os.path.join(save_path,"diffatac_clu_exp_avg_ordered.csv"))
plot_exp_data = plot_exp_data.T[rearranged_row,:]
plot_exp_data = plot_exp_data[:,rearranged_col]
gene_index_output = np.array(list(joint_diff_gene_index_promoter_map.keys()))
df = pd.DataFrame(data=plot_exp_data,index= gene_names[gene_index_output[rearranged_row]],columns=rearranged_col)
df.to_csv(os.path.join(save_path,"diffgene_clu_exp_avg_ordered.csv"))


marker_gene_anno = pd.read_csv(save_path+'brain_marker_snare.txt', header=0, sep='\t')
marker_genes = marker_gene_anno['Cell Marker'].values
marker_celltypes = marker_gene_anno['Cell Type'].values
for i in range(len(gene_names)):
    gene_names[i] = gene_names[i][2:-3]
gene_names = gene_names
marker_gene_index = None
for i in range(len(marker_genes)):
    temp_index = np.where(gene_names == marker_genes[i])
    if not temp_index[0].size:
        continue
    if not marker_gene_index is None:
        marker_gene_index = np.append(marker_gene_index,temp_index[0])
    else:
        marker_gene_index = temp_index[0]

joint_marker_gene_promoter_map = {}
for i in range(len(marker_gene_index)):
    temp_gene = gene_names[marker_gene_index[i]]
    if temp_gene in promoter_gene_dic:
        temp_promoters_loc = promoter_gene_dic[temp_gene]
        temp_promoter_chr = promoter_chr_dic[temp_gene]
        temp_atac_index = np.where(atac_chr_set == temp_promoter_chr[0])[0]
        flag = False
        for j in range(len(temp_atac_index)):
            atac_loc = atac_loc_set[temp_atac_index[j],:]
            for temp_promoter_loc in temp_promoters_loc:
                if temp_promoter_loc >= atac_loc[0] and temp_promoter_loc <= atac_loc[1]:
                    joint_marker_gene_promoter_map[marker_gene_index[i]] = temp_atac_index[j]
                    flag = True
                    break
            if flag:
                break
    else:
        print(temp_gene+'\n')

plot_exp_data = None
plot_atac_data = None
plot_exp_rate = None
for cell_clu in np.unique(cell_clu_index):
    temp_exp = gene_exp.values[:,cell_clu_index == cell_clu]
    temp_exp = temp_exp[list(joint_marker_gene_promoter_map.keys()),:]
    #temp_exp = np.log1p(temp_exp)
    temp_atac = atac_exp.values[:, cell_clu_index == cell_clu]
    temp_atac = temp_atac[list(joint_marker_gene_promoter_map.values()),:]

    temp_exp_ave = np.ones(temp_exp.shape[0])
    temp_exp_rate = np.ones(temp_exp.shape[0])
    for i in range(temp_exp.shape[0]):
        temp = temp_exp[i, :]
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
    plot_exp_data[:, i] = (plot_exp_data[:, i]-np.min(plot_exp_data[:, i])) / (np.max(plot_exp_data[:, i]) - np.min(plot_exp_data[:, i]))
for i in range(plot_atac_data.shape[1]):
#    plot_atac_data[:,i] = plot_atac_data[:,i]/np.max(plot_atac_data[:,i])
    plot_atac_data[:, i] = (plot_atac_data[:, i]-np.min(plot_atac_data[:, i])) / (np.max(plot_atac_data[:, i]) - np.min(plot_atac_data[:, i]))

ax = sns.heatmap(plot_exp_data.T,yticklabels=gene_names[list(joint_marker_gene_promoter_map.keys())],cmap='rainbow')
plt.show()
ax = sns.heatmap(plot_atac_data.T,yticklabels=gene_names[list(joint_marker_gene_promoter_map.keys())],cmap='rainbow')
plt.show()

df = pd.DataFrame(data=plot_exp_data.T, index= gene_names[list(joint_marker_gene_promoter_map.keys())])
df.to_csv(os.path.join(save_path,"marker_gene_clu_exp_avg.csv"))
df = pd.DataFrame(data=plot_atac_data.T, index= gene_names[list(joint_marker_gene_promoter_map.keys())])
df.to_csv(os.path.join(save_path,"marker_atac_clu_exp_avg.csv"))

column_1 = None
column_2 = None
column_3 = None
column_4 = None
column_5 = None
for i in range(plot_exp_data.shape[1]):
    for j in range(plot_exp_data.shape[0]):
        if column_1 is None:
            column_1 = gene_names[list(joint_marker_gene_promoter_map.keys())[i]]
            column_2 = gene_order_index[j] # rearranging the trajectory order
            column_3 = plot_exp_data[j,i]
            column_4 = plot_atac_data[j,i]
            column_5 = plot_exp_rate[j,i]
        else:
            column_1 = np.append(column_1,gene_names[list(joint_marker_gene_promoter_map.keys())[i]])
            column_2 = np.append(column_2,gene_order_index[j]) # rearranging the trajectory order
            column_3 = np.append(column_3,plot_exp_data[j, i])
            column_4 = np.append(column_4,plot_atac_data[j, i])
            column_5 = np.append(column_5, plot_exp_rate[j, i])
cm = plt.cm.get_cmap('Blues')
plt.scatter(column_1,column_2,c=column_3,s=column_4*200,cmap=cm,alpha=0.6)
plt.show()
plt.scatter(column_1,column_2,c=column_3,s=column_5*500,cmap=cm,alpha=0.6)
plt.show()

cell_clu_anno = marker_gene_top50.columns.values.tolist()
marker_gene_set  = marker_gene_top50.values
marker_gene_name_map = {}
for i in range(len(cell_clu_anno)):
    temp_marker_gene_set = marker_gene_set[:,i]
    temp_marker_index = None
    for temp_marker in temp_marker_gene_set:
        temp_index = np.where(gene_names == temp_marker)[0]
        if not temp_index.size:
            continue
        if temp_marker_index is None:
            temp_marker_index = temp_index
        else:
            temp_marker_index = np.append(temp_marker_index,temp_index)
    marker_gene_name_map[cell_clu_anno[i]] = temp_marker_index

diff_gene_clu_anno = {}
diff_gene_clu_anno_index = {}
cell_clu_marker_map = {}
cell_clu_marker_matrix = np.zeros((diff_gene_matrix.shape[0],len(list(marker_gene_name_map.keys())))).astype(np.int32)
cell_clu_marker_map_anno = {}
for i in range(diff_gene_matrix.shape[0]):
    for j in list(marker_gene_name_map.keys()):
        temp_index = np.array([var for var in marker_gene_name_map[j] if var in diff_gene_matrix[i,:]])
        if not temp_index.size:
            continue
        temp_j_index = np.where(np.array(list(marker_gene_name_map.keys())) == j)[0]
        cell_clu_marker_matrix[i,temp_j_index] = len(temp_index)
        '''
        if i in cell_clu_marker_map.keys():
            if len(cell_clu_marker_map[i]) < len(temp_index):
                cell_clu_marker_map[i] = temp_index
                cell_clu_marker_map_anno[i] = j
        else:
            cell_clu_marker_map[i] = temp_index
            cell_clu_marker_map_anno[i] = j
        '''

        if j in diff_gene_clu_anno.keys():
            if len(diff_gene_clu_anno[j]) < len(temp_index):
                diff_gene_clu_anno[j] = temp_index
                diff_gene_clu_anno_index[j] = i
        else:
            diff_gene_clu_anno[j] = temp_index
            diff_gene_clu_anno_index[j] = i
df = pd.DataFrame(data=cell_clu_marker_matrix.T, index= list(marker_gene_name_map.keys()))
df.to_csv(os.path.join(save_path,"cell_clu_anno_matrix.csv"))

cell_types = pd.read_csv(save_path+'cell_clu_anno.csv', header=0, index_col=0)
extra_marker_genes_dict = {}
for i in cell_types.index.values:
    temp_index = np.array([j for j in diff_gene_matrix[i,:] if j in marker_gene_name_map[list(cell_types.loc[i])[0]]])
    extra_marker_genes_dict[list(cell_types.loc[i])[0]] = temp_index
cell_anno_markers_output = None
cell_anno_markers_output_index = np.array([])
for (key, vaule) in extra_marker_genes_dict.items():
    clu_anno = key
    markers = gene_names[vaule]
    cell_anno_markers_output_index = np.append(cell_anno_markers_output_index,vaule)
    for j in markers:
        if cell_anno_markers_output is None:
            cell_anno_markers_output = np.array([clu_anno,j])
        else:
            cell_anno_markers_output = np.vstack((cell_anno_markers_output,[clu_anno,j]))
df = pd.DataFrame(data=cell_anno_markers_output[:,1], index= cell_anno_markers_output[:,0])
df.to_csv(os.path.join(save_path,"cell_type_anno_markers.csv"))
cell_anno_markers_output_index = cell_anno_markers_output_index.astype(np.int)

plot_exp_data = None
plot_exp_rate = None
for cell_clu in np.unique(cell_clu_index):
    temp_exp = gene_exp.values[:,cell_clu_index == cell_clu]
    temp_exp = temp_exp[cell_anno_markers_output_index,:]
    #temp_exp = np.log1p(temp_exp)

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
    if plot_exp_data is None:
        plot_exp_data = temp_exp
        plot_exp_rate = temp_exp_rate
    else:
        plot_exp_data = np.vstack((plot_exp_data,temp_exp))
        plot_exp_rate = np.vstack((plot_exp_rate,temp_exp_rate))
for i in range(plot_exp_data.shape[1]):
    #plot_exp_data[:,i] = plot_exp_data[:,i]/np.max(plot_exp_data[:,i])
    plot_exp_data[:, i] = (plot_exp_data[:, i]-np.min(plot_exp_data[:, i])) / (np.max(plot_exp_data[:, i]) - np.min(plot_exp_data[:, i]))
ax = sns.heatmap( 1- plot_exp_data[gene_order_index,:].T,xticklabels=cell_type_annotation[gene_order_index],yticklabels=False,cmap='RdYlBu')
plt.xticks(rotation=90)
plt.show()
ax = sns.clustermap( 1- plot_exp_data[gene_order_index,:].T,row_cluster=True,col_cluster=False, xticklabels=cell_type_annotation[gene_order_index],
                     metric="correlation",yticklabels=gene_names[cell_anno_markers_output_index.astype(np.int)],cmap='RdYlBu')
plt.xticks(rotation=90)
plt.show()
ax = sns.clustermap( 1- plot_exp_data[gene_order_index,:].T,xticklabels=cell_type_annotation[gene_order_index],
                     metric="correlation",yticklabels=gene_names[cell_anno_markers_output_index.astype(np.int)],cmap='RdYlBu')
plt.xticks(rotation=90)
plt.show()
ax = sns.heatmap( 1- plot_exp_rate.T,xticklabels=cell_type_annotation,yticklabels=False,cmap='RdYlBu')
plt.xticks(rotation=90)
plt.show()

# for the joint of new marker genes from science paper and promoters
joint_marker_gene_promoter_map = {}
for i in range(len(cell_anno_markers_output_index)):
    temp_gene = gene_names[cell_anno_markers_output_index[i]]
    if temp_gene in promoter_gene_dic:
        temp_promoters_loc = promoter_gene_dic[temp_gene]
        temp_promoter_chr = promoter_chr_dic[temp_gene]
        temp_atac_index = np.where(atac_chr_set == temp_promoter_chr[0])[0]
        flag = False
        for j in range(len(temp_atac_index)):
            atac_loc = atac_loc_set[temp_atac_index[j],:]
            for temp_promoter_loc in temp_promoters_loc:
                if temp_promoter_loc >= atac_loc[0] and temp_promoter_loc <= atac_loc[1]:
                    joint_marker_gene_promoter_map[cell_anno_markers_output_index[i]] = temp_atac_index[j]
                    flag = True
                    break
            if flag:
                break
    else:
        print(temp_gene+'\n')
# for the joint expression of new marker genes from science paper and promoters
plot_exp_data = None
plot_atac_data = None
plot_exp_rate = None
for cell_clu in np.unique(cell_clu_index):
    temp_exp = gene_exp.values[:,cell_clu_index == cell_clu]
    temp_exp = temp_exp[list(joint_marker_gene_promoter_map.keys()),:]
    #temp_exp = np.log1p(temp_exp)
    temp_atac = atac_exp.values[:, cell_clu_index == cell_clu]
    temp_atac = temp_atac[list(joint_marker_gene_promoter_map.values()),:]

    temp_exp_ave = np.ones(temp_exp.shape[0])
    temp_exp_rate = np.ones(temp_exp.shape[0])
    for i in range(temp_exp.shape[0]):
        temp = temp_exp[i, :]
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
    plot_exp_data[:, i] = (plot_exp_data[:, i]-np.min(plot_exp_data[:, i])) / (np.max(plot_exp_data[:, i]) - np.min(plot_exp_data[:, i]))
for i in range(plot_atac_data.shape[1]):
#    plot_atac_data[:,i] = plot_atac_data[:,i]/np.max(plot_atac_data[:,i])
    plot_atac_data[:, i] = (plot_atac_data[:, i]-np.min(plot_atac_data[:, i])) / (np.max(plot_atac_data[:, i]) - np.min(plot_atac_data[:, i]))

ax = sns.clustermap(1-plot_exp_data.T,yticklabels=gene_names[list(joint_marker_gene_promoter_map.keys())],cmap='RdYlBu')
plt.show()
rearranged_row = ax.dendrogram_row.reordered_ind
rearranged_col = ax.dendrogram_col.reordered_ind
temp_plot_atac_data = plot_atac_data.T[rearranged_row,:]
temp_plot_atac_data = temp_plot_atac_data[:,rearranged_col]
rearranged_index = np.array(list(joint_marker_gene_promoter_map.keys()))
rearranged_index = rearranged_index[rearranged_row]
ax = sns.heatmap(1 - temp_plot_atac_data,xticklabels= rearranged_col,yticklabels=gene_names[rearranged_index],cmap='RdYlBu')
plt.show()
ax = sns.clustermap(1-plot_atac_data.T,yticklabels=gene_names[list(joint_marker_gene_promoter_map.keys())],cmap='RdYlBu')
plt.show()
yticklabels = gene_names[list(joint_marker_gene_promoter_map.keys())]
yticklabels = np.append(yticklabels,yticklabels)
ax = sns.clustermap(1 - np.vstack((plot_exp_data.T, plot_atac_data.T)), metric="correlation",yticklabels=yticklabels,cmap='RdYlBu')
plt.show()
ax = sns.clustermap(1 - np.vstack((plot_exp_data.T, plot_atac_data.T)), yticklabels=yticklabels,cmap='RdYlBu')
plt.show()

column_1 = None
column_2 = None
column_3 = None
column_4 = None
column_5 = None
for i in rearranged_row:
    for j in gene_order_index: # rearranged_col
        if column_1 is None:
            column_1 = gene_names[list(joint_marker_gene_promoter_map.keys())[i]]
            column_2 = gene_order_index[j] # rearranging the trajectory order
            column_3 = plot_exp_data[j,i]
            column_4 = plot_atac_data[j,i]
            column_5 = plot_exp_rate[j,i]
        else:
            column_1 = np.append(column_1,gene_names[list(joint_marker_gene_promoter_map.keys())[i]])
            column_2 = np.append(column_2,gene_order_index[j]) # rearranging the trajectory order
            column_3 = np.append(column_3,plot_exp_data[j, i])
            column_4 = np.append(column_4,plot_atac_data[j, i])
            column_5 = np.append(column_5, plot_exp_rate[j, i])
column_union = np.vstack((column_2,column_1))
column_union = np.vstack((column_union,column_3))
column_union = np.vstack((column_union,column_4))
column_union = column_union.T

cm = plt.cm.get_cmap('Blues')
plt.scatter(column_1,cell_type_annotation[column_2],c=column_3,s=column_4*200,cmap=cm,alpha=0.6)
plt.xticks(rotation=90)
plt.show()
plt.scatter(column_1,cell_type_annotation[column_2],c=column_3,s=column_5*200,cmap=cm,alpha=0.6)
plt.xticks(rotation=90)
plt.show()

cm = plt.cm.get_cmap('YlOrRd')
plt.scatter(column_1,cell_type_annotation[column_2],c=column_3,s=column_4*200,cmap=cm,alpha=0.6)
plt.xticks(rotation=90)
plt.show()
plt.scatter(column_1,cell_type_annotation[column_2],c=column_3,s=column_5*200,cmap=cm,alpha=0.6)
plt.xticks(rotation=90)
plt.show()

df = pd.DataFrame(data=column_union,columns=['cell clu','marker gene','gene exp avg','atac exp avg'])
df.to_csv(os.path.join(save_path,"diff_markergene_exp_avg.csv"))



cell_clu_marker_x_flag = np.array([i for i in range(cell_clu_marker_matrix.shape[0])])
cell_clu_marker_y_flag = np.array(list(marker_gene_name_map.keys()))
temp_cell_clu_marker_maxtrix = cell_clu_marker_matrix
for i in range(cell_clu_marker_matrix.shape[0]):
    if np.max(temp_cell_clu_marker_maxtrix) == 0:
        break
    temp_max = np.argmax(temp_cell_clu_marker_maxtrix)
    temp_max_x = temp_max//temp_cell_clu_marker_maxtrix.shape[1]
    temp_max_y = temp_max%temp_cell_clu_marker_maxtrix.shape[1]
    cell_clu_marker_map_anno[cell_clu_marker_x_flag[temp_max_x]] = cell_clu_marker_y_flag[temp_max_y]
    cell_clu_marker_map[cell_clu_marker_x_flag[temp_max_x]] = np.array([var for var in marker_gene_name_map[cell_clu_marker_y_flag[temp_max_y]] if var in diff_gene_matrix[cell_clu_marker_x_flag[temp_max_x], :]])
    temp_index_x = np.setdiff1d([i for i in range(temp_cell_clu_marker_maxtrix.shape[0])],temp_max_x)
    temp_index_y = np.setdiff1d([i for i in range(temp_cell_clu_marker_maxtrix.shape[1])], temp_max_y)
    if temp_index_x.size == 0 or temp_index_y.size == 0:
        break
    cell_clu_marker_x_flag = cell_clu_marker_x_flag[temp_index_x]
    cell_clu_marker_y_flag = cell_clu_marker_y_flag[temp_index_y]
    temp_cell_clu_marker_maxtrix = temp_cell_clu_marker_maxtrix[temp_index_x,:]
    temp_cell_clu_marker_maxtrix = temp_cell_clu_marker_maxtrix[:, temp_index_y]
cell_anno_markers_output = None
for (key, vaule) in cell_clu_marker_map.items():
    clu_anno = cell_clu_marker_map_anno[key]
    markers = gene_names[vaule]
    for j in markers:
        if cell_anno_markers_output is None:
            cell_anno_markers_output = np.array([clu_anno,j])
        else:
            cell_anno_markers_output = np.vstack((cell_anno_markers_output,[clu_anno,j]))
df = pd.DataFrame(data=cell_anno_markers_output[:,1], index= cell_anno_markers_output[:,0])
df.to_csv(os.path.join(save_path,"cell_clu_anno_markers.csv"))



cell_snare_clu_anno = list(marker_snare_gene_clu.keys())
marker_snare_gene_name_map = {}
for i in cell_snare_clu_anno:
    temp_marker_gene_set = marker_snare_gene_clu[i]
    temp_marker_index = None
    for temp_marker in temp_marker_gene_set:
        temp_index = np.where(gene_names == temp_marker)[0]
        if not temp_index.size:
            continue
        if temp_marker_index is None:
            temp_marker_index = temp_index
        else:
            temp_marker_index = np.append(temp_marker_index,temp_index)
    marker_snare_gene_name_map[i] = temp_marker_index

marker_snare_gene_index = None
for temp_index in list(marker_snare_gene_name_map.values()):
    if marker_snare_gene_index is None:
        marker_snare_gene_index = np.array(temp_index)
    else:
        marker_snare_gene_index = np.append(marker_snare_gene_index,temp_index)
marker_snare_interaction = np.array([i for i in marker_snare_gene_index if i in (diff_gene_matrix.flatten())])
diff_gene_matrix_index = np.unique(diff_gene_matrix.flatten())

diff_snare_gene_clu_anno = {}
diff_snare_gene_clu_anno_index = {}
cell_snare_clu_marker_map = {}
cell_snare_clu_marker_map_anno = {}
cell_snare_clu_marker_matrix = np.zeros((diff_gene_matrix.shape[0],len(list(marker_snare_gene_name_map.keys())))).astype(np.float)
for i in range(diff_gene_matrix.shape[0]):
    for j in list(marker_snare_gene_name_map.keys()):
        temp_index = np.array([var for var in marker_snare_gene_name_map[j] if var in diff_gene_matrix[i,:]])
        if not temp_index.size:
            continue
        temp_j_index = np.where(np.array(list(marker_snare_gene_name_map.keys())) == j)[0]
        cell_snare_clu_marker_matrix[i, temp_j_index] = len(temp_index)/len(marker_snare_gene_name_map[j])
        '''
        if i in cell_snare_clu_marker_map.keys():
            if len(cell_snare_clu_marker_map[i])/len(marker_snare_gene_name_map[cell_snare_clu_marker_map_anno[i]]) < len(temp_index)/len(marker_snare_gene_name_map[j]):
                cell_snare_clu_marker_map[i] = temp_index
                cell_snare_clu_marker_map_anno[i] = j
        else:
            cell_snare_clu_marker_map[i] = temp_index
            cell_snare_clu_marker_map_anno[i] = j
        '''

        if j in diff_snare_gene_clu_anno.keys():
            if len(diff_snare_gene_clu_anno[j]) < len(temp_index):
                diff_snare_gene_clu_anno[j] = temp_index
                diff_snare_gene_clu_anno_index[j] = i
        else:
            diff_snare_gene_clu_anno[j] = temp_index
            diff_snare_gene_clu_anno_index[j] = i

cell_clu_marker_x_flag = np.array([i for i in range(cell_snare_clu_marker_matrix.shape[0])])
cell_clu_marker_y_flag = np.array(list(marker_snare_gene_name_map.keys()))
temp_cell_clu_marker_maxtrix = cell_snare_clu_marker_matrix
for i in range(cell_snare_clu_marker_matrix.shape[0]):
    if np.max(temp_cell_clu_marker_maxtrix) == 0:
        break
    temp_max = np.argmax(temp_cell_clu_marker_maxtrix)
    temp_max_x = temp_max//temp_cell_clu_marker_maxtrix.shape[1]
    temp_max_y = temp_max%temp_cell_clu_marker_maxtrix.shape[1]
    cell_snare_clu_marker_map_anno[cell_clu_marker_x_flag[temp_max_x]] = cell_clu_marker_y_flag[temp_max_y]
    cell_snare_clu_marker_map[cell_clu_marker_x_flag[temp_max_x]] = np.array([var for var in marker_snare_gene_name_map[cell_clu_marker_y_flag[temp_max_y]] if var in diff_gene_matrix[cell_clu_marker_x_flag[temp_max_x], :]])
    temp_index_x = np.setdiff1d([i for i in range(temp_cell_clu_marker_maxtrix.shape[0])],temp_max_x)
    temp_index_y = np.setdiff1d([i for i in range(temp_cell_clu_marker_maxtrix.shape[1])], temp_max_y)
    if temp_index_x.size == 0 or temp_index_y.size == 0:
        break
    cell_clu_marker_x_flag = cell_clu_marker_x_flag[temp_index_x]
    cell_clu_marker_y_flag = cell_clu_marker_y_flag[temp_index_y]
    temp_cell_clu_marker_maxtrix = temp_cell_clu_marker_maxtrix[temp_index_x,:]
    temp_cell_clu_marker_maxtrix = temp_cell_clu_marker_maxtrix[:, temp_index_y]
cell_anno_snare_markers_output = None
for (key, vaule) in cell_snare_clu_marker_map.items():
    clu_anno = cell_snare_clu_marker_map_anno[key]
    markers = gene_names[vaule]
    for j in markers:
        if cell_anno_snare_markers_output is None:
            cell_anno_snare_markers_output = np.array([clu_anno,j])
        else:
            cell_anno_snare_markers_output = np.vstack((cell_anno_snare_markers_output,[clu_anno,j]))
df = pd.DataFrame(data=cell_anno_snare_markers_output[:,1], index= cell_anno_snare_markers_output[:,0])
df.to_csv(os.path.join(save_path,"cell_clu_snaure_anno_markers.csv"))

gene_names = gene_names
