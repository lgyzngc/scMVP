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
gene_names = gene_exp.index.values
atac_names = atac_exp.index.values
gene_exp = None
atac_exp = None
promoter_location = pd.read_csv(save_path+'promoter/promoter_ucsc.txt', sep='\t', header=0, index_col=0)
enhancer_data_path = "E:/data/qiliu/single-cell program/ATAC/result/whole figure/top 5000 knn 3 pvalue 1/validation data for mouse brain/"
enhancer_location = pd.read_csv(enhancer_data_path+'GSE86150_Mouse.ESF123.placseq.H3k4me3.fithic.interactions.bed', sep='\t', header=0, index_col=None)
enhancer_location_H3K27ac = pd.read_csv(enhancer_data_path+'ENCFF676TSV.bed', sep='\t', header=None, index_col=None)

enhancer_location_H3K27ac_dict = {}
enhancer_location_H3K27ac_peaks = np.array([])
for i in enhancer_location_H3K27ac.index.values:
    if enhancer_location_H3K27ac.iloc[i,0] in enhancer_location_H3K27ac_dict.keys():
        temp_location_set = enhancer_location_H3K27ac_dict[enhancer_location_H3K27ac.iloc[i,0]]
        temp_location = str(enhancer_location_H3K27ac.iloc[i, 0]) + "_" + str(enhancer_location_H3K27ac.iloc[i, 1]) + "_" + str(enhancer_location_H3K27ac.iloc[i, 2])
        temp_location_set = np.append(temp_location_set,temp_location)
        enhancer_location_H3K27ac_dict[enhancer_location_H3K27ac.iloc[i, 0]] = temp_location_set
        enhancer_location_H3K27ac_peaks = np.append(enhancer_location_H3K27ac_peaks,temp_location)
    else:
        temp_location = str(enhancer_location_H3K27ac.iloc[i, 0]) + "_" + str(enhancer_location_H3K27ac.iloc[i, 1]) + "_" + str(enhancer_location_H3K27ac.iloc[i, 2])
        enhancer_location_H3K27ac_dict[enhancer_location_H3K27ac.iloc[i, 0]] = np.array([temp_location])
        enhancer_location_H3K27ac_peaks = np.append(enhancer_location_H3K27ac_peaks,temp_location)



diff_gene_clu_index = np.array([0,2,4,5,6,9,13,15,16,19,20])
regression_peak_dict = {}
for i in diff_gene_clu_index:
    regression_peak_dict['the_peakclu_'+str(i)] = pd.read_csv(save_path+'the'+str(i)+'atac_pls_cluster_index_rerun.csv', header=0, index_col=0)



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
# mapping promoter into diff gene clu
joint_promoter_diff_Gene_clu_dict = {}
joint_promoter_diff_Gene_clu_peak_dict = {}
joint_promoter_diff_Gene_clu_regression_eapk_dict = {}
joint_diff_Gene_regression_peaks = None
joint_diff_Gene_regression_peaks_index = np.array([])

for i in diff_gene_clu_index:
    clugene = regression_peak_dict['the_peakclu_'+str(i)]
    for j in clugene.index.values:
        temp_j = j
        j = j[2:-3]
        if j in joint_gene_promoter_map.keys():
            joint_promoter_diff_Gene_clu_dict[j] = joint_gene_promoter_map[j]
            joint_promoter_diff_Gene_clu_peak_dict[joint_gene_promoter_map[j]] = j
            joint_promoter_diff_Gene_clu_regression_eapk_dict[j] = np.array(clugene.loc[temp_j].tolist())
            if joint_diff_Gene_regression_peaks is None:
                joint_diff_Gene_regression_peaks = np.array(clugene.loc[temp_j].tolist())[0:2000]
            else:
                joint_diff_Gene_regression_peaks = np.vstack((joint_diff_Gene_regression_peaks,np.array(clugene.loc[temp_j].tolist())[0:2000]))
            joint_diff_Gene_regression_peaks_index = np.append(joint_diff_Gene_regression_peaks_index,j)
# promoter_enhancer_regresssion_mapping_H3K27ac
promoter_enhancer_regresssion_H3K27ac_dict = {}
enhancer_location_array = None
joint_diff_Gene_regression_peaks_unique = np.unique(joint_diff_Gene_regression_peaks.flatten())
joint_diff_Gene_regression_peaks_unique_loaction = None
for i in joint_diff_Gene_regression_peaks_unique:
    temp_atac_name = atac_names[i]
    temp_atac_name = temp_atac_name[2:-3]
    temp = temp_atac_name.split(':')
    if len(temp) != 2:
        continue
    temp_loc = temp[1].split('-')
    if len(temp_loc) != 2:
        continue
    if joint_diff_Gene_regression_peaks_unique_loaction is None:
        joint_diff_Gene_regression_peaks_unique_loaction = np.array([temp[0],temp_loc[0],temp_loc[1]])
    else:
        joint_diff_Gene_regression_peaks_unique_loaction = np.vstack((joint_diff_Gene_regression_peaks_unique_loaction,np.array([temp[0],temp_loc[0],temp_loc[1]])))
    for j in enhancer_location_H3K27ac_peaks:
        temp_enhancer_loc = j.split('_')
        if len(temp_enhancer_loc) < 3:
            continue
        if temp_enhancer_loc[0] != temp[0]:
            continue
        # temp_enhancer_loc = temp_enhancer_loc[1:-1]
        if (float(temp_enhancer_loc[1]) > float(temp_loc[1])) or (float(temp_enhancer_loc[2]) < float(temp_loc[0])):
            continue
        promoter_enhancer_regresssion_H3K27ac_dict[i] = j
        if enhancer_location_array is None:
            enhancer_location_array = np.array([temp[0],temp_loc[0],temp_loc[1]])
        else:
            enhancer_location_array = np.vstack((enhancer_location_array,np.array([temp[0],temp_loc[0],temp_loc[1]])))
        break
    print(i)

df = pd.DataFrame(enhancer_location_array)
df.to_csv(os.path.join(save_path, "validated_regression_peak_location.csv"))
df = pd.DataFrame(joint_diff_Gene_regression_peaks_unique_loaction)
df.to_csv(os.path.join(save_path, "regression_peak_location.csv"))



promoter_enhancer_regresssion_H3K27ac_peak = np.array(list(promoter_enhancer_regresssion_H3K27ac_dict.keys()))
gene_name_regression_H3K27ac_peak_dict = {}
for i in range(joint_diff_Gene_regression_peaks.shape[0]):
    gene_name_regression_H3K27ac_peak_dict[joint_diff_Gene_regression_peaks_index[i]] = \
        np.array([j for j in joint_diff_Gene_regression_peaks[i,:] if j in promoter_enhancer_regresssion_H3K27ac_peak])

joint_diff_Gene_regression_peak_counts = {}
for i in gene_name_regression_H3K27ac_peak_dict.keys():
    joint_diff_Gene_regression_peak_counts[i] = len(gene_name_regression_H3K27ac_peak_dict[i])
# save as csv
psudo_validated_peaks = np.zeros((len(list(gene_name_regression_H3K27ac_peak_dict.keys())),2000)).astype(np.int)
index=0
for i in gene_name_regression_H3K27ac_peak_dict.keys():
    temp_peaks = gene_name_regression_H3K27ac_peak_dict[i]
    psudo_validated_peaks[index,0:len(temp_peaks)] = temp_peaks
    index=index+1
df = pd.DataFrame(psudo_validated_peaks,index=list(gene_name_regression_H3K27ac_peak_dict.keys()))
df.to_csv(os.path.join(save_path, "validated_regression_peaks.csv"))
# venn plot
'''
pro_enhan_regress_H3K27ac = set([i for i in range(len(list(promoter_enhancer_regresssion_H3K27ac_dict.keys())))])
pro_enhan_regress = set([(i+len(enhancer_location_H3K27ac_peaks)-len(list(promoter_enhancer_regresssion_H3K27ac_dict.keys()))) for i in range(len(joint_diff_Gene_regression_peaks_unique))])
enhancer_counts = set([i for i in range(len(enhancer_location_H3K27ac_peaks))])
marker_enhan_corr_H3K27ac = set([i for i in range(2799)])
marker_enhan_corr = set([(i+len(enhancer_location_H3K27ac_peaks)-2799) for i in range(6750)])
'''
# enhancer mapping：
enhancer_location_Dict = {}
for i in enhancer_location.index.values:
    if enhancer_location.iloc[i,0] in enhancer_location_Dict.keys():
        temp_dict = enhancer_location_Dict[enhancer_location.iloc[i,0]]
        temp_str_source = str(enhancer_location.iloc[i,0])+"_"+str(enhancer_location.iloc[i,1]) +"_"+ str(enhancer_location.iloc[i,2])
        temp_str_target = str(enhancer_location.iloc[i,3]) +"_"+ str(enhancer_location.iloc[i,4]) +"_"+ str(enhancer_location.iloc[i,5])
        if temp_str_source in temp_dict.keys():
            temp_dict[temp_str_source] = np.unique(np.append(temp_dict[temp_str_source],temp_str_target))
        else:
            temp_dict[temp_str_source] = temp_str_target
        enhancer_location_Dict[enhancer_location.iloc[i, 0]] = temp_dict
    else:
        temp_dict = {}
        temp_str_source = str(enhancer_location.iloc[i, 0]) + "_" + str(enhancer_location.iloc[i, 1]) + "_" + str(enhancer_location.iloc[i, 2])
        temp_str_target = str(enhancer_location.iloc[i, 3]) + "_" + str(enhancer_location.iloc[i, 4]) + "_" + str(enhancer_location.iloc[i, 5])
        temp_dict[temp_str_source] = temp_str_target
        enhancer_location_Dict[enhancer_location.iloc[i, 0]] = temp_dict

for i in enhancer_location.index.values:
    if enhancer_location.iloc[i,3] in enhancer_location_Dict.keys():
        temp_dict = enhancer_location_Dict[enhancer_location.iloc[i,3]]
        temp_str_source = str(enhancer_location.iloc[i,3])+"_"+str(enhancer_location.iloc[i,4]) +"_"+ str(enhancer_location.iloc[i,5])
        temp_str_target = str(enhancer_location.iloc[i,0]) +"_"+ str(enhancer_location.iloc[i,1]) +"_"+ str(enhancer_location.iloc[i,2])
        if temp_str_source in temp_dict.keys():
            temp_dict[temp_str_source] = np.unique(np.append(temp_dict[temp_str_source],temp_str_target))
        else:
            temp_dict[temp_str_source] = np.array([temp_str_target])
        enhancer_location_Dict[enhancer_location.iloc[i, 3]] = temp_dict
    else:
        temp_dict = {}
        temp_str_source = str(enhancer_location.iloc[i, 3]) + "_" + str(enhancer_location.iloc[i, 4]) + "_" + str(enhancer_location.iloc[i, 5])
        temp_str_target = str(enhancer_location.iloc[i, 0]) + "_" + str(enhancer_location.iloc[i, 1]) + "_" + str(enhancer_location.iloc[i, 2])
        temp_dict[temp_str_source] = np.array([temp_str_target])
        enhancer_location_Dict[enhancer_location.iloc[i, 3]] = temp_dict
# mapping promoter into protential enhancer
promoter_enhancer_location_dict = {}
for i in joint_promoter_diff_Gene_clu_peak_dict.keys():
    atac_key = atac_names[i]
    atac_key = atac_key[2:-3]
    temp = atac_key.split(':')
    if len(temp) != 2:
        continue
    temp_loc = temp[1].split('-')
    if len(temp_loc) != 2:
        continue
    if temp[0] in enhancer_location_Dict.keys():
        temp_chr_enhancer = enhancer_location_Dict[temp[0]]
        for j in temp_chr_enhancer.keys():
            temp_enhancer_loc = j.split('_')
            if len(temp_enhancer_loc) < 3:
                continue
            if temp[0] != temp_enhancer_loc[0]:
                continue
            #temp_enhancer_loc = temp_enhancer_loc[1:-1]
            if (float(temp_enhancer_loc[1]) > float(temp_loc[1])) or (float(temp_enhancer_loc[2]) < float(temp_loc[0])):
                continue
            promoter_enhancer_location_dict[i] = temp_chr_enhancer[j]


# promoter_enhancer_regression_mapping
promoter_enhancer_regression_dict = {}
for i in promoter_enhancer_location_dict.keys():
    temp_gene_name = joint_promoter_diff_Gene_clu_peak_dict[i]
    temp_enhancer_peak = promoter_enhancer_location_dict[i]
    temp_regression_peak = joint_promoter_diff_Gene_clu_regression_eapk_dict[temp_gene_name]
    for j in temp_regression_peak:
        temp_atac_name = atac_names[j]
        temp_atac_name = temp_atac_name[2:-3]
        temp = temp_atac_name.split(':')
        if len(temp) != 2:
            continue
        temp_loc = temp[1].split('-')
        if len(temp_loc) != 2:
            continue

        for k in temp_enhancer_peak:
            temp_enhancer_loc = k.split('_')
            if len(temp_enhancer_loc) < 3:
                continue
            if temp_enhancer_loc[0] != temp[0]:
                continue
            #temp_enhancer_loc = temp_enhancer_loc[1:-1]
            if (float(temp_enhancer_loc[1]) > float(temp_loc[1]) + 10000) or (float(temp_enhancer_loc[2]) < float(temp_loc[0])-10000):
                continue
            if i in promoter_enhancer_regression_dict.keys():
                promoter_enhancer_regression_dict[i] = np.append(promoter_enhancer_regression_dict[i],j)
            else:
                promoter_enhancer_regression_dict[i] = np.array([j])
            #break

promoter_enhancer_regression_interaction_peaks = np.array([])
promoter_enhancer_regression_interaction_index = np.array([])
for i in promoter_enhancer_regression_dict.keys():
    promoter_enhancer_regression_interaction_peaks = np.append(promoter_enhancer_regression_interaction_peaks,promoter_enhancer_regression_dict[i])
    temp_index = None
    if len(promoter_enhancer_regression_dict[i]) == 1:
        temp_index = i
    else:
        temp_index = np.repeat(i,len(promoter_enhancer_regression_dict[i]))
    promoter_enhancer_regression_interaction_index = np.append(promoter_enhancer_regression_interaction_index,temp_index)
df = pd.DataFrame(np.vstack((promoter_enhancer_regression_interaction_index,promoter_enhancer_regression_interaction_peaks)).T)
df.to_csv(os.path.join(save_path,"promoter_enhancer_regression_interaction.csv"))


promoter_enhancer_peaks = np.array([])
promoter_enhancer_index = np.array([])
for i in promoter_enhancer_location_dict.keys():
    promoter_enhancer_peaks = np.append(promoter_enhancer_peaks, promoter_enhancer_location_dict[i])
    temp_index = None
    if type(promoter_enhancer_location_dict[i]) is not np.ndarray:
        temp_index = i
    elif len(promoter_enhancer_location_dict[i]) == 1:
        temp_index = i
    else:
        temp_index = np.repeat(i, len(promoter_enhancer_location_dict[i]))
    promoter_enhancer_index = np.append(promoter_enhancer_index, temp_index)
df = pd.DataFrame(np.vstack((promoter_enhancer_index, promoter_enhancer_peaks)).T)
df.to_csv(os.path.join(save_path, "promoter_enhancer.csv"))

promoter_enhancer_peaks = len(promoter_enhancer_peaks)
promoter_enhancer_regression_interaction_peaks = len(promoter_enhancer_regression_interaction_peaks)
ratio_regpeak = promoter_enhancer_regression_interaction_peaks / promoter_enhancer_peaks
print(ratio_regpeak)

promoter_enhancer_peaks_unique = len(np.unique(promoter_enhancer_index))
promoter_enhancer_regression_interaction_peaks_unique = len(np.unique(promoter_enhancer_regression_interaction_index))
ratio_promoter = promoter_enhancer_regression_interaction_peaks_unique / promoter_enhancer_peaks_unique
print(ratio_promoter)

# mapping enhancer location into promoter
promoter_enhancer_dict = {}
promoter_enhancer_regression_dict = {}
for i in regression_peak_dict.keys():
    diff_gene_clu = regression_peak_dict[i]
    for j in diff_gene_clu.index.values:
        regression_peaks = np.array(diff_gene_clu.loc[j].tolist()).astype(np.int)
        j = j[2:-3]
        if j not in promoter_gene_dic.keys():
            continue
        gene_promoter_loc = promoter_gene_dic[j]
        gene_promoter_chr = promoter_chr_dic[j]


        temp_chr_1 = enhancer_location.values[:,0]
        temp_chr_1_index = np.where(temp_chr_1 == gene_promoter_chr[0])[0]
        if len(temp_chr_1_index) < 1:
            continue
        for k in temp_chr_1_index:
            for temp_gene_promoter_loc in gene_promoter_loc:
                if temp_gene_promoter_loc >= enhancer_location.values[k,1] and temp_gene_promoter_loc <= enhancer_location.values[k,2]:
                    if j in promoter_enhancer_dict.keys():
                        promoter_enhancer_dict[j] = np.append(promoter_enhancer_dict[j], k)
                    else:
                        promoter_enhancer_dict[j] = k

                    atac_chrs_index = np.where(atac_chr_set[regression_peaks] == enhancer_location.values[k, 3])[0]
                    if len(atac_chrs_index) < 1:
                        continue

                    atac_chrs_index = regression_peaks[atac_chrs_index]
                    atac_locs = atac_loc_set[atac_chrs_index, :]
                    enhancer_loc = enhancer_location.values[k, 4:6]
                    for atac_index in range(atac_locs.shape[0]):
                        if  atac_locs[atac_index,0] > enhancer_loc[1] or atac_locs[atac_index,1] < enhancer_loc[0]:
                            continue
                        else:
                            if j in promoter_enhancer_regression_dict.keys():
                                promoter_enhancer_regression_dict[j] = np.append(promoter_enhancer_regression_dict[j],atac_chrs_index[atac_index])
                            else:
                                promoter_enhancer_regression_dict[j] = atac_chrs_index[atac_index]


        temp_chr_2 = enhancer_location.values[:, 3]
        temp_chr_2_index = np.where(temp_chr_2 == gene_promoter_chr[0])[0]
        if len(temp_chr_2_index) < 1:
            continue
        for k in temp_chr_2_index:
            for temp_gene_promoter_loc in gene_promoter_loc:
                if temp_gene_promoter_loc >= enhancer_location.values[k,4] and temp_gene_promoter_loc <= enhancer_location.values[k,5]:
                    if j in promoter_enhancer_dict.keys():
                        promoter_enhancer_dict[j] = np.append(promoter_enhancer_dict[j],k)
                    else:
                        promoter_enhancer_dict[j] = k

                    atac_chrs_index = np.where(atac_chr_set[regression_peaks] == enhancer_location.values[k,0])[0]
                    if len(atac_chrs_index) < 1:
                        continue
                    atac_chrs_index = regression_peaks[atac_chrs_index]
                    atac_locs = atac_loc_set[atac_chrs_index, :]
                    enhancer_loc = enhancer_location.values[k, 1:3]
                    for atac_index in range(atac_locs.shape[0]):
                        if  atac_locs[atac_index,0] > enhancer_loc[1] or atac_locs[atac_index,1] < enhancer_loc[0]:
                            continue
                        else:
                            if j in promoter_enhancer_regression_dict.keys():
                                promoter_enhancer_regression_dict[j] = np.append(promoter_enhancer_regression_dict[j],atac_chrs_index[atac_index])
                            else:
                                promoter_enhancer_regression_dict[j] = atac_chrs_index[atac_index]

promoter_enhancer_regression_interaction_peaks = np.array([])
promoter_enhancer_regression_interaction_index = np.array([])
for i in promoter_enhancer_regression_dict.keys():
    promoter_enhancer_regression_interaction_peaks = np.append(promoter_enhancer_regression_interaction_peaks,promoter_enhancer_regression_dict[i])
    temp_index = np.repeat(i,len(promoter_enhancer_regression_dict[i]))
    promoter_enhancer_regression_interaction_index = np.append(promoter_enhancer_regression_interaction_index,temp_index)
df = pd.DataFrame(np.vstack((promoter_enhancer_regression_interaction_index,promoter_enhancer_regression_interaction_peaks)).T)
df.to_csv(save_path,"promoter_enhancer_regression_interaction.csv")


promoter_enhancer_peaks = np.array([])
promoter_enhancer_index = np.array([])
for i in promoter_enhancer_dict.keys():
    promoter_enhancer_peaks = np.append(promoter_enhancer_peaks,promoter_enhancer_dict[i])
    temp_index = np.repeat(i, len(promoter_enhancer_dict[i]))
    promoter_enhancer_index = np.append(promoter_enhancer_index,temp_index)
df = pd.DataFrame(np.vstack((promoter_enhancer_index,promoter_enhancer_peaks)).T)
df.to_csv(save_path,"promoter_enhancer.csv")

promoter_enhancer_peaks_unique = np.unique(promoter_enhancer_peaks)
promoter_enhancer_regression_interaction_peaks_unique = np.unique(promoter_enhancer_regression_interaction_peaks)
ratio = promoter_enhancer_regression_interaction_peaks_unique/promoter_enhancer_peaks_unique
print("ok!")
