# need to modify the csv file
import pandas as pd
import re
import numpy as np
gene_imputation_names = pd.read_csv(
            "E:\data\qiliu\single-cell program\ATAC\snare data\gene_imputation.csv",
            header = 0,
            index_col=0,
            na_values = "NA",
            )

gene_name_list = pd.read_csv(
                             "E:\data\qiliu\single-cell program\ATAC\snare data\GSE126074_P0_BrainCortex_SNAREseq_cDNA.genes.tsv",
                             header = None,
                            )
pattern = re.compile(r'[a-z0-9.\-\\(\\)]+', re.I)
index = 0
genename_dict = {}
for i in range(gene_imputation_names.index.size):
    gene_name_imp = gene_imputation_names.index[i]
    m = pattern.match(gene_name_imp,2)
    gene_name_imp = m.group(0)
    while True:
        gene_name = gene_name_list.values[index]
        gene_name = gene_name[0].upper()
        if index >= gene_name_list.size:
            break
        elif gene_name.strip() == gene_name_imp.strip():
            genename_dict[gene_name_list.values[index][0]] = gene_name_imp
            break
        index = index+1
f = open('E:\data\qiliu\single-cell program\ATAC\snare data\genename_map.txt', 'w')
for (k,v) in  genename_dict.items():
    f.write(str(k)+'\t'+str(v)+'\n')
f.close()
print("ok")
