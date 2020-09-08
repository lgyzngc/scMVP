# testing the multi-modal vae method by scRNA-seq and scATAC data date: 11/12/2019
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
#pygui(True)
#mpl.use('WX')
import matplotlib.pyplot as plt
from scMVP.dataset import CortexDataset, snareDataset, scienceDataset, ATACDataset, geneDataset
from scMVP.models import VAE
from scMVP.inference import UnsupervisedTrainer
from scMVP.inference import MultiPosterior, MultiTrainer
import torch

from scMVP.models.multi_vae import Multi_VAE

## Visualizing the latent space with scanpy
import scanpy as sc
import seaborn as sns
import anndata


def allow_mmvae_for_test():
    print("Testing the basic tutorial mmvae")


test_mode = False
save_path = "data/"
n_epochs_all = None
show_plot = True

if not test_mode:
    save_path = "E:/data/qiliu/single-cell program/ATAC/science data/"

## Loading data should implenment the detail of data requirement function
#dataset = snareDataset(dataset_name="P0_BrainCortex", save_path=save_path, measurement_names_column=1, is_binary = True)
#dataset = snareDataset(dataset_name="CellLineMixture", save_path=save_path, measurement_names_column=1, is_binary = True)
dataset = scienceDataset(dataset_name="CellLineMixture", save_path=save_path, measurement_names_column=1, is_binary = True)

#ATAC_dataset = CortexDataset(save_path=save_path, total_genes=558)
# We do some light filtering for cells without many genes expressed and cells with low protein counts
def filter_dataset(dataset):
    high_count_genes = (dataset.X > 0).sum(axis=0).ravel() > 0.01 * dataset.X.shape[0]
    dataset.update_genes(high_count_genes)
    dataset.subsample_genes(new_n_genes=10000)

    adata_source = anndata.AnnData(X=dataset.X)
    #sc.pp.normalize_per_cell(adata_source, counts_per_cell_after=1e4)
    sc.pp.log1p(adata_source)
    sc.pp.highly_variable_genes(adata_source)
    matplotlib.use('TkAgg')
    sc.pl.highly_variable_genes(adata_source)
    #sc.pl.highly_variable_genes(adata_source)
    dataset.update_genes((adata_source.var['highly_variable']))




    # Filter atac data
    high_count_atacs = (dataset.atac_expression > 0).sum(axis=0).ravel() > 0.03 * dataset.atac_expression.shape[0]
    dataset.atac_expression = dataset.atac_expression[:, high_count_atacs]
    dataset.atac_names = dataset.atac_names[high_count_atacs]


    high_gene_count_cells = (dataset.X > 0).sum(axis=1).ravel() > 50
    #high_atac_cells = dataset.atac_expression.sum(axis=1) >= np.percentile(dataset.atac_expression.sum(axis=1), 10)
    high_atac_cells = dataset.atac_expression.sum(axis=1) >= np.percentile(dataset.atac_expression.sum(axis=1), 1)
    inds_to_keep = np.logical_and(high_gene_count_cells, high_atac_cells)
    dataset.update_cells(inds_to_keep)
    return dataset, inds_to_keep

if test_mode is False:
    dataset, inds_to_keep = filter_dataset(dataset)
## Training
# __n_epochs__: Maximum number of epochs to train the model. If the likelihood change is small than a set threshold training will stop automatically.
# __lr__: learning rate. Set to 0.001 here.
# __use_batches__: If the value of true than batch information is used in the training. Here it is set to false because the cortex data only contains one batch.
# __use_cuda__: Set to true to use CUDA (GPU required)

#n_epochs = 400 if n_epochs_all is None else n_epochs_all
'''
n_epochs = 50 if n_epochs_all is None else n_epochs_all # p0 contex data
lr = 1e-3
use_batches = False
use_cuda = True
n_centroids = 19
n_alfa = 1.0
'''
# mixture cell line
n_epochs = 50 if n_epochs_all is None else n_epochs_all
lr = 1e-3
use_batches = False
use_cuda = True
n_centroids = 5
n_alfa = 1.0

'''
# AdBrainCortex
n_epochs = 50 if n_epochs_all is None else n_epochs_all
lr = 1e-3
use_batches = False
use_cuda = True
n_centroids = 22
n_alfa = 1.0
'''
#We now create the model and the trainer object. We train the model and output model likelihood every 5 epochs. In order to evaluate the likelihood on a test set, we split the datasets (the current code can also so train/validation/test).
#If a pre-trained model already exist in the save_path then load the same model rather than re-training it. This is particularly useful for large datasets.


# pretrainning vae with single guassian prior for initializing the mixture gaussian prior using a GMM model
#pre_vae = VAE(dataset.nb_genes, n_batch=dataset.n_batches * use_batches)
pre_vae = VAE(dataset.nb_genes, n_batch=256)
pre_trainer = UnsupervisedTrainer(
    pre_vae,
    dataset,
    train_size=0.75,
    use_cuda=use_cuda,
    frequency=5,
)

is_test_pragram = False
if is_test_pragram:
    pre_trainer.train(n_epochs=n_epochs, lr=lr)
    torch.save(pre_trainer.model.state_dict(), '%s/pre_trainer_varianceGene_1.pkl' % save_path)

if os.path.isfile('%s/pre_trainer_varianceGene_1.pkl' % save_path):
    pre_trainer.model.load_state_dict(torch.load('%s/pre_trainer_varianceGene_1.pkl' % save_path))
    pre_trainer.model.eval()
else:
    #pre_trainer.model.init_gmm_params(dataset)
    pre_trainer.train(n_epochs=n_epochs, lr=lr)
    torch.save(pre_trainer.model.state_dict(), '%s/pre_trainer_varianceGene_1.pkl' % save_path)

# pretrainer_posterior:
full = pre_trainer.create_posterior(pre_trainer.model, dataset, indices=np.arange(len(dataset)))
latent, batch_indices, labels = full.sequential().get_latent()
batch_indices = batch_indices.ravel()
imputed_values = full.sequential().imputation()
# visulization
prior_adata = anndata.AnnData(X=dataset.X)
prior_adata.obsm["X_multi_vi"] = latent
prior_adata.obs['cell_type'] = torch.tensor(labels.reshape(-1,1))
sc.pp.neighbors(prior_adata, use_rep="X_multi_vi", n_neighbors=15)
sc.tl.umap(prior_adata, min_dist=0.1)
matplotlib.use('TkAgg')
fig, ax = plt.subplots(figsize=(7, 6))
sc.pl.umap(prior_adata, color=["cell_type"], ax=ax, show=show_plot)
# save data as csv file
df = pd.DataFrame(data=prior_adata.obsm["X_umap"],  columns=["umap_dim1","umap_dim2"] , index=dataset.barcodes )
df.insert(0,"labels",labels)
df.to_csv(os.path.join(save_path,"scvi_umap_varianceGene.csv"))

sample_latents = torch.tensor([])
samples = torch.tensor([])
sample_labels = torch.tensor([])
for tensors_list in range(int(len(imputed_values)/256)+1):
    if tensors_list == range(int(len(imputed_values)/256)):
        x = torch.zeros((256,len(imputed_values[0])))
        x[0:len(x)-256*tensors_list,:] = torch.tensor(imputed_values[tensors_list * 256:len(imputed_values), :])
        y = torch.zeros((256))
        y[0:len(x)-256*tensors_list,:]  = torch.tensor(dataset.labels[tensors_list * 256:len(imputed_values)].astype(int))
        temp_samples = pre_trainer.model.get_latents(x,y)
        for temp_sample in temp_samples:
            sample_latents = torch.cat((sample_latents, temp_sample[0:len(x)-256*tensors_list,:].float()))


    temp_samples = pre_trainer.model.get_latents(
        x=torch.tensor(imputed_values[tensors_list * 256:(1 + tensors_list) * 256, :]),
        y=torch.tensor(dataset.labels[tensors_list * 256:(1 + tensors_list) * 256].astype(int)))
    for temp_sample in temp_samples:
        sample_latents = torch.cat((sample_latents, temp_sample.float()))
#        sample_labels = torch.cat((sample_labels, torch.tensor(dataset.labels[tensors_list * 256:(1 + tensors_list) * 256].astype(int)).float()))
# visulization
prior_adata = anndata.AnnData(X=dataset.X)
prior_adata.obsm["X_multi_vi"] = sample_latents.detach().numpy()
prior_adata.obs['cell_type'] = torch.tensor(dataset.labels[0:len(sample_latents)].astype(int))
sc.pp.neighbors(prior_adata, use_rep="X_multi_vi", n_neighbors=15)
sc.tl.umap(prior_adata, min_dist=0.1)
matplotlib.use('TkAgg')
fig, ax = plt.subplots(figsize=(7, 6))
sc.pl.umap(prior_adata, color=["cell_type"], ax=ax, show=show_plot)
# save data as csv file
df = pd.DataFrame(data=prior_adata.obsm["X_umap"],  columns=["umap_dim1","umap_dim2"] , index=dataset.barcodes[0:len(sample_latents)])
df.insert(0,"labels",dataset.labels[0:len(sample_latents)])
df.to_csv(os.path.join(save_path,"scvi_umap_imputation_varianceGene.csv"))

sample_latents = torch.tensor([])
samples = torch.tensor([])
sample_labels = torch.tensor([])
for tensors_list in pre_trainer.data_loaders_loop():
    sample_batch, local_l_mean, local_l_var, batch_index, y = zip(*tensors_list)
    temp_samples = pre_trainer.model.get_latents(*sample_batch) #check this expression
    samples = torch.cat((samples, sample_batch[0].float()))
    for temp_sample in temp_samples:
        sample_latents = torch.cat((sample_latents, temp_sample.float()))
        sample_labels = torch.cat((sample_labels, y[0].float()))
# end the pre-training

#multi_vae = Multi_VAE(dataset.nb_genes, len(dataset.atac_names), n_batch=dataset.n_batches * use_batches, n_centroids=n_centroids, n_alfa = n_alfa, mode="mm-vae") # should provide ATAC num, alfa, mode and loss type
multi_vae = Multi_VAE(dataset.nb_genes, len(dataset.atac_names), n_batch=256, n_centroids=n_centroids, n_alfa = n_alfa, mode="mm-vae") # should provide ATAC num, alfa, mode and loss type

# begin the multi-vae training
trainer = MultiTrainer(
    multi_vae,
    dataset,
    train_size=0.75,
    use_cuda=use_cuda,
    frequency=5,
)
# gmm cluster visulization
clust_index_gmm = trainer.model.init_gmm_params(sample_latents.detach().numpy())

prior_adata = anndata.AnnData(X=samples.detach().numpy())
#post_adata = anndata.AnnData(X=dataset.atac_expression)
prior_adata.obsm["X_multi_vi"] = sample_latents.detach().numpy()
#post_adata.obs['cell_type'] = np.array([dataset.cell_types[dataset.labels[i][0]]
#                                        for i in range(post_adata.n_obs)])
# scvi labeled
prior_adata.obs['cell_type'] = torch.tensor(sample_labels.numpy().reshape(-1,1))
sc.pp.neighbors(prior_adata, use_rep="X_multi_vi", n_neighbors=15)
sc.tl.umap(prior_adata, min_dist=0.1)
matplotlib.use('TkAgg')
fig, ax = plt.subplots(figsize=(7, 6))
sc.pl.umap(prior_adata, color=["cell_type"], ax=ax, show=show_plot)
# scvi gmm clustering
prior_adata.obs['cell_type'] = torch.tensor(clust_index_gmm.reshape(-1,1))
sc.pp.neighbors(prior_adata, use_rep="X_multi_vi", n_neighbors=15)
sc.tl.umap(prior_adata, min_dist=0.1)
#%%
matplotlib.use('TkAgg')
fig, ax = plt.subplots(figsize=(7, 6))
sc.pl.umap(prior_adata, color=["cell_type"], ax=ax, show=show_plot)


sc.tl.louvain(prior_adata)
sc.pl.umap(prior_adata, color=['louvain'])

is_test_pragram = False
if is_test_pragram:
    trainer.train(n_epochs=n_epochs, lr=lr)
    torch.save(trainer.model.state_dict(), '%s/multi_vae_varianceGene_12.pkl' % save_path)

#if os.path.isfile('%s/multi_vae_1.pkl' % save_path):
#    trainer.model.load_state_dict(torch.load('%s/multi_vae_1.pkl' % save_path))
#    trainer.model.eval()
if os.path.isfile('%s/multi_vae_varianceGene_12.pkl' % save_path):
    trainer.model.load_state_dict(torch.load('%s/multi_vae_varianceGene_12.pkl' % save_path))
    trainer.model.eval()
    #trainer.train(n_epochs=n_epochs, lr=lr)
    #torch.save(trainer.model.state_dict(), '%s/multi_vae_3.pkl' % save_path)
else:
    #trainer.model.init_gmm_params(dataset)
    trainer.train(n_epochs=n_epochs, lr=lr)
    torch.save(trainer.model.state_dict(), '%s/multi_vae_varianceGene_12.pkl' % save_path)

# Plotting the likelihood change across the 500 epochs of training: blue for training error and orange for testing error.**
elbo_train_set = trainer.history["elbo_train_set"]
elbo_test_set = trainer.history["elbo_test_set"]
x = np.linspace(0, 500, (len(elbo_train_set)))
#matplotlib.use('TkAgg')
plt.plot(x, elbo_train_set)
plt.plot(x, elbo_test_set)
plt.ylim(1150, 1600)
plt.show()
print(matplotlib.get_backend())

## Obtaining the posterior object and sample latent space as well as imputation from it
full = trainer.create_posterior(trainer.model, dataset, indices=np.arange(len(dataset)),type_class=MultiPosterior)
latent, latent_rna, latent_atac, cluster_gamma, cluster_index, batch_indices, labels = full.sequential().get_latent()
batch_indices = batch_indices.ravel()

prior_adata = anndata.AnnData(X=dataset.X)
prior_adata.obsm["X_multi_vi"] = latent
prior_adata.obs['cell_type'] = torch.tensor(labels)
sc.pp.neighbors(prior_adata, use_rep="X_multi_vi", n_neighbors=15)
sc.tl.umap(prior_adata, min_dist=0.1)
matplotlib.use('TkAgg')
fig, ax = plt.subplots(figsize=(7, 6))
sc.pl.umap(prior_adata, color=["cell_type"], ax=ax, show=show_plot)

#Similarly, it is possible to query the imputed values via the `imputation` method of the posterior object. **Note for advanced users:** imputation is an ambiguous term and there are two ways to perform imputation in scVI. The first way is to query the **mean of the negative binomial** distribution modeling the counts. This is referred to as `sample_rate` in the codebase and can be reached via the `imputation` method. The second is to query the **normalized mean of the same negative binomial** (please refer to the scVI manuscript). This is referred to as `sample_scale` in the codebase and can be reached via the `get_sample_scale` method. In differential expression for example, we of course rely on the normalized latent variable which is corrected for variations in sequencing depth.
imputed_values = full.sequential().imputation()
#normalized_values = full.sequential().get_sample_scale()
sample_latents = torch.tensor([])
sample_labels = torch.tensor([])
temp_rna = imputed_values[0]
#temp_atac = imputed_values[1]
temp_atac = imputed_values[3]
temp_label = []
sample_latents = torch.tensor([])
samples = torch.tensor([])
sample_labels = torch.tensor([])
if len(imputed_values) >= 3:
    temp_label = imputed_values[2]
for tensors_list in range(int(len(imputed_values[0])/256)+1):
    if temp_label.any():
        temp_samples = trainer.model.get_latents(x_rna=torch.tensor(temp_rna[tensors_list*256:(1+tensors_list)*256,:]),
                                                x_atac=torch.tensor(temp_atac[tensors_list*256:(1+tensors_list)*256,:]),
                                                y=torch.tensor(temp_label[tensors_list*256:(1+tensors_list)*256])) #check this expression
    else:
        temp_samples = trainer.model.get_latents(x_rna=torch.tensor(temp_rna[tensors_list*256:(1+tensors_list)*256,:]),
                                                x_atac=torch.tensor(temp_atac[tensors_list*256:(1+tensors_list)*256,:]),
                                                y=torch.tensor(np.zeros(256))) #check this expression
    for temp_sample in temp_samples:
        #sample_latents = torch.cat((sample_latents, temp_sample[2].float()))
        sample_latents = torch.cat((sample_latents, temp_sample[0][0].float()))
        sample_labels = torch.cat((sample_labels, torch.tensor(temp_label[tensors_list*256:(1+tensors_list)*256]).float()))

clust_index_gmm = trainer.model.init_gmm_params(sample_latents.detach().numpy())
#prior_adata = anndata.AnnData(X=temp_rna[0:int(len(imputed_values[0])/256)*256,:])
prior_adata = anndata.AnnData(X=temp_rna)
prior_adata.obsm["X_multi_vi"] = sample_latents.detach().numpy()
prior_adata.obs['cell_type'] = torch.tensor(clust_index_gmm.reshape(-1,1))
sc.pp.neighbors(prior_adata, use_rep="X_multi_vi", n_neighbors=15)
sc.tl.umap(prior_adata, min_dist=0.1)
matplotlib.use('TkAgg')
fig, ax = plt.subplots(figsize=(7, 6))
sc.pl.umap(prior_adata, color=["cell_type"], ax=ax, show=show_plot)
# imputation labels
prior_adata.obs['cell_type'] = torch.tensor(sample_labels.reshape(-1,1))
sc.pp.neighbors(prior_adata, use_rep="X_multi_vi", n_neighbors=15)
sc.tl.umap(prior_adata, min_dist=0.1)
matplotlib.use('TkAgg')
fig, ax = plt.subplots(figsize=(7, 6))
sc.pl.umap(prior_adata, color=["cell_type"], ax=ax, show=show_plot)
sc.tl.louvain(prior_adata)
sc.pl.umap(prior_adata, color=['louvain'])
# use multi-vae cluster
gmm_clus_index = clust_index_gmm.reshape(-1,1)
for i in range(len(np.unique(gmm_clus_index))):
    if len(gmm_clus_index[gmm_clus_index == i]) <= 10:
        for j in range(len(np.unique(gmm_clus_index))):
            if len(gmm_clus_index[gmm_clus_index == j]) > 100:
                gmm_clus_index[gmm_clus_index == i] = j
                break
unique_gmm_clus_index = np.unique(gmm_clus_index)
for i in range(len(unique_gmm_clus_index)):
    gmm_clus_index[gmm_clus_index == unique_gmm_clus_index[i]] = i
#prior_adata.obs['louvain'] = pd.Categorical((gmm_clus_index.T)[0].astype(np.str))
sample_labels_np = sample_labels.numpy().astype(np.int32)
#sample_labels_np = sample_labels_np.reshape(-1,1)
sample_labels_unique = np.unique(sample_labels_np)
for i in range(len(sample_labels_unique)):
    sample_labels_np[sample_labels_np==sample_labels_unique[i]] = i
prior_adata.obs['louvain'] = pd.Categorical(sample_labels_np.astype(np.str))
prior_adata.obs['louvain2'] = torch.tensor(gmm_clus_index.reshape(-1,1))
is_tensor= False
# save data as csv file
df = pd.DataFrame(data=prior_adata.obsm["X_umap"],  columns=["umap_dim1","umap_dim2"] , index=dataset.barcodes[0:len(sample_labels)])
df.insert(0,"labels",sample_labels)
df.to_csv(os.path.join(save_path,"multivae_umap_imputation_varianceGene.csv"))
# save impute data
df = pd.DataFrame(data=prior_adata.obsm["X_multi_vi"],  columns=["dim1","dim2","dim3","dim4","dim5","dim6","dim7","dim8","dim9","dim10"] , index=dataset.barcodes[0:len(sample_labels)])
df.to_csv(os.path.join(save_path,"multivae_latent_imputation_varianceGene.csv"))
#diff gene analysis
sc.tl.rank_genes_groups(prior_adata, 'louvain')
sc.pl.rank_genes_groups(prior_adata, n_genes=10, sharey=False)
diff_top_gene_set = prior_adata.uns['rank_genes_groups']
diff_top_gene_set = (diff_top_gene_set['names'])
diff_top_gene_set = diff_top_gene_set[0:25]
diff_top_gene_pvalue_set = prior_adata.uns['rank_genes_groups']
diff_top_gene_pvalue_set = (diff_top_gene_pvalue_set['pvals_adj'])
diff_top_gene_pvalue_set = diff_top_gene_pvalue_set[0:25]
diff_top_gene_foldchange_set = prior_adata.uns['rank_genes_groups']
diff_top_gene_foldchange_set = (diff_top_gene_foldchange_set['logfoldchanges'])
diff_top_gene_foldchange_set = diff_top_gene_foldchange_set[0:25]
diff_top_gene_matrix = np.array([])
diff_top_gene_pvalue_matrix = np.array([])
diff_top_gene_foldchange_matrix = np.array([])
for i in range(len(diff_top_gene_set.dtype.descr)):
    if i == 0:
        if is_tensor:
            diff_top_gene_matrix = diff_top_gene_set['(tensor('+str(i)+'),)'].astype(np.int32)
            diff_top_gene_pvalue_matrix = diff_top_gene_pvalue_set['(tensor('+str(i)+'),)'].astype(np.float)
            diff_top_gene_foldchange_matrix = diff_top_gene_foldchange_set['(tensor('+str(i)+'),)'].astype(np.float)
        else:
            diff_top_gene_matrix = diff_top_gene_set[str(i)].astype(np.int32)
            diff_top_gene_pvalue_matrix = diff_top_gene_pvalue_set[str(i)].astype(np.float)
            diff_top_gene_foldchange_matrix = diff_top_gene_foldchange_set[str(i)].astype(np.float)
    else:
        if is_tensor:
            diff_top_gene_matrix = np.vstack((diff_top_gene_matrix, diff_top_gene_set['(tensor('+str(i)+'),)'].astype(np.int32)))
            diff_top_gene_pvalue_matrix = np.vstack(
                (diff_top_gene_pvalue_matrix, diff_top_gene_pvalue_set['(tensor('+str(i)+'),)'].astype(np.float)))
            diff_top_gene_foldchange_matrix = np.vstack(
                (diff_top_gene_foldchange_matrix, diff_top_gene_foldchange_set['(tensor('+str(i)+'),)'].astype(np.float)))
        else:
            diff_top_gene_matrix = np.vstack((diff_top_gene_matrix,diff_top_gene_set[str(i)].astype(np.int32)))
            diff_top_gene_pvalue_matrix = np.vstack((diff_top_gene_pvalue_matrix, diff_top_gene_pvalue_set[str(i)].astype(np.float)))
            diff_top_gene_foldchange_matrix = np.vstack(
                (diff_top_gene_foldchange_matrix, diff_top_gene_foldchange_set[str(i)].astype(np.float)))
diff_top_gene_unique = np.unique(diff_top_gene_matrix.flatten())
# save diff atac matrix
df = pd.DataFrame(data=diff_top_gene_matrix)
df.to_csv(os.path.join(save_path,"gene_diff_matrix_adj_varianceGene.csv"))
df = pd.DataFrame(data=diff_top_gene_pvalue_matrix)
df.to_csv(os.path.join(save_path,"gene_diff_pvalue_matrix_adj_varianceGene.csv"))
df = pd.DataFrame(data=diff_top_gene_foldchange_matrix)
df.to_csv(os.path.join(save_path,"gene_diff_foldchange_matrix_adj_varianceGene.csv"))
df = pd.DataFrame(data=temp_rna.T, columns=dataset.barcodes, index=dataset.gene_names[:,0])
df.to_csv(os.path.join(save_path,"gene_imputation_varianceGene.csv"))
#gene clustering
gene_expression = temp_rna.T
for i in range(len(gene_expression)):
    gene_expression[i, :] = np.log1p(gene_expression[i, :]) / np.log1p(np.max(gene_expression[i, :])) + 0.0001
gene_name = dataset.gene_names[:,0]
cell_name = dataset.barcodes
gene_exp_data = geneDataset(gene_expression, gene_name, cell_name)
gene_exp_vae = VAE(gene_exp_data.nb_genes, n_batch=256)
gene_exp_trainer = UnsupervisedTrainer(
    gene_exp_vae,
    gene_exp_data,
    train_size=0.75,
    use_cuda=use_cuda,
    frequency=5,
)
is_test_pragram = False
if is_test_pragram:
    gene_exp_trainer.train(n_epochs=n_epochs, lr=lr)
    torch.save(gene_exp_trainer.model.state_dict(), '%s/gene_exp_trainer_p0_0.pkl' % save_path)

if os.path.isfile('%s/gene_exp_trainer_p0_0.pkl' % save_path):
    gene_exp_trainer.model.load_state_dict(torch.load('%s/gene_exp_trainer_p0_0.pkl' % save_path))
    gene_exp_trainer.model.eval()
else:
    gene_exp_trainer.train(n_epochs=n_epochs, lr=lr)
    torch.save(gene_exp_trainer.model.state_dict(), '%s/gene_exp_trainer_p0_0.pkl' % save_path)
# visulization
full = gene_exp_trainer.create_posterior(gene_exp_trainer.model, gene_exp_data, indices=np.arange(len(gene_exp_data)))
latent, batch_indices, labels = full.sequential().get_latent()
batch_indices = batch_indices.ravel()
imputed_values = full.sequential().imputation()
gene_exp_adata = anndata.AnnData(X=gene_exp_data.X)
gene_exp_adata.obsm["X_multi_vi"] = latent
sc.pp.neighbors(gene_exp_adata, use_rep="X_multi_vi", n_neighbors=15)
sc.tl.umap(gene_exp_adata, min_dist=0.1)
matplotlib.use('TkAgg')
fig, ax = plt.subplots(figsize=(7, 6))
sc.tl.louvain(gene_exp_adata)
sc.pl.umap(gene_exp_adata, color=['louvain'])

df = pd.DataFrame(data=gene_exp_adata.obsm["X_umap"],  columns=["umap_dim1","umap_dim2"] , index=gene_exp_data.gene_name_formulation)
df.insert(0,"gene_cluster",gene_exp_adata.obs['louvain'].as_matrix())
df.to_csv(os.path.join(save_path,"gene_cluster_umap_varianceGene.csv"))



#%% atac expression analysis
atac_expression = temp_atac.T
atac_name = dataset.atac_names
cell_name = dataset.barcodes
knn_matrix = prior_adata.add['neighbors']
knn_matrix = knn_matrix['distances']
cluster_ID = prior_adata.obs['louvain'].as_matrix()
cluster_ID_unique = np.unique(cluster_ID)
is_cluster_dataset = False
if is_cluster_dataset:
    cluster_ID_index = cluster_ID == cluster_ID_unique[0]
    knn_cluster_matrix = knn_matrix[np.ix_(cluster_ID_index, cluster_ID_index)].A
    #rank_knn = np.argpartition(knn_cluster_matrix, -10)[:, -10:]
    rank_knn = np.argpartition(knn_cluster_matrix, -3)[:, -3:]
    atac_cluster_expression = atac_expression[:, cluster_ID_index]
    for i in range(np.size(atac_cluster_expression, 1)):
        atac_cluster_expression[:, i] += np.sum(atac_cluster_expression[:, rank_knn[i, :]], axis=1)
    atac_cluster_expression[atac_cluster_expression > 1] = 1
else:
    for i in range(len(cluster_ID_unique)):
        cluster_ID_index = cluster_ID == cluster_ID_unique[i]
        knn_cluster_matrix = knn_matrix[np.ix_(cluster_ID_index, cluster_ID_index)].A
        #rank_knn = np.argpartition(knn_cluster_matrix, -10)[:, -10:]
        rank_knn = np.argpartition(knn_cluster_matrix, -3)[:, -3:]
        atac_cluster_expression = atac_expression[:, cluster_ID_index]
        for j in range(np.size(atac_cluster_expression, 1)):
            atac_cluster_expression[:, j] += np.sum(atac_cluster_expression[:, rank_knn[j, :]], axis=1)
        atac_cluster_expression[atac_cluster_expression > 1] = 1
        atac_expression[:,cluster_ID_index] = atac_cluster_expression

atac_exp_data = ATACDataset(atac_expression, atac_name, cell_name)
# atac top rank peak analysis
atac_prior_adata = anndata.AnnData(X=atac_exp_data.X.T)
atac_prior_adata.obs['louvain'] = prior_adata.obs['louvain']
#sc.tl.filter_rank_genes_groups(atac_prior_adata, min_fold_change=1.5)
sc.tl.rank_genes_groups(atac_prior_adata, 'louvain',n_genes=1000)
sc.pl.rank_genes_groups(atac_prior_adata, n_genes=10, sharey=False)
atac_diff_top_gene_set = atac_prior_adata.uns['rank_genes_groups']
atac_diff_top_gene_set = (atac_diff_top_gene_set['names'])
atac_diff_top_gene_set = atac_diff_top_gene_set[0:1000]
atac_diff_top_gene_pvalue_set = atac_prior_adata.uns['rank_genes_groups']
atac_diff_top_gene_pvalue_set = (atac_diff_top_gene_pvalue_set['pvals_adj'])
atac_diff_top_gene_pvalue_set = atac_diff_top_gene_pvalue_set[0:1000]
atac_diff_top_gene_foldchange_set = atac_prior_adata.uns['rank_genes_groups']
atac_diff_top_gene_foldchange_set = (atac_diff_top_gene_foldchange_set['logfoldchanges'])
atac_diff_top_gene_foldchange_set = atac_diff_top_gene_foldchange_set[0:1000]
atac_diff_top_gene_matrix = None
atac_diff_top_gene_pvalue_matrix = None
atac_diff_top_gene_foldchange_matrix = None
for i in range(len(atac_diff_top_gene_set.dtype.descr)):
    if i == 0:
        if is_tensor:
            atac_diff_top_gene_matrix = atac_diff_top_gene_set['(tensor('+str(i)+'),)'].astype(np.int32)
            atac_diff_top_gene_pvalue_matrix = atac_diff_top_gene_pvalue_set['(tensor('+str(i)+'),)'].astype(np.float)
            atac_diff_top_gene_foldchange_matrix = atac_diff_top_gene_foldchange_set['(tensor('+str(i)+'),)'].astype(np.float)
        else:
            atac_diff_top_gene_matrix = atac_diff_top_gene_set[str(i)].astype(np.int32)
            atac_diff_top_gene_pvalue_matrix = atac_diff_top_gene_pvalue_set[str(i)].astype(np.float)
            atac_diff_top_gene_foldchange_matrix = atac_diff_top_gene_foldchange_set[str(i)].astype(np.float)
    else:
        if is_tensor:
            atac_diff_top_gene_matrix = np.vstack((atac_diff_top_gene_matrix, atac_diff_top_gene_set['(tensor('+str(i)+'),)'].astype(np.int32)))
            atac_diff_top_gene_pvalue_matrix = np.vstack(
                (atac_diff_top_gene_pvalue_matrix, atac_diff_top_gene_pvalue_set['(tensor('+str(i)+'),)'].astype(np.float)))
            atac_diff_top_gene_foldchange_matrix = np.vstack(
                (atac_diff_top_gene_foldchange_matrix, atac_diff_top_gene_foldchange_set['(tensor('+str(i)+'),)'].astype(np.float)))
        else:
            atac_diff_top_gene_matrix = np.vstack((atac_diff_top_gene_matrix,atac_diff_top_gene_set[str(i)].astype(np.int32)))
            atac_diff_top_gene_pvalue_matrix = np.vstack((atac_diff_top_gene_pvalue_matrix,atac_diff_top_gene_pvalue_set[str(i)].astype(np.float)))
            atac_diff_top_gene_foldchange_matrix = np.vstack((atac_diff_top_gene_foldchange_matrix, atac_diff_top_gene_foldchange_set[str(i)].astype(np.float)))
atac_diff_clu_matrix = atac_diff_top_gene_matrix
atac_diff_clu_unique = np.unique(atac_diff_clu_matrix.flatten())
atac_clu_semi_supvised_index = np.ones((1,len(atac_name))).flatten()*-1
for i in range(len(atac_diff_clu_matrix)):
    atac_clu_semi_supvised_index[atac_diff_clu_matrix[i,:]] = i
atac_diff_top_gene_matrix =atac_diff_top_gene_matrix[:,0:200]
atac_diff_top_gene_unique = np.unique(atac_diff_top_gene_matrix.flatten())
# save diff atac
df = pd.DataFrame(data=atac_diff_clu_unique,  columns=["diffatac"] , index=atac_exp_data.atac_name_formulation[atac_diff_clu_unique])
df.to_csv(os.path.join(save_path,"atac_diff_set_varianceGene.csv"))
# save diff atac matrix
df = pd.DataFrame(data=atac_diff_clu_matrix)
df.to_csv(os.path.join(save_path,"atac_diff_matrix_varianceGene.csv"))
df = pd.DataFrame(data=atac_diff_top_gene_pvalue_matrix)
df.to_csv(os.path.join(save_path,"atac_diff_pvalue_matrix_varianceGene.csv"))
df = pd.DataFrame(data=atac_diff_top_gene_foldchange_matrix)
df.to_csv(os.path.join(save_path,"atac_diff_foldchange_matrix_varianceGene.csv"))
df = pd.DataFrame(data=atac_exp_data.X, columns=atac_exp_data.cell_name_formulation, index=atac_exp_data.atac_name_formulation)
df.to_csv(os.path.join(save_path,"atac_aggregation_varianceGene.csv"))



# diff gene cluster
comp_atac_diff_clu_unique = np.array([i for i in range(len(atac_expression)) if i not in atac_diff_clu_unique])
cell_clu_index = prior_adata.obs['louvain'].as_matrix().astype(np.int32)
gene_clu_index = np.ones(np.max(diff_top_gene_matrix.flatten().flatten())+1)
atac_clu_semi_supvised_index[comp_atac_diff_clu_unique] = len(cluster_ID_unique)+1
atac_clu_index = atac_clu_semi_supvised_index.astype(np.int32)
#atac_clu_index = atac_exp_adata.obs['louvain'].as_matrix().astype(np.int32)
diff_gene_clu_index = gene_clu_index[diff_top_gene_matrix.flatten()].reshape(diff_top_gene_matrix.shape)
diff_atac_clu_index = atac_clu_index[atac_diff_top_gene_matrix.flatten()].reshape(atac_diff_top_gene_matrix.shape)
diff_gene_exp = prior_adata.X[:,diff_top_gene_matrix.flatten()].T
for i in range(np.shape(diff_gene_exp)[0]):
    diff_gene_exp[i,:] = np.log1p(diff_gene_exp[i,:])/np.max(np.log1p(diff_gene_exp[i,:]))
norm_diff_gene_exp = diff_gene_exp
diff_atac_exp = atac_prior_adata.X[:,atac_diff_top_gene_matrix.flatten()].T
max_value = np.max(diff_gene_clu_index.flatten())
for i in range(100):
    diff_gene_exp = np.hstack((diff_gene_exp,diff_gene_clu_index.reshape(-1,1)/max_value))
cell_marker_index = np.array([i//diff_top_gene_matrix.shape[1] for i in range(diff_top_gene_matrix.size)])
cell_marker_index = cell_marker_index/np.max(cell_marker_index)
for i in range(100):
    diff_gene_exp = np.hstack((diff_gene_exp,cell_marker_index.reshape(-1,1)))
for i in range(10):
    diff_gene_exp = np.vstack((diff_gene_exp,np.append(cell_clu_index/np.max(cell_clu_index),np.ones(200)*-1/np.max(cell_clu_index))))
diff_gene_exp_sort = diff_gene_exp.T[diff_gene_exp.T[:,-1].argsort()].T
ax = sns.heatmap(diff_gene_exp_sort,xticklabels=False,cmap='rainbow')
plt.show()
ax = sns.clustermap(diff_gene_exp_sort,xticklabels=False,metric="correlation",cmap='rainbow')
plt.show()
ax = sns.clustermap(diff_gene_exp_sort,xticklabels=False,cmap='rainbow')
plt.show()

diff_gene_atac_clu_index = np.append(diff_gene_clu_index.flatten(),np.max(diff_gene_clu_index.flatten())+diff_atac_clu_index.flatten())
diff_gene_atac_exp = np.vstack((norm_diff_gene_exp,diff_atac_exp))
max_mixed_value = np.max(diff_gene_atac_clu_index)
for i in range(100):
    diff_gene_atac_exp = np.hstack((diff_gene_atac_exp, diff_gene_atac_clu_index.reshape(-1, 1) / max_mixed_value))
cell_gene_marker_index = np.array([i//(diff_top_gene_matrix.shape[1]) for i in range(diff_top_gene_matrix.size)])
cell_atac_marker_index = np.max(cell_gene_marker_index)+np.array([i//(atac_diff_top_gene_matrix.shape[1]) for i in range(atac_diff_top_gene_matrix.size)])
cell_marker_index = np.append(cell_gene_marker_index,cell_atac_marker_index)/np.max(cell_atac_marker_index)
for i in range(100):
    diff_gene_atac_exp = np.hstack((diff_gene_atac_exp,cell_marker_index.reshape(-1,1)))
for i in range(50):
    diff_gene_atac_exp = np.vstack((diff_gene_atac_exp,np.append(cell_clu_index/np.max(cell_clu_index),np.ones(200)*-1/np.max(cell_clu_index))))
diff_gene_atac_exp_sort = diff_gene_atac_exp.T[diff_gene_atac_exp.T[:,-1].argsort()].T
ax = sns.heatmap(diff_gene_atac_exp_sort,xticklabels=False,cmap='rainbow')
plt.show()
ax = sns.clustermap(diff_gene_atac_exp_sort,xticklabels=False,metric="correlation",cmap='rainbow')
plt.show()
ax = sns.clustermap(diff_gene_atac_exp_sort,xticklabels=False,cmap='rainbow')
plt.show()
#%% atac clustering
atac_exp_filter_data = ATACDataset(atac_expression[comp_atac_diff_clu_unique,:], atac_name[comp_atac_diff_clu_unique], cell_name)
atac_exp_vae = VAE(atac_exp_filter_data.nb_genes, n_batch=256)
atac_exp_trainer = UnsupervisedTrainer(
    atac_exp_vae,
    atac_exp_filter_data,
    train_size=0.75,
    use_cuda=use_cuda,
    frequency=5,
)
is_test_pragram = False
atac_exp_trainer.train(n_epochs=n_epochs, lr=lr)
torch.save(atac_exp_trainer.model.state_dict(), '%s/atac_exp_trainer_p0_0.pkl' % save_path)
# visulization
full = atac_exp_trainer.create_posterior(atac_exp_trainer.model, atac_exp_filter_data, indices=np.arange(len(atac_exp_filter_data)))
latent, batch_indices, labels = full.sequential().get_latent()
batch_indices = batch_indices.ravel()
imputed_values = full.sequential().imputation()
atac_exp_adata = anndata.AnnData(X=atac_exp_filter_data.X)
atac_exp_adata.obsm["X_multi_vi"] = latent
sc.pp.neighbors(atac_exp_adata, use_rep="X_multi_vi", n_neighbors=15)
sc.tl.umap(atac_exp_adata, min_dist=0.1)
matplotlib.use('TkAgg')
fig, ax = plt.subplots(figsize=(7, 6))
sc.tl.louvain(atac_exp_adata)
sc.pl.umap(atac_exp_adata, color=['louvain'])
# save data as csv file
df = pd.DataFrame(data=atac_exp_adata.obsm["X_umap"],  columns=["umap_dim1","umap_dim2"] , index=atac_exp_filter_data.atac_name_formulation)
df.insert(0,"atac_cluster",atac_exp_adata.obs['louvain'].as_matrix())
df.to_csv(os.path.join(save_path,"atac_cluster_umap_varianceGene.csv"))

post_adata = anndata.AnnData(X=dataset.X)
#post_adata = anndata.AnnData(X=dataset.atac_expression)
post_adata.obsm["X_multi_vi"] = latent
#post_adata.obs['cell_type'] = np.array([dataset.cell_types[dataset.labels[i][0]]
#                                        for i in range(post_adata.n_obs)])
post_adata.obs['cell_type'] = torch.tensor(np.array(cluster_index).reshape(-1,1))
sc.pp.neighbors(post_adata, use_rep="X_multi_vi", n_neighbors=15)
sc.tl.umap(post_adata, min_dist=0.1)

#sc.pp.neighbors(post_adata, use_rep="X_multi_vi", n_neighbors=20, metric="correlation")
#sc.tl.umap(post_adata, min_dist=0.3)

#%% postior gmm
matplotlib.use('TkAgg')
fig, ax = plt.subplots(figsize=(7, 6))
sc.pl.umap(post_adata, color=["cell_type"], ax=ax, show=show_plot)
print(matplotlib.get_backend())

clust_index_posterior_gmm = trainer.model.init_gmm_params(latent)
post_adata.obs['cell_type'] = torch.tensor(clust_index_posterior_gmm.reshape(-1,1))
sc.pp.neighbors(post_adata, use_rep="X_multi_vi", n_neighbors=15)
sc.tl.umap(post_adata, min_dist=0.1)
fig, ax = plt.subplots(figsize=(7, 6))
sc.pl.umap(post_adata, color=["cell_type"], ax=ax, show=show_plot)

#%% postior label


post_adata.obs['cell_type'] = torch.tensor(labels.reshape(-1,1))
sc.pp.neighbors(post_adata, use_rep="X_multi_vi", n_neighbors=15)
sc.tl.umap(post_adata, min_dist=0.1)
fig, ax = plt.subplots(figsize=(7, 6))
sc.pl.umap(post_adata, color=["cell_type"], ax=ax, show=show_plot)
# save data as csv file
df = pd.DataFrame(data=post_adata.obsm["X_umap"],  columns=["umap_dim1","umap_dim2"] , index=dataset.barcodes)
df.insert(0,"labels",labels )
df.to_csv(os.path.join(save_path,"multivae_umap_varianceGene.csv"))

# train set clustering
sample_latents = torch.tensor([])
samples = torch.tensor([])
for tensors_list in trainer.data_loaders_loop():
    sample_batch, local_l_mean, local_l_var, batch_index, y, sample_atac_batch = zip(*tensors_list)
    temp_samples = trainer.model.get_latents(*sample_batch, *y, *sample_atac_batch) #check this expression
    samples = torch.cat((samples, sample_batch[0].float()))
    for temp_sample in temp_samples:
        sample_latents = torch.cat((sample_latents, temp_sample[2].float()))

clust_index_gmm = trainer.model.init_gmm_params(sample_latents.detach().numpy())
prior_adata = anndata.AnnData(X=samples.detach().numpy())
prior_adata.obsm["X_multi_vi"] = sample_latents.detach().numpy()
prior_adata.obs['cell_type'] = torch.tensor(clust_index_gmm.reshape(-1,1))
sc.pp.neighbors(prior_adata, use_rep="X_multi_vi", n_neighbors=15)
sc.tl.umap(prior_adata, min_dist=0.1)
#%%
matplotlib.use('TkAgg')
fig, ax = plt.subplots(figsize=(7, 6))
sc.pl.umap(prior_adata, color=["cell_type"], ax=ax, show=show_plot)


sc.tl.louvain(post_adata)
sc.pl.umap(post_adata, color=['louvain'])
print(matplotlib.get_backend())

#%% imputation

