# testing the multi-modal vae method by scRNA-seq and scATAC data date: 11/12/2019
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
#pygui(True)
#mpl.use('WX')
import matplotlib.pyplot as plt
from scvi.dataset import CortexDataset, snareDataset, scienceDataset
from scvi.models import VAE
from scvi.inference import UnsupervisedTrainer
from scvi.inference import MultiPosterior, MultiTrainer
import torch

from scvi.models.multi_vae import Multi_VAE

## Visualizing the latent space with scanpy
import scanpy as sc
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

    '''
    # Filter atac data
    high_count_atacs = (dataset.atac_expression > 0).sum(axis=0).ravel() > 0.01 * dataset.atac_expression.shape[0]
    dataset.atac_expression = dataset.atac_expression[:, high_count_atacs]
    dataset.atac_names = dataset.atac_names[high_count_atacs]
    '''

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
    torch.save(pre_trainer.model.state_dict(), '%s/pre_trainer6.pkl' % save_path)

if os.path.isfile('%s/pre_trainer6.pkl' % save_path):
    pre_trainer.model.load_state_dict(torch.load('%s/pre_trainer6.pkl' % save_path))
    pre_trainer.model.eval()
else:
    #pre_trainer.model.init_gmm_params(dataset)
    pre_trainer.train(n_epochs=n_epochs, lr=lr)
    torch.save(pre_trainer.model.state_dict(), '%s/pre_trainer6.pkl' % save_path)

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
prior_adata.obs['cell_type'] = sample_labels.numpy()
sc.pp.neighbors(prior_adata, use_rep="X_multi_vi", n_neighbors=15)
sc.tl.umap(prior_adata, min_dist=0.1)
matplotlib.use('TkAgg')
fig, ax = plt.subplots(figsize=(7, 6))
sc.pl.umap(prior_adata, color=["cell_type"], ax=ax, show=show_plot)
# scvi gmm clustering
prior_adata.obs['cell_type'] = clust_index_gmm
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
    torch.save(trainer.model.state_dict(), '%s/multi_vae_21.pkl' % save_path)

#if os.path.isfile('%s/multi_vae_1.pkl' % save_path):
#    trainer.model.load_state_dict(torch.load('%s/multi_vae_1.pkl' % save_path))
#    trainer.model.eval()
if os.path.isfile('%s/multi_vae_21.pkl' % save_path):
    trainer.model.load_state_dict(torch.load('%s/multi_vae_21.pkl' % save_path))
    trainer.model.eval()
    #trainer.train(n_epochs=n_epochs, lr=lr)
    #torch.save(trainer.model.state_dict(), '%s/multi_vae_3.pkl' % save_path)
else:
    #trainer.model.init_gmm_params(dataset)
    trainer.train(n_epochs=n_epochs, lr=lr)
    torch.save(trainer.model.state_dict(), '%s/multi_vae_21.pkl' % save_path)

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

#Similarly, it is possible to query the imputed values via the `imputation` method of the posterior object. **Note for advanced users:** imputation is an ambiguous term and there are two ways to perform imputation in scVI. The first way is to query the **mean of the negative binomial** distribution modeling the counts. This is referred to as `sample_rate` in the codebase and can be reached via the `imputation` method. The second is to query the **normalized mean of the same negative binomial** (please refer to the scVI manuscript). This is referred to as `sample_scale` in the codebase and can be reached via the `get_sample_scale` method. In differential expression for example, we of course rely on the normalized latent variable which is corrected for variations in sequencing depth.
imputed_values = full.sequential().imputation()
#normalized_values = full.sequential().get_sample_scale()
sample_latents = torch.tensor([])
sample_labels = torch.tensor([])
temp_rna = imputed_values[0]
temp_atac = imputed_values[1]
temp_label = []
if len(imputed_values) == 3:
    temp_label = imputed_values[2]
for tensors_list in range(int(len(imputed_values[0])/256)):
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
prior_adata = anndata.AnnData(X=temp_rna[0:int(len(imputed_values[0])/256)*256,:])
prior_adata.obsm["X_multi_vi"] = sample_latents.detach().numpy()
prior_adata.obs['cell_type'] = clust_index_gmm
sc.pp.neighbors(prior_adata, use_rep="X_multi_vi", n_neighbors=15)
sc.tl.umap(prior_adata, min_dist=0.1)
matplotlib.use('TkAgg')
fig, ax = plt.subplots(figsize=(7, 6))
sc.pl.umap(prior_adata, color=["cell_type"], ax=ax, show=show_plot)
# imputation labels
prior_adata.obs['cell_type'] = sample_labels
sc.pp.neighbors(prior_adata, use_rep="X_multi_vi", n_neighbors=15)
sc.tl.umap(prior_adata, min_dist=0.1)
matplotlib.use('TkAgg')
fig, ax = plt.subplots(figsize=(7, 6))
sc.pl.umap(prior_adata, color=["cell_type"], ax=ax, show=show_plot)


#%% clustering

post_adata = anndata.AnnData(X=dataset.X)
#post_adata = anndata.AnnData(X=dataset.atac_expression)
post_adata.obsm["X_multi_vi"] = latent
#post_adata.obs['cell_type'] = np.array([dataset.cell_types[dataset.labels[i][0]]
#                                        for i in range(post_adata.n_obs)])
post_adata.obs['cell_type'] = np.array(cluster_index)
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
post_adata.obs['cell_type'] = clust_index_posterior_gmm
sc.pp.neighbors(post_adata, use_rep="X_multi_vi", n_neighbors=15)
sc.tl.umap(post_adata, min_dist=0.1)
fig, ax = plt.subplots(figsize=(7, 6))
sc.pl.umap(post_adata, color=["cell_type"], ax=ax, show=show_plot)

#%% postior label


post_adata.obs['cell_type'] = labels
sc.pp.neighbors(post_adata, use_rep="X_multi_vi", n_neighbors=15)
sc.tl.umap(post_adata, min_dist=0.1)
fig, ax = plt.subplots(figsize=(7, 6))
sc.pl.umap(post_adata, color=["cell_type"], ax=ax, show=show_plot)

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
prior_adata.obs['cell_type'] = clust_index_gmm
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

