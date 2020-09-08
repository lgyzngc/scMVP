## scMVP - single cell multi-view processor

scMVP is a python toolkit for joint profiling of scRNA and scATAC data profiling and analysis
with multi-modal VAE model.

### Installation
**Environment requirements:**<br>
scMVP requires Python3.7.x and [**Pytorch**](http://pytorch.org).<br>
For example, use [**miniconda**](https://conda.io/miniconda.html) to install python and pytorch of CPU or GPU version.
```Bash
conda install -c pytorch python3.7 pytorch
```

Then you can install scMVP from github repo:<br>
```Bash
# first move to your target directory
git clone https://github.com/bm2-lab/scMVP.git
cd scMVP/
python setup.py install
```

Try ```import scMVP``` in your python console and start your first [**tutorial**](demos/scMVP_tutorial.ipynb) with scMVP!

### User tutorial

1. Using scMVP for sci-CAR cell line mixture. [demo](demos/scMVP_tutorial.ipynb)
- Basic analysis modules with multi-VAE.

2. Using scMVP for snare-seq mouse cerebral cortex P0 dataset.[demo](demos/scMVP_regress_tutorial.ipynb)
- Perform CRE-gene analysis with PLS-regression.

3. Using scMVP on customize joint profiling dataset.[demo]
- Load and analyze your own data.

### Reference
scMVP: an integrative generative model for joint profiling of single cell RNA-seq and ATAC-seq data. 2020

