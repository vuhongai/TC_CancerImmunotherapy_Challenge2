# Challenge 2 submission - Ai VU HONG (vuhongai)

## 1 Overview
In order to predict the 5-state proportions of different knockout in the test set, I learned the minimal presentation of all 15077 genes (gene embedding - 32-dimension vector) from provided scRNA-seq data. The [scETM model](https://github.com/hui2000ji/scETM) was used to trained and extract gene embedding. Next, multiple fully-connected neural networks was trained mapping 32-D gene embedding vector to 5-D output vector, resulting multiple predictions for each gene. These predictions were then filtered and averaged for final submission. 

By running the notebook [step3_prediction.ipynb](./step3_prediction.ipynb), you will able to reproduce the 3 required .csv files. 

## 2 Installation
Python version: 3.10.8

Create and activate new virtual environment for the project:
```bash
anaconda3/bin/conda create -n tcells python=3.10.8
source anaconda3/bin/activate tcells
```

Install dependencies

Note: Make sure that torch is already installed before installing scETM
If you don't want to re-train the scETM model to extract gene embedding, you can ignore the step 1 and not install torch or scETM, the precomputed embedding for all 15077 genes are available.

```bash
pip install torch torchvision torchaudio
pip install tensorflow
pip install scETM anndata scanpy pandas numpy
pip install -U scikit-learn
```

## 3 Usage

### Training scETM model for gene and topic embeddings
The code used for training scETM model can be found in [here](./step1_gene_embedding_extraction.ipynb)
A notable modification from original model is that the size of gene embedding reduced from 400 to 32 in this case. The model was trained for 12000 epoches, the checkpoints can be found in [here](./submission/checkpoints/scETM_01_14-12_57_32/model-12000)

Before running the notebook, please make sure that the scRNA-seq data (sc_training.h5ad file) is downloaded in the [data folder](./data/).

Gene embedding of all 15077 genes was precomputed and save in [here](./submission/embedding/gene_embedding_32.npy). However, it can also be done by calling scETM model as a Pytorch model:
```python
import torch
from scETM import scETM

# define scETM model
model = scETM(adata.n_vars, # number of genes/variables
              n_batches=4, 
              trainable_gene_emb_dim=32,
             )

# load pretrained model
model.load_state_dict(torch.load("./submission/checkpoints/scETM_01_14-12_57_32/model-12000"))
model.eval()

# calculate and save the gene embedding vector
model.get_all_embeddings_and_nll(adata)
gene_embedding = np.array(adata.varm['rho'])
np.save(f"./submission/embedding/gene_embedding_{emb_dim}", gene_embedding)
```

### Training multiple perceptron to predict 5-state proportion
The code used for training, evaluating and predicting cell states can be found in [here](./step2_neural_network.ipynb). Briefly, 10-fold cross-validation of the training set of 64 genes was trained to map 32-dimension gene embedding vector to 5-dimension output vector. Each k-fold train/val set generated 60 different models, by tuning different hyperparameters, which can be found in the notebook. In total, 600 models were generated, and ranked by validation MAE [score](./submission/predictions/), its weights can be found in this [folder](./submission/checkpoints/NN/).

### Select best prediction based on similarity
Despite the score on validation set from previous step is quite good (mae_val<0.1), the predictions on each gene varied a lot from model to model. Therefore, it is neccesary to filter rather directly averaging all generated predictions. I did that by selecting 2 predictions from 2 best models for each k-fold training based on mae_val (with mae_val < 0.1), resulting a list of total 20 predictions. Each of these predictions (5-dimension vector) was calculated L1 distance (in fact I used mean_absolute_error in the script so it'll be L1_distance/5) with other 19 vectors, counted how many other vectors with mae<0.06 and selecting the one(s) with highest number of similar predictions. This selection is iteratively repeated if the final list contains vectors that are not similar in our definition. The average of the this selection will be submitted. By running [step3_prediction.ipynb notebook](./step3_prediction.ipynb), you can reproduce the all the required .csv files.