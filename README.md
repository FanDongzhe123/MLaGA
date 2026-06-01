# MLaGA
## Installation
```bash
conda create -n mlaga python=3.10
conda activate mlaga
pip3 install torch  --index-url https://download.pytorch.org/whl/cu118

#install required packages
pip install -r requirements.txt

#install flash attention, you can also install the right version from https://github.com/dao-ailab/flash-attention 
pip install flash-attn --no-build-isolation

#install pyg
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

#install dgl
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html
```

## Train SMA
```bash
conda activate mlaga
cd SMA
bash Train_mix_SMA.sh
```

## Prepare CLIP features
```bash
conda activate mlaga
cd SMA
python CLIP_feature.py
```

## Train MMGIT
```bash
conda activate mlaga
bash train_MLaGA.sh

#for evaluation
bash eval_MLaGA.sh
```
