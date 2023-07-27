# BiasedUserHistorySynthesis

Source for Biased User History Synthesis for Personalized Long Tail Item Recommendation

## Create the conda environment

conda create --name ENV_NAME --file conda_requirements.txt python=3.7 \
conda activate ENV_NAME \
conda install pip \
python3 -m pip install -r requirements.txt

## Downloading the datasets

We use two public benchmark datasets: [MovieLens1m](https://grouplens.org/datasets/movielens/1m/) and [BookCrossing](http://www2.informatik.uni-freiburg.de/~cziegler/BX/). Download the datasets into a datasets subdirectory as follows: PATH/TO/DATASETS_DIR/ml-1m and PATH/TO/DATASETS/BookCrossing, and subsequently seting the following environment variable:

export DATASETS_DIR=PATH/TO/DATASETS


## Reproducing Results

The default branch of this repository is called **BaseModel**. It contains the code for the Two Tower Neural Network Base Recommendation System. The other branch in this repository is called **BiasedUserHistorySynthesis**. It contains the code for BiasedUserHistorySynthesis built on top of a Two Tower Neural Network. The main results in the paper for the Base Model and all variants of BiasedUserHistorySynthesis can be reproduced by running the following shell scripts:

| Dataset | Model | Command |
| ------- | ----- | ------- |
| MovieLens-1m | Base Two Tower Neural Network | bash ml1m_ttnn_basemodel.sh |
| MovieLens-1m | BUHS-Mean + TTNN | bash ml1m_ttnn_buhs_mean.sh |
| MovieLens-1m | BUHS-Attn + TTNN | bash ml1m_ttnn_buhs_attn.sh |
| MovieLens-1m | BUHS-GRU + TTNN | bash ml1m_ttnn_buhs_gru.sh |
| BookCrossing | Base Two Tower Neural Network | bash bx_ttnn_basemodel.sh |
| BookCrossing | BUHS-Mean + TTNN | bash bx_ttnn_buhs_mean.sh |
| BookCrossing | BUHS-Mean + TTNN | bash bx_ttnn_buhs_attn.sh |
| BookCrossing | BUHS-Mean + TTNN | bash bx_ttnn_buhs_gru.sh |





