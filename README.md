# BiasedUserHistorySynthesis

Source for Biased User History Synthesis for Personalized Long Tail Item Recommendation

## Create the conda environment

conda create --name ENV_NAME --file conda_requirements.txt python=3.7 \
conda activate ENV_NAME \
conda install pip \
python3 -m pip install -r pip_requirements.txt


## Reproducing Results

The default branch of this repository is called **BaseModel**. It contains the code for the Two Tower Neural Network Base Recommendation System. The other branch in this repository is called **BiasedUserHistorySynthesis**. It contains the code for BiasedUserHistorySynthesis built on top of a Two Tower Neural Network. The main results in the paper for the Base Model and all variants of BiasedUserHistorySynthesis can be reproduced by running the following shell scripts:

### Base Model

**Base Two Tower Neural Network on MovieLens1m** -  bash ml1m_ttnn_basemodel.sh\
**Base Two Tower Neural Network on BookCrossing** - bash bx_ttnn_basemodel.sh

### Biased User History Synthesis

**BUHS-Mean on MovieLens1m**  -  bash ml1m_ttnn_buhs_mean.sh \
**BUHS-Attn on MovieLens1m**  -  bash ml1m_ttnn_buhs_attn.sh \
**BUHS-GRU on MovieLens1m**   -  bash ml1m_ttnn_buhs_gru.sh \
**BUHS-Mean on BookCrossing** -  bash bx_ttnn_buhs_mean.sh \
**BUHS-Attn on BookCrossing** -  bash bx_ttnn_buhs_attn.sh \
**BUHS-GRU on BookCrossing**  -  bash bx_ttnn_buhs_gru.sh \





