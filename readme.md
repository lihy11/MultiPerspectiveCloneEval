# Multi-Perspective Evaluation for Code Clone Detection
## 1. Artifact Description:
this is the source code and dataset of paper "Assessing and Improving Dataset and Evaluation
Methodology in Deep Learning for Code Clone
Detection". The paper is published on ISSRE 2023([Camera Ready Version](./paper.pdf)).
### 1.1  Repo Structure
- dataset: the source code of the two datasets.
  - cbcb: Conbigclonebench
    - code0 ~ code3: codes after code abstract described in the paper.
  - gcj:  Googlecodejam2
    - code0 ~ code3 similar to cbcb.
  - train_split: split the data into train test eval from different perspective described in the paper.
  - pre_train:  data for pretrained models.
- ASTNN: a rnn model based on ast and its subtrees.
- TBCCD: a cnn model based on ast with well-designed token embedding named PACE.
- FAAST: a gnn model based on ast which argumented by control flow and data flow.
- MLPModel: MLP model using bag-of-words and MLP models to detect clones.
- UnixCoder: pre-trained multi modal code model.
- GraphCodeBert: pre-trained model using dataflow information.


## 2. Environment Setup:
We conduct experiments with 6 deeplearning models, each of them need different running environment. We list all the dependencies in the experiment as below:
##### ASTNN
python == 3.7.13
pytorch == 1.9.0 + cuda10.2
javalang == 0.13.0
pandas == 1.3.5
numpy == 1.21.5
scikit-learn == 1.0.2
##### FAAST
python == 3.7.13
pytorch == 1.9.0 + cuda10.2
javalang == 0.13.0
numpy == 1.21.5
torch-geometric == 2.0.3
anytree == 2.8.0
##### TBCCD
python == 3.6.13
javalang == 0.13.0
numpy == 1.21.5
gensim == 3.5.0
tensorflow-gpu == 1.15.0
scikit-learn == 0.24.2
##### MLP Model
python=3.7.13
pytorch == 1.9.0 + cuda10.2
javalang == 0.13.0
numpy == 1.21.5
scikit-learn == 1.0.2
gensim == 3.5.0
##### Pre-trained Models(UnixCoder and GraphCodeBERT)
python == 3.7.13
pytorch == 1.11.0 + cuda11.3
huggingfacehub == 0.14.1


## 3.  Getting Started:
This section, We use MLP Model to illustrate the usage of the dataset and evaluation method.
### 3.1  unzip raw dataset.
For linux system, you can use tool 7z and commands below to unzip cbcb.7z, to run the models, you need to unzip all the .7z file the the folder.
```bash
sudo apt install p7zip-full
cd dataset
7z x cbcb.7z
```
For windows system, you can use software 7-Zip or other tools to unzip the archives.
### 3.2  experiment settings
All the dependency is list in the section **Enviroment Settup**.  We recommand environment management software like *conda* or *virtual env* to organize running environments.  We will show an example using conda to create an python enviroment for the MLP Model on the Linux.
#### 3.2.1 install conda
You can install conda with Anaconda or MiniConda easily.
#### 3.2.2 create environment
```bash
conda create -n mlp python=3.7.13
conda activate mlp
``` 
#### 3.2.3 install dependencies
```bash
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install javalang==0.13.0 numpy==1.21.5 gensim==3.5.0
pip install -U scikit-learn=1.0.2
```
#### 3.2.4 train model
You can use **run.sh** to start training for every model. 
```bash
cd MLPModel
sh run.sh
``` 
There are 4 arguments in run.sh:
- dataset: cbcb | gcj, choose a dataset
- split: random0 | pro0 | fun0 | all0, choose a evaluation perspective
- cross: 0 | 1 | 2 | 3, choose a code abstract level
- cuda: cuda device


## 4. Reproducibility Instructions: 
We conducted experiments on 6 models and 4 evaluation perspectives on a machine with 128GB RAM, 3\*Nvidia 1080ti in 6 weeks, every model can work within 32GB RAM and 1\*Nvidia 1080ti.  Reproduce all results may take a few weeks dependending on your device.
If you want to reproduce the result of MLP Model in random-view on dataset gcj， you can simply run the bash in `MLPModel/run.sh`， set the arguments to`DATASET="gcj", CROSS=0, SPLIT="random0"` 



## 5. Addition Information: F1 score on cross-dataset experiment

Here we conduct cross-dataset experiments between cbcb and bcb, gcj2 and gcj.  where gcj represents GoogleCodeJam, gcj2 represents GoogleCodeJam2, bcb represents BigCloneBench, cbcb represents ConBigCloneBench.  We select 3 task specific model: ASTNN, TBCCD, FA-AST

||ASTNN|TBCCD|FA-AST|
|-|-|-|-|
|train: gcj; test: gcj2|0.319|0.141|0.386|
|train: gcj2; test: gcj|0.873|0.512|0.886|
|$\Delta$ gcj   |0.436|0.369|0.500|
|train bcb test: cbcb|0.796|0.805|0.789 |
|train:cbcb test:bcb | 0.935|0.956| 0.963|
|$\Delta$ bcb   | 0.139|0.149|0.174|




