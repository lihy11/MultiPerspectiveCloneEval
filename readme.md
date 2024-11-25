# Multi-Perspective Evaluation for Code Clone Detection[![DOI](https://zenodo.org/badge/681138002.svg)](https://zenodo.org/badge/latestdoi/681138002)
# Something New!
This is the extended version (4 language) of GCJ2, please refer to this dataset for the meta info of all these problems.
https://huggingface.co/datasets/lihy11/GCJ2-4lang
In subsequent work, we further filtered out a small amount of low-quality data (encoding issues, excessively long or short lengths), leaving 12,447 Java data entries. This is 13 samples fewer compared to that in the paper. Our experiments indicate that these data are insufficient to impact the experimental conclusions of the paper. You can use `pandas` to read this data and filter the Java language data we used. You can also try using data from more languages for cross-language clone detection.
```python
# pip install pandas
df = pandas.read_pickle('./dataset/gcj4.pkl')
df.head()
```
meta data for the df
```json
{
'index': "index of the sample in the table",
'year': "competetion year",
'round': "competetion round",
'username': "user name",
'task': "task name of the problem, we collected them from the GCJ website manualy",
'solution': "1 means that this solution is passed.",
'file': "file path when upload to GCJ website",
'flines': "code content",
'lan': "program language",
'funid': "id from 0 to 12446",
'lines': "line count of code"
}
```
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
  - train_split: split the data into train test eval from different perspectives described in the paper.
  - pre_train:  data for pre-trained models.
- ASTNN: a rnn model based on AST and its subtrees.
- TBCCD: a cnn model based on AST with well-designed token embedding named PACE.
- FAAST: a gnn model based on AST, which is argumented by control flow and data flow.
- MLPModel: MLP model using bag-of-words and MLP models to detect clones.
- UnixCoder: pre-trained multi-modal code model.
- GraphCodeBert: pre-trained model using dataflow information.

## 2. Environment Setup:
### 2.1 Hardware requirement
We conducted experiments on a machine with 128GB RAM and 3\*Nvidia 1080ti(11GB memory each).  ASTNN, FAAST, TBCCD and MLP Model can work within 32GB RAM and 1\*Nvidia 1080ti.  GraphCodeBERT and UnixCoder need 32GB RAM and 3\*Nvidia 1080ti to run.  
### 2.2 OS requirement
All models can run on  Windows and Linux. If you need to use CUDA, we recommend using Linux for experiments. Our experiments were conducted on Ubuntu 18.04.
### 2.3 Software requirement
We conduct experiments with 6 deep learning models, each of them needs different running environment. We list all the dependencies in the experiment as below:
##### ASTNN
- python == 3.7.13
- pytorch == 1.9.0 + cuda10.2
- javalang == 0.13.0
- pandas == 1.3.5
- numpy == 1.21.5
- scikit-learn == 1.0.2
##### FAAST
- python == 3.7.13
- pytorch == 1.9.0 + cuda10.2
- javalang == 0.13.0
- numpy == 1.21.5
- torch-geometric == 2.0.3
- anytree == 2.8.0
##### TBCCD
- python == 3.6.13
- javalang == 0.13.0
- numpy == 1.19.5
- gensim == 3.5.0
- tensorflow-gpu == 1.15.0
- cudatoolkit==10.0.130 
- scikit-learn == 0.24.2
##### MLP Model
- python=3.7.13
- pytorch == 1.9.0 + cuda10.2
- javalang == 0.13.0
- numpy == 1.21.5
- scikit-learn == 1.0.2
- gensim == 3.5.0
##### Pre-trained Models(UnixCoder and GraphCodeBERT)
- python == 3.7.13
- pytorch == 1.11.0 + cuda11.3
- huggingface-hub == 0.13.4


## 3.  Getting Started:
We provide a script for building the runtime environment for each model, which is located in build_enviroment.sh in the folder of the corresponding model.  This section, We use MLP Model to illustrate the usage of the dataset and evaluation method. You can use a similar approach to build a runtime environment for the model you want to run.
### 3.1 download repository
Download and unzip this repository.
### 3.2  unzip raw dataset.
First of all, you need to unzip the dataset archives in folder `dataset/`.  For Linux, you can use tool 7z to unzip .7z files:
```bash
cd dataset
sudo apt install p7zip-full
7z x dataset.7z
```
For windows system, you can use software 7-Zip or other tools to unzip the archives.  You will get 4 folders named `cbcb`, `gcj`, `pre_train` and `train_split` in `dataset/` after decompression..
### 3.3  experiment settings
All the dependency is list in the section **Enviroment Setup**.  We recommend environment management software like *conda* or *virtual env* to organize running environments.  We will show an example using conda to create an python enviroment for the MLP Model on Linux. 
#### 3.3.1 install conda
If you already have Conda installed, you can skip this step.  You can install conda with Anaconda or MiniConda easily.  For example, MiniConda can be installed using the following command：
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc
```
#### 3.3.2 create environment
We provide a configure file named `environment.yml` for building the runtime environment for each model in corresponding folder.  For MLP Model, you can create environment using the following command: 
```bash
cd MLPModel
conda env create -f ./environment.yml
conda activate torch19
``` 

#### 3.3.3 train model
You can use **run.sh** to start training for every model. **Please note that before starting this step, your terminal should be able to use  `conda` command and the `dataset/` folder should contain the 4 folders described in Section 3.2.**
```bash
cd MLPModel
sh run.sh
``` 
There are 4 arguments in run.sh:
- dataset: cbcb | gcj, choose a dataset
- split: random0 | pro0 | fun0 | all0, choose a evaluation perspective
- cross: 0 | 1 | 2 | 3, choose a code abstract level
- cuda: cuda device

If you see the following output, it means that MLPModel is running successfully.
```bash
training:
train test data shape: (332255，3),(37135，3)，(38057，3)parsing source code: total count 9323parse finished， success parsing count 9323build sentences
build vocab dict
build bow vec
build onr hot vec, vocab:65332
one hot vec shape:(9323，65332)----- 11851011514663696
```

## 4. Reproducibility Instructions: 
We conducted experiments on 6 models and 4 evaluation perspectives on a machine with 128GB RAM, 3\*Nvidia 1080ti in 6 weeks, every model can work within 32GB RAM and 1\*Nvidia 1080ti.  Reproduce all results may take a few weeks dependending on your device.
If you want to reproduce the result of MLP Model in random-view on dataset gcj， you can simply run the bash in `MLPModel/run.sh`， set the arguments to `DATASET="gcj", CROSS=0, SPLIT="random0"` 



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

## Contact
Haiyang Li(lihaiyang@pku.edu.cn)


