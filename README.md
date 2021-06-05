# GePGCN

## Description
GePGCN is a framework based on a GCN that uses genomic gene synteny information, from which the graph topological pattern and gene node characteristics can be learned, to disseminate node attributes in the network and make predictions about metabolic pathway assignment. Our graph neural network framework is implemented based on Pytorch Geometric package in Python 3.7.     
     
## Citing


## Usage
### Requirements
- Python 3.7
- Pytorch
- Pytorch Geometric
- numpy
- pandas
- matplotlib
- seaborn
- networkx
- scipy
- scikit-learn
- node2vec

### Require inputs
- sequences fasta file
- protein-K file
- CDS file
- blastp result file

### Steps
#### Step1: construct the network
\> mkdir result  
\> mkdir others  
\> cat test_data/faa/*.faa > whole.faa  
\> python construction.py  
> **Note there are several parameters can be tuned. Please refer to the construction.py file for detailed description of all parameters**  

#### Step2: run the framework
\> python model_and_train.py   
> **Note there are several parameters can be tuned. Please refer to the model_and_train.py file for detailed description of all parameters**  
