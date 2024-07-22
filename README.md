# Lessons from Generalization Error Analysis of Federated Learning: You May Communicate Less Often!
Contains source code for reproducing experimental results of the ICML2024 paper "Lessons from Generalization Error Analysis of Federated Learning: You May Communicate Less Often!" by Milad Sefidgaran, Romain Chor, Abdellatif Zaidi and Yijun Wan.
The paper can be found [here.](https://arxiv.org/abs/2306.05862)

Our experiments use the public datasets MNIST and CIFAR-10. 

## Code for experiments on the generalization of FSVM 
- run_experiments.py: File to run to reproduce presented experiments in the paper.  
  Arguments:  
  - `--data_path`: Path to directory containing MNIST data (type: str)  
  - `--save_path`: Path to directory where to save experiments results and plots (type: str)  
  - `--mode`: Whether to run simulations ('train') or to plot figures ('plot'). Set to 'train', run the file then set to 'plot' and run again to get plots (type: str)  
  - `--comparison`: To run simulations with 'K' or 'n' fixed (type: str)  
  - `--MC`: Number of runs (Monte-Carlo approximation) (type: int) 
  - `--classes`: MNIST classes (type: sequence of ints)  
  - `--frac_iid`: Fraction of clients with AWGN (type: float)  
  - `--iid_std`: Standard deviation of the AWGN (type: float)

- utils/: Contains utility files for the simulations.  
  - dataloaders.py
  - models.py

 
## Code for additional experiments on the generalization of FL 
- run_training.sh: Shell file to run training.  
  Arguments:
  - `--nproc_per_node`: Number of GPUs used (Multi-GPUs training) (type: int)
  - `--seed`: Seed for reproducibility (type: int)
  - `--data-pth`: Path to directory containing the CIFAR-10 data (will download and install the data there otherwise) (type: str)
  - `--log-pth`: Path to directory where to save the models, (type: str)

- inference.py: Script file to run inference  
  - `--save-pth`: Path to directory containing the saved models, type: str

- models/: Contains the models classes


## Reproducing our results

### FSVM
- Download MNIST data at https://yann.lecun.com/exdb/mnist/.
- To reproduce Fig. 2 and Fig. 4:
  - Launch run_experiments.py with `--data_path` and `--save_path` set properly. Some .pickle files should be saved in the 
   directory indicated in `--save_path`.
  - Launch run_experiments.py again with `--mode 'plot'`. 
- To reproduce Fig. 5 and Fig. 6:
  - Launch run_experiments.py with `--compare 'n'` and with `--data_path` and `--save_path` set properly. Some .pickle files should be saved in the 
   directory indicated in `--save_path`.
  - Launch run_experiments.py again with `--mode 'plot'`. 
- To reproduce Fig. 7 and Fig. 8:
  - Launch run_experiments.py with `--frac_iid 0.2` and `--data_path` and `--save_path` set properly. Some .pickle files should be saved in the 
   directory indicated in `--save_path`.
  - Launch run_experiments.py again with `--mode 'plot'`. 
 To reproduce Fig. 9:	
   - Launch run_experiments.py with `--mode 'estimation'` and `--data_path` and `--save_path` set properly. Some .pickle files should be saved in the 
     directory indicated in `--save_path`.

### Additional experiments on NNs
To reproduce Fig. 3:
  - Open run_training.sh and set the arguments described above accordingly. 
  - Run run_training.sh in a terminal.
  - Launch inference.py in a terminal by correctly setting `--save_pth`.
  - Read the .pickle file using Pandas library and plot the values.


## Requirements
idx2numpy==1.2.3  
joblib==1.2.0  
matplotlib==3.7.1  
numpy==1.23.5  
pandas==1.5.3  
scikit_learn==1.2.2  
seaborn==0.12.2  
tqdm==4.65.0  
torch==2.0.0  
torchvision==0.15.0


## References & Credits
Our implementation for the additional experiments is partially based on the code found at https://github.com/hmgxr128/Local-SGD.
