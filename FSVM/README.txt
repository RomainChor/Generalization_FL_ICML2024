########################################################################################
#                                                                                      #
#                            Code for ICML 2024 submission                             #
# 		 Federated Learning: Lessons from Generalization Error Analysis        #
#                                  FSVM experiments                                    #
########################################################################################

 This folder contains code and instructions for experiments in the submitted paper titled:
 Federated Learning, Lessons from Generalization Study: Communicate Less, Learn More


#################
# Requirements  #
#################
 The code is tested with Python 3.10.10 on Linux. Please see requirements.txt.


##################
# Code structure #
##################
 run_experiments.py  # File to run to reproduce presented experiments in the paper.
 Arguments:
 --data_path  # Path to directory containing MNIST data, type: str
 --save_path  # Path to directory where to save experiments results and plots, type: str
 --mode # Whether to run simulations ('train') or to plot figures ('plot'). Set to 'train', run the file 
          then set to 'plot' and run again to get plots, type: str
 --comparison  # To run simulations with 'K' or 'n' fixed, type: str
 --MC  # Number of runs (Monte-Carlo simulations), type: int
 --classes  # MNIST classes, type: sequence of ints
 --frac_iid  # Fraction of clients with AWGN, type: float
 --iid_std  # Standard deviation of the AWGN, type: float

 utils/  # Contains utility files for the simulations.
  >>> dataloaders.py
  >>> models.py


#######################
# Running experiments #
#######################  
 Download MNIST data from: http://yann.lecun.com/exdb/mnist/.
 Extract the files and place in the current directory (or make sure to modify PATH environment variable)

 To reproduce Fig. 2 and Fig. 4:
   - Launch run_experiments.py with "--data_path" and "--save_path" set properly. Some .pickle files should be saved in the 
     directory indicated in "--save_path".
   - Launch run_experiments.py again with "--mode 'plot'". 

 To reproduce Fig. 5 and Fig. 6:
   - Launch run_experiments.py with "--compare 'n'" and with "--data_path" and "--save_path" set properly. Some .pickle files should be saved in the 
     directory indicated in "--save_path".
   - Launch run_experiments.py again with "--mode 'plot'". 

 To reproduce Fig. 7 and Fig. 8:
   - Launch run_experiments.py with "--frac_iid 0.2" and "--data_path" and "--save_path" set properly. Some .pickle files should be saved in the 
     directory indicated in "--save_path".
   - Launch run_experiments.py again with "--mode 'plot'". 

 To reproduce Fig. 9:	
   - Launch run_experiments.py with "--mode 'estimation'" and "--data_path" and "--save_path" set properly. Some .pickle files should be saved in the 
     directory indicated in "--save_path". 