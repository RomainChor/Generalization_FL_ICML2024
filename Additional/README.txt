########################################################################################
#                                                                                      #
#                            Code for ICML 2024 submission                             #
#  	Federated Learning: Lessons from Generalization Error Analysis		       #
#                               Additional experiments                                 #
########################################################################################

 This folder contains code and instructions for the additional experiments on ResNet-56 in the submitted paper titled:
 Federated Learning, Lessons from Generalization Study: Communicate Less, Learn More


#################
# Requirements  #
#################
 The code is tested with Python 3.10.10 on Linux. Please see requirements.txt.


##################
# Code structure #
##################
 run_training.sh # Shell file to run training 
 Arguments:
 --nproc_per_node # Number of GPUs used (Multi-GPUs training)
 --seed # Seed for reproducibility
 --data-pth # Path to directory containing the CIFAR-10 data (will download and install the data there otherwise)
 --log-pth # Path to directory where to save the models

 inference.py # Script file to run inference
 --save-pth # Path to directory containing the saved models

 models/ # Contains the models classes


#######################
# Running experiments #
#######################  
 To reproduce Fig. 3: 
   - Open run_training.sh and set the arguments described above accordingly. 
   - Run run_training.sh in a terminal.
   - Launch inference.py in a terminal by setting --save_pth correctly.
   - Read the .pickle file using Pandas library and plot the values.

#######################
#      Credits        #
####################### 
Our implementation is partially based on the code found at https://github.com/hmgxr128/Local-SGD.
