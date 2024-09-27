# Evaluating the Performance of Optimizers on the Loss Landscape of Quantum Neural Networks
Experiment/Code for reproduction of results for "Evaluating the Performance of Optimizers on the Loss Landscape of Quantum Neural Networks" research project by Serhat Ceri, Jonathan Klenk, Alina Mürwald, Anna Volpe (2024).
The Code is adapted from Victor Ülger's master thesis "Analyzing the Effect of Entanglement of Training Samples on the Loss Landscape of Quantum Neural Networks" (2024).

**TODO: Update as we go**: 

Code/entangled_qnn_training-main/project_qnn_experiments_main.py contains the (initial) main experiment methods. 

experimental_results/results/2024-07-19_allConfigs_allOpt contains initial experimental results (for all optimizers) 

experimental_results/results/optimizer_results/bounds_2024-07-29 contains experimental results for different bounds (for x). 

qnn-experiments/plots contains all plots (fun vs. max. iterations (boxplot), average fun vs. (actual) iterations, min and max x-value per bounds (boxplots) and convergence plots)

So far only convergence plots for maxiter = 1000 exist.

Convergence Plots are sorted such that folder names indicate which parameter (datatype, num-data-points and s-rank) are fixed and which value they have, i.e. all convergence plots where datatype is "random", num-data-points is 1 and s-rank is variable are in /qnn-experiments/plots/convergence_plots/maxiter = 1000/datatype/random/num_data_points/1.

## Dependencies

The code contained in this repository requires the following dependencies for reproducing the experiments:
- matplotlib==3.8.0
- networkx==3.1
- numpy==1.26.4
- orqviz==0.6.0 
- PennyLane==0.27.0
- scipy==1.11.4
- torch==2.3.1

Install dependencies using ``pip install -r requirements.txt``  
Python 3.11.7 is the version the experiments were run on.

#### Disclaimer of Warranty 
Unless required by applicable law or agreed to in writing, Licensor provides the Work (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.

### Experiment Settings
For detailed descriptions of the attributes and parameters used in this project, please refer to the experiment_settings.txt file. This file contains information about the attributes we utilized, as well as the available parameters for each optimizer employed in the experiments.
