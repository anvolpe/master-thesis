# Evaluating the Performance of Optimizers on the Loss Landscape of Quantum Neural Networks

Experiment/Code for reproduction of results for "Evaluating the Performance of Optimizers on the Loss Landscape of Quantum Neural Networks" research project by Serhat Ceri, Jonathan Klenk, Alina Mürwald, Anna Volpe (2024).
The Code is adapted from Victor Ülger's Master Thesis "Analyzing the Effect of Entanglement of Training Samples on the Loss Landscape of Quantum Neural Networks" (2024).

Employed optimizers: Nelder-Mead, COBYLA, Powell, Dual Annealing, Genetic Algorithm, Differential Evolution, Particle Swarm Optimization (PSO), BFGS, SLSQP, SGD with Momentum, Adam and RMSprop.

## Directory

Files that start with `project_qnn_` were made for this project. All other files were made for the Master Thesis by Victor Ülger or for Alexander Mandl, Johanna Barzen, Frank Leymann, Daniel Vietz. On Reducing the Amount of Samples Required for Training of QNNs: Constraints on the Linear Structure of the Training Data. [arXiv:2309.13711 [quant-ph]](https://arxiv.org/abs/2309.13711).

The main experiment methods are contained in Code/entangled_qnn_training-main/project_qnn_experiments_main.py. You can adapt and execute the experiments in the `run_all_optimizer_experiments` method.

Results of the final experiment (used in the paper) can be found in qnn-experiments/optimizer_results/final_experiment_2024-10.

Results for the preliminary tests regarding different bounds (for x) for Dual Annealing can be found in qnn-experiments/optimizer_results/bounds_2024-07-29.

Results for preliminary hyperparameter tests regarding differnt hyperparameters for Genetic Algorithm, Differential Evolution and PSO can be found inqnn-experiments/optimizer_results/hyperparameter_tests_2024-10-26

### Plots

Plots used in the paper can be found in qnn-experiments/plots. This contains boxplots, convergence plots and information in the form of csv and txt files for all convergence plots. All functions used to analyze the experiment data is contained in python scripts starting with "project_qnn_analysis".
Only convergence plots for maxiter = 1000 exist.

Convergence plots are sorted such that folder names indicate which parameter (datatype, num-data-points and s-rank) are fixed and which value they have, i.e. all convergence plots where datatype is "random", num-data-points is 1 and s-rank is variable are in qnn-experiments/plots/convergence_plots/maxiter/1000/datatype/random/num_data_points/1.

### Experiment Settings

For detailed descriptions of the attributes and parameters used in this project, please refer to the experiment_settings.txt file. This file contains information about the attributes we utilized, as well as the available parameters for each optimizer employed in the experiments.

## Dependencies

The code contained in this repository requires the following dependencies for reproducing the experiments:

- matplotlib==3.8.0
- networkx==3.1
- numpy==1.26.4
- orqviz==0.6.0
- PennyLane==0.27.0
- scipy==1.11.4
- torch==2.3.1
- pygad == 3.3.1
- pyswarms == 1.3.0

Use "pip install -r Code/entangled_qnn_training-main/requirements.txt" to install all needed python packages.

The experiments were run on Python 3.11.7, and no errors were encountered when using Python 3.13.0 either.

#### Disclaimer of Warranty

Unless required by applicable law or agreed to in writing, Licensor provides the Work (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.