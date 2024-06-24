# Evaluating the Performance of Optimizers on the Loss Landscape of Quantum Neural Networks
Experiment/Code for reproduction of results for "Evaluating the Performance of Optimizers on the Loss Landscape of Quantum Neural Networks" research project by Serhat Ceri, Jonathan Klenk, Alina Mürwald, Anna Volpe (2024).

**TODO**: 
Files with names starting with ``victor_thesis_`` are the core of the thesis. All other files were originally made for  
Alexander Mandl, Johanna Barzen, Frank Leymann, Daniel Vietz. On Reducing the Amount of Samples Required for Training of QNNs: Constraints on the Linear Structure of the Training Data. [arXiv:2309.13711 [quant-ph]](https://arxiv.org/abs/2309.13711)

victor_thesis_experiments_main.py contains the main experiment methods. You can adapt and execute the experiments in the ``run_full_experiment()`` method.  
  
victor_thesis_experiments_nb.ipynb is a jupyter notebook which includes examples of how to evaluate and visualize the raw experiment results.  
  
The experiment results from the thesis can be found in /experimental_results under the  run id ``16_6_4_10_13_3_14``.  
Graphics used in the thesis can be found in /Graphics  

## Dependencies
The code contained in this repository requires the following dependencies for reproducing the experiments:
- matplotlib==3.5.2
- networkx==2.8.8
- numpy==1.24.1
- orqviz==0.5.0
- PennyLane==0.27.0
- scipy==1.13.1
- torch==2.0.0

Install dependencies using ``pip install -r requirements.txt``  
Python 3.9.13 is the version the experiments were run on.

#### Disclaimer of Warranty 
Unless required by applicable law or agreed to in writing, Licensor provides the Work (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.
