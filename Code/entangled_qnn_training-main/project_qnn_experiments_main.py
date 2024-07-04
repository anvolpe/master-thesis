import csv
import gc
import json
import time
from datetime import datetime
from qnns.cuda_qnn import CudaPennylane

from victor_thesis_utils import *
from victor_thesis_landscapes import *
from victor_thesis_plots import *
from victor_thesis_metrics import *
from victor_thesis_experiments_main import *
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import os
import pandas as pd
from scipy.optimize import minimize

num_layers = 1
num_qubits = 2
dimensions = 6

def single_optimizer_experiment():
    # specifications of qnn
    #num_layers = 1 # 2??
    #num_qubits = 2 # 1??
    #dimensions = 6 # geht nicht: alles außer 6
    type_of_data = 4 # random data
    deg_of_entanglement = 1 # high, (low = 1)
    num_data_points = 1 # low

    qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
    
    # generate a random unitary with num_qubits qubits (why are they the same?)
    unitary = torch.tensor(
            np.array(random_unitary_matrix(num_qubits)),
            dtype=torch.complex128,
            device="cpu",
        )
    data_points = generate_data_points(
        type_of_data,
        deg_of_entanglement,
        num_data_points,
        unitary, num_qubits
    )
    #print(data_points.shape)
    #print(data_points)
    expected_output = torch.matmul(unitary, data_points)
    y_true = expected_output.conj()
    
    # objective function based on cost function of qnn 
    def objective(x):
        qnn.params = torch.tensor(x, dtype=torch.float64, requires_grad=True).reshape(qnn.params.shape) # stimmt das???????
        cost = cost_func(data_points, y_true, qnn, device="cpu") 
        return cost.item()
    
    
    optimizers = ['COBYLA', 'BFGS', 'Nelder-Mead', 'Powell', 'SLSQP']
    results = {}
    # verschiedene inital_param_values ausprobieren und avg bilden? 
    initial_param_values = np.random.uniform(0, 2*np.pi, size=dimensions) # [0,2pi] siehe victor_thesis_landscapes.py, bei allen optimierern gleich
    for opt in optimizers:
        start = time.time()
        res = minimize(objective, initial_param_values, method=opt,
                       options={"maxiter": 10000, "ftol":1e-10, "xtol":1e-10})
        duration = time.time() - start
        results[opt] = {'result': res, 'duration': duration}
        print(f"Optimizer: {opt}")
        print(res)
        print(f"Duration: {duration}s\n")

    print("Results:", results)

    with open('experimental_results/results/optimization_results.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(["Optimizer", "Result", "Duration"])

        for opt, result in results.items():
            writer.writerow([opt, result['result'], result['duration']])

def run_single_optimizer_experiment_batch(conf_id, data_type, num_data_points, s_rank, data_batch, unitary):
    # TODO: für jede Reihe an Datenpunkten ein Experiment für alle Optimierer
    for i in range(len(data_batch)):
        data_points = data_batch[i]


def run_all_optimizer_experiments():
    filename = "Code/entangled_qnn_training-main/experimental_results/configs/configurations_16_6_4_10_13_3_14.txt"
    # TODO: aus Code/entangled_qnn_training-main/experimental_results/configs/configurations_16_6_4_10_13_3_14.txt configs einlesen
    # TODO: Für jede config einmal run_single_experiment laufen lassen
    
    # so könnten Daten aussehen --> weiter verarbeiten für qnn
    conf_id=0
    data_type=random
    num_data_points=1
    s_rank=1
    unitary=[[-0.45070455-0.02711853j,0.78437395-0.06613086j,0.06358678+0.19963393j,-0.07343613+0.35668523j],
            [-0.01890143-0.03813363j,0.32408202+0.25557629j,0.05872864-0.68979805j,0.55466693-0.20227297j],
            [-0.11215405+0.64023111j,-0.13344055+0.29565494j,-0.49012687-0.19046288j,-0.04241254+0.44046348j],
            [0.55771659+0.24656916j,0.31851997-0.05798805j,0.28761525-0.34294258j,-0.56718418+0.03616933j]]
    data_batch_0=[[[0.09314128-0.12946863j,0.39382838-0.19799267j,0.05133879+0.12112185j,0.08106995-0.04021906j],
                [0.07622026-0.09754417j,0.31152873-0.14143589j,0.03608905+0.09551662j,0.06411194-0.02869752j],
                [0.11804856-0.19626647j,0.54031288-0.32976236j,0.08774511+0.16729872j,0.1112873-0.06711204j],
                [-0.01827577+0.10086995j,-0.17383409+0.2237231j,-0.06326177-0.05610261j,-0.03593256+0.04574145j]]]
    data_batch_1=[[[0.11174606-0.02974311j,0.0062696-0.05280092j,-0.01978072+0.02987687j,0.16841097+0.03037302j],
                [-0.14204144+0.20092099j,0.06172976+0.094819j,-0.00840543-0.07578002j,-0.31657303+0.17993658j],
                [0.15064607-0.0773127j,-0.00745024-0.07750219j,-0.01901218+0.04890241j,0.25042374-0.0089162j],
                [-0.14602579-0.40250224j,-0.19679115-0.00596365j,0.1166284+0.0632495j,0.05729205-0.63104682j]]]
    data_batch_2=[[[0.12574507+0.14295216j,-0.06600188+0.17882274j,0.5097803-0.21337109j,0.03062141+0.16953985j],
                [0.04134762+0.01061072j,0.01034856+0.04146659j,0.06751534-0.10389842j,0.02707945+0.02754715j],
                [-0.01994428-0.12673754j,0.10211291-0.07792621j,-0.36710336-0.06262313j,0.04378125-0.10752594j],
                [-0.18503833+0.01204989j,-0.09874132-0.15721565j,-0.138382+0.52015265j,-0.14901135-0.07714729j]]]
    data_batch_3=[[[0.00938174-0.03915841j,-0.09861291-0.06432623j,-0.39565405-0.01500526j,0.18159026-0.15552714j],
                [0.06185613+0.02658457j,0.1340102-0.14420851j,0.13714247-0.64766039j,0.20466824+0.34339872j],
                [0.01549031-0.00427146j,0.00346342-0.04685593j,-0.07103547-0.14113085j,0.08934859+0.03346192j],
                [-0.0255441-0.00158969j,-0.02948625+0.06878092j,0.03389378+0.24936634j,-0.11724566-0.0966799j]]]
    data_batch_4=[[[-0.11458899+0.03928362j,0.0435563+0.05453538j,0.02294759-0.11649752j,0.14355952+0.05525074j],
                [0.239814+0.27457382j,0.10239304-0.18339806j,-0.35068906+0.06859816j,-0.00938315-0.46283845j],
                [0.1138846+0.30497249j,0.14333106-0.12098589j,-0.31463533-0.05315627j,0.13796413-0.38968998j],
                [0.0272455-0.04205309j,-0.02810213-0.00661599j,0.02229414+0.04376376j,-0.06082023+0.01869768j]]]
    # TODO: Wie unitary und data_points speichern, damit verwendbar? (man braucht torch.tensor, aber bestimmte Shapes??)



if __name__ == "__main__":
    single_optimizer_experiment()