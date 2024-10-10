import multiprocessing
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
from multiprocessing.pool import Pool
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import re
'''
def generate_and_save_testLandscape():
    print("####### Generate and Save test landscape")
    num_layers = 1
    num_qubits = 2
    #num_unitaries = 1 # was 5
    #num_tries = 1 # was 5
    grid_size = 16
    dimensions = 6 # geht nicht: alles außer 6
    type_of_data = 1 # random data
    deg_of_entanglement = 4 # low, (high = 4)
    num_data_points = 1 # low

    # generate an experiment id (based on time) to identify which results and configs belong to which experiment run
    current_time = datetime.now()
    exp_id = (
        str(grid_size)
        + "_"
        + str(dimensions)
        + "_"
        + str(current_time.month)
        + "_"
        + str(current_time.day)
        + "_"
        + str(current_time.hour)
        + "_"
        + str(current_time.minute)
        + "_"
        + str(current_time.second)
    )
    # create directories for results and configs

    #os.makedirs("experimental_results/configs", exist_ok=True)
    #os.makedirs("experimental_results/results", exist_ok=True)
    # generate a U3 ansatz containing 2 layers -> 6 params
    qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
    #unitaries = []
    # [type_of_data][num_data_points][deg_of_entanglement][id_unitary][id_try]
    #configurations = []
    
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
    print(data_points.shape)
    #start = time.time()
    # generate configurations (5 datapoint sets = 5 runs per config)
    conf_id = 0

    TV_arr = []
    FD_arr = []
    IGSD_arr = []
    SC_metrics = []
    start = time.time()
    landscape = generate_loss_landscape(grid_size, dimensions, data_points, unitary, qnn)
    t = time.time()-start
    print("time [generate loss landscape] (sec): ", np.round(t,2))
    start = time.time()
    TV_arr.append(calc_total_variation(landscape))
    FD_arr.append(calc_fourier_density(landscape))
    IGSD_arr.append(calc_IGSD(landscape))
    SC = calc_scalar_curvature(landscape)
    SC_metrics.append(process_sc_metrics(SC))
    t = time.time()-start
    print("time [calc metrics] (sec): ", np.round(t,2))
    del SC
    # save landscape as JSON file
    os.makedirs("experimental_results/landscapes", exist_ok=True)
    start = time.time()
    np.savez_compressed("experimental_results/landscapes/testLandscape.npz", np.array(grid_size),landscape)
    t = time.time()-start
    print("time [write npz file] (sec): ", np.round(t,2))

    gc.collect() # garbage collector
        
    metrics = []
    metrics.append(TV_arr)
    metrics.append(FD_arr)
    metrics.append(IGSD_arr)
    metrics.append(SC_metrics)
    process_and_store_metrics(metrics, 1, conf_id, exp_id)
    # TODO: speichern: Optimierer, Zeiten, Spezifikation
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"[{now}] Finished run: {conf_id}")

def run_single_optimizer_experiment():
    # TODO: def objective basierend auf landscape
    # TODO: optimize.minimize(...) Aufruf für einen Optimierer + Options (Spezifikationen)
    # TODO: Zeit messen + Zeit speichern

    # load test loss landscape from npz file (67mb)
    start = time.time()
    landscapeZ = np.load("experimental_results/landscapes/testLandscape.npz") # falls 
    grid_size = landscapeZ['arr_0']
    landscape = landscapeZ['arr_1']
    t = time.time()-start
    print("time [load npz file] (sec): ", np.round(t,2))

    # define objective basd on loss landscape and grid_size
    # input x: is a 1D array with 6 values (since: 6 parameters)
    # output: corresponding loss 
    def objective(x):
        
            objective to minimize based on loss landscape

            Args:
                x (1D array): is a 1D array with 6 values (since: 6 parameters)
            Returns:
                float: corresponding loss 
        
        obj_value = 0

        # TODO: Index mithilfe von x und grid_size berechnen
        idx = 0,0,0,0,0,0
        obj_value = landscape[idx]
        return obj_value
    print(objective([1,0,5,5,0,0]))
    #[0][0][0][0][0][0]
    print(landscape.shape)

def optimizer_experiment():
    num_layers = 1
    num_qubits = 2
    #num_unitaries = 1 # was 5
    #num_tries = 1 # was 5
    #grid_size = 16
    dimensions = 6 # geht nicht: alles außer 6
    type_of_data = 1 # random data
    deg_of_entanglement = 4 # high, (low = 1)
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
    print(data_points.shape)
    print(data_points)
    expected_output = torch.matmul(unitary, data_points)
    y_true = expected_output.conj()
    
    def objective(x):
        qnn.params = torch.tensor(x, dtype=torch.float64, requires_grad=True).reshape(qnn.params.shape) # stimmt das???????
        cost = cost_func(data_points, y_true, qnn, device="cpu") 
        return cost.item()
    
    optimizers = ['COBYLA', 'BFGS', 'Nelder-Mead', 'Powell', 'SLSQP']
    results = {}
    initial_param_values = np.random.uniform(-np.pi, np.pi, size=dimensions)
    bounds = list(zip(np.zeros(6), np.ones(6)*2*np.pi))
    for opt in optimizers:
        start = time.time()
        res = minimize(objective, initial_param_values, method=opt,
                       options={"maxiter": 100, "bounds": bounds})
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

def testMinimizeBounds():
    x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    bounds = list(zip(np.zeros(5), np.ones(6)*2))
    res = minimize(rosen, x0, method='BFGS', jac=rosen_der, bounds=bounds, options={'gtol': 1e-6, 'disp': True})'''

def func1():
    time.sleep(3)
    print("done with func1")
    return None

def func2():
    time.sleep(4)
    print("done with func2")
    return None

def pso_test():
    num_layers = 1
    num_qubits = 2
    dimensions = 6
    max_iters = [100,500,1000]
        # setup qnn configuration
    conf_id = 0
    data_type = "random" # random, orthogonal, non_lin_ind, var_s_rank
    num_data_points = 1
    s_rank = 1 # Schmidt-Rank
    var = "data_batch_0"
    databatch_id = int(var.strip("datbch_"))

    # setup dictionary for dumping info into json file later
    result_dict = {"conf_id":conf_id, "data_type":data_type, "num_data_points":num_data_points, "s_rank":s_rank}
    val = "[[-0.45070455-0.02711853j,0.78437395-0.06613086j,0.06358678+0.19963393j,-0.07343613+0.35668523j],[-0.01890143-0.03813363j,0.32408202+0.25557629j,0.05872864-0.68979805j,0.55466693-0.20227297j],[-0.11215405+0.64023111j,-0.13344055+0.29565494j,-0.49012687-0.19046288j,-0.04241254+0.44046348j],[0.55771659+0.24656916j,0.31851997-0.05798805j,0.28761525-0.34294258j,-0.56718418+0.03616933j]]"
    result_dict["unitary"] = val#.strip("\"")
    val,_ = re.subn('\[|\]|\\n', '', val)
    unitary = torch.from_numpy(np.fromstring(val,dtype=complex,sep=',').reshape(-1,4))#unitary: 4x4 tensor
    val = '[[[0.09314128-0.12946863j,0.39382838-0.19799267j,0.05133879+0.12112185j,0.08106995-0.04021906j],[0.07622026-0.09754417j,0.31152873-0.14143589j,0.03608905+0.09551662j,0.06411194-0.02869752j],[0.11804856-0.19626647j,0.54031288-0.32976236j,0.08774511+0.16729872j,0.1112873-0.06711204j],[-0.01827577+0.10086995j,-0.17383409+0.2237231j,-0.06326177-0.05610261j,-0.03593256+0.04574145j]]]'
    result_dict[var] = {}
    result_dict[var]["databatch"] = val#.strip("\"")
    val,_ = re.subn('\[|\]|\\n', '', val) 
    data_points = torch.from_numpy(np.fromstring(val,dtype=complex,sep=',').reshape(-1,4,4)) #data_points: 1x4x4 tensor


    # setup qnn
    qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
    expected_output = torch.matmul(unitary, data_points)
    y_true = expected_output.conj()
    
    # objective function based on cost function of qnn 
    def objective(x):
        '''
            x  is of size num_particles x dimensions
            returns array of length num_particles (cost-value for each particle)
        '''
        n_particles = x.shape[0]
        cost_values = []
        for i in range(n_particles):
            qnn.params = torch.tensor(x[i], dtype=torch.float64, requires_grad=True).reshape(qnn.params.shape) # stimmt das???????
            cost = cost_func(data_points, y_true, qnn, device="cpu") 
            cost_values.append(cost.item())
        return cost_values
    results = {}
    # verschiedene inital_param_values ausprobieren und avg bilden? 
    # TODO: gleich für alle Konfigs oder nur gleich für eine Konfig aber da für alle Optmimierer?
    initial_param_values = np.random.uniform(0, 2*np.pi, size=dimensions) # [0,2pi] siehe victor_thesis_landscapes.py, bei allen optimierern gleich
    initial_param_values_tensor = torch.tensor(initial_param_values)

    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
    run_n = 0
    for max_iter in max_iters:
        #for tol in tols:
        #for catol in tols:
                # Call instance of PSO
                optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=dimensions, options=options)

                # Perform optimization
                start = time.time()
                res, pos = optimizer.optimize(objective, iters=max_iter)
                duration = time.time() - start
                # fill results dict
                # specifications of this optimizer run
                results[run_n] = {"maxiter": max_iter, "duration":duration}
                # result info
                '''for attribute in res.keys():
                    results[run_n][attribute] = str(res[attribute])'''
                results[run_n]["fun"] = res
                results[run_n]["x"] = pos
                results[run_n]["callback"] = list(optimizer.cost_history) # stimmt das??
                global nit 
                nit = 0
                run_n += 1
    print(results)

if __name__ == "__main__":
    pso_test()





    
    