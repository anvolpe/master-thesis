import gc
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

def single_test_run():
    print("single_test_run")
    num_layers = 1
    num_qubits = 2
    num_unitaries = 1 # was 5
    num_tries = 1 # was 5
    grid_size = 16
    dimensions = 6
    type_of_data = 1 # random data
    deg_of_entanglement = 1 # low, (high = 4)
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

    os.makedirs("experimental_results/configs", exist_ok=True)
    os.makedirs("experimental_results/results", exist_ok=True)
    # generate a U3 ansatz containing 2 layers -> 6 params
    qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
    print("single_test_run")
    #unitaries = []
    # [type_of_data][num_data_points][deg_of_entanglement][id_unitary][id_try]
    #configurations = []
    
    # generate a random unitary with num_qubits qubits (why are they the same?)
    unitary = torch.tensor(
            np.array(random_unitary_matrix(num_qubits)),
            dtype=torch.complex128,
            device="cpu",
        )
    print("single_test_run")
    data_points = generate_data_points(
        type_of_data,
        deg_of_entanglement,
        num_data_points,
        unitary, num_qubits
    )
    print("single_test_run")
    #start = time.time()
    # generate configurations (5 datapoint sets = 5 runs per config)
    conf_id = 0

    TV_arr = []
    FD_arr = []
    IGSD_arr = []
    SC_metrics = []
    print("before landscape")
    landscape = generate_loss_landscape(grid_size, dimensions, data_points, unitary, qnn)
    print("after landscape")
    TV_arr.append(calc_total_variation(landscape))
    FD_arr.append(calc_fourier_density(landscape))
    IGSD_arr.append(calc_IGSD(landscape))
    SC = calc_scalar_curvature(landscape)
    SC_metrics.append(process_sc_metrics(SC))
    del SC
    # del landscape # weglassen?
    # TODO: def objective basierend auf landscape
    # TODO: optimize.minimize(...) Aufruf für einen Optimierer + Options (Spezifikationen)
    # TODO: Zeit messen + Zeit speichern

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


def run_experiment():
    start = time.time()
    with ProcessPoolExecutor(cpu_count()) as exe:
        exe.submit(single_test_run)
    end = time.time()
    print(f"total runtime run_experiment: {np.round(end-start,2)}s")


if __name__ == "__main__":
    # one thread per core
    #torch.set_num_threads(1)
    #torch.multiprocessing.set_sharing_strategy("file_system")
    run_experiment()
    #run_full_experiment()