from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import os

def func():
    for i in range(100000000000000):
        if(i == 83985348958):
            break

def concurrentTest():
    print("before")
    with ProcessPoolExecutor(cpu_count()) as exe:
        exe.submit(func)
        print("submit done")
    print("after processpoolexec")
if __name__ == "__main__":
    # one thread per core
    #torch.set_num_threads(1)
    #torch.multiprocessing.set_sharing_strategy("file_system")
    concurrentTest()
    #run_full_experiment()