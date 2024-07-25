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
from sgd_for_scipy import *
from jax import jacrev
import os
import pandas as pd
from scipy.optimize import minimize, dual_annealing
import re



def boxplot_fun_values_per_optimizers():
    #TODO: read conf_[conf_id]_opt.json files: all fun values + optimizer
    #one violin plot (boxplot??) per optimizer
    print("TODO")

if __name__ == "__main__":
    print("TODO")