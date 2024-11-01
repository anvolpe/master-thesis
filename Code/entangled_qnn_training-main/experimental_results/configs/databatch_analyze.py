import numpy as np
import sys

def skip_to_next(pattern, file):
    for line in file:
        if line.startswith(pattern):
            return line
    print("REACHED END")
    sys.exit(0)
    
def handle_batch(bnum, batchline):
    batchdata = batchline.split("=")[1]
    batch = np.array(eval(batchdata))
    num_samples = batch.shape[0]
    #print("Got %s samples" % str(num_samples))
    
    # check each sample if its normalized 
    for i in range(0, num_samples):
        sample = batch[i]
        norm = np.linalg.norm(sample)
        if not np.isclose(norm, 1.):
            print("Samplenorm not 1: ", norm)
            print("Samplenum %d in batch %d" % (i, bnum))

with open('configurations_16_6_4_10_13_3_14.txt', 'r') as file:
    
    while True:
        conf_line = skip_to_next('conf_id', file)
        print(conf_line)
        
        dbline = skip_to_next('data_batch_0', file)
        handle_batch(0, dbline)
        
        dbline = skip_to_next('data_batch_1', file)
        handle_batch(1, dbline)
        
        dbline = skip_to_next('data_batch_2', file)
        handle_batch(2, dbline)
        
        dbline = skip_to_next('data_batch_3', file)
        handle_batch(3, dbline)
        
        print("-----------")
        print("")
    
    