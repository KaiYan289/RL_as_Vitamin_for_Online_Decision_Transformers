import gym
import d4rl
import os
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
file_traj = {}
for data_name in os.listdir():
    #print("file name:", data_name)
    if data_name.find(".pkl") == -1: continue
    if data_name.find("hopper") == -1: continue  
    with open(data_name, "rb") as f:
        trajectories = pickle.load(f)
        totrets, totshape = [], []
        
        for i in range(len(trajectories)):
            totrets.append(trajectories[i]['rewards'].sum())
            totshape.append(trajectories[i]['rewards'].shape)
        
        totrets, totshape = np.array(totrets), np.array(totshape)
        # print("totrets-max:", totrets.max(), "totrets-mean:", totrets.mean()) 
        
        def count_values_in_array(arr):
            # Find unique values and their counts
            values, counts = np.unique(arr, return_counts=True)
            
            # Combine the counts and values into a list of tuples
            count_value_pairs = list(zip(values, counts))
            
            # Sort the list of tuples by the value (the first element of each tuple)
            sorted_count_value_pairs = sorted(count_value_pairs, key=lambda x: x[0])
            
            # Print the sorted counts and values
            
            values, counts = [], []
            for value, count in sorted_count_value_pairs:
                print(f"Value: {value}, Count: {count}")
                values.append(value)
                counts.append(count)
            plt.scatter(values, counts)
            """
            plt.bar(values, counts)
            """

        plt.savefig(data_name.split(".")[0]+".png")
        plt.cla()

        print(data_name, len(trajectories))
        count_values_in_array(totshape)

