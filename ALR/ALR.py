""""""

import os
import numpy as np
import torch
import time

""""""
def make_oracle(model, xy, temperature=1.0):
    
    num_nodes = len(xy)
    
    xyt = torch.tensor(xy).float()[None]  # Add batch dimension
    
    with torch.no_grad():  # Inference only
        embeddings, _ = model.embedder(model._init_embed(xyt))

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = model._precompute(embeddings)
    
    def oracle(tour):
        with torch.no_grad():  # Inference only
            # Input tour with 0 based indices
            # Output vector with probabilities for locations not in tour
            tour = torch.tensor(tour).long()
            if len(tour) == 0:
                step_context = model.W_placeholder
            else:
                step_context = torch.cat((embeddings[0, tour[0]], embeddings[0, tour[-1]]), -1)

            # Compute query = context node embedding, add batch and step dimensions (both 1)
            query = fixed.context_node_projected + model.project_step_context(step_context[None, None, :])

            # Create the mask and convert to bool depending on PyTorch version
            mask = torch.zeros(num_nodes, dtype=torch.uint8) > 0
            mask[tour] = 1
            mask = mask[None, None, :]  # Add batch and step dimension

            log_p, _ = model._one_to_many_logits(query, fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key, mask)
            p = torch.softmax(log_p / temperature, -1)[0, 0]
            assert (p[tour] == 0).all()
            assert (p.sum() - 1).abs() < 1e-5
            #assert np.allclose(p.sum().item(), 1)
        return p.numpy()
    
    return oracle

def run_ALR(coordinates: np.array, temperature: float):
    # all_dataset_results = []
    
    xy = coordinates[0, :, :]
    num_nodes = xy.shape[0]
    if num_nodes == 100:
        pretrained_file = 'ALR/pretrained/tsp_100/'
    elif num_nodes == 50:
        pretrained_file = 'ALR/pretrained/tsp_50/'
    elif num_nodes == 20:
        pretrained_file = 'ALR/pretrained/tsp_20/'
    else:
        print(f"ERROR NUM NODES:\t{num_nodes}")
        return -1

    from ALR.utils.functions import load_model
    model, _ = load_model(pretrained_file)
    model.eval()  # Put in evaluation mode to not track gradients

    oracle = make_oracle(model, xy, temperature=temperature)

    sample = False
    tour = []
    tour_p = []
    start_time = time.time()
    while(len(tour) < len(xy)):
        p = oracle(tour)
        
        
        if sample:
            # Advertising the Gumbel-Max trick
            g = -np.log(-np.log(np.random.rand(*p.shape)))
            i = np.argmax(np.log(p) + g)
            # i = np.random.multinomial(1, p)
        else:
            # Greedy
            i = np.argmax(p)
        tour.append(i)
        tour_p.append(p)
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    total_distance = 0
    for node_index in range(num_nodes):
        # print(f"Current:\t{tour[node_index]}\tNext:\t{tour[(node_index+1)%num_nodes]}")
        curr_node = xy[tour[node_index], :]
        next_node = xy[tour[(node_index+1)%num_nodes], :]
        distance = np.sqrt((curr_node[0] - next_node[0])**2 + (curr_node[1] - next_node[1])**2)
        total_distance = total_distance + distance
        
    # all_dataset_results.append((total_distance, tour, time_taken))
            
        
    
    # print(total_distance)
    # print(tour)
    # print(tour_p)
    # print(time_taken)
    
    # While still doing 1 dataset
    return total_distance, tour, time_taken

