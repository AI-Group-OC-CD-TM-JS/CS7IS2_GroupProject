from typing import List, Tuple
from scipy import spatial
import numpy as np
import matplotlib.pyplot as plt
from sko.ACA import ACA_TSP
import time

DISTANCE_MATRIX = np.zeros


def cal_total_distance(routine):
    """Get the total distance between all nodes"""
    (num_points,) = routine.shape
    return sum(
        [
            DISTANCE_MATRIX[routine[i % num_points], routine[(i + 1) % num_points]]
            for i in range(num_points)
        ]
    )

"""
aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
              size_pop=50, max_iter=200,
              distance_matrix=distance_matrix)"""

def run_ACA(
    coordinates: np.array, iterations: int, population:int
):
    """
    Run Ant Colony Optimisation Algorithm on a set of coords and their distance matrix.

    Parameters
    ----------
    coordinates : np.array
                    A numpy array of coordinates. Expecting coordinates in the form that is produced
                by the attention module. (3D array)
    iterations : int
        Max number of iterations for ACA
    population : int
        Number of ants

    Returns
    -------
    best_distance : float
        The best distance found
    best_points : List[int]
        The best order of nodes follow
    """
    global DISTANCE_MATRIX
    DISTANCE_MATRIX = spatial.distance.cdist(
        coordinates[0], coordinates[0], metric="euclidean"
    )

    # initialise ACA, using euclidean distance as objective function
    aca_tsp = ACA_TSP(
        func = cal_total_distance,
        n_dim = len(coordinates[0]),
        size_pop=population,
        max_iter = iterations,
        distance_matrix=DISTANCE_MATRIX
    )

    start_time = time.time()
    best_points, best_distance = aca_tsp.run()

    print("--- %s seconds ---" % (time.time() - start_time))

    return best_distance, best_points
