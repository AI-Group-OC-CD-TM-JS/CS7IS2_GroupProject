from typing import List, Tuple
from scipy import spatial
import numpy as np
import matplotlib.pyplot as plt
from sko.SA import SA_TSP
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


def run_SA(
    coordinates: np.array, iterations: int, t_max: int, t_min: int
) -> Tuple[float, Tuple[List]]:
    """
    Run Simulated Annealing on a set of coords and their distance matrix.

    Parameters
    ----------
    coordinates : np.array
                A numpy array of coordinates. Expecting coordinates in the form that is produced
                by the attention module. (3D array)
    iterations : int
                Number of iterations for SA
    t_max : int
            Temperature to being SA at.
    t_min : int
            Cut-off point for temperature

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

    # Initialise SA, using euclidean distance as objective function
    sa_tsp = SA_TSP(
        func=cal_total_distance,
        x0=range(len(coordinates[0])),
        T_max=t_max,
        T_min=t_min,
        L=iterations,
    )

    # Run SA and report total time as well as final distance
    start_time = time.time()
    best_points, best_distance = sa_tsp.run()

    # print("--- %s seconds ---" % (time.time() - start_time))
    time_taken = (time.time() - start_time)

    return best_distance, best_points, time_taken
