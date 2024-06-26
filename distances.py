# distances.py

"""
Distância de Manhattan
A1 = [v1, v2, v3]
A2 = [v4, v5, v6]

dm = |v4-v1| + |v5-v2| + |v6-v3|
"""

def manhattan_distance(
    p_a1,
    p_a2
):
    sum = 0
    b_can_compute = len(p_a1)==len(p_a2)
    if(b_can_compute):
        for idx in range(len(p_a1)):
            el1 = p_a1[idx]
            el2 = p_a2[idx]
            dif = abs(el2-el1)
            sum += dif
        # for
        return sum
    # if
    return "No se puede"
# def manhattan_distance

v1 = [1,2,3]
v2 = [1,1,4]
v3 = [10, 1, 8]

d_v1_v2 = manhattan_distance(v1, v2)
print(f"dm(v1,v2)={d_v1_v2}")
d_v1_v3 = manhattan_distance(v1, v3)
print(f"dm(v1,v3)={d_v1_v3}")

import numpy as np

def manhattan_distance_vNP(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    dif = np.abs(v2-v1)
    return np.sum(dif)
# def manhattan_distance_vNP

d_v1_v2 = manhattan_distance_vNP(v1, v2)
print(f"dm(v1,v2)={d_v1_v2}")
d_v1_v3 = manhattan_distance_vNP(v1, v3)
print(f"dm(v1,v3)={d_v1_v3}")


