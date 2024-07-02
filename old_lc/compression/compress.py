import numpy as np
from collections import defaultdict
import zlib, math

#####


def generate_decomposed_names(original_layers):
    """
    @param original_layers : List of original layer names (pre-decomposition)

    @return New decomposed layers name with respect to the decomposition of the model.
        <original>.weight will be returned as <original>.beta and <original>.alpha.
    """
    decomposed_layers = []
    for dcl in original_layers:
        newname = ".".join(dcl.split(".")[:-1])
        decomposed_layers.append(newname + ".alpha")
        decomposed_layers.append(newname + ".beta")
    return decomposed_layers

#####

# def compress_data(δt, num_bits = 3, threshhold=True):
#     """
#     @param δt : The delta to compress.
#     @param num_bits : The number of bits to limit huffman encoded variables to.
#     @param treshold : Enabler for priority promotion process.

#     @return Zlib compressed promoted delta and uncompressed version.
#     """
#     _, δt_exp = np.frexp(δt)
#     δt_sign = np.sign(δt)
#     δt_sign[δt_sign > 0] = 0
#     δt_sign[δt_sign < 0] = 1    
#     mp =  defaultdict(list)
#     for i in range(len(δt)):
#         mp[(δt_exp[i], δt_sign[i])].append((i, δt[i]))
#     for k in mp:
#         mp[k] = (np.average(np.array([x[-1] for x in mp[k]])), 
#                  [x[0] for x in mp[k]])
#     mp = list(mp.values())
#     if threshhold:
#         allowed_buckets = int(math.pow(2, num_bits) - 1)
#         mp = sorted(mp, key = lambda x : abs(x[0]), reverse = True)[:min(allowed_buckets, len(mp))]
#     new_δt= [0 for x in range(len(δt))]
#     for qtVal, pos in mp:
#         for p in pos:
#             new_δt[p] = qtVal
#     new_δt = np.array(new_δt, dtype = np.float32)
#     return zlib.compress(new_δt), new_δt