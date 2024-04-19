import os
import pickle, zlib
import numpy as np
from decimal import *
from collections import defaultdict
import torch
import old_lc.decompression.decompress as decompress
import old_lc.compression.compress as compress

# Trainer handles the training
# Restore handles restoration
# CheckpointManager handles the checkpointing
# LoRAManager handles the LoRA merge / injection
# DeltaManager handles the LC-Checkpoint component of the process.


# old_lc was a recreation of the LC-Checkpoint mechanism from the effective checkpoint paper: https://arxiv.org/abs/2009.13003.

# Elements of it can be seen: https://github.com/yeoshuheng/LC-LoRA/blob/main/src/compression/deltaCompress.py 

# The only distinction between the old_lc and the source is that old_lc considers the delta across the full rank weights. 

# Whereas the delta manager in src considers the delta across full rank weights for the layers which we do not decompose, 
# and the delta across the low rank decompositions for the layers we decompose into LoRA. 

# Otherwise, the rationale between both the delta components of src and the MLP repo remains the same as per the efficient checkpoint paper.


# Functions in old_lc are as per the original effective checkpoint paper I've sent awhile back. 
# The quantization and delta promotion efforts are the same as per deltaManager. 

# It's actually just converting the model weights into a flat tensor, taking the delta, passing it through deltamanager and 
# using the resultant tensor as the checkpoint. The resultant tensor is then used to create a delta for the next timestep.

# For the flattening / unflattening of the model you could look into torch.unflatten and torch.flatten, those should ensure the model 
# weights are properly extracted / rebuilt in the same manner.

import numpy as np
from collections import defaultdict
import zlib, math
import sys
import os
import pickle

def compress_set(filename : str, saveloc : str):
    """
    @param filename : Filename of the set of MLP models to compress. (.tar)
    @param saveloc : Save location for the MLP models.

    Writes compressed set into a seperate directory.
    """
    models = []
    for model in os.listdir(filename):
        if "._" in model:
            continue # Ignore hidden tar files
        models.append(model)
    compressed_deltas = {}
    base, bias = extract_weights(get_state_dict(filename + "/" + models[0]))
    compressed_deltas[models[0]] = (zlib.compress(base), bias)
    for m_ in models[1:]:
        print("Delta Compression on: {}".format(m_))
        curr, bias = extract_weights(get_state_dict(filename + "/" + m_))
        δt = np.subtract(curr, base)
        compressed_δt, lossy_δt = compress.compress_data(δt)
        compressed_deltas[m_] = (compressed_δt, bias)
        base = np.add(base, lossy_δt)
    if not os.path.exists(saveloc):
        os.makedirs(saveloc)
    for key, val in compressed_deltas.items(): # Save process
        print("Saving Compressed Format: {}".format(key))
        idiv_file_path = os.path.join(saveloc, 'compressed_{}.pt'.format(key[:-4]))
        with open(idiv_file_path, 'wb') as f:
            pickle.dump(val, f)

def compress_set_torch(filename : str, saveloc : str):
    """
    @param filename : Filename of the set of MLP models to compress. (.ckpt / .pt)
    @param saveloc : Save location for the MLP models.

    Writes compressed set into a seperate directory.
    """
    models = []
    for model in os.listdir(filename):
        if "._" in model:
            continue # Ignore hidden tar files
        models.append(model)
    compressed_deltas = {}
    base, bias = extract_weights(get_state_dict_torch(filename + "/" + models[0]))
    compressed_deltas[models[0]] = (zlib.compress(base), bias)
    for m_ in models[1:]:
        print("Delta Compression on: {}".format(m_))
        curr, bias = extract_weights(get_state_dict_torch(filename + "/" + m_))
        δt = np.subtract(curr, base)
        compressed_δt, lossy_δt = compress.compress_data(δt)
        compressed_deltas[m_] = (compressed_δt, bias)
        base = np.add(base, lossy_δt)
    if not os.path.exists(saveloc):
        os.makedirs(saveloc)
    for key, val in compressed_deltas.items(): # Save process
        print("Saving Compressed Format: {}".format(key))
        idiv_file_path = os.path.join(saveloc, 'compressed_{}.pt'.format(key.split(".")[0]))
        with open(idiv_file_path, 'wb') as f:
            pickle.dump(val, f)

def read_decompressed_state_dict(filepath):
    """
    @param filepath : Filepath where decompressed state_dict resides.

    @return The decompressed state_dict in dictionary format.
    """
    with open(filepath, 'rb') as f:
            return pickle.load(f)
        
def extract_weights_gpu(sd, saveloc, decomposed_layers):
    """
    @param sd : Initial state_dict of the model.

    @return The base for all delta calculations.
    """
    # print(saveloc)

    if not os.path.exists(saveloc):
        os.makedirs(saveloc)
    # fp = os.path.join(saveloc, "/initial_model.pt")
    print("old_lc | saving full base model @ {}".format(saveloc + "/initial_model.pt"))
    torch.save(sd, saveloc + "/initial_model.pt")

    weights = []
    bias = {}
    for layer_name, weight in sd.items():
        if 'bias' in layer_name:
            continue

        # if layer_name in decomposed_layers:
        #     weights.append(weight)
        #     continue
        
        # elif "classifier" in layer_name:
        #     continue

        else:
            weights.append(weight)
    # return np.concatenate([tensor.flatten().numpy() for tensor in weights]), bias
    return np.concatenate([tensor.cpu().detach().flatten().numpy() for tensor in weights])

      
def extract_weights(sd, saveloc, decomposed_layers):
    """
    @param sd : Initial state_dict of the model.

    @return The base for all delta calculations.
    """
    # print(saveloc)

    if not os.path.exists(saveloc):
        os.makedirs(saveloc)
    # fp = os.path.join(saveloc, "/initial_model.pt")
    print("old_lc | saving full base model @ {}".format(saveloc + "/initial_model.pt"))
    torch.save(sd, saveloc + "/initial_model.pt")

    weights = []
    bias = {}
    for layer_name, weight in sd.items():
        if 'bias' in layer_name:
            continue

        # if layer_name in decomposed_layers:
        #     weights.append(weight)
        #     continue
        
        # elif "classifier" in layer_name:
        #     continue

        else:
            weights.append(weight)
    # return np.concatenate([tensor.flatten().numpy() for tensor in weights]), bias
    return np.concatenate([tensor.flatten().numpy() for tensor in weights])


def get_state_dict(filename : str) -> dict:
    """
    @param filename : Model checkpoint file. (If .tar)

    @return State dictionary of original model.
    """
    print("Loading: {}".format(filename))
    return torch.load(filename, map_location = torch.device('cpu'))["network_fn_state_dict"]

def get_state_dict_torch(filename : str) -> dict:
    """
    @param filename : Model checkpoint file. (If torch.pt / .ckpt)

    @return State dictionary of original model.
    """
    print("Loading: {}".format(filename))
    return torch.load(filename)


def load_compressed_set(filepath : str, saveloc : str, original_weight_dict : dict):
    """
    @param filepath : Filepath of set of MLP to load checkpoints from.
    @param saveloc : Filepath to save restored models to.
    @param original_weight_dict : The structure for the model to load into;
        Can be a randomly initialised weight dictionary.
    
    @return The decompressed MLP state dicts.
    """
    compressed_models = {}
    for model in os.listdir(filepath):
        with open(filepath + "/" + model, 'rb') as f:
            model_ = pickle.load(f)
        compressed_models[model] = model_
    base = None
    start = False
    for name, encoded_checkpoint in compressed_models.items():
        print("Decompressing for: {}".format(name))
        decoded_checkpoint = decompress.decode_data(encoded_checkpoint[0])
        bias = encoded_checkpoint[1]
        if not start: 
            base = decoded_checkpoint
            start = True
        else:
            base = np.add(base, decoded_checkpoint)
        base_dict = original_weight_dict.copy()
        compressed_models[name] = decompress.restore_state_dict(base, bias, base_dict)
    if not os.path.exists(saveloc):
        os.makedirs(saveloc)
    for name, full_state_dict in compressed_models.items():
        new_name = "decompressed_" + name.split("compressed_")[-1]
        print("Saving Decompressed Model at: {}".format(new_name))
        idiv_file_path = os.path.join(saveloc, new_name)
        with open(idiv_file_path, 'wb') as f:
            pickle.dump(full_state_dict, f)



############################################################################################################

def generate_delta_gpu(weight_prev, sd_curr, decomposed_layers):
    """
    @param base : The base for all delta calculations.
    @param curr : The current state of the model.

    @return The delta between the base and the current model state.
    """
    weights_curr = []

    full = {}

    for k in sd_curr:

        # If bias, skip
        if "bias" in k:
            full[k] = sd_curr[k]
            continue
        
        # if k in decomposed_layers:
        #     weights_curr.append(sd_curr[k]) #TO COMMENT IF PROBLEM
        #     continue

        # # if layer classifier, add it to the full dictionary
        # elif "classifier" in k:
        #     full[k] = sd_curr[k]
        #     continue

        # Otherwise, extract weights for prev and current layer
        else:
            weights_curr.append(sd_curr[k])


    # Generate weight delta

    curr_flatten = np.concatenate([tensor.cpu().detach().numpy().flatten() for tensor in weights_curr])

    # Generate delta btw current and previous
    weight_delta = np.subtract(curr_flatten, weight_prev)

    return weight_delta, full


def generate_delta(weight_prev, sd_curr, decomposed_layers):
    """
    @param base : The base for all delta calculations.
    @param curr : The current state of the model.

    @return The delta between the base and the current model state.
    """
    weights_curr = []

    full = {}

    for k in sd_curr:

        # If bias, skip
        if "bias" in k:
            full[k] = sd_curr[k]
            continue
        
        # if k in decomposed_layers:
        #     weights_curr.append(sd_curr[k]) #TO COMMENT IF PROBLEM
        #     continue

        # # if layer classifier, add it to the full dictionary
        # elif "classifier" in k:
        #     full[k] = sd_curr[k]
        #     continue

        # Otherwise, extract weights for prev and current layer
        else:
            weights_curr.append(sd_curr[k])


    # Generate weight delta

    curr_flatten = np.concatenate([tensor.numpy().flatten() for tensor in weights_curr])

    # Generate delta btw current and previous
    weight_delta = np.subtract(curr_flatten, weight_prev)

    return weight_delta, full

# def represent_buckets(deltas):
#     """
#     Répartit les deltas en buckets en fonction de leur exposant et représente
#     chaque bucket par la moyenne des valeurs max et min.
    
#     @param deltas: Liste des deltas à traiter.
    
#     @return: Un dictionnaire avec l'exposant comme clé et la représentation moyenne comme valeur.
#     """
#     bucket_dict = {}
#     for delta in deltas:
#         # Calcul de s, m, et e pour chaque delta, cela dépend de la représentation de vos deltas
#         s = np.sign(delta)
#         m, e = np.frexp(delta)  # m est la mantisse et e est l'exposant
        
#         if e not in bucket_dict:
#             bucket_dict[e] = {'max': delta, 'min': delta}
#         else:
#             if delta > bucket_dict[e]['max']:
#                 bucket_dict[e]['max'] = delta
#             if delta < bucket_dict[e]['min']:
#                 bucket_dict[e]['min'] = delta
    
#     # Calcul de la moyenne des valeurs max et min pour chaque bucket
#     for e in bucket_dict:
#         bucket_dict[e] = np.mean([bucket_dict[e]['max'], bucket_dict[e]['min']])
    
#     return bucket_dict

# def priority_promotion(bucket_dict, x):
#     """
#     Garde les 2^x - 1 buckets avec les plus grands exposants et fusionne les autres.
    
#     @param bucket_dict: Dictionnaire des buckets et de leurs représentations.
#     @param x: Nombre de bits pour la promotion prioritaire.
    
#     @return: Dictionnaire des buckets après promotion prioritaire.
#     """
#     # Trier les buckets par exposant (clé) en ordre décroissant
#     sorted_buckets = sorted(bucket_dict.items(), key=lambda item: item[0], reverse=True)
    
#     # Garder les 2^x - 1 buckets avec les plus grands exposants
#     promoted_buckets = dict(sorted_buckets[:2**x - 1])
    
#     # Fusionner les buckets restants, si nécessaire
#     if len(sorted_buckets) > 2**x - 1:
#         merged_value = 0  # Valeur pour les buckets fusionnés
#         for e, value in sorted_buckets[2**x - 1:]:
#             merged_value += value  # Vous pourriez vouloir ajuster cette opération
#         merged_value /= len(sorted_buckets[2**x - 1:])  # Moyenne, ajustez si nécessaire
#         promoted_buckets[0] = merged_value  # Utilisez un identifiant spécial pour le bucket fusionné
#     # print length of promoted_buckets
#     print(len(promoted_buckets))
#     return promoted_buckets


# def compress_delta(delta, num_bits):
#     """
#     Compresse un delta en utilisant la représentation des buckets et la promotion prioritaire.
    
#     @param delta: Delta à compresser.
#     @param num_bits: Nombre de bits pour la promotion prioritaire.
    
#     @return: Delta compressé.
#     """
#     bucket_dict = represent_buckets(delta)
#     promoted_buckets = priority_promotion(bucket_dict, num_bits)
#     # Sérialisation des buckets promus pour compression
#     serialized_buckets = pickle.dumps(promoted_buckets)
    
#     # Étape 3: Compression avec zlib
#     compressed_delta = zlib.compress(serialized_buckets)
    
#     return compressed_delta
#     # compressed_delta = []
#     # for d in delta:
#     #     e = np.frexp(d)[1]  # Exposant du delta
#     #     if e in promoted_buckets:
#     #         compressed_delta.append(promoted_buckets[e])
#     #     else:
#     #         compressed_delta.append(promoted_buckets[0])  # Utilisez l'identifiant spécial pour les buckets fusionnés
    
#     # return np.array(compressed_delta)

# def save_checkpoint(saveloc, checkpoint_weights, checkpoint_bias, checkpoint_id):
#     """
#     @param saveloc : Save location for the checkpoint.
#     @param checkpoint_weights : The checkpoint weights.
#     @param checkpoint_bias : The checkpoint bias.
#     @param epch : The epoch of the checkpoint.
#     @param checkpoint_id : The checkpoint id.

#     Saves the checkpoint.
#     """
#     if not os.path.exists(saveloc):
#         os.makedirs(saveloc)
#     checkpoint_name = "old_lc_checkpoint_{}.pt".format(checkpoint_id)
#     print("Saving Checkpoint: {} @ {}".format(checkpoint_name, saveloc))
#     fp = os.path.join(saveloc, checkpoint_name)
#     with open(fp, 'wb') as f:
#         print("from old_lc")
#         print("Weight size in kilo octets:", len(checkpoint_weights) / 1024)
#         print("Bias size in kilo octets:", len(checkpoint_bias) / 1024)
#         print("Total size in kilo octets:", (len(checkpoint_weights) + len(checkpoint_bias)) / 1024)
#         # print("Checkpoint weights", checkpoint_weights)
#         # print("Checkpoint bias", checkpoint_bias)
#         pickle.dump((checkpoint_weights, checkpoint_bias), f)

#     # checkpoint_name_weight = "old_lc_checkpoint_weight_{}_{}.pt".format(epch, checkpoint_id)
#     # fp_weights = os.path.join(saveloc, checkpoint_name_weight)
#     # with open(fp_weights, 'wb') as f:
#     #     pickle.dump(checkpoint_weights, f)

#     # checkpoint_name_bias = "old_lc_checkpoint_bias_{}_{}.pt".format(epch, checkpoint_id)
#     # fp_bias = os.path.join(saveloc, checkpoint_name_bias)
#     # with open(fp_bias, 'wb') as f:
#     #     pickle.dump(checkpoint_bias, f)



import os
import pickle
import sys

def save_checkpoint(saveloc, checkpoint_weights, checkpoint_bias, checkpoint_id):
    """
    Saves the checkpoint.

    @param saveloc : Save location for the checkpoint.
    @param checkpoint_weights : The checkpoint weights.
    @param checkpoint_bias : The checkpoint bias.
    @param checkpoint_id : The checkpoint id.
    """

    if not os.path.exists(saveloc):
        os.makedirs(saveloc)

    checkpoint_name = "old_lc_checkpoint_{}.pt".format(checkpoint_id)
    print("Saving Checkpoint: {} @ {}".format(checkpoint_name, saveloc))
    fp = os.path.join(saveloc, checkpoint_name)

    # Calculate sizes before serialization
    # weights_size = sys.getsizeof(checkpoint_weights)
    # bias_size = sys.getsizeof(checkpoint_bias)
    # total_size = (weights_size + bias_size) / 1024

    # print("from old_lc")
    # print("Weight size in kilo octets:", weights_size / 1024)
    # print("Bias size in kilo octets:", bias_size / 1024)
    # print("Total size in kilo octets before serialization:", total_size)

    # print("Checkpoint bias:",checkpoint_bias.keys())
    # print("Checkpoint weights:",checkpoint_weights)

    with open(fp, 'wb') as f:
        pickle.dump((checkpoint_weights, checkpoint_bias), f)

    # Get size of the serialized file
    # file_size = os.path.getsize(fp) / 1024
    # print(f"Size of serialized file: {file_size} kilo octets")







# import pickletools

# def total_size(o, handlers={}):
#     """ Calculate the total memory footprint of an object including the content of containers. """
#     from types import ModuleType, FunctionType
#     from gc import get_referents

#     # Custom objects know their class.
#     # Function objects seem to know way too much, including modules.
#     # Exclude modules as well.
#     blacklist = type, ModuleType, FunctionType

#     if isinstance(o, blacklist):
#         # Basic type or not an object we want to deal with.
#         raise TypeError("getsize() does not take argument of type: " + str(type(o)))
#     seen = set()
#     default_size = sys.getsizeof(0)  # Default size of zero, used for base case.

#     def inner(obj):
#         if id(obj) in seen:
#             # If the object has already been processed, do not process it again
#             return 0
#         seen.add(id(obj))
#         size = sys.getsizeof(obj, default_size)

#         if isinstance(obj, (str, bytes, bytearray)):
#             return size

#         if isinstance(obj, dict):
#             size += sum(inner(k) + inner(v) for k, v in obj.items())

#         elif hasattr(obj, '__dict__'):
#             size += inner(vars(obj))

#         elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
#             size += sum(inner(i) for i in obj)

#         return size

#     return inner(o)

# def save_checkpoint(saveloc, checkpoint_weights, checkpoint_bias, checkpoint_id):
#     """
#     @param saveloc : Save location for the checkpoint.
#     @param checkpoint_weights : The checkpoint weights.
#     @param checkpoint_bias : The checkpoint bias.
#     @param checkpoint_id : The checkpoint id.

#     Saves the checkpoint.
#     """
#     if not os.path.exists(saveloc):
#         os.makedirs(saveloc)
#     checkpoint_name = "old_lc_checkpoint_{}.pt".format(checkpoint_id)
#     print("Saving Checkpoint: {} @ {}".format(checkpoint_name, saveloc))
#     fp = os.path.join(saveloc, checkpoint_name)
#     with open(fp, 'w+') as f:
#         print("from old_lc")
#         data = f.read()
#         pickletools.dis(data)
#         # print("Weight size in kilo octets:", total_size(checkpoint_weights) / 1024)
#         # print("Bias size in kilo octets:", total_size(checkpoint_bias) / 1024)
#         # total = total_size(checkpoint_weights) + total_size(checkpoint_bias)
#         # print("Total size in kilo octets:", total / 1024)
#         pickle.dump((checkpoint_weights, checkpoint_bias), f)

#     file_size_in_kb = os.path.getsize(fp) / 1024
#     # print(f"Total size in kilo octets (after pickle): {file_size_in_kb:.2f} Ko")

# You would call save_checkpoint with the required parameters like this:
# save_checkpoint('/path/to/save', weights, biases, checkpoint_id)


def load_checkpoint(filepath):
    """
    @param full_path : The full_path of the checkpoint to be reloaded from.
    """
    with open(filepath, "rb") as f:
        checkpoint_weights, checkpoint_bias = pickle.load(f)
    decompressed_weights = decompress.decode_data(checkpoint_weights)
    return decompressed_weights, checkpoint_bias


# def compress_data(δt, num_bits = 10, threshhold=True):
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

# def compress_data(δt, num_bits=10, threshold=True):
#     """
#     @param δt: The delta to compress.
#     @param num_bits: The number of bits to limit Huffman encoded variables to.
#     @param threshold: Enabler for priority promotion process.

#     @return: Zlib compressed promoted delta and uncompressed version.
#     """
#     # Extraction de l'exposant et du signe de δt
#     _, δt_exp = np.frexp(δt)
#     δt_sign = np.sign(δt).astype(int)
#     δt_sign[δt_sign > 0] = 0
#     δt_sign[δt_sign < 0] = 1

#     # Groupement des valeurs par exposant et signe
#     mp = defaultdict(list)
#     for i in range(len(δt)):
#         mp[(δt_exp[i], δt_sign[i])].append(δt[i])

#     # Calcul de la moyenne pour chaque groupe
#     for k in mp:
#         mp[k] = np.mean(mp[k])

#     # Conversion du dictionnaire en liste et application du seuil si activé
#     mp_items = list(mp.items())
#     if threshold:
#         allowed_buckets = int(math.pow(2, num_bits) - 1)
#         mp_items = sorted(mp_items, key=lambda x: abs(x[1]), reverse=True)[:allowed_buckets]

#     # Reconstruction des deltas basés sur les moyennes calculées
#     new_δt = np.zeros_like(δt, dtype=np.float32)
#     for (exp, sign), val in mp_items:
#         for i, (exp_i, sign_i) in enumerate(zip(δt_exp, δt_sign)):
#             if exp == exp_i and sign == sign_i:
#                 new_δt[i] = val

#     # Sérialisation de la structure de données avant compression
#     serialized_data = pickle.dumps(new_δt)
    
#     # Compression avec zlib
#     compressed_data = zlib.compress(serialized_data)

#     return compressed_data, new_δt


# def compress_data(δt, num_bits=10, threshold=True):
#     """
#     Compresse δt en utilisant la quantification basée sur l'exposant et la promotion prioritaire,
#     puis applique la compression zlib.

#     @param δt : Les deltas à compresser.
#     @param num_bits : Le nombre de bits pour la promotion prioritaire.
#     @param threshold : Activer ou non la promotion prioritaire.

#     @return : Les données compressées et la nouvelle version de δt.
#     """
#     # Quantification basée sur l'exposant
#     _, δt_exp = np.frexp(δt)  # Extrait les exposants
#     δt_buckets = defaultdict(list)
#     for val, exp in zip(δt, δt_exp):
#         δt_buckets[exp].append(val)
    
#     # Représentation de chaque bucket par la moyenne des valeurs max et min
#     for exp in δt_buckets:
#         δt_buckets[exp] = np.mean([max(δt_buckets[exp]), min(δt_buckets[exp])])
    
#     # Promotion prioritaire
#     if threshold:
#         # Sélection des 2^num_bits - 1 plus grands exposants
#         allowed_keys = sorted(δt_buckets.keys(), reverse=True)[:2**num_bits - 1]
#         new_δt_buckets = {k: δt_buckets[k] for k in allowed_keys}
#         # Fusion des autres buckets en leur attribuant la valeur 0
#         for k in set(δt_buckets) - set(allowed_keys):
#             new_δt_buckets[k] = 0
#     else:
#         new_δt_buckets = δt_buckets
    
#     # Reconstruction de δt basée sur les buckets quantifiés et promus
#     new_δt = np.array([new_δt_buckets[exp] for exp in δt_exp])
    
#     # Compression
#     serialized_data = pickle.dumps(new_δt)  # Sérialisation
#     compressed_data = zlib.compress(serialized_data)  # Compression zlib
    
#     return compressed_data, new_δt


# def compress_data(δt, num_bits=2, threshold=True):
#     """
#     @param δt : The delta to compress.
#     @param num_bits : The number of bits to limit the huffman encoded variables to.
#     @param threshold : Enabler for priority promotion process.

#     @return : Zlib compressed promoted delta and uncompressed version.
#     """
#     print("length of δt: ", len(δt))
#     # Promote the most significant changes (deltas) based on their magnitude and sign.
#     _, δt_exp = np.frexp(δt)  # Extract the exponent of δt, ignoring mantissa.
#     δt_sign = np.sign(δt)  # Get the sign of δt.
#     # Convert sign information to binary (0 for positive, 1 for negative).
#     δt_sign[δt_sign > 0] = 0
#     δt_sign[δt_sign < 0] = 1

#     # Group deltas by their exponent and sign.
#     mp = defaultdict(list)
#     for i in range(len(δt)):
#         mp[(δt_exp[i], δt_sign[i])].append(δt[i])

#     # Average the deltas within each group.
#     for k in mp:
#         # mp[k] = np.mean(mp[k])
#         mp[k] = np.mean([max(mp[k]), min(mp[k])])

#     # If threshold is enabled, select the most significant changes.
#     if threshold:
#         allowed_buckets = int(math.pow(2, num_bits) - 1)
#         # Sort groups by their absolute mean value and select top significant groups.
#         significant_deltas = sorted(mp.items(), key=lambda x: abs(x[1]), reverse=True)[:allowed_buckets]
#         mp = {k: v for k, v in significant_deltas}
#     # Apply the averaged value to all elements in each group.
#     promoted_δt = np.array([mp.get((exp, sign), 0) for exp, sign in zip(δt_exp, δt_sign)], dtype=np.float32)

#     # Zlib compression of the promoted delta.
#     compressed_promoted_δt = zlib.compress(promoted_δt)

#     return compressed_promoted_δt, promoted_δt

def compress_data(δt, num_bits = 3, threshhold=True):
    """
    @param δt : The delta to compress.
    @param num_bits : The number of bits to limit huffman encoded variables to.
    @param treshold : Enabler for priority promotion process.

    @return Zlib compressed promoted delta and uncompressed version.
    """
    _, δt_exp = np.frexp(δt)
    δt_sign = np.sign(δt)
    δt_sign[δt_sign > 0] = 0
    δt_sign[δt_sign < 0] = 1    
    mp =  defaultdict(list)
    for i in range(len(δt)):
        mp[(δt_exp[i], δt_sign[i])].append((i, δt[i]))
    for k in mp:
        mp[k] = (np.average(np.array([x[-1] for x in mp[k]])), 
                 [x[0] for x in mp[k]])
    mp = list(mp.values())
    # print("len(mp):", len(mp))
    if threshhold:
        allowed_buckets = int(math.pow(2, num_bits)-1)
        # print("allowed_buckets:", allowed_buckets)
        # print("min between len(mp) and allowed_buckets:", min(allowed_buckets, len(mp)))
        mp = sorted(mp, key = lambda x : abs(x[0]), reverse = True)[:min(allowed_buckets, len(mp))]
        # print("len(mp):", len(mp))
    new_δt= [0 for x in range(len(δt))]
    for qtVal, pos in mp:
        for p in pos:
            new_δt[p] = qtVal
    new_δt = np.array(new_δt, dtype = np.float32)
    # # Compress the data
    # compressed_data = zlib.compress(new_δt)
    # # Get the size of the original data
    # original_size = len(new_δt)
    # # Get the size of the compressed data
    # compressed_size = len(compressed_data)
    # # Calculate the compression ratio
    # compression_ratio = original_size / compressed_size
    # print("from old_lc")
    # print("Original size:", original_size, "bytes")
    # print("Compressed size:", compressed_size, "bytes")
    # print("Compression ratio:", compression_ratio)
    return zlib.compress(new_δt), new_δt

# def compress_data(δt, num_bits=10, threshold=True):
#     """
#     @param delta: The delta obtained between normal layers.
#     @param decomposed_delta: The delta obtained between decomposed layers.

#     @return Quantized deltas as well as new deltas to replace the initial base.
#     """
#     compressed_weight_delta, full_delta = compress.compress_data(δt, num_bits = 3)
#     return compressed_weight_delta, full_delta


def restore_state_dict(decoded_checkpoint, bias, base_dict, decomposed_layers):
    """
    @param decoded_checkpoint: The decoded_checkpoint from zlib.
    @param bias : The bias dictionary of the model.
    @param base_dict : The base dictionary of the model which helps us understand its structure.

    @return Restored state_dict.
    """
    last_idx = 0
    last_idx_dcomp = 0
    for layer_name, init_tensor in base_dict.items():
        dim = init_tensor.numpy().shape
        if not dim:
            continue
        if "bias" in layer_name:
            base_dict[layer_name] = bias[layer_name]
            continue
        # if layer_name in decomposed_layers: #for the fully connected layers
        #     # t_element_decomp = dim[0] * dim[1]
        #     # needed_ele_decomp = decoded_checkpoint[last_idx_dcomp : last_idx_dcomp + t_element_decomp]
        #     # last_idx_dcomp += t_element_decomp
        #     # base_dict[layer_name] = torch.unflatten(torch.from_numpy(np.copy(needed_ele_decomp)), -1, dim)
        #     continue
        # elif "classifier" in layer_name:
        #     base_dict[layer_name] = bias[layer_name]
            # continue
        else:
            t_elements = np.prod(dim)
            needed_ele = decoded_checkpoint[last_idx : last_idx + t_elements]
            last_idx += t_elements
            base_dict[layer_name] = torch.unflatten(torch.from_numpy(np.copy(needed_ele)), -1, dim)
    return base_dict