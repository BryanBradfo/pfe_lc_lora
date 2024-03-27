import os
import pickle
import numpy as np
from decimal import *
import torch
import src.decompression.decompress as decompress
import src.compression.deltaCompress as compress
from src.utils.utils import lazy_restore

def compress_delta(weight_delta, decomposed_delta):
    """
    @param delta: The delta obtained between normal layers.
    @param decomposed_delta: The delta obtained between decomposed layers.

    @return Quantized deltas as well as new deltas to replace the initial base.
    """
    compressed_weight_delta, full_delta = compress.compress_data(weight_delta, num_bits = 3)
    compressed_decomposed_delta, decomp_full_delta = compress.compress_data(decomposed_delta, num_bits = 3)
    return compressed_weight_delta, full_delta, compressed_decomposed_delta, decomp_full_delta

def extract_weights(initmodel, saveloc, decomposed_layers, restoring = False):
    """
    @param initmodel : Initial run of the model.
    @param saveloc : The save location for the current model training process.
    @param restoring : If it is currently being used for model restoration, 
        which does not require another full-save
    @param decomposed_layers : Names of the decomposed layers.
    @return The base for all delta calculations.
    """
    if not restoring:
        wd = initmodel.state_dict()
    else:
        wd = initmodel # If we are restoring, this will already be a state dictionary.

    if not restoring:
        # Save current model state_dict for restoration of weights.
        if not os.path.exists(saveloc):
            os.makedirs(saveloc)
        fp = os.path.join(saveloc, "base_model.pt")
        print("saving full base model @ {}".format(fp))
        torch.save(wd, fp)

    # Generate base layer of weights (0-th state) for delta to build on.
    decomposed_layers = compress.generate_decomposed_names(decomposed_layers)
    weights, decomposed_weights = [], []
    for k, v in wd.items():
        if "bias" in k:
            continue
        if k in decomposed_layers:
            decomposed_weights.append(v)
            continue
        elif "classifier" in k:
            continue
        else:
            weights.append(v)
    weights = np.concatenate([tensor.flatten().numpy() for tensor in weights])
    decomposed_weights = np.concatenate([tensor.flatten().numpy() for tensor in decomposed_weights])
    return weights, decomposed_weights

def full_snapshot(current_base, decomp_base, bias, 
                  clean_model, rank, org, clean_model_lora_f, saveloc, 
                  decomposed_layers, decomposed_bias):
    """
    @param base_weights : The current non-decomposition weights base used in the delta calculations.
    @param decomp_base : The current decomposed weights used in the delta calculations.
    @param bias : The current bias used in the delta calculations.
    @param clean_model : A clean version of the non-loRA model to merge the loRA layers into.
    @param rank : The rank used in the loRA decomposition.
    @param original : The origin model (base model in the set the LoRA to be saved came from).
    @param clean_model_lora_f : A function to build the clean version of the 
        loRA model to load the current weights into with a fresh set of low-rank weights.
        Can be written as a anon functions, eg. lambda x : model(x)
    @param saveloc : The save location to store the full snapshot in.
    @param decomposed_layers : The layers which contain decomposed weights.
    @param decomposed_bias : The bias layers affected by the decomposition of the weights.

    @return Saves a full snapshot of the current loRA model as a full model whilst 
        rebuilding a new LoRA model for training.
    """
    clean_model = lazy_restore(current_base, decomp_base, bias, clean_model, rank, org, decomposed_layers)
    if not os.path.exists(saveloc):
            os.makedirs(saveloc)
    fp = os.path.join(saveloc, "base_model.pt")
    torch.save(clean_model.state_dict(), fp)
    weights, bias = [], []
    for k, v in clean_model.state_dict().items():
        if k in decomposed_layers:
            weights.append(v)
            continue
        if k in decomposed_bias:
            bias.append(v)
    clean_model_lora = clean_model_lora_f(weights, bias, rank = rank)
    wd = clean_model_lora.items()
    for k, v in clean_model.state_dict().items():
        if k in decomposed_bias or k in decomposed_layers:
            continue
        wd[k] = v
    clean_model_lora.load_state_dict(wd)
    return clean_model_lora

def generate_delta(weights_prev : np.array, decomposed_weights_prev : np.array, sd_curr, decomposed_layers):
    """
    @param weights_prev : The previous weights of non-decomposed layers.
    @param decomposed_weights_prev : The previous weights of decomposed layers.
    @param sd_curr : The state dictionary of the current weights.
    @param decomposed_layers : layers that have undergone decomposition.

    @return The delta for the weights of the normal and decomposed layers.
    Also returns the full dictionary, which holds the bias.
    """
    weights_curr, decomposed_weights_curr = [], []
    decomposed_layers = compress.generate_decomposed_names(decomposed_layers)
    full = {} # Store layers that require full save (bias layers)
    for k in sd_curr:
        if "bias" in k:
            full[k] = sd_curr[k]
            continue
        if k in decomposed_layers:
            decomposed_weights_curr.append(sd_curr[k])
            continue
        elif "classifier" in k:
            full[k] = sd_curr[k]
            continue
        else: # Extract weights for prev and current layer.
            weights_curr.append(sd_curr[k])
        
    # Generate weight delta.
    curr_flatten = np.concatenate([tensor.numpy().flatten() for tensor in weights_curr])
    decomposed_curr_flatten = np.concatenate([tensor.numpy().flatten() for tensor in decomposed_weights_curr])
    weight_delta = np.subtract(curr_flatten, weights_prev)
    decomposed_weight_delta = np.subtract(decomposed_curr_flatten, decomposed_weights_prev)
    
    return weight_delta, decomposed_weight_delta, full

def save_checkpoint(checkpoint_weights, decomposed_weights, checkpoint_bias, checkpoint_id, saveloc):
    """
    @param checkpoint_weights : The delta of the weights data of the checkpoint (post GZIP Compression).
    @param decomposed_weights : The delta of the decomposed weights of the model (post GZIP Compression).
    @param checkpoint_bias : The bias of the checkpoint to be saved.
    @param checkpoint_id : The ID of the checkpoint to be saved.
    @param saveloc : The filepath for the training process to be saved in.
    """
    if not os.path.exists(saveloc):
        os.makedirs(saveloc)
    checkpoint_name = "lc_checkpoint_{}.pt".format(checkpoint_id)
    print("Saving Checkpoint {} @ {}".format(checkpoint_name, saveloc))
    fp = os.path.join(saveloc, checkpoint_name)
    with open(fp, "wb") as f:
        pickle.dump((checkpoint_weights, decomposed_weights, checkpoint_bias), f)
    
def load_checkpoint(full_path):
    """
    @param full_path : The full_path of the checkpoint to be reloaded from.
    """
    with open(full_path, "rb") as f:
        checkpoint_weights, decomposed_weights, checkpoint_bias = pickle.load(f)
    decompressed_weights = decompress.decode_data(checkpoint_weights)
    decompressed_dcomp = decompress.decode_data(decomposed_weights)
    return decompressed_weights, decompressed_dcomp, checkpoint_bias

def restore_checkpoint(model, saveloc_, set_id, id, decomposed_layers, rank = -1, scaling = -1):
    """
    @param model : The model to load the checkpoint weights into.
    @param saveloc_ : The filepath for the training process to be restored from.
    @param set_id : The set id to restore the model from.
    @param id : The ID with respect to the current file for 
            the training process to be restored from, ID given is treated as inclusive.
            Note that valid IDs starts from 1 onwards (0-th ID is the full save).
    @param rank : The rank specified within the decomposition process.
    @param scaling : The scaling factor used.
    @param decomposed_layers : Names of the decomposed layers.
    @return model with restored weights included.
    """
    saveloc = os.path.join(saveloc_, "set_{}".format(set_id))
    fp = os.path.join(saveloc, "base_model.pt")
    
    if id == 0: # id 0 == base model.
        model.load_state_dict(torch.load(fp))
        return model
    
    og_sd = torch.load(fp)
    org_weight, org_decomposed_weights = extract_weights(og_sd, saveloc, 
                                                         decomposed_layers, restoring = True)
    base = org_weight.copy()
    base_decomposed = org_decomposed_weights.copy()

    # Adding delta back to base.
    for i in range(0, id + 1):
        fp = os.path.join(saveloc, "lc_checkpoint_{}.pt".format(i))
        decompressed_weights, decompressed_dcomp_weights, _ = load_checkpoint(fp)
        base = np.add(base, decompressed_weights)
        base_decomposed = np.add(base_decomposed, decompressed_dcomp_weights)
    
    # Get full bias of final model.
    fp = os.path.join(saveloc, "lc_checkpoint_{}.pt".format(id))
    _, _, full_bias = load_checkpoint(fp)

    # Get the LoRA base of the model.
    lora_fp = os.path.join(saveloc, "lora_bases.pt")
    lora_bases = torch.load(lora_fp)

    new_sd = decompress.restore_state_dict(decoded_checkpoint=base, decoded_decomp_checkpoint=base_decomposed, 
                                           lora_bases=lora_bases, bias=full_bias, 
                                           base_dict=model.state_dict(),
                                           decomposed_layers=decomposed_layers, 
                                           rank=rank, scaling=scaling)
    model.load_state_dict(new_sd)
    return model