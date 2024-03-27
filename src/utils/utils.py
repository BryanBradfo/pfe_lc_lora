import torch
import numpy as np
from src.decompression.decompress import restoreLinearLayer
from src.compression.LowRankLinear import generate_rank

def evaluate_accuracy(model, test_ds):
    """
    @param model : PyTorch model to evaluate accuracy.
    @param test_ds : Test dataset.

    @return model accuracy.
    """
    model.eval() # Set to eval state.
    t_correct, t_dp = 0, 0
    with torch.no_grad():
        for i, label in test_ds:
            _, opt = torch.max(model(i), dim = 1)
            t_dp += label.size(0)
            t_correct += (opt == label).sum().item()
    acc = t_correct / t_dp
    print("model accuracy: {}".format(acc))
    model.train() # Revert to training state.
    return acc

def lazy_restore(weights, weights_decomp, bias, clean_model, org, decomposed_layers, rank : int = -1, scaling : int = -1):
    """
    @param weights : Decompressed weights of normal model layers.
    @param weights_decomp : Decompressed weights of the decomposed layers.
    @param bias : full bias save.
    @param clean_model : A clean seperate model to test the model on.
    @param rank : The rank of the decomposition used.
    @param scaling : The scaling factor of the model.
    @param org : The original model.
    @param decomposed_layers : list of layers that have undergone decomposition. 
    
    @return lazily restored model (restoration with weights vector settled) for evaluating checkpoint accuracy.
    """
    base_dict = clean_model.state_dict()
    last_idx, last_idx_dcomp = 0, 0
    for layer_name, init_tensor in base_dict.items():
        if "bias" in layer_name:
            base_dict[layer_name] = bias[layer_name]
            continue
        dim = init_tensor.numpy().shape
        if not dim:
            continue
        if layer_name in decomposed_layers: # Restoration procedure for dense layers.
            if rank == -1:
                rr = generate_rank(dim[0], dim[1])
            else:
                rr = rank
            t_element_alpha = dim[0] * rr
            t_element_beta = dim[1] * rr
            alpha = weights_decomp[last_idx_dcomp : last_idx_dcomp + t_element_alpha]
            last_idx_dcomp += t_element_alpha
            beta = weights_decomp[last_idx_dcomp : last_idx_dcomp + t_element_beta]
            last_idx_dcomp += t_element_beta
            alpha = torch.unflatten(torch.from_numpy(np.copy(alpha)), -1, (dim[0], rr))
            beta = torch.unflatten(torch.from_numpy(np.copy(beta)), -1, (rr, dim[1]))
            restored_decomp = restoreLinearLayer(alpha, beta, org[layer_name], scaling)
            base_dict[layer_name] = restored_decomp
        elif "classifier" in layer_name:
            base_dict[layer_name] = bias[layer_name]
        else: # Restoration procedure for convolutional layers.
            t_elements = np.prod(dim)
            needed_ele = weights[last_idx : last_idx + t_elements]
            base_dict[layer_name] = torch.unflatten(torch.from_numpy(np.copy(needed_ele)), -1, dim)
            last_idx += t_elements
    clean_model.load_state_dict(base_dict)
    return clean_model

def evaluate_compression(uncompressed_size, compressed_size):
    """
    @param uncompressed_size : Size of the uncompressed model.
    @param compressed_size : Size of compressed model.

    @return compression_ratio & space savings ratio.
    """
    compression_ratio = round((uncompressed_size / compressed_size), 5) * 100
    space_savings = round(1 - (compressed_size / uncompressed_size), 5) * 100
    return compression_ratio, space_savings
