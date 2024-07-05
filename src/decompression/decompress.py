import numpy as np
import pathos.multiprocessing as pmp
import torch, zlib
from src.compression.LowRankLinear import generate_rank

def decode_data(checkpoint):
    """
    @param checkpoint : GZIP Encoded checkpoint

    @return : Decoded checkpoint.
    """

    decoded = zlib.decompress(checkpoint)
    return np.frombuffer(decoded, dtype = np.float32)

def restoreLinearLayer(alpha, beta, base, scaling):
    """
    @param alpha : Left component of the decomposition.
    @param beta : Right component of the decomposition.
    @param scaling : The scaling factor of the model.

    @return The converted weights of the original model according to the decomposition.
    """
    device = base.device  # Get the device of the base tensor
    if scaling == -1:
        scaling = 0.5
    return torch.add(base, scaling * torch.matmul(alpha.to(device), beta.to(device)))


    # if scaling == -1:
    #     scaling = 0.5
    # return torch.add(base, scaling * torch.matmul(alpha, beta))

def restore_state_dict(decoded_checkpoint, decoded_decomp_checkpoint, lora_bases, bias, base_dict,
                        decomposed_layers, rank = -1, scaling = -1):
    """
    @param decoded_checkpoint: The decoded checkpoint of normal weights from zlib.
    @param decoded_decomp_checkpoint: The decoded checkpoint of decomposed weights from zlib.
    @param lora_bases : 
    @param bias : The bias dictionary of the model.
    @param base_dict : The base dictionary of the model which helps us understand its structure.
    @param rank : The rank of the decomposition used for the linear layers.
    @param scaling : The scaling factor of the linear layers (default put -1)
    @param decomposed_layers : list of layers that have undergone decomposition. 

    @return Restored state_dict.
    """
    last_idx, last_idx_dcomp = 0, 0
    for layer_name, init_tensor in base_dict.items():
        if "bias" in layer_name:
            base_dict[layer_name] = bias[layer_name]
            continue
        dim = init_tensor.cpu().detach().numpy().shape
        if not dim:
            continue
        if layer_name in decomposed_layers: # Restoration procedure for dense layers.
            if rank == -1:
                rr = rr = generate_rank(dim[0], dim[1])
            else:
                rr = rank
            t_element_alpha = dim[0] * rr
            t_element_beta = dim[1] * rr
            alpha = decoded_decomp_checkpoint[last_idx_dcomp : last_idx_dcomp + t_element_alpha]
            last_idx_dcomp += t_element_alpha
            beta = decoded_decomp_checkpoint[last_idx_dcomp : last_idx_dcomp + t_element_beta]
            last_idx_dcomp += t_element_beta
            alpha = torch.unflatten(torch.from_numpy(np.copy(alpha)), -1, (dim[0], rr))
            beta = torch.unflatten(torch.from_numpy(np.copy(beta)), -1, (rr, dim[1]))
            base = lora_bases[layer_name]
            restored_decomp = restoreLinearLayer(alpha, beta, base, scaling)
            base_dict[layer_name] = restored_decomp
        elif "classifier" in layer_name:
            base_dict[layer_name] = bias[layer_name]
        else: # Restoration procedure for convolutional layers.
            t_elements = np.prod(dim)
            needed_ele = decoded_checkpoint[last_idx : last_idx + t_elements]
            base_dict[layer_name] = torch.unflatten(torch.from_numpy(np.copy(needed_ele)), -1, dim)
            last_idx += t_elements
    return base_dict