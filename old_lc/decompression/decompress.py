import numpy as np
import pathos.multiprocessing as pmp
import torch, zlib

def decode_data(checkpoint):
    """
    @param checkpoint : GZIP Encoded checkpoint

    @return : Decoded checkpoint.
    """
    return np.frombuffer(zlib.decompress(checkpoint), dtype = np.float32)

def restore_state_dict(decoded_checkpoint, bias, base_dict):
    """
    @param decoded_checkpoint: The decoded_checkpoint from zlib.
    @param bias : The bias dictionary of the model.
    @param base_dict : The base dictionary of the model which helps us understand its structure.

    @return Restored state_dict.
    """
    last_idx = 0
    for layer_name, init_tensor in base_dict.items():
        if "bias" in layer_name:
            base_dict[layer_name] = bias[layer_name]
            continue
        dim = init_tensor.numpy().shape
        if not dim:
            continue
        t_elements = np.prod(dim)
        needed_ele = decoded_checkpoint[last_idx : last_idx + t_elements]
        last_idx += t_elements
        base_dict[layer_name] = torch.unflatten(torch.from_numpy(np.copy(needed_ele)), -1, dim)
    return base_dict