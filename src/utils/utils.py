import torch
import numpy as np
from src.decompression.decompress import restoreLinearLayer
from src.compression.LowRankLinear import generate_rank
from src.utils.criterions import LabelSmoothingCrossEntropyLoss

def evaluate_accuracy_gpu(model, test_ds, device):
    """
    @param model : PyTorch model to evaluate accuracy.
    @param test_ds : Test dataset.

    @return model accuracy.
    """

    model.eval() # Set to eval state.
    t_correct, t_dp = 0, 0
    with torch.no_grad():
        for i, label in test_ds:
            _, opt = torch.max(model(i.to(device)), dim = 1)
            t_dp += label.size(0)
            t_correct += (opt == label.to(device)).sum().item()
    acc = t_correct / t_dp
    # print("model accuracy: {}".format(acc))
    model.train() # Revert to training state.
    return acc

def evaluate_accuracy_vit_gpu(model, test_ds, device):
    """
    @param model : PyTorch model to evaluate accuracy.
    @param test_ds : Test dataset.

    @return model accuracy.
    """
    
    model.eval()  # Set to eval state.
    t_correct, t_dp = 0, 0
    with torch.no_grad():
        for i, label in test_ds:
            i, label = i.to(device), label.to(device)
            opt = model(i)
            # Extract the logits from ImageClassifierOutput
            logits = opt.logits if hasattr(opt, 'logits') else opt
            # Assuming logits is the tensor containing model predictions
            predicted = torch.argmax(logits, dim=1)
            t_dp += label.size(0)
            t_correct += (predicted == label).sum().item()
    acc = t_correct / t_dp
    model.train()  # Revert to training state.
    return acc

def evaluate_accuracy_distilbert_gpu(model, test_ds, device):
    """
    @param model : PyTorch model to evaluate accuracy.
    @param test_ds : Test dataset.

    @return model accuracy.
    """
    acc = lambda x, y: (torch.argmax(x, dim=1) == y).sum().item() / y.size(0)
    avg_acc = 0
    model.eval()  # Set to eval state.
    total_acc = 0
    with torch.no_grad():
        for data in test_ds:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)
            opt = model(input_ids=input_ids, attention_mask=attention_mask)
            # Extract the logits from ImageClassifierOutput
            logits = opt.logits if hasattr(opt, 'logits') else opt
            # Assuming logits is the tensor containing model predictions
            predicted = torch.argmax(logits, dim=1)
            total_acc += acc(logits, labels)
    avg_acc = total_acc / len(test_ds)
    model.train()  # Revert to training state.
    return avg_acc

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
    # print("model accuracy: {}".format(acc))
    model.train() # Revert to training state.
    return acc

def evaluate_accuracy_vit(model, test_ds):
    """
    @param model : PyTorch model to evaluate accuracy.
    @param test_ds : Test dataset.

    @return model accuracy.
    """
    model.eval() # Set to eval state.
    t_correct, t_dp = 0, 0
    with torch.no_grad():
        for i, label in test_ds:
            opt = model(i)
            logits = opt.logits if hasattr(opt, 'logits') else opt
            predicted = torch.argmax(logits, dim=1)
            t_dp += label.size(0)
            t_correct += (predicted == label).sum().item()
    acc = t_correct / t_dp
    # print("model accuracy: {}".format(acc))
    model.train() # Revert to training state.
    return acc

def evaluate_accuracy_distilbert(model, test_ds):
    """
    @param model : PyTorch model to evaluate accuracy.
    @param test_ds : Test dataset.

    @return model accuracy.
    """
    acc = lambda x, y: (torch.argmax(x, dim=1) == y).sum().item() / y.size(0)
    avg_acc = 0
    model.eval()  # Set to eval state.
    total_acc = 0
    with torch.no_grad():
        for data in test_ds:
            input_ids = data['input_ids']
            attention_mask = data['attention_mask']
            labels = data['labels']
            opt = model(input_ids=input_ids, attention_mask=attention_mask)
            # Extract the logits from ImageClassifierOutput
            logits = opt.logits if hasattr(opt, 'logits') else opt
            # Assuming logits is the tensor containing model predictions
            predicted = torch.argmax(logits, dim=1)
            total_acc += acc(logits, labels)
    avg_acc = total_acc / len(test_ds)
    model.train()  # Revert to training state.
    return avg_acc
# def evaluate_accuracy_gpu(model, test_ds, device):
#     """
#     @param model : PyTorch model to evaluate accuracy.
#     @param test_ds : Test dataset.

#     @return model accuracy.
#     """

#     model.eval() # Set to eval state.
#     valid_loss = 0
#     valid_acc = 0
#     with torch.no_grad():
#         for i, label in test_ds:
#             output = model(i.to(device))
#             loss = LabelSmoothingCrossEntropyLoss(classes=10)(output, label.to(device))
#             valid_loss += loss.item() * i.size(0)
#             valid_acc = torch.eq(output.argmax(-1), label.to(device)).float().mean()
    
#     valid_loss /= len(test_ds.dataset)

#     print("model accuracy: {}".format(valid_acc))
#     model.train() # Revert to training state.
#     return valid_acc, valid_loss

# def evaluate_accuracy(model, test_ds):
#     """
#     @param model : PyTorch model to evaluate accuracy.
#     @param test_ds : Test dataset.

#     @return model accuracy.
#     """

#     model.eval() # Set to eval state.
#     valid_loss = 0
#     valid_acc = 0
#     with torch.no_grad():
#         for i, label in test_ds:
#             output = model(i)
#             loss = LabelSmoothingCrossEntropyLoss(classes=10)(output, label)
#             valid_loss += loss.item() * i.size(0)
#             valid_acc = torch.eq(output.argmax(-1), label).float().mean()
    
#     valid_loss /= len(test_ds.dataset)

#     print("model accuracy: {}".format(valid_acc))
#     model.train() # Revert to training state.
#     return valid_acc, valid_loss


def lazy_restore_gpu(weights, weights_decomp, bias, clean_model, org, decomposed_layers, rank : int = -1, scaling : int = -1):
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
        dim = init_tensor.cpu().detach().numpy().shape
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
            restored_decomp = restoreLinearLayer(alpha, beta, org[layer_name], scaling)  # Remove .to(device)
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
