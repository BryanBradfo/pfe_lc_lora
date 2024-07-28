import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="torch.utils._pytree._register_pytree_node is deprecated")
import glob
import sys
import os
import shutil
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy as spy
import torchvision
import copy
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import ssl
from src.utils.autoaugment import CIFAR10Policy
from src.compression.LowRankLinear import LowRankLinear
import pickle, json
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from datasets import load_dataset
from torch.nn.functional import cross_entropy
import src.main as lc
import old_lc.main as olc
import torch.optim as optim
import src.compression.deltaCompress as lc_compress
from src.utils.utils import evaluate_accuracy, evaluate_accuracy_vit, evaluate_accuracy_distilbert, evaluate_accuracy_gpu, evaluate_accuracy_vit_gpu, evaluate_accuracy_distilbert_gpu,lazy_restore,lazy_restore_gpu, evaluate_compression

import wandb
# Connect to W&B
wandb.login(key="beb938fdf67db528128a4298e19b9997afd83dfd")

train_batch_size = 128
test_batch_size = 1024
num_work = 16

def data_loader():

    # Load the dataset
    dataset = load_dataset('imdb')

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    context_length = 128
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=context_length)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    def collate_fn(batch):
        input_ids = torch.tensor([item['input_ids'] for item in batch])
        attention_mask = torch.tensor([item['attention_mask'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


    # Prepare the dataset for training
    train_dataset = tokenized_datasets['train']
    test_dataset = tokenized_datasets['test']

    # DataLoaders
    trainloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
    testloader = DataLoader(test_dataset, batch_size=test_batch_size, collate_fn=collate_fn)

    return trainloader, testloader
def getBase(model, basepath=""):
    wd = model.state_dict()
    w = [
        wd['distilbert.transformer.layer.0.attention.q_lin.weight'],
        wd['distilbert.transformer.layer.0.attention.v_lin.weight'],

        wd['distilbert.transformer.layer.1.attention.q_lin.weight'],
        wd['distilbert.transformer.layer.1.attention.v_lin.weight'],

        wd['distilbert.transformer.layer.2.attention.q_lin.weight'],
        wd['distilbert.transformer.layer.2.attention.v_lin.weight'],

        wd['distilbert.transformer.layer.3.attention.q_lin.weight'],
        wd['distilbert.transformer.layer.3.attention.v_lin.weight'],

        wd['distilbert.transformer.layer.4.attention.q_lin.weight'],
        wd['distilbert.transformer.layer.4.attention.v_lin.weight'],

        wd['distilbert.transformer.layer.5.attention.q_lin.weight'],
        wd['distilbert.transformer.layer.5.attention.v_lin.weight']]
    b = [
        wd['distilbert.transformer.layer.0.attention.q_lin.bias'],
        wd['distilbert.transformer.layer.0.attention.v_lin.bias'],

        wd['distilbert.transformer.layer.1.attention.q_lin.bias'],
        wd['distilbert.transformer.layer.1.attention.v_lin.bias'],

        wd['distilbert.transformer.layer.2.attention.q_lin.bias'],
        wd['distilbert.transformer.layer.2.attention.v_lin.bias'],

        wd['distilbert.transformer.layer.3.attention.q_lin.bias'],
        wd['distilbert.transformer.layer.3.attention.v_lin.bias'],

        wd['distilbert.transformer.layer.4.attention.q_lin.bias'],
        wd['distilbert.transformer.layer.4.attention.v_lin.bias'],

        wd['distilbert.transformer.layer.5.attention.q_lin.bias'],
        wd['distilbert.transformer.layer.5.attention.v_lin.bias']]
    base_dict = {
        'distilbert.transformer.layer.0.attention.q_lin.weight': wd['distilbert.transformer.layer.0.attention.q_lin.weight'],
        'distilbert.transformer.layer.0.attention.q_lin.bias': wd['distilbert.transformer.layer.0.attention.q_lin.bias'],
        'distilbert.transformer.layer.0.attention.v_lin.weight': wd['distilbert.transformer.layer.0.attention.v_lin.weight'],
        'distilbert.transformer.layer.0.attention.v_lin.bias': wd['distilbert.transformer.layer.0.attention.v_lin.bias'],
        
        'distilbert.transformer.layer.1.attention.q_lin.weight': wd['distilbert.transformer.layer.1.attention.q_lin.weight'],
        'distilbert.transformer.layer.1.attention.q_lin.bias': wd['distilbert.transformer.layer.1.attention.q_lin.bias'],
        'distilbert.transformer.layer.1.attention.v_lin.weight': wd['distilbert.transformer.layer.1.attention.v_lin.weight'],
        'distilbert.transformer.layer.1.attention.v_lin.bias': wd['distilbert.transformer.layer.1.attention.v_lin.bias'],
        
        'distilbert.transformer.layer.2.attention.q_lin.weight': wd['distilbert.transformer.layer.2.attention.q_lin.weight'],
        'distilbert.transformer.layer.2.attention.q_lin.bias': wd['distilbert.transformer.layer.2.attention.q_lin.bias'],
        'distilbert.transformer.layer.2.attention.v_lin.weight': wd['distilbert.transformer.layer.2.attention.v_lin.weight'],
        'distilbert.transformer.layer.2.attention.v_lin.bias': wd['distilbert.transformer.layer.2.attention.v_lin.bias'],
        
        'distilbert.transformer.layer.3.attention.q_lin.weight': wd['distilbert.transformer.layer.3.attention.q_lin.weight'],
        'distilbert.transformer.layer.3.attention.q_lin.bias': wd['distilbert.transformer.layer.3.attention.q_lin.bias'],
        'distilbert.transformer.layer.3.attention.v_lin.weight': wd['distilbert.transformer.layer.3.attention.v_lin.weight'],
        'distilbert.transformer.layer.3.attention.v_lin.bias': wd['distilbert.transformer.layer.3.attention.v_lin.bias'],
        
        'distilbert.transformer.layer.4.attention.q_lin.weight': wd['distilbert.transformer.layer.4.attention.q_lin.weight'],
        'distilbert.transformer.layer.4.attention.q_lin.bias': wd['distilbert.transformer.layer.4.attention.q_lin.bias'],
        'distilbert.transformer.layer.4.attention.v_lin.weight': wd['distilbert.transformer.layer.4.attention.v_lin.weight'],
        'distilbert.transformer.layer.4.attention.v_lin.bias': wd['distilbert.transformer.layer.4.attention.v_lin.bias'],
        
        'distilbert.transformer.layer.5.attention.q_lin.weight': wd['distilbert.transformer.layer.5.attention.q_lin.weight'],
        'distilbert.transformer.layer.5.attention.q_lin.bias': wd['distilbert.transformer.layer.5.attention.q_lin.bias'],
        'distilbert.transformer.layer.5.attention.v_lin.weight': wd['distilbert.transformer.layer.5.attention.v_lin.weight'],
        'distilbert.transformer.layer.5.attention.v_lin.bias': wd['distilbert.transformer.layer.5.attention.v_lin.bias']
        }

    if basepath != "":
        if not os.path.exists(basepath):
            os.makedirs(basepath)
        fp = os.path.join(basepath, "lora_bases.pt")
        torch.save(base_dict, fp)

    return w, b

def load_sd_decomp(org_sd, model, decomposed_layers):
    """
    @param org_sd : The state_dict when the model is ongoing.
    @param model : The decomp model with decomposed layers.
    @param decomposed_layers : The decomposed layers in decomp model.

    @return The new model with the old state dictionary loaded in.
    """
    new_sd = model.state_dict()
    for k, v in org_sd.items():
        if k not in decomposed_layers:
            new_sd[k] = v
    model.load_state_dict(new_sd)

def replace_linear_with_lowrank(module, additional_weights, additional_bias):
    for i, layer in enumerate(module.transformer.layer):
        layer.attention.q_lin = LowRankLinear(768, 768, additional_weights[2*i], additional_bias[2*i], rank=-1)

        layer.attention.v_lin = LowRankLinear(768, 768, additional_weights[2*i+1], additional_bias[2*i+1], rank=-1)

# Load DataLoaders
train_loader, test_loader = data_loader()

# Vérification des tailles des DataLoaders
print(f'Size of train_loader1: {len(train_loader.dataset)}')
print(f'Size of test_loader: {len(test_loader.dataset)}')

HDFP = "./volumes/Ultra Touch" # Load HHD

SAVE_LOC = HDFP + "/lobranch-snapshot/diffbitwidth-adaptive-rank/distilbert/lobranch"
if not os.path.exists(SAVE_LOC):
    os.makedirs(SAVE_LOC)

SAVE_LOC_OLC = HDFP + "/lobranch-snapshot/diffbitwidth-adaptive-rank/distilbert/old-lc"
if not os.path.exists(SAVE_LOC_OLC):
    os.makedirs(SAVE_LOC_OLC)

SAVE_BRANCH_PRETRAINING = HDFP + "/lobranch-snapshot/branchpoints/distilbert/pretraining"
if not os.path.exists(SAVE_BRANCH_PRETRAINING):
    os.makedirs(SAVE_BRANCH_PRETRAINING)

SAVE_BRANCH = HDFP + "/lobranch-snapshot/branchpoints/distilbert/branch"
if not os.path.exists(SAVE_BRANCH):
    os.makedirs(SAVE_BRANCH)

DECOMPOSED_LAYERS = [
    'distilbert.transformer.layer.0.attention.q_lin.weight',
    'distilbert.transformer.layer.0.attention.v_lin.weight',

    'distilbert.transformer.layer.1.attention.q_lin.weight',
    'distilbert.transformer.layer.1.attention.v_lin.weight',

    'distilbert.transformer.layer.2.attention.q_lin.weight',
    'distilbert.transformer.layer.2.attention.v_lin.weight',

    'distilbert.transformer.layer.3.attention.q_lin.weight',
    'distilbert.transformer.layer.3.attention.v_lin.weight',

    'distilbert.transformer.layer.4.attention.q_lin.weight',
    'distilbert.transformer.layer.4.attention.v_lin.weight',

    'distilbert.transformer.layer.5.attention.q_lin.weight',
    'distilbert.transformer.layer.5.attention.v_lin.weight'
]
RANK = -1
SCALING = -1
learning_rate = 5e-5
learning_rate_dloralc = 5e-5
NUM_EPOCHES = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
original = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased').to(device)
model_original = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased').to(device)
model_no_touch = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased').to(device)
model_hf = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased').to(device)
w, b = getBase(original)
model = copy.deepcopy(model_hf).to(device)
replace_linear_with_lowrank(model.distilbert, w, b)

optimizer = AdamW(model.parameters(), lr=learning_rate_dloralc)
optimizer_lc_only = AdamW(model_original.parameters(), lr=learning_rate)
optimizer_no_touch = AdamW(model_no_touch.parameters(), lr=learning_rate)

full_accuracy = []
decomposed_full_accuracy = []
restored_accuracy = []
lc_accuracy = []

# Initialize the current iteration and set to 0
current_iter = 0
current_set = 0

# Initialize the current iteration and set to 0 for the old LC method
current_iter_old_lc = 0
current_set_old_lc = 0

acc = lambda x, y : (torch.max(x, 1)[1] == y).sum().item() / y.size(0)

# Initialiser wandb
wandb.init(project="DistilBert_pretrained",
           name="DistilBert-Without-Incremental-Learning",
           tags=["DistilBert", "Without-Incremental-Learning", "IMDb"],
           config={"num_epochs": NUM_EPOCHES,
                "model_hf": "DistilBert",
                "train dataset": "IMDb train dataset[:]",
                "test dataset": "IMDb test dataset[:]",
                "train_batch_size": train_batch_size,
                "test_batch_size": test_batch_size,
                "num_workers": num_work,
                "learning_rate": learning_rate,
                "learning_rate_dloralc": learning_rate_dloralc,
                "optimizer": "AdamW",
           })

train_loader_list = [train_loader]
for j in range(len(train_loader_list)):
    train_loader_txt = "train_loader{}".format(j+1)
    print("--------------------------")
    print("Beginning of model training on {}...".format(train_loader_txt))

    full_accuracy_dloralc = 0
    decomposed_full_accuracy_dloralc = 0
    restored_accuracy_dloralc = 0
    lc_accuracy_dloralc = 0

    for epch in range(NUM_EPOCHES):
        for i, data in enumerate(train_loader_list[j], 0):

            SAVE_LOC_j = SAVE_LOC + "/"+train_loader_txt
            if not os.path.exists(SAVE_LOC_j):
                os.makedirs(SAVE_LOC_j)
                
            SAVE_LOC_OLC_j = SAVE_LOC_OLC + "/"+train_loader_txt
            if not os.path.exists(SAVE_LOC_OLC_j):
                os.makedirs(SAVE_LOC_OLC_j)
            print("Epoch: {}, Iteration: {}".format(epch, i))
            
            set_path = "/set_{}".format(current_set)
            if not os.path.exists(SAVE_LOC_j + set_path):
                os.makedirs(SAVE_LOC_j + set_path)

            if i == 0 and epch == 0: # first iteration, create baseline model
                base, base_decomp = lc.extract_weights_gpu(model, SAVE_LOC_j + 
                                                        "/set_{}".format(current_set), DECOMPOSED_LAYERS)
            else:
                if i % 10 == 0: 
                    # full snapshot!
                    new_model = lazy_restore_gpu(base, base_decomp, bias, DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased'), 
                                            original.state_dict(), DECOMPOSED_LAYERS, rank = RANK, scaling = SCALING)
                    original = new_model # Changing previous "original model" used to restore the loRA model.
                    
                    current_set += 1
                    current_iter = 0

                    set_path = "/set_{}".format(current_set)
                    if not os.path.exists(SAVE_LOC_j + set_path):
                        os.makedirs(SAVE_LOC_j + set_path)
                    
                    # Rebuilding LoRA layers => reset model!
                    w, b = getBase(original)
                    model = copy.deepcopy(model_hf).to(device)
                    replace_linear_with_lowrank(model.distilbert, w, b)
                    optimizer = AdamW(model.parameters(), lr = learning_rate)
                    load_sd_decomp(original.state_dict(), model, DECOMPOSED_LAYERS)
                    base, base_decomp = lc.extract_weights_gpu(model, SAVE_LOC_j + 
                                                        "/set_{}".format(current_set), DECOMPOSED_LAYERS)

                else:
                    # Delta-compression
                    delta, decomp_delta, bias = lc.generate_delta_gpu(base, 
                                                                    base_decomp, model.state_dict(), DECOMPOSED_LAYERS)
                    compressed_delta, full_delta, compressed_dcomp_delta, full_dcomp_delta  = lc.compress_delta(delta, 
                                                                                                                decomp_delta)
                    
                    # Saving checkpoint
                    lc.save_checkpoint(compressed_delta, compressed_dcomp_delta, bias, current_iter, SAVE_LOC_j + 
                                    "/set_{}".format(current_set))
        
                    base = np.add(base, full_delta) # Replace base with latest for delta to accumulate.
                    base_decomp = np.add(full_dcomp_delta, base_decomp)

                    current_iter += 1
                
            # ==========================
            # Saving using LC-Checkpoint
            # ==========================
                    
            if i == 0 and epch == 0:
                cstate = model_original.state_dict()
                set_path = "/set_{}".format(current_set_old_lc)
                if not os.path.exists(SAVE_LOC_OLC_j + set_path):
                    os.makedirs(SAVE_LOC_OLC_j + set_path)
                torch.save(cstate, SAVE_LOC_OLC_j + set_path + "/initial_model.pt")
                prev_state = olc.extract_weights_gpu(cstate, SAVE_LOC_OLC_j + set_path, DECOMPOSED_LAYERS)
            else:
                if i % 10 == 0:
                    cstate = model_original.state_dict()
                    current_set_old_lc += 1
                    current_iter_old_lc = 0
                    set_path = "/set_{}".format(current_set_old_lc)
                    if not os.path.exists(SAVE_LOC_OLC_j + set_path):
                        os.makedirs(SAVE_LOC_OLC_j + set_path)
                    torch.save(cstate, SAVE_LOC_OLC_j + set_path + "/initial_model.pt")
                    prev_state = olc.extract_weights_gpu(cstate, SAVE_LOC_OLC_j + set_path, DECOMPOSED_LAYERS)
                else:
                    cstate = model_original.state_dict()
                    old_lc_delta, old_lc_bias = olc.generate_delta_gpu(prev_state, cstate, DECOMPOSED_LAYERS)
                    olc_compressed_delta, update_prev = olc.compress_data(old_lc_delta, num_bits = 3)
                    olc.save_checkpoint(SAVE_LOC_OLC_j + "/set_{}".format(current_set_old_lc), olc_compressed_delta, 
                                        old_lc_bias, current_iter_old_lc)
                    prev_state = np.add(prev_state, update_prev)
                    current_iter_old_lc += 1
            
            # ==========================
            # Training on Low-Rank Model
            # ==========================

            # Get the inputs and labels
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            print("LoRA+LC Training Loss (Decomposed): {}".format(loss.item()))
            optimizer.step()
                
            # ======================
            # Training on Full Model
            # ======================

            # Zero the parameter gradients
            optimizer_lc_only.zero_grad()

            # Forward + backward + optimize
            outputs_full = model_original(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits
            loss_full = torch.nn.CrossEntropyLoss()(outputs_full,labels)
            loss_full.backward()
            print("LC Training Loss (Full): {}".format(loss_full.item()))
            optimizer_lc_only.step()

            if i % 20 == 0:
                print("Training Accuracy | Decomposed: {}, Full : {}".format(acc(outputs, labels), 
                                                                            acc(outputs_full, labels)))

            if i != 0  and i % 5 == 0: # Evaluation on testing set
                full_accuracy_dloralc = evaluate_accuracy_distilbert_gpu(model_original, test_loader, device)
                full_accuracy.append(full_accuracy_dloralc)

                decomposed_full_accuracy_dloralc = evaluate_accuracy_distilbert_gpu(model, test_loader, device)
                decomposed_full_accuracy.append(decomposed_full_accuracy_dloralc)
                
                restored_model = lazy_restore_gpu(base, base_decomp, bias, DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased'), 
                                            original.state_dict(), DECOMPOSED_LAYERS, 
                                            rank = RANK, scaling = SCALING)
                restored_accuracy_dloralc_restored = evaluate_accuracy_distilbert(restored_model, test_loader)
                restored_accuracy.append(restored_accuracy_dloralc_restored)

                restored_lc_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased').to(device)
                restored_lc_model.load_state_dict(olc.restore_state_dict(prev_state, old_lc_bias, 
                                                                    restored_model.state_dict(), DECOMPOSED_LAYERS))
                lc_accuracy_lc = evaluate_accuracy_distilbert_gpu(restored_lc_model, test_loader, device)
                lc_accuracy.append(lc_accuracy_lc)
                print("Full accuracy (w/o dLoRA+LC): {}, LC accuracy: {}, Decomposed-Full (w/dLoRA+LC) accuracy: {}, Decomposed-Restored (w/dLoRA+LC restored) accuracy: {}".format(
                    full_accuracy[-1], lc_accuracy[-1], decomposed_full_accuracy[-1], restored_accuracy[-1]))
                
                wandb.log({
                    "accuracy w/o dLoRALC": full_accuracy[-1],
                    "accuracy w/ LC": lc_accuracy[-1],
                    "accuracy w/ dLoRALC": decomposed_full_accuracy[-1],
                    "accuracy w/ dLoRALC after restoration": restored_accuracy[-1]
                })
                

    print("End of model training on {}...".format(train_loader_txt))

    rounded_valid_acc = decomposed_full_accuracy_dloralc
    torch.save(model.state_dict(), HDFP + "/lobranch-snapshot/branchpoints/distilbert/branch/branch_{}.pt".format(rounded_valid_acc))
    print("Model saved at accuracy: {:.4f}".format(rounded_valid_acc))

    w, b = getBase(original)
    model = copy.deepcopy(model_hf).to(device)
    replace_linear_with_lowrank(model.distilbert, w, b)

    model.load_state_dict(torch.load(HDFP + "/lobranch-snapshot/branchpoints/distilbert/branch/branch_{}.pt".format(rounded_valid_acc)))
    optimizer = AdamW(model.parameters(), lr=learning_rate_dloralc)

    # Initialize the current iteration and set to 0
    current_iter = 0
    current_set = 0

    # Initialize the current iteration and set to 0 for the old LC method
    current_iter_old_lc = 0
    current_set_old_lc = 0

torch.save(model_original.state_dict(), './distilbert.pt')

import math
def getsize(sl):
    dir = [x for x in os.listdir(sl)]
    csize, usize = 0, 0
    for dataloader in dir:
        for set in os.listdir(sl + "/" + dataloader):
            # print(set)
            for f in os.listdir(sl + "/" + dataloader + "/" + set):
                # print(f)
                fp = sl + "/" + dataloader + "/{}/{}".format(set, f)
                csize += os.path.getsize(fp)
                usize += 255.0 * math.pow(2, 20) # torch checkpoint same size
    return csize, usize

compressed_size, uncompressed_size = getsize(SAVE_LOC)
a, b = evaluate_compression(uncompressed_size, compressed_size)
compressed_size, uncompressed_size = getsize(SAVE_LOC_OLC)
a1, b1 = evaluate_compression(uncompressed_size, compressed_size)

print("LC-Checkpoint + GZIP")
print("Compression Ratio: {}%, Space Savings: {}%".format(a1, b1))
print("LoRA + LC-Checkpoint + GZIP")
print("Compression Ratio: {}%, Space Savings: {}%".format(a, b))