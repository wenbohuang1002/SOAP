import torch
import torch.nn.functional as F
import os
import math
from enum import Enum
import sys
import shutil
import numpy as np

def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.
    """
    numerator = torch.matmul(x, y.transpose(-1,-2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1,-2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists


def extract_class_indices(labels, which_class):
    """
    Helper method to extract the indices of elements which have the specified label.
    :param labels: (torch.tensor) Labels of the context set.
    :param which_class: Label for which indices are extracted.
    :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
    """
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask, as_tuple=False)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

class TestAccuracies:
    """
    Determines if an evaluation on the validation set is better than the best so far.
    In particular, this handles the case for meta-dataset where we validate on multiple datasets and we deem
    the evaluation to be better if more than half of the validation accuracies on the individual validation datsets
    are better than the previous best.
    """
    def __init__(self, validation_datasets):
        self.datasets = validation_datasets
        self.dataset_count = len(self.datasets)

    def print(self, logfile, accuracy_dict, mode="val"):
        print_and_log(logfile, "")  # add a blank line
        print_and_log(logfile, "{} accuracies:".format(mode))
        for dataset in self.datasets:
            print_and_log(logfile, "{0:}: {1:.1f}+/-{2:.1f}".format(dataset, accuracy_dict[dataset]["accuracy"],
                                                                    accuracy_dict[dataset]["confidence"]))
        print_and_log(logfile, "")  # add a blank line



def verify_checkpoint_dir(checkpoint_dir, resume, test_mode):
    if resume:  # verify that the checkpoint directory and file exists
        if not os.path.exists(checkpoint_dir):
            print("Can't resume for checkpoint. Checkpoint directory ({}) does not exist.".format(checkpoint_dir), flush=True)
            sys.exit()

        checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pt')
        if not os.path.isfile(checkpoint_file):
            print("Can't resume for checkpoint. Checkpoint file ({}) does not exist.".format(checkpoint_file), flush=True)
            sys.exit()
    else:
        if os.path.exists(checkpoint_dir):
            print("Checkpoint directory ({}) already exists. Deleting directory...".format(checkpoint_dir), flush=True)
            shutil.rmtree(checkpoint_dir)
            # print("Checkpoint directory ({}) already exits.".format(checkpoint_dir), flush=True)
            # print("If starting a new training run, specify a directory that does not already exist.", flush=True)
            # print("If you want to resume a training run, specify the -r option on the command line.", flush=True)
            # sys.exit()


def print_and_log(log_file, message):
    """
    Helper function to print to the screen and the log.txt file.
    """
    print(message, flush=True)
    log_file.write(message + '\n')


def get_log_files(checkpoint_dir, resume, test_mode):
    """
    Function that takes a path to a checkpoint directory and returns a reference to a logfile and paths to the
    fully trained model and the model with the best validation score.
    """
    # verify_checkpoint_dir(checkpoint_dir, resume, test_mode)
    if not resume and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path_validation = os.path.join(checkpoint_dir, 'best_validation.pt')
    checkpoint_path_final = os.path.join(checkpoint_dir, 'fully_trained.pt')
    logfile_path = os.path.join(checkpoint_dir, 'log.txt')
    if os.path.isfile(logfile_path):
        logfile = open(logfile_path, "a", buffering=1)
    else:
        logfile = open(logfile_path, "w", buffering=1)

    return checkpoint_dir, logfile, checkpoint_path_validation, checkpoint_path_final


def aggregate_accuracy(test_logits_sample, test_labels):
    """
    Compute classification accuracy.
    """
    # print(test_labels)
    # print(torch.argmax(test_logits_sample, dim=-1))
    return torch.mean(torch.eq(test_labels, torch.argmax(test_logits_sample, dim=-1)).float())


def loss(test_logits_sample, test_labels, device):
    """
    Compute the classification loss.
    """
    size = test_logits_sample.size()
    sample_count = size[0]  # scalar for the loop counter
    num_samples = torch.tensor([sample_count], dtype=torch.float, device=device, requires_grad=False)

    log_py = torch.empty(size=(size[0], size[1]), dtype=torch.float, device=device)
    for sample in range(sample_count):
        log_py[sample] = -F.cross_entropy(test_logits_sample[sample], test_labels, reduction='none')
    score = torch.logsumexp(log_py, dim=0) - torch.log(num_samples)
    return -torch.sum(score, dim=0)


def split_first_dim_linear(x, first_two_dims):
    """
    Undo the stacking operation
    """
    x_shape = x.size()
    new_shape = first_two_dims
    if len(x_shape) > 1:
        new_shape += [x_shape[-1]]
    return x.view(new_shape)

def theta_add1(x):
    a,b=x[:,0],x[:,1]
    return torch.stack([a+1,b],-1)

def gen_mask(offsets,resolution):
    '''
    g in (-1,1) for 7 pix
    '''
    with torch.no_grad():
        L=resolution
        shape=offsets.shape[:-1]
    offsets=offsets.reshape(-1,1,2)
    with torch.no_grad():
        N=offsets.size(0)
        grid=torch.arange(0,L).cuda()/(L-1)*2-1
        grid=grid.reshape(1,L,1).expand(N,-1,2)
    grid=grid+offsets
    full = (-1<grid)*(grid<1)*1.0
    margin= (1-full)*(1-(grid.abs()-1)*((L-1)/2))
    margin=torch.nn.functional.relu(margin)
    mask_xy=full+margin
    mask_x,mask_y=mask_xy[...,0],mask_xy[...,1]
    mask=mask_y.unsqueeze(-1)*mask_x.unsqueeze(-2)
    mask=mask.reshape(*shape,resolution,resolution)
    return mask

class Squeeze(torch.nn.Module):
    def __init__(self,*dim):
        super().__init__()
        if all(v>=0 for v in dim):
            self.dim=sorted(dim,reverse=True)
        elif all(v<0 for v in dim):
            self.dim=sorted(dim)

    def forward(self,x):
        for d in self.dim:
            x=torch.squeeze(x,dim=d)
        return x

def softmax(x):
    """Compute softmax values for each row of x."""
    # Subtract the maximum value for numerical stability
    max_val = np.max(x, axis=-1, keepdims=True)
    x_exp = np.exp(x - max_val)
    x_sum = np.sum(x_exp, axis=-1, keepdims=True)
    softmax_output = x_exp / x_sum
    return softmax_output