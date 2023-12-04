"""
Trains and evaluates a model a single time for given hyperparameters.
"""

import random
import time 
import os

import wandb
import torch
import numpy as np

from Exp.parser import parse_args
from Misc.config import config
from Misc.utils import list_of_dictionary_to_dictionary_of_lists
from Exp.preparation import load_dataset, get_model, get_optimizer_scheduler, get_loss
from Exp.training_loop_functions import train, eval, step_scheduler

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def track_epoch(epoch, metric_name, train_result, val_result, test_result, lr):
    wandb.log({
        "Epoch": epoch,
        "Train/Loss": train_result["total_loss"],
        "Val/Loss": val_result["total_loss"],
        f"Val/{metric_name}": val_result[metric_name],
        "Test/Loss": test_result["total_loss"],
        f"Test/{metric_name}": test_result[metric_name],
        "LearningRate": lr
        })
    Exp/run_model.py
def print_progress(train_loss, val_loss, test_loss, metric_name, val_metric, test_metric):
    print(f"\tTRAIN\t loss: {train_loss:6.4f}")
    print(f"\tVAL\t loss: {val_loss:6.4f}\t  {metric_name}: {val_metric:10.4f}")
    print(f"\tTEST\t loss: {test_loss:6.4f}\t  {metric_name}: {test_metric:10.4f}")


## new stuff

def to_aids_format(
    g, # graph, in pyg format
    i, # graph name
    file, # BytesIO object
    undirected: bool = True # whether to assume that g is undirected (no guarantees if set to False :D )
):

    row, col = g.edge_index.cpu().numpy()
    edge_attr = np.ones(row.shape[0])

    # graph header
    file.write(f'# {i} {0} {g.num_nodes} {g.num_edges // 2}\n'.encode('latin1'))

    # node handling
    np.savetxt(file, np.ones([1,g.num_nodes]), fmt='%d')

    # edge handling
    if undirected:
        mask = row < col
        row = row[mask]
        col = col[mask]
        edge_attr = edge_attr[mask]
    E = np.hstack([1 + row.reshape([-1,1]), 1 + col.reshape([-1,1]), edge_attr.reshape([-1,1])])
    EL = E.reshape([1, -1])
    np.savetxt(file, EL, fmt='%d')


def batch_to_aids_format(batch, # a batch of pyg graphs
                        undirected=True, # whether to assume that g is undirected (no guarantees if set to False :D )
                        add_terminator=True # whether to add the final $ to the string 
                        ):
    import io   
    bio = io.BytesIO()

    for i in range(batch.num_graphs):
        g = batch.get_example(i)
        to_aids_format(g, i, bio, undirected=undirected)
    
    if add_terminator:
        bio.write('$\n'.encode('latin1'))

    return bio.getvalue().decode('latin1')


def plain_bagels(loader, config={'binfile': './outerplanaritytest', 'verbose': True}):
    '''
    This is the important function

    Given a dataloader, compute for each batch individually
    1) if the graphs in the batch are outerplanar
    2)  the Hamiltonian cycles of the outerplanar blocks

    To this end, each batch is transformed to a textual format, piped to an external program 
    which pipes its results back which is then parsed and stored in the tensors (TODO).

    The function uses the (linux) executable ``outerplanaritytest`` which might need to
    be recompiled from source if your system is different.
    The source code is available at https://github.com/pwelke/GraphMiningTools
    By cloning the repository and running 
    ``make outerplanaritytest``
    on your system, the binary can be recompiled to run on your Posix system.
    '''
    import json
    import subprocess

    t_c_code = 0

    t_graph_conversion = 0
    t_torch_conversion = 0
    t_json_conversion = 0

    n_outerplanar = 0
    n_total = 0
    for batch in loader:

        tic = time.time()
        graphstring = ''
        
        # graph conversion to textual input format for all graphs in the current batch
        # assumes undirected graphs and ignores labels
        graphstring = batch_to_aids_format(batch, undirected=True, add_terminator=True)

        toc = time.time()
        t_graph_conversion += toc - tic

        tic = time.time()
        # the actual computation of outerplanarity and Hamiltonian cycles in a subprocess
        cmd = [config['binfile'], '-sw', '-']
        proc = subprocess.run(args=cmd, capture_output=True, input=graphstring.encode("utf-8"))
        toc = time.time()
        t_c_code += toc - tic

        tic = time.time()
        # parsing of the results (directly from stdout of the process)
        jstr = proc.stdout.decode("utf-8")
        # print(jstr)
        jsobjects = json.loads(jstr)

        toc = time.time()
        t_json_conversion += toc - tic

        for g in jsobjects:
            if g['isOuterplanar']:
                n_outerplanar += 1
            n_total += 1

        # TODO: don't know, yet, how to best store this information in node or edge features
    
    if config['verbose']:
        print(f'time spent for torch conversion: {t_torch_conversion:.2f}s')
        print(f'time spent for graph conversion: {t_graph_conversion:.2f}s')
        print(f'time spent for json parsing: {t_json_conversion:.2f}s')
        print(f'time spent abroad: {t_c_code:.2f}s')
        print(f'{n_outerplanar} out of {n_total} graphs are outerplanar')

    return n_outerplanar, n_total

    
def main(args):
    print(args)
    device = args.device
    use_tracking = args.use_tracking
    
    set_seed(args.seed)
    train_loader, val_loader, test_loader = load_dataset(args, config)
    num_classes, num_vertex_features = train_loader.dataset.num_classes, train_loader.dataset.num_node_features
    
    if args.dataset.lower() == "zinc" or "ogb" in args.dataset.lower():
        num_classes = 1
   
    try:
        num_tasks = train_loader.dataset.num_tasks
    except:
        num_tasks = 1
        
    print(f"#Features: {num_vertex_features}")
    print(f"#Classes: {num_classes}")
    print(f"#Tasks: {num_tasks}")

    import time

    tic = time.time()

    n_outerplanar, n_total = plain_bagels(train_loader)
    a, b = plain_bagels(val_loader)
    c, d = plain_bagels(test_loader)
    n_outerplanar += a + c
    n_total += b + d

    toc = time.time()

    print(f'time for conversion and computation {toc -  tic}')

    return n_outerplanar, n_total


def run(passed_args = None):
    args = parse_args(passed_args)
    return main(args)
