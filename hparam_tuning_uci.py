from uci_test import run_once
#from uci_baseline import run_once as run_once_lang
from datetime import datetime
import numpy as np

import sys
# 'mlp', 'json-nn', or 'set-nn-max'
model = 'set-nn-max'#'json-nn'

#learning_rates = [ 0.0025, 0.001, 0.0005, 0.00025, 0.0001, ]
#batch_sizes = [ 1, 4, 16, 64, 128 ]
#mem_dims = [ 32, 64, 128, 256 ]
#layers = [ 1, 2, 3, 5, 10 ]
#epochss = [ 1, 2, 3, 5, 10, 15, 25, 50 ]

learning_rates = [0.001, 0.0005, 0.00025]
batch_sizes = [ 4, 16, 64 ]
mem_dims = [ 32, 64, 128 ]
layers = [ 1, 3, 5 ]

#json-nn
dataset_params = {
    "car": {
        "lr": [0.0005,0.00025,0.0001], "mem_dim": [32,64,128], "batch_size": [1,4,16],
        "epochs": 20
    },
    "nursery": {
        "lr": [0.0025,0.001,0.0005], "mem_dim": [32,64,128], "batch_size": [1,4,16],
        "epochs": 5
    },
    "mushroom": {
        "lr": [0.0005,0.00025,0.0001], "mem_dim": [32,64,128], "batch_size": [1,4,16],
        "epochs": 5
    },
    "contraceptive": {
        "lr": [0.0005,0.00025,0.0001], "mem_dim": [32,64,128], "batch_size": [64,128,256],
        "epochs": 50
    },
    "automobile": { #*
        "lr": [0.001, 0.0005, 0.00025], "mem_dim": [32,64,128], "batch_size": [4, 16, 64],
        "epochs": 50, "cross_val": True
    },
    "bank": {
        "lr": [0.0005, 0.00025, 0.0001], "mem_dim": [32,64,128], "batch_size": [1, 4, 16],
        "epochs": 5
    },
    "seismic": { #*
        "lr": [0.0025, 0.001, 0.0005], "mem_dim": [32,64,128], "batch_size": [1, 4, 16],
        "epochs": 20, "cross_val": True
    },
    "student": { #*
        "lr": [0.0025, 0.001, 0.0005], "mem_dim": [32,64,128], "batch_size": [64, 128, 256],
        "epochs": 20, "cross_val": True
    },
}


datasets = [ 'car', 'nursery', 'mushroom', 'contraceptive',
             'automobile', 'bank', 'seismic', 'student' ]

datasets = ['student']

test_folds = 5
val_folds = 3


for dataset in datasets:

  #learning_rates = dataset_params[dataset]['lr']
  #batch_sizes = dataset_params[dataset]['batch_size']
  #mem_dims = dataset_params[dataset]['mem_dim']
  #layers = dataset_params[dataset]['layers']
  #epochss = dataset_params[dataset]['epochs']
  for fold in range(test_folds):
   for learning_rate in learning_rates:
    for batch_size in batch_sizes:
     for mem_dim in mem_dims:
      #for layer in layers:
        starttime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        np.random.seed(int(datetime.now().timestamp()))

        args = {}
        if model == 'json-nn-homogenous':
            model = 'json-nn'
            args['homogenous_types'] = True

        args['model_type'] = model
        args['dataset'] = dataset

        args['lr'] = learning_rate
        args['mem_dim'] = mem_dim
        args['batch_size'] = batch_size
        #args['epochs'] = dataset_params[dataset]['epochs']

        if model == 'mlp':
            args['layers'] = layer

        args['test_fold'] = fold
        args['test_folds'] = test_folds

        if dataset_params[dataset].get('cross_val') is not None:
            args['cross_val'] = True
            args['val_folds'] = val_folds
            loss = []
            acc = []
            for val_fold in range(val_folds):
                args['val_fold'] = val_fold
                print("{}, args:{}\n".format(model, args))
                l, a = run_once(**args)
                loss.append(l)
                acc.append(a)
        else:
            print("{}, args:{}\n".format(model, args))
            loss, acc = run_once(**args)

        outfile = "hparam_uci_{}.txt".format(dataset)

        endtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(outfile, 'a+') as f:
            f.write("{} -> {}\n".format(starttime, endtime))
            f.write("args: {}\n".format(args))
            f.write("test_loss: {}\n".format(loss))
            f.write("test_acc: {}\n\n".format(acc))
