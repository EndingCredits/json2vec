#from dota_test import run_once
from uci_test import run_once as run_once_poker
from datetime import datetime
import numpy as np

#N.B: LR for these results was tuned incrorectly, it should be 0.001 for all results
hparams = [
    #{ 'model_type': 'set-nn-max', 'dataset': 'teamfights', 'lr': 0.0005,
    #'mem_dim': 64, 'batch_size': 4, 'epochs': 1 },

    #{'model_type': 'json-nn', 'dataset': 'teamfights', 'lr': 0.0005,
    #'mem_dim': 128, 'batch_size': 4, 'epochs': 1 },

    #{'model_type': 'json-nn', 'dataset': 'poker', 'lr': 0.0005, #55.11
    #'mem_dim': 128, 'batch_size': 64, 'epochs': 50, 'poker_frac_test': 0.05 },
    #{'model_type': 'json-nn', 'dataset': 'poker', 'lr': 0.00025, #91.6
    #'mem_dim': 128, 'batch_size': 4, 'epochs': 100, 'poker_frac_test': 0.2 },
    #{'model_type': 'json-nn', 'dataset': 'poker', 'lr': 0.0005, #98.60
    #'mem_dim': 128, 'batch_size': 16, 'epochs': 50, 'poker_frac_test': 0.5}
    #{'model_type': 'json-nn', 'dataset': 'poker', 'lr': 0.001, #97.76
    #'mem_dim': 32, 'batch_size': 4, 'epochs': 50, 'poker_frac_test': 1.0},#0.9}

    #{'model_type': 'set-nn-max', 'dataset': 'poker', 'lr': 0.001, #83.4
    #'mem_dim': 32, 'batch_size': 4, 'epochs': 100, 'poker_frac_test': 0.05},
    #{'model_type': 'set-nn-max', 'dataset': 'poker', 'lr': 0.001, #92.24
    #'mem_dim': 128, 'batch_size': 64, 'epochs': 100, 'poker_frac_test': 0.2},
    #{'model_type': 'set-nn-max', 'dataset': 'poker', 'lr': 0.00025, #97.05
    #'mem_dim': 128, 'batch_size': 4, 'epochs': 50, 'poker_frac_test': 0.5},
    #{'model_type': 'set-nn-max', 'dataset': 'poker', 'lr': 0.00025, #97.12
    #'mem_dim': 64, 'batch_size': 4, 'epochs': 50, 'poker_frac_test': 1.0},#0.9}

    #{'model_type': 'mlp', 'dataset': 'poker', 'lr': 0.001, 'mem_dim': 128, #47.8
    #'batch_size': 4, 'epochs': 100, 'layers': 5, 'poker_frac_test': 0.05},
    #{'model_type': 'mlp', 'dataset': 'poker', 'lr': 0.00025, 'mem_dim': 128, #84.3
    #'batch_size': 16, 'epochs': 100, 'layers': 3, 'poker_frac_test': 0.2},
    #{'model_type': 'mlp', 'dataset': 'poker', 'lr': 0.00025, 'mem_dim': 32, #97.04
    #'batch_size': 4, 'epochs': 50, 'layers': 5, 'poker_frac_test': 0.5},
    #{'model_type': 'mlp', 'dataset': 'poker', 'lr': 0.001, 'mem_dim': 32, #98.44
    #'batch_size': 4, 'epochs': 50, 'layers': 3, 'poker_frac_test': 1.0}, #0.9}

    #{'model_type': 'json-nn-modified', 'dataset': 'poker', 'lr': 0.001, #96.69
    #'mem_dim': 64, 'batch_size': 4, 'epochs': 50, 'poker_frac_test': 0.05},
    #{'model_type': 'json-nn-modified', 'dataset': 'poker', 'lr': 0.001, #97.30
    #'mem_dim': 128, 'batch_size': 16, 'epochs': 50, 'poker_frac_test': 0.2},
    #{'model_type': 'json-nn-modified', 'dataset': 'poker', 'lr': 0.001, #97.44
    #'mem_dim': 64, 'batch_size': 4, 'epochs': 50, 'poker_frac_test': 0.5},
    #{'model_type': 'json-nn-modified', 'dataset': 'poker', 'lr': 0.001, #97.32
    #'mem_dim': 128, 'batch_size': 64, 'epochs': 30, 'poker_frac_test': 1.0}, #0.9}


]

seed = "12345678901234567890"

for params in hparams:
    for i in range(5):
        starttime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        args = params.copy()
        args['test'] = True
        args['seed'] = int(seed[:i+3])
        print("args:{}\n".format(args))

        if args['dataset'] == 'poker':
            loss, acc = run_once_poker(**args)
        else:
            loss, acc = run_once(**args)

        outfile = "results_other_{}_{}.txt".format(args['dataset'], args['model_type'])

        endtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(outfile, 'a+') as f:
            f.write("{} -> {}\n".format(starttime, endtime))
            f.write("args: {}\n".format(args))
            f.write("test_loss: {}\n".format(loss))
            f.write("test_acc: {}\n\n\n".format(acc))