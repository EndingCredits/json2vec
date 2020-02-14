from uci_test import run_once
from datetime import datetime
import numpy as np

import sys
# 'mlp', 'json-nn', or 'set-nn-max'
model = 'json-nn-modified'

learning_rates = [ 0.001 ]#, 0.0005, 0.00025 ]
batch_sizes = [ 4, 16, 64 ]
mem_dims = [ 32, 64, 128 ]
layers = [ 1, 3, 5 ]


for fraction in [ 0.9 ]:#, 0.2, 0.5 ]:
    for i in range(5):
        starttime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        np.random.seed(int(datetime.now().timestamp()))

        args = {}

        args['model_type'] = model
        args['dataset'] = 'poker'

        args['lr'] = np.random.choice(learning_rates)
        args['mem_dim'] = np.random.choice(mem_dims)
        args['batch_size'] = np.random.choice(batch_sizes)

        args['epochs'] = 50

        if model == 'mlp':
            args['layers'] = np.random.choice(layers)

        args['poker_frac_test'] = fraction

        print("{}, args:{}\n".format(model, args))
        loss, acc = run_once(**args)

        outfile = "hparam_uci_poker3.txt"

        endtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(outfile, 'a+') as f:
            f.write("{} -> {}\n".format(starttime, endtime))
            f.write("args: {}\n".format(args))
            f.write("test_loss: {}\n".format(loss))
            f.write("test_acc: {}\n\n".format(acc))
