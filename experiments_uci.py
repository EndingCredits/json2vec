from uci_test import run_once
from datetime import datetime
import numpy as np

hparams = [
    #LSTM
    #car
    #{ 'model_type': 'json-nn', 'dataset': 'car', 'lr': 0.001, 'mem_dim': 64,
    #'batch_size': 4, 'test_fold': 0, 'test_folds': 5, 'epochs': 10 },
    #{'model_type': 'json-nn', 'dataset': 'car', 'lr': 0.001, 'mem_dim': 64,
    #'batch_size': 4, 'test_fold': 1, 'test_folds': 5, 'epochs': 10 },
    #{'model_type': 'json-nn', 'dataset': 'car', 'lr': 0.001, 'mem_dim': 128,
    #'batch_size': 4, 'test_fold': 2, 'test_folds': 5, 'epochs': 10 },
    #{'model_type': 'json-nn', 'dataset': 'car', 'lr': 0.001, 'mem_dim': 128,
    #'batch_size': 4, 'test_fold': 3, 'test_folds': 5, 'epochs': 10 },
    #{'model_type': 'json-nn', 'dataset': 'car', 'lr': 0.001, 'mem_dim': 128,
    #'batch_size': 4, 'test_fold': 4, 'test_folds': 5, 'epochs': 10 },

    #nursery
    #{'model_type': 'json-nn', 'dataset': 'nursery', 'lr': 0.00025, 'mem_dim': 64,
    #'batch_size': 16, 'test_fold': 0, 'test_folds': 5, 'epochs': 3 },
    #{'model_type': 'json-nn', 'dataset': 'nursery', 'lr': 0.001, 'mem_dim': 128,
    #'batch_size': 64, 'test_fold': 1, 'test_folds': 5, 'epochs': 4 },
    #{'model_type': 'json-nn', 'dataset': 'nursery', 'lr': 0.001, 'mem_dim': 64,
    #'batch_size': 4, 'test_fold': 2, 'test_folds': 5, 'epochs': 3 },
    #{'model_type': 'json-nn', 'dataset': 'nursery', 'lr': 0.0005, 'mem_dim': 128,
    #'batch_size': 4, 'test_fold': 3, 'test_folds': 5, 'epochs': 4 },
    #{'model_type': 'json-nn', 'dataset': 'nursery', 'lr': 0.0005, 'mem_dim': 128,
    #'batch_size': 16, 'test_fold': 4, 'test_folds': 5, 'epochs': 3 },

    #mushroom
    #{'model_type': 'json-nn', 'dataset': 'mushroom', 'lr': 0.001, 'mem_dim': 32,
    #'batch_size': 4, 'test_fold': 0, 'test_folds': 5, 'epochs': 2 },
    #{'model_type': 'json-nn', 'dataset': 'mushroom', 'lr': 0.001, 'mem_dim': 32,
    #'batch_size': 4, 'test_fold': 1, 'test_folds': 5, 'epochs': 2 },
    #{'model_type': 'json-nn', 'dataset': 'mushroom', 'lr': 0.001, 'mem_dim': 32,
    #'batch_size': 4, 'test_fold': 2, 'test_folds': 5, 'epochs': 2 },
    #{'model_type': 'json-nn', 'dataset': 'mushroom', 'lr': 0.001, 'mem_dim': 64,
    #'batch_size': 4, 'test_fold': 3, 'test_folds': 5, 'epochs': 2 },
    #{'model_type': 'json-nn', 'dataset': 'mushroom', 'lr': 0.001, 'mem_dim': 32,
    #'batch_size': 4, 'test_fold': 4, 'test_folds': 5, 'epochs': 2 },

    #contraceptive
    #{'model_type': 'json-nn', 'dataset': 'contraceptive', 'lr': 0.001, 'mem_dim': 128,
    #'batch_size': 4, 'test_fold': 0, 'test_folds': 5, 'epochs': 2 },
    #{'model_type': 'json-nn', 'dataset': 'contraceptive', 'lr': 0.0005, 'mem_dim': 64,
    #'batch_size': 64, 'test_fold': 1, 'test_folds': 5, 'epochs': 5 },
    #{'model_type': 'json-nn', 'dataset': 'contraceptive', 'lr': 0.0005, 'mem_dim': 32,
    #'batch_size': 64, 'test_fold': 2, 'test_folds': 5, 'epochs': 3 },
    #{'model_type': 'json-nn', 'dataset': 'contraceptive', 'lr': 0.001, 'mem_dim': 64,
    #'batch_size': 4, 'test_fold': 3, 'test_folds': 5, 'epochs': 5 },
    #{'model_type': 'json-nn', 'dataset': 'contraceptive', 'lr': 0.0005, 'mem_dim': 64,
    #'batch_size': 16, 'test_fold': 4, 'test_folds': 5, 'epochs': 5 },

    #automobile
    #{'model_type': 'json-nn', 'dataset': 'automobile', 'lr': 0.00025, 'mem_dim': 128,
    #'batch_size': 4, 'test_fold': 0, 'test_folds': 5, 'epochs': 30 },
    #{'model_type': 'json-nn', 'dataset': 'automobile', 'lr': 0.0005, 'mem_dim': 128,
    #'batch_size': 4, 'test_fold': 1, 'test_folds': 5, 'epochs': 10 },
    #{'model_type': 'json-nn', 'dataset': 'automobile', 'lr': 0.001, 'mem_dim': 128,
    #'batch_size': 4, 'test_fold': 2, 'test_folds': 5, 'epochs': 30 },
    #{'model_type': 'json-nn', 'dataset': 'automobile', 'lr': 0.0005, 'mem_dim': 128,
    #'batch_size': 4, 'test_fold': 3, 'test_folds': 5, 'epochs': 30 },
    #{'model_type': 'json-nn', 'dataset': 'automobile', 'lr': 0.00025, 'mem_dim': 128,
    #'batch_size': 4, 'test_fold': 4, 'test_folds': 5, 'epochs': 30 },

    #bank
    #{'model_type': 'json-nn', 'dataset': 'bank', 'lr': 0.00025, 'mem_dim': 128,
    #'batch_size': 64, 'test_fold': 0, 'test_folds': 5, 'epochs': 4 },
    #{'model_type': 'json-nn', 'dataset': 'bank', 'lr': 0.00025, 'mem_dim': 128,
    #'batch_size': 16, 'test_fold': 1, 'test_folds': 5, 'epochs': 3 },
    #{'model_type': 'json-nn', 'dataset': 'bank', 'lr': 0.00025, 'mem_dim': 32,
    #'batch_size': 16, 'test_fold': 2, 'test_folds': 5, 'epochs': 4 },
    #{'model_type': 'json-nn', 'dataset': 'bank', 'lr': 0.001, 'mem_dim': 128,
    #'batch_size': 4, 'test_fold': 3, 'test_folds': 5, 'epochs': 2 },
    #{'model_type': 'json-nn', 'dataset': 'bank', 'lr': 0.001, 'mem_dim': 64,
    #'batch_size': 4, 'test_fold': 4, 'test_folds': 5, 'epochs': 4 },

    #seismic
    #{'model_type': 'json-nn', 'dataset': 'seismic', 'lr': 0.00025, 'mem_dim': 32,
    #'batch_size': 16, 'test_fold': 0, 'test_folds': 5, 'epochs': 5 },
    #{'model_type': 'json-nn', 'dataset': 'seismic', 'lr': 0.001, 'mem_dim': 64,
    #'batch_size': 64, 'test_fold': 1, 'test_folds': 5, 'epochs': 5 },
    #{'model_type': 'json-nn', 'dataset': 'seismic', 'lr': 0.0005, 'mem_dim': 64,
    #'batch_size': 4, 'test_fold': 2, 'test_folds': 5, 'epochs': 3 },
    #{'model_type': 'json-nn', 'dataset': 'seismic', 'lr': 0.001, 'mem_dim': 128,
    #'batch_size': 16, 'test_fold': 3, 'test_folds': 5, 'epochs': 1 },
    #{'model_type': 'json-nn', 'dataset': 'seismic', 'lr': 0.00025, 'mem_dim': 128,
    #'batch_size': 64, 'test_fold': 3, 'test_folds': 5, 'epochs': 3 },


    #student
    #{'model_type': 'json-nn', 'dataset': 'student', 'lr': 0.00025, 'mem_dim': 128,
    #'batch_size': 16, 'test_fold': 0, 'test_folds': 5, 'epochs': 3 },
    #{'model_type': 'json-nn', 'dataset': 'student', 'lr': 0.0005, 'mem_dim': 64,
    #'batch_size': 16, 'test_fold': 1, 'test_folds': 5, 'epochs': 5 },
    #{'model_type': 'json-nn', 'dataset': 'student', 'lr': 0.0005, 'mem_dim': 64,
    #'batch_size': 16, 'test_fold': 2, 'test_folds': 5, 'epochs': 10 },
    #{'model_type': 'json-nn', 'dataset': 'student', 'lr': 0.001, 'mem_dim': 64,
    #'batch_size': 64, 'test_fold': 3, 'test_folds': 5, 'epochs': 5 },
    #{'model_type': 'json-nn', 'dataset': 'student', 'lr': 0.0005, 'mem_dim': 32,
    #'batch_size': 4, 'test_fold': 4, 'test_folds': 5, 'epochs': 5 },

    #student-reg
    #{'model_type': 'json-nn', 'dataset': 'student-regression', 'lr': 0.001, 'mem_dim': 32,
    #'batch_size': 16, 'test_fold': 0, 'test_folds': 5, 'epochs': 10 },
    #{'model_type': 'json-nn', 'dataset': 'student-regression', 'lr': 0.00025, 'mem_dim': 32,
    #'batch_size': 16, 'test_fold': 1, 'test_folds': 5, 'epochs': 5 },
    #{'model_type': 'json-nn', 'dataset': 'student-regression', 'lr': 0.00025, 'mem_dim': 64,
    #'batch_size': 16, 'test_fold': 2, 'test_folds': 5, 'epochs': 5 },
    #{'model_type': 'json-nn', 'dataset': 'student-regression', 'lr': 0.0005, 'mem_dim': 32,
    #'batch_size': 16, 'test_fold': 3, 'test_folds': 5, 'epochs': 10 },
    #{'model_type': 'json-nn', 'dataset': 'student-regression', 'lr': 0.001, 'mem_dim': 32,
    #'batch_size': 4, 'test_fold': 4, 'test_folds': 5, 'epochs': 3 },

    #set-nn (max)
    #car
    #{'model_type': 'set-nn-max', 'dataset': 'car', 'lr': 0.001, 'mem_dim': 32,
    #'batch_size': 4, 'test_fold': 0, 'test_folds': 5, 'epochs': 20},
    #{'model_type': 'set-nn-max', 'dataset': 'car', 'lr': 0.001, 'mem_dim': 64, 
    #'batch_size': 4, 'test_fold': 1, 'test_folds': 5, 'epochs': 20},
    #{'model_type': 'set-nn-max', 'dataset': 'car', 'lr': 0.001, 'mem_dim': 32, 
    #'batch_size': 4, 'test_fold': 2, 'test_folds': 5, 'epochs': 20},
    #{'model_type': 'set-nn-max', 'dataset': 'car', 'lr': 0.001, 'mem_dim': 32, 
    #'batch_size': 4, 'test_fold': 3, 'test_folds': 5, 'epochs': 20},
    #{'model_type': 'set-nn-max', 'dataset': 'car', 'lr': 0.001, 'mem_dim': 64, 
    #'batch_size': 4, 'test_fold': 4, 'test_folds': 5, 'epochs': 10},

    #nursery
    #{'model_type': 'set-nn-max', 'dataset': 'nursery', 'lr': 0.00025, 'mem_dim': 128,
    #'batch_size': 16, 'test_fold': 0, 'test_folds': 5, 'epochs': 3 },
    #{'model_type': 'set-nn-max', 'dataset': 'nursery', 'lr': 0.00025, 'mem_dim': 128,
    #'batch_size': 16, 'test_fold': 1, 'test_folds': 5, 'epochs': 3 },
    #{'model_type': 'set-nn-max', 'dataset': 'nursery', 'lr': 0.00025, 'mem_dim': 128,
    #'batch_size': 16, 'test_fold': 2, 'test_folds': 5, 'epochs': 3 },
    #{'model_type': 'set-nn-max', 'dataset': 'nursery', 'lr': 0.0005, 'mem_dim': 128,
    #'batch_size': 64, 'test_fold': 3, 'test_folds': 5, 'epochs': 4 },
    #{'model_type': 'set-nn-max', 'dataset': 'nursery', 'lr': 0.00025, 'mem_dim': 64,
    #'batch_size': 4, 'test_fold': 4, 'test_folds': 5, 'epochs': 3 },

    #mushroom
    #{'model_type': 'set-nn-max', 'dataset': 'mushroom', 'lr': 0.001, 'mem_dim': 128,
    #'batch_size': 4, 'test_fold': 0, 'test_folds': 5, 'epochs': 2 },
    #{'model_type': 'set-nn-max', 'dataset': 'mushroom', 'lr': 0.001, 'mem_dim': 64,
    #'batch_size': 16, 'test_fold': 1, 'test_folds': 5, 'epochs': 3 },
    #{'model_type': 'set-nn-max', 'dataset': 'mushroom', 'lr': 0.001, 'mem_dim': 64,
    #'batch_size': 16, 'test_fold': 2, 'test_folds': 5, 'epochs': 3 },
    #{'model_type': 'set-nn-max', 'dataset': 'mushroom', 'lr': 0.001, 'mem_dim': 64,
    #'batch_size': 16, 'test_fold': 3, 'test_folds': 5, 'epochs': 2 },
    #{'model_type': 'set-nn-max', 'dataset': 'mushroom', 'lr': 0.001, 'mem_dim': 128,
    #'batch_size': 16, 'test_fold': 4, 'test_folds': 5, 'epochs': 3 },

    #contraceptive
    #{'model_type': 'set-nn-max', 'dataset': 'contraceptive', 'lr': 0.00025, 'mem_dim': 64,
    #'batch_size': 4, 'test_fold': 0, 'test_folds': 5, 'epochs': 10 },
    #{'model_type': 'set-nn-max', 'dataset': 'contraceptive', 'lr': 0.0005, 'mem_dim': 32,
    #'batch_size': 16, 'test_fold': 1, 'test_folds': 5, 'epochs': 10 },
    #{'model_type': 'set-nn-max', 'dataset': 'contraceptive', 'lr': 0.0005, 'mem_dim': 32,
    #'batch_size': 4, 'test_fold': 2, 'test_folds': 5, 'epochs': 20 },
    #{'model_type': 'set-nn-max', 'dataset': 'contraceptive', 'lr': 0.001, 'mem_dim': 64,
    #'batch_size': 16, 'test_fold': 3, 'test_folds': 5, 'epochs': 10 },
    #{'model_type': 'set-nn-max', 'dataset': 'contraceptive', 'lr': 0.001, 'mem_dim': 128,
    #'batch_size': 64, 'test_fold': 4, 'test_folds': 5, 'epochs': 10 },

    #automobile
    #{'model_type': 'set-nn-max', 'dataset': 'automobile', 'lr': 0.0005, 'mem_dim': 64,
    #'batch_size': 4, 'test_fold': 0, 'test_folds': 5, 'epochs': 50 },
    #{'model_type': 'set-nn-max', 'dataset': 'automobile', 'lr': 0.001, 'mem_dim': 64,
    #'batch_size': 4, 'test_fold': 1, 'test_folds': 5, 'epochs': 50 },
    #{'model_type': 'set-nn-max', 'dataset': 'automobile', 'lr': 0.0005, 'mem_dim': 64,
    #'batch_size': 4, 'test_fold': 2, 'test_folds': 5, 'epochs': 50 },
    #{'model_type': 'set-nn-max', 'dataset': 'automobile', 'lr': 0.001, 'mem_dim': 64,
    #'batch_size': 4, 'test_fold': 3, 'test_folds': 5, 'epochs': 50 },
    #{'model_type': 'set-nn-max', 'dataset': 'automobile', 'lr': 0.001, 'mem_dim': 64,
    #'batch_size': 4, 'test_fold': 4, 'test_folds': 5, 'epochs': 50 },

    #bank
    #{'model_type': 'set-nn-max', 'dataset': 'bank', 'lr': 0.001, 'mem_dim': 64,
    #'batch_size': 16, 'test_fold': 0, 'test_folds': 5, 'epochs': 2 },
    #{'model_type': 'set-nn-max', 'dataset': 'bank', 'lr': 0.00025, 'mem_dim': 64,
    #'batch_size': 4, 'test_fold': 1, 'test_folds': 5, 'epochs': 4 },
    #{'model_type': 'set-nn-max', 'dataset': 'bank', 'lr': 0.0005, 'mem_dim': 128,
    #'batch_size': 64, 'test_fold': 2, 'test_folds': 5, 'epochs': 1 },
    #{'model_type': 'set-nn-max', 'dataset': 'bank', 'lr': 0.0005, 'mem_dim': 64,
    #'batch_size': 4, 'test_fold': 3, 'test_folds': 5, 'epochs': 4 },
    #{'model_type': 'set-nn-max', 'dataset': 'bank', 'lr': 0.0005, 'mem_dim': 128,
    #'batch_size': 4, 'test_fold': 4, 'test_folds': 5, 'epochs': 3 },

    #seismic
    #{'model_type': 'set-nn-max', 'dataset': 'seismic', 'lr': 0.001, 'mem_dim': 64,
    #'batch_size': 4, 'test_fold': 0, 'test_folds': 5, 'epochs': 20 },
    #{'model_type': 'set-nn-max', 'dataset': 'seismic', 'lr': 0.0005, 'mem_dim': 128,
    #'batch_size': 4, 'test_fold': 1, 'test_folds': 5, 'epochs': 30 },
    #{'model_type': 'set-nn-max', 'dataset': 'seismic', 'lr': 0.00025, 'mem_dim': 32, 
    #'batch_size': 4, 'test_fold': 2, 'test_folds': 5, 'epochs': 10 },
    #{'model_type': 'set-nn-max', 'dataset': 'seismic', 'lr': 0.001, 'mem_dim': 64,
    #'batch_size': 16, 'test_fold': 3, 'test_folds': 5, 'epochs': 10 },
    #{'model_type': 'set-nn-max', 'dataset': 'seismic', 'lr': 0.0005, 'mem_dim': 128,
    #'batch_size': 16, 'test_fold': 4, 'test_folds': 5, 'epochs': 50 },

    #student
    #{'model_type': 'set-nn-max', 'dataset': 'student', 'lr': 0.001, 'mem_dim': 128,
    #'batch_size': 4, 'test_fold': 0, 'test_folds': 5, 'epochs': 3 },
    #{'model_type': 'set-nn-max', 'dataset': 'student', 'lr': 0.00025, 'mem_dim': 128,
    #'batch_size': 64, 'test_fold': 1, 'test_folds': 5, 'epochs': 50 },
    #{'model_type': 'set-nn-max', 'dataset': 'student', 'lr': 0.001, 'mem_dim': 128,
    #'batch_size': 16, 'test_fold': 2, 'test_folds': 5, 'epochs': 10 },
    #{'model_type': 'set-nn-max', 'dataset': 'student', 'lr': 0.001, 'mem_dim': 64,
    #'batch_size': 16, 'test_fold': 3, 'test_folds': 5, 'epochs': 20 },
    #{'model_type': 'set-nn-max', 'dataset': 'student', 'lr': 0.00025, 'mem_dim': 64,
    #'batch_size': 4, 'test_fold': 4, 'test_folds': 5, 'epochs': 10 },

    #student-regression
    #{'model_type': 'set-nn-max', 'dataset': 'student-regression', 'lr': 0.00025,
    #'mem_dim': 64, 'batch_size': 64, 'test_fold': 0, 'test_folds': 5, 'epochs': 30 },
    #{'model_type': 'set-nn-max', 'dataset': 'student-regression', 'lr': 0.00025,
    #'mem_dim': 128, 'batch_size': 64, 'test_fold': 1, 'test_folds': 5, 'epochs': 30 },
    #{'model_type': 'set-nn-max', 'dataset': 'student-regression', 'lr': 0.00025,
    #'mem_dim': 64, 'batch_size': 4, 'test_fold': 2, 'test_folds': 5, 'epochs': 5 },
    #{'model_type': 'set-nn-max', 'dataset': 'student-regression', 'lr': 0.00025,
    #'mem_dim': 128, 'batch_size': 64, 'test_fold': 3, 'test_folds': 5, 'epochs': 30 },
    #{'model_type': 'set-nn-max', 'dataset': 'student-regression', 'lr': 0.0005,
    #'mem_dim': 64, 'batch_size': 16, 'test_fold': 4, 'test_folds': 5, 'epochs': 10 },



    #mlp
    #car
    #{'model_type': 'mlp', 'dataset': 'car', 'lr': 0.0005, 'mem_dim': 32,
    #'batch_size': 4, 'layers': 3, 'test_fold': 0, 'test_folds': 5, 'epochs': 10},
    #{'model_type': 'mlp', 'dataset': 'car', 'lr': 0.00025, 'mem_dim': 128,
    #'batch_size': 4, 'layers': 1, 'test_fold': 1, 'test_folds': 5, 'epochs': 3},
    #{'model_type': 'mlp', 'dataset': 'car', 'lr': 0.0005, 'mem_dim': 64,
    #'batch_size': 16, 'layers': 1, 'test_fold': 2, 'test_folds': 5, 'epochs': 5},
    #{'model_type': 'mlp', 'dataset': 'car', 'lr': 0.001, 'mem_dim': 32,
    #'batch_size': 4, 'layers': 3, 'test_fold': 3, 'test_folds': 5, 'epochs': 10},
    #{'model_type': 'mlp', 'dataset': 'car', 'lr': 0.00025, 'mem_dim': 128,
    #'batch_size': 4, 'layers': 5, 'test_fold': 4, 'test_folds': 5, 'epochs': 20},

    #nursery
    #{'model_type': 'mlp', 'dataset': 'nursery', 'lr': 0.00025, 'mem_dim': 64,
    #'batch_size': 16, 'layers': 5, 'test_fold': 0, 'test_folds': 5, 'epochs': 4},
    #{'model_type': 'mlp', 'dataset': 'nursery', 'lr': 0.001, 'mem_dim': 128,
    #'batch_size': 64, 'layers': 3, 'test_fold': 1, 'test_folds': 5, 'epochs': 4},
    #{'model_type': 'mlp', 'dataset': 'nursery', 'lr': 0.001, 'mem_dim': 64,
    #'batch_size': 4, 'layers': 3, 'test_fold': 2, 'test_folds': 5, 'epochs': 4},
    #{'model_type': 'mlp', 'dataset': 'nursery', 'lr': 0.0005, 'mem_dim': 128, 
    #'batch_size': 16, 'layers': 3, 'test_fold': 3, 'test_folds': 5, 'epochs': 3},
    #{'model_type': 'mlp', 'dataset': 'nursery', 'lr': 0.0005, 'mem_dim': 64, 
    #'batch_size': 4, 'layers': 1, 'test_fold': 4, 'test_folds': 5, 'epochs': 4},

    #mushroom
    #{'model_type': 'mlp', 'dataset': 'mushroom', 'lr': 0.001, 'mem_dim': 32, 
    #'batch_size': 4, 'epochs': 1, 'layers': 1, 'test_fold': 0, 'test_folds': 5},
    #{'model_type': 'mlp', 'dataset': 'mushroom', 'lr': 0.001, 'mem_dim': 32, 
    #'batch_size': 4, 'epochs': 1, 'layers': 1, 'test_fold': 1, 'test_folds': 5},
    #{'model_type': 'mlp', 'dataset': 'mushroom', 'lr': 0.001, 'mem_dim': 32, 
    #'batch_size': 4, 'epochs': 1, 'layers': 1, 'test_fold': 2, 'test_folds': 5},
    #{'model_type': 'mlp', 'dataset': 'mushroom', 'lr': 0.001, 'mem_dim': 32, 
    #'batch_size': 4, 'epochs': 1, 'layers': 1, 'test_fold': 3, 'test_folds': 5},
    #{'model_type': 'mlp', 'dataset': 'mushroom', 'lr': 0.001, 'mem_dim': 32, 
    #'batch_size': 4, 'epochs': 2, 'layers': 1, 'test_fold': 4, 'test_folds': 5},

    #contraceptive
    #{'model_type': 'mlp', 'dataset': 'contraceptive', 'lr': 0.0005, 'mem_dim': 32, 
    #'batch_size': 4, 'layers': 5, 'test_fold': 0, 'test_folds': 5, 'epochs': 3},
    #{'model_type': 'mlp', 'dataset': 'contraceptive', 'lr': 0.0005,'mem_dim': 64, 
    #'batch_size': 4, 'layers': 3, 'test_fold': 1, 'test_folds': 5, 'epochs': 3},
    #{'model_type': 'mlp', 'dataset': 'contraceptive', 'lr': 0.001, 'mem_dim': 128, 
    #'batch_size': 4, 'layers': 5, 'test_fold': 2, 'test_folds': 5, 'epochs': 3},
    #{'model_type': 'mlp', 'dataset': 'contraceptive', 'lr': 0.001, 'mem_dim': 32, 
    #'batch_size': 64, 'layers': 3, 'test_fold': 3, 'test_folds': 5, 'epochs': 30},
    #{'model_type': 'mlp', 'dataset': 'contraceptive', 'lr': 0.001, 'mem_dim': 64, 
    #'batch_size': 64, 'layers': 5, 'test_fold': 4, 'test_folds': 5, 'epochs': 10},

    #automobile
    #{'model_type': 'mlp', 'dataset': 'automobile', 'lr': 0.00025, 'mem_dim': 128, 
    #'batch_size': 16, 'layers': 3, 'test_fold': 0, 'test_folds': 5, 'epochs': 30},
    #{'model_type': 'mlp', 'dataset': 'automobile', 'lr': 0.00025, 'mem_dim': 128, 
    #'batch_size': 16, 'layers': 1, 'test_fold': 1, 'test_folds': 5, 'epochs': 30},
    #{'model_type': 'mlp', 'dataset': 'automobile', 'lr': 0.00025, 'mem_dim': 128, 
    #'batch_size': 4, 'layers': 1, 'test_fold': 2, 'test_folds': 5, 'epochs': 30},
    #{'model_type': 'mlp', 'dataset': 'automobile', 'lr': 0.001, 'mem_dim': 64, 
    #'batch_size': 4, 'layers': 1, 'test_fold': 3, 'test_folds': 5, 'epochs': 50},
    #{'model_type': 'mlp', 'dataset': 'automobile', 'lr': 0.001, 'mem_dim': 128, 
    #'batch_size': 16, 'layers': 3, 'test_fold': 4, 'test_folds': 5, 'epochs': 30},

    #bank
    #{'model_type': 'mlp', 'dataset': 'bank', 'lr': 0.0005, 'mem_dim': 128, 
    #'batch_size': 64, 'layers': 3, 'test_fold': 0, 'test_folds': 5, 'epochs': 3},
    #{'model_type': 'mlp', 'dataset': 'bank', 'lr': 0.001, 'mem_dim': 64, 
    #'batch_size': 16, 'layers': 1, 'test_fold': 1, 'test_folds': 5, 'epochs': 4},
    #{'model_type': 'mlp', 'dataset': 'bank', 'lr': 0.00025, 'mem_dim': 64, 
    #'batch_size': 4, 'layers': 3, 'test_fold': 2, 'test_folds': 5, 'epochs': 1},
    #{'model_type': 'mlp', 'dataset': 'bank', 'lr': 0.00025, 'mem_dim': 32, 
    #'batch_size': 4, 'layers': 5, 'test_fold': 3, 'test_folds': 5, 'epochs': 4},
    #{'model_type': 'mlp', 'dataset': 'bank', 'lr': 0.001, 'mem_dim': 64, 
    #'batch_size': 4, 'layers': 3, 'test_fold': 4, 'test_folds': 5, 'epochs': 4},

    #seismic
    #{'model_type': 'mlp', 'dataset': 'seismic', 'lr': 0.00025, 'mem_dim': 64,
    #'batch_size': 16, 'layers': 5, 'test_fold': 0, 'test_folds': 5, 'epochs': 5},
    #{'model_type': 'mlp', 'dataset': 'seismic', 'lr': 0.00025, 'mem_dim': 64, 
    #'batch_size': 64, 'layers': 5, 'test_fold': 1, 'test_folds': 5, 'epochs': 10},
    #{'model_type': 'mlp', 'dataset': 'seismic', 'lr': 0.00025, 'mem_dim': 32, 
    #'batch_size': 64, 'layers': 1, 'test_fold': 2, 'test_folds': 5, 'epochs': 5},
    #{'model_type': 'mlp', 'dataset': 'seismic', 'lr': 0.001, 'mem_dim': 64, 
    #'batch_size': 16, 'layers': 1, 'test_fold': 3, 'test_folds': 5, 'epochs': 3},
    #{'model_type': 'mlp', 'dataset': 'seismic', 'lr': 0.00025, 'mem_dim': 64, 
    #'batch_size': 64, 'layers': 3, 'test_fold': 4, 'test_folds': 5, 'epochs': 10},


    #student
    #{'model_type': 'mlp', 'dataset': 'student', 'lr': 0.001, 'mem_dim': 32, 
    #'batch_size': 4, 'layers': 1, 'test_fold': 0, 'test_folds': 5, 'epochs': 5},
    #{'model_type': 'mlp', 'dataset': 'student', 'lr': 0.00025, 'mem_dim': 32, 
    #'batch_size': 16, 'layers': 1, 'test_fold': 1, 'test_folds': 5, 'epochs': 20},
    #{'model_type': 'mlp', 'dataset': 'student', 'lr': 0.001, 'mem_dim': 128, 
    #'batch_size': 4, 'layers': 5, 'test_fold': 2, 'test_folds': 5, 'epochs': 20},
    #{'model_type': 'mlp', 'dataset': 'student', 'lr': 0.001, 'mem_dim': 64, 
    #'batch_size': 4, 'layers': 1, 'test_fold': 3, 'test_folds': 5, 'epochs': 3},
    #{'model_type': 'mlp', 'dataset': 'student', 'lr': 0.001, 'mem_dim': 128, 
    #'batch_size': 64, 'layers': 5, 'test_fold': 4, 'test_folds': 5, 'epochs': 10},


    #student-regression
    #{'model_type': 'mlp', 'dataset': 'student-regression', 'lr': 0.0005, 'mem_dim': 64,
    #'batch_size': 64, 'layers': 1, 'test_fold': 0, 'test_folds': 5, 'epochs': 20},
    #{'model_type': 'mlp', 'dataset': 'student-regression', 'lr': 0.001, 'mem_dim': 64, 
    #'batch_size': 16, 'layers': 3, 'test_fold': 1, 'test_folds': 5,'epochs': 5},
    #{'model_type': 'mlp', 'dataset': 'student-regression', 'lr': 0.001, 'mem_dim': 64, 
    #'batch_size': 4, 'layers': 5, 'test_fold': 2, 'test_folds': 5, 'epochs': 3},
    #{'model_type': 'mlp', 'dataset': 'student-regression', 'lr': 0.001, 'mem_dim': 32, 
    #'batch_size': 16, 'layers': 5, 'test_fold': 3, 'test_folds': 5, 'epochs': 10},
    #{'model_type': 'mlp', 'dataset': 'student-regression', 'lr': 0.001, 'mem_dim': 64, 
    #'batch_size': 4, 'layers': 5, 'test_fold': 4, 'test_folds': 5, 'epochs': 3}


]

seed = "12345678901234567890"

updates = [
    { 'tie_weights_containers': True, 'tie_weights_primitives': True },
    { 'homogenous_types': True },
    { 'tie_weights_containers': True, 'tie_weights_primitives': True, 'homogenous_types': True },
]

#for update in updates:
for params in hparams:
    starttime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    args = params.copy()
    args['test'] = True
    args['seed'] = 123#int(seed[:i+3])
    #args.update(update)
    print("args:{}\n".format(args))
    loss, acc = run_once(**args)

    outfile = "results_uci_{}_{}.txt".format(args['dataset'], args['model_type'])

    endtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(outfile, 'a+') as f:
        f.write("{} -> {}\n".format(starttime, endtime))
        f.write("args:{}\n".format(args))
        f.write("test_loss: {}\n".format(loss))
        f.write("test_acc: {}\n\n\n".format(acc))