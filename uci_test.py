import csv
import numpy as np
import json
#import scipy.misc
import tqdm

from json2vec import JSONTreeLSTM, SetJSONNN
#from json2vec import JSONTreeLSTM
import torch

from datasets import load_car_dataset, load_nursery_dataset, load_seismic_dataset
from datasets import load_poker_dataset, load_mushroom_dataset, load_contraceptive_dataset
from datasets import load_automobile_dataset, load_bank_dataset, load_census_dataset
from datasets import load_student_dataset, load_poker_dataset_train_test
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

def get_fold(X, y, fold, num_folds):
    folds_X = np.array_split(X, num_folds)
    test_X = folds_X.pop(fold)
    train_X = np.concatenate(folds_X)
    
    folds_y = np.array_split(y, num_folds)
    test_y = folds_y.pop(fold)
    train_y = np.concatenate(folds_y)
    return train_X, test_X, train_y, test_y

def run_once(dataset="car", model_type='json-nn', seed=123, epochs=None, lr=0.000125,
    mem_dim=128, batch_size=64, tie_weights_containers = False, tie_weights_primitives = False,
    homogenous_types=False, layers=3,
    test=False, test_fold=0, test_folds=5,
    cross_val=False, val_fold=0, val_folds=3,
    poker_frac_test=None):


    regression = False
    if poker_frac_test is None:
        # Get data
        dataloader = { 'car': load_car_dataset,
                       'nursery': load_nursery_dataset,
                       'seismic': load_seismic_dataset,
                       'poker': load_poker_dataset,
                       'mushroom': load_mushroom_dataset,
                       'contraceptive': load_contraceptive_dataset,
                       'automobile': load_automobile_dataset,
                       'bank': load_bank_dataset,
                       'student': load_student_dataset,
                       'student-regression':
                           lambda: load_student_dataset(regression=True),
                       'census': load_census_dataset }
        jsons, vectors, labels = (dataloader[dataset])()

        regression = 'regression' in dataset

        # Get new dataframe
        if model_type == 'mlp':
            X = np.array(vectors)
        else:
            X = jsons

        if regression:
            y = np.array(labels)
        else:
            encoder = LabelEncoder()
            y = encoder.fit_transform(labels)

        if epochs is None:
            epochs = min(50000 // len(X) + 1, 50)

        # split into train and test sets
        np.random.seed(123)
        X, y = shuffle(X, y)

        # Get test fold
        #frac = max(100, len(X) * 0.1)
        X_train, X_test, y_train, y_test = get_fold(X, y, test_fold, test_folds)

        # Validation
        if not test and cross_val:
            X_train, X_test, y_train, y_test = get_fold(X_train, y_train, val_fold, val_folds)
        elif not test:
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)

        test_epochs = [1, 2, 3, 5, 10, 20, 30, 50, epochs]

    else:
        print("USING POKER DATSASET")
        # Load poker split
        jsons, vectors, labels, j_test, v_test, l_test = load_poker_dataset_train_test()

        # Get new dataframe
        num_items = int(len(labels) * poker_frac_test)
        print(len(labels))
        print(num_items)
        if model_type == 'mlp':
            X_train = np.array(vectors)
            X_test = np.array(v_test)
        else:
            X_train = jsons
            X_test = j_test
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(labels)
        y_test = encoder.transform(l_test)
        
        np.random.seed(seed)
        X_train, y_train = shuffle(X_train, y_train)
        
        if test:
            X_test, y_test = shuffle(X_test, y_test)
            X_test = X_test[:100000]
            y_test = y_test[:100000]
            test_epochs = [epochs]
        else:
            X_test = X_train[num_items:]
            y_test = y_train[num_items:]
            test_epochs = [1, 2, 3, 5, 10, 20, 30, 50, epochs]
            

        X_train = X_train[:num_items]
        y_train = y_train[:num_items]

        

    print(dataset)
    print(model_type)
    np.random.seed(seed)
    num_classes = 1 if regression else len(encoder.classes_)
    if model_type == 'json-nn':
        embedder = JSONTreeLSTM(mem_dim, tie_weights_containers,
            tie_weights_primitives, homogenous_types)
        output_layer = torch.nn.Linear(mem_dim*2, num_classes)
        model = torch.nn.Sequential(embedder, output_layer)
    if model_type == 'set-nn-max':
        embedder = SetJSONNN(mem_dim, tie_weights_containers, tie_weights_primitives)
        output_layer = torch.nn.Linear(mem_dim, num_classes)
        model = torch.nn.Sequential(embedder, output_layer)
    elif model_type == 'mlp':
        hidden_size = mem_dim
        input_layer = torch.nn.Linear(X_train.shape[1], hidden_size)
        layers_ = [ input_layer ]
        for i in range(layers):
            layers_ = layers_ + [ torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU() ]
        output_layer = torch.nn.Linear(hidden_size, num_classes)
        layers_.append(output_layer)
        model = torch.nn.Sequential(*layers_)
    criterion = torch.nn.MSELoss() if regression else torch.nn.CrossEntropyLoss()

    #print([ m  for m in model.parameters()])

    def test():
        running_test_loss = 0.0
        running_test_acc = 0.0
        for i, data in tqdm.tqdm(enumerate(X_test), total=len(X_test), leave=False):
            # get the inputs
            inputs = torch.Tensor(data) if model_type=='mlp' else data
            labels = torch.Tensor([[y_test[i]]]) if regression else torch.LongTensor([y_test[i]]) 

            outputs = model(inputs).view(1, -1)
            loss = criterion(outputs, labels)
            
            if not regression:
                _, predicted = torch.max(outputs, 1)
                acc = (predicted == labels)
                running_test_acc += acc.item()

            # print statistics
            running_test_loss += loss.item()
        avr_loss = running_test_loss / len(X_test)
        avr_acc = running_test_acc / len(X_test) * 100.0
        return avr_loss, avr_acc

    # Start experiments
    tqdm.tqdm.write('Training for {:d} epochs'.format(epochs))
    if poker_frac_test is None:
        l, a = test()
        tqdm.tqdm.write('test_loss: %.3f, test_acc: %.3f' % (l,a))
    else:
        inputs = torch.Tensor(X_train[0]) if model_type=='mlp' else X_train[0]
        model(inputs)
        l, a = 0, 0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses, accs = {0: l}, {0: a}

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        running_acc = 0.0
        test_step = batch_size * (2048 // batch_size)

        X_train, y_train = shuffle(X_train, y_train)
        for i, data in tqdm.tqdm(enumerate(X_train), total=len(X_train), leave=False,
            desc='Epoch {:d}: '.format(epoch)):

            # get the inputs
            inputs = torch.Tensor(data) if model_type=='mlp' else data
            labels = torch.Tensor([[y_train[i]]]) if regression else torch.LongTensor([y_train[i]]) 
            
            if i % batch_size == 0:
                optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = model(inputs).view(1, -1)
            loss = criterion(outputs, labels) / batch_size
            loss.backward()

            if (i+1) % batch_size == 0:
                optimizer.step()
            
            # print statistics
            if not regression:
                _, predicted = torch.max(outputs, 1)
                acc = (predicted == labels)
                running_acc += acc.item()
                
                running_loss += loss.item() * batch_size

            if i % 5000 == 4999:    # print every 2000 mini-batches
                tqdm.tqdm.write('[%d, %5d] loss: %.3f, acc: %.3f' %
                      (epoch + 1, i + 1, running_loss / 5000, running_acc / 50))
                running_loss = 0.0
                running_acc = 0.0

        if (epoch+1) in test_epochs:
            l, a = test()
            tqdm.tqdm.write('test_loss: %.3f, test_acc: %.3f' % (l,a))
            losses[epoch+1] = l
            accs[epoch+1] = a


        
    print('Finished Training')
    
    return losses, accs


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="car",
                       help='Dataset to use, choices are:' + \
                       'car, nursery, seismic,  poker, mushroom, contraceptive, automobile, bank, student')
    parser.add_argument('--model', type=str, default="json-nn",
                       help='Model to use, choices are: mlp, json-nn, set-nn-max')

    parser.add_argument('--test', action='store_true',
                       help='Set to use helf-out test set')

    parser.add_argument('--learning_rate', type=float, default=0.00025,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Effective batch size')
    parser.add_argument('--num_epoch', type=int, default=4,
                       help='Number of iterations over full dataset')
                       
    parser.add_argument('--dataset_fraction', type=float, default=0.05,
                       help='Fraction of dataset to use (only valid for poker dataset)')

   
    args = parser.parse_args()


    if args.dataset == 'poker':
        print(run_once(model_type=args.model, dataset='poker', batch_size=args.batch_size,
            lr=args.learning_rate, epochs=args.num_epoch, poker_frac_test=args.dataset_fraction,
            test=args.test))
    else:
        print(run_once(model_type=args.model, dataset=args.dataset, batch_size=args.batch_size,
            lr=args.learning_rate, epochs=args.num_epoch, poker_frac_test=None,
            test=args.test))