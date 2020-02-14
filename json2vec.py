import json
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import numbers
import string
from collections import defaultdict

all_characters = string.printable
n_characters = len(all_characters)

class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError( key )
        else:
            ret = self[key] = self.default_factory(key)
            return ret



# module for childsumtreelstm
class JSONNN(nn.Module):
    def __init__(self, mem_dim=128):
        super(JSONNN, self).__init__()
        self.mem_dim = mem_dim

    def forward(self, node):
        return self.embedNode(node)

    def embedNode(self, node, path=[]):

        if isinstance(node, dict): # DICT
            child_states = []
            for child_name, child in node.items():
                child_path = path + [child_name]
                child_state = self.embedNode(child, child_path)
                if child_state is None:
                    #print(name + "." + childname + " skipped...?")
                    #print(child)
                    continue # Skip any
                child_states.append(child_state)

            if not child_states:
                return None
            return self.embedObject(child_states, path)


        if isinstance(node, list): # DICT
            child_states = []
            for i, child in enumerate(node):
                child_path = path + [i]
                child_state = self.embedNode(child, child_path)
                if child_state is None:
                    #print(name + "." + childname + " skipped...?")
                    #print(child)
                    continue # Skip any
                child_states.append(child_state)

            if not child_states:
                return None
            return self.embedArray(child_states, path)

        
        elif isinstance(node, str): # STRING
            return self.embedString(node, path)
            
        elif isinstance(node, numbers.Number): # NUMBER
            return self.embedNumber(node, path)

        else:
            # not impl error
            return None




class JSONTreeLSTM(JSONNN):
    def __init__(self, mem_dim=128,
                 tie_weights_containers=False,
                 tie_weights_primitives=False,
                 homogenous_types=False ):
        super(JSONTreeLSTM, self).__init__(mem_dim)

        self.iouh = keydefaultdict(self._newiouh)
        self.fh = keydefaultdict(self._newfh)
        self.lout = keydefaultdict(self._newlout)

        self.LSTMs = keydefaultdict(self._newLSTM)

        self.string_encoder = nn.Embedding(n_characters, self.mem_dim)
        self.string_rnn = keydefaultdict(self._newStringRNN)
        #nn.LSTM(self.mem_dim, self.mem_dim, 1)

        self.numberEmbeddings = keydefaultdict(self._newNumber)
        self.numberStats = defaultdict(lambda: [])

        self.tie_primitives = tie_weights_primitives
        self.tie_containers = tie_weights_containers
        self.homogenous_types = homogenous_types

    def forward(self, node):
        if isinstance(node, str):
            try:
                node = json.loads(node)
            except:
                raise Exception(node)
        #return self.embedNode(node, path=["___root___"])[0]

        try:
            node = self.embedNode(node, path=["___root___"])
            if node is None: return torch.cat([self._initC()]*2, 1)  
            return torch.cat(node, 1) 
        except:
            print(node)

    def _canonical(self, path):
        # Restrict to last two elements
        path = path[-2:]
        
        # Replace ints with placeholder
        for i in range(len(path)):
            if isinstance(path[i], numbers.Number):
                path[i] = "___list___"
        return tuple(path)

    def _initC(self):
        return torch.empty(1, self.mem_dim).fill_(0.).requires_grad_()


    def _newiouh(self, key):
        layer = nn.Linear(self.mem_dim, self.mem_dim * 3)
        self.add_module(str(key)+"_iouh", layer)
        return layer


    def _newfh(self, key):
        layer = nn.Linear(self.mem_dim, self.mem_dim)
        self.add_module(str(key)+"_fh", layer)
        return layer

    def _newlout(self, key):
        layer = nn.Linear(self.mem_dim, self.mem_dim)
        self.add_module(str(key)+"_lout", layer)
        return layer

    def embedObject(self, child_states, path):
        canon_path = self._canonical(path) if not self.tie_containers else "___default___"

        hs = [] ; fcs = []
        for child_c, child_h in child_states:
            hs.append(child_h)

            #f = self.fh[self._canonical(path)](child_h)
            f = self.fh["___default___"](child_h)
            fc = torch.mul(torch.sigmoid(f), child_c)
            fcs.append(fc)

        if not hs:
            return None

        hs_bar = torch.sum(torch.cat(hs), dim=0, keepdim=True)
        #iou = self.iouh[self._canonical(path)](hs_bar)
        iou = self.iouh["___default___"](hs_bar)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        c = torch.mul(i, u) + torch.sum(torch.cat(fcs), dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))

        h_hat = self.lout[canon_path](h)

        return c, h_hat


    def _newLSTM(self, key):
        lstm = nn.LSTMCell(input_size = self.mem_dim * 2, hidden_size = self.mem_dim)
        self.add_module(str(key)+"_LSTM", lstm)
        return lstm

    def embedArray(self, child_states, path):
        if self.homogenous_types:
            return self.embedObject(child_states, path)

        canon_path = self._canonical(path) if not self.tie_containers else "___default___"
        lstm = self.LSTMs[canon_path]
        c = self._initC() ; h = self._initC()
        for child_c, child_h in child_states:
            child_ch = torch.cat([child_c, child_h], dim=1)
            c, h = lstm(child_ch, (c, h))
        return c, h 


    def _newStringRNN(self, key):
        lstm = nn.LSTM(self.mem_dim, self.mem_dim, 1)
        self.add_module(str(key)+"_StringLSTM", lstm)
        return lstm

    def embedString(self, string, path):
        canon_path = self._canonical(path) if not self.tie_primitives else "___default___"
        if string == "":
            return self._initC(), self._initC()

        tensor_input = torch.zeros(len(string)).long()
        for c in range(len(string)):
            try:
                tensor_input[c] = all_characters.index(string[c])
            except:
                continue
        encoded_input = self.string_encoder(tensor_input.view(1, -1)).view(-1, 1, self.mem_dim)
        output, hidden = self.string_rnn[canon_path](encoded_input)
        return self._initC(), hidden[0].mean(dim=1)
        #hidden[1].mean(dim=1), hidden[0].mean(dim=1)


    def _newNumber(self, key):
        layer = nn.Linear(1, self.mem_dim)
        self.add_module(str(key), layer)
        return layer

    def embedNumber(self, num, path):
        if self.homogenous_types:
            return self.embedString(str(num), path)

        canon_path = self._canonical(path) if not self.tie_primitives else "___default___"

        if len(self.numberStats[canon_path]) < 100:
            self.numberStats[canon_path].append(num)

        num -= np.mean(self.numberStats[canon_path])

        if len(self.numberStats[canon_path]) > 3:
            std = np.std(self.numberStats[canon_path])
            if not np.allclose(std, 0.0):
                num /= std

        encoded = self.numberEmbeddings[canon_path](torch.Tensor([[num]]))
        return self._initC(), encoded



class SetJSONNN(JSONNN):
    def __init__(self, mem_dim=128,
                 tie_weights_containers=False,
                 tie_weights_primitives=False,
                 homogenous_types=False):
        super(SetJSONNN, self).__init__(mem_dim)

        self.mem_dim = mem_dim
        self.embedder = keydefaultdict(self._new_embedder)
        self.out = keydefaultdict(self._new_out)

        self.string_encoder = nn.Embedding(n_characters, self.mem_dim)
        self.string_rnn = keydefaultdict(self._newStringRNN)

        self.numberEmbeddings = keydefaultdict(self._newNumber)
        self.numberStats = defaultdict(lambda: [])

        self.tie_primitives = tie_weights_primitives
        self.tie_containers = tie_weights_containers
        self.homogenous_types = homogenous_types

    def forward(self, node):
        if isinstance(node, str):
            try:
                node = json.loads(node)
            except:
                raise Exception(node)

        node = self.embedNode(node, path=["___root___"])
        if node is None: return self._initC()
        return node


    def _canonical(self, path):
        # Restrict to last two elements
        path = path[-2:]
        
        # Replace ints with placeholder
        for i in range(len(path)):
            if isinstance(path[i], numbers.Number):
                path[i] = "___list___"
        return tuple(path)

    def _initC(self):
        return torch.empty(1, self.mem_dim).fill_(0.).requires_grad_()

    def _new_embedder(self, key):
        l1 = torch.nn.Linear(self.mem_dim, self.mem_dim)
        l2= torch.nn.Linear(self.mem_dim, self.mem_dim)
        self.add_module(str(key)+"_l1", l1)
        self.add_module(str(key)+"_l2", l2)
        model = torch.nn.Sequential(l1, torch.nn.ReLU(), l2)
        return model

    def _new_out(self, key):
        l1 = torch.nn.Linear(self.mem_dim, self.mem_dim)
        self.add_module(str(key)+"_out_l1", l1)
        #l2 = torch.nn.Linear(self.mem_dim, self.mem_dim)
        #self.add_module(str(key)+"_out_l2", l2)
        #l3 = torch.nn.Linear(self.mem_dim, self.mem_dim)
        #self.add_module(str(key)+"_out_l3", l3)
        model = torch.nn.Sequential(torch.nn.ReLU(),
            l1, torch.nn.ReLU(), )
            #l2, torch.nn.ReLU(),
            #l3, torch.nn.ReLU() )
        return model


    def embedObject(self, child_states, path):
        canon_path = self._canonical(path) if not self.tie_containers else "___default___"

        c_ts = []
        for c in child_states:
            c_t = self.embedder["___default___"](c)
            c_ts.append(c_t)

        if not c_ts:
            return None

        return self.out[canon_path](torch.max(torch.cat(c_ts),
            dim=0, keepdim=True)[0])



    def embedArray(self, child_states, path):
        canon_path = self._canonical(path) if not self.tie_containers else "___default___"

        c_ts = []
        for c in child_states:
            c_t = self.embedder[canon_path](c)
            c_ts.append(c_t)

        if not c_ts:
            return None

        return self.out[canon_path](torch.max(torch.cat(c_ts),
            dim=0, keepdim=True)[0])



    def _newStringRNN(self, key):
        lstm = nn.LSTM(self.mem_dim, self.mem_dim, 1)
        self.add_module(str(key)+"_StringLSTM", lstm)
        return lstm

    def embedString(self, string, path):
        canon_path = self._canonical(path) if not self.tie_primitives else "___default___"
        if string == "":
            return self._initC()

        tensor_input = torch.zeros(len(string)).long()
        for c in range(len(string)):
            try:
                tensor_input[c] = all_characters.index(string[c])
            except:
                continue

        encoded_input = self.string_encoder(tensor_input.view(1, -1)).view(-1, 1, self.mem_dim)
        output, hidden = self.string_rnn[canon_path](encoded_input)
        return hidden[0].mean(dim=1)


    def _newNumber(self, key):
        layer = nn.Linear(1, self.mem_dim)
        self.add_module(str(key), layer)
        return layer

    def embedNumber(self, num, path):
        if self.homogenous_types:
            return self.embedString(str(num), path)

        canon_path = self._canonical(path) if not self.tie_primitives else "___default___"

        if len(self.numberStats[canon_path]) < 100:
            self.numberStats[canon_path].append(num)

        num -= np.mean(self.numberStats[canon_path])

        if len(self.numberStats[canon_path]) > 3:
            std = np.std(self.numberStats[canon_path])
            if not np.allclose(std, 0.0):
                num /= std

        encoded = self.numberEmbeddings[canon_path](torch.Tensor([[num]]))
        return encoded



class FlatJSONNN(JSONNN):
    def __init__(self, mem_dim=128):
        super(FlatJSONNN, self).__init__(mem_dim)

        self.string_encoder = nn.Embedding(n_characters, self.mem_dim)
        self.string_rnn = keydefaultdict(self._newStringRNN)

        self.numberEmbeddings = keydefaultdict(self._newNumber)
        self.numberStats = defaultdict(lambda: [])

    def forward(self, node):
        return self.embedNode(node, path=["___root___"])

    def _canonical(self, path):
        # Restrict to last two elements
        path = path[-2:]
        
        # Replace ints with placeholder
        for i in range(len(path)):
            if isinstance(path[i], numbers.Number):
                path[i] = "___list___"
        return tuple(path)


    def embedObject(self, child_states, path):
        return torch.sum(torch.cat(child_states), dim=0, keepdim=True)


    def embedArray(self, child_states, path):
        return torch.sum(torch.cat(child_states), dim=0, keepdim=True)


    def _newStringRNN(self, key):
        lstm = nn.LSTM(self.mem_dim, self.mem_dim, 1)
        self.add_module(str(key)+"_StringLSTM", lstm)
        return lstm

    def embedString(self, string, path):
        if string == "":
            return torch.empty(1, self.mem_dim).fill_(0.).requires_grad_()

        tensor_input = torch.zeros(len(string)).long()
        for c in range(len(string)):
            try:
                tensor_input[c] = all_characters.index(string[c])
            except:
                continue
        encoded_input = self.string_encoder(tensor_input.view(1, -1)).view(-1, 1, self.mem_dim)
        output, hidden = self.string_rnn[self._canonical(path)](encoded_input)
        return hidden[0][:,-1,:]


    def _newNumber(self, key):
        layer = nn.Linear(1, self.mem_dim)
        self.add_module(str(key), layer)
        return layer

    def embedNumber(self, num, path):
        if len(self.numberStats[self._canonical(path)]) < 100:
            self.numberStats[self._canonical(path)].append(num)

        num -= np.mean(self.numberStats[self._canonical(path)])

        if len(self.numberStats[self._canonical(path)]) > 3:
            std = np.std(self.numberStats[self._canonical(path)])
            if not np.allclose(std, 0.0):
                num /= std

        encoded = self.numberEmbeddings[self._canonical(path)](torch.Tensor([[num]]))
        return encoded


        
if __name__ == "__main__":

    some_json = """
    {"menu": {
        "header": "SVG Viewer",
        "id": 23.5,
        "items": [
            {"id": "Open"},
            {"id": "OpenNew", "label": "Open New"},
            null,
            {"id": "ZoomIn", "label": "Zoom In"},
            {"id": "ZoomOut", "label": "Zoom Out"},
            {"id": "OriginalView", "label": "Original View"},
            null,
            {"id": "Quality"},
            {"id": "Pause"},
            {"id": "Mute"},
            null,
            {"id": "Find", "label": "Find..."},
            {"id": "FindAgain", "label": "Find Again"},
            {"id": "Copy"},
            {"id": "CopyAgain", "label": "Copy Again"},
            {"id": "CopySVG", "label": "Copy SVG"},
            {"id": "ViewSVG", "label": "View SVG"},
            {"id": "ViewSource", "label": "View Source"},
            {"id": "SaveAs", "label": "Save As"},
            null,
            {"id": "Help"},
            {"id": "About", "label": "About Adobe CVG Viewer..."}
        ]
    }}
    """

    #json_decoded = json.loads(some_json)
    #print(" \n\n\n ")
    #print(json_decoded)
    #print("\n\n\nStarting Json decoding... ")

    #model = JSONTreeLSTM()
    #x = model(json_decoded, "some json")

    #print("Representation:")
    #print(x.forward())


    import tqdm

    hidden_size = 128
    embedder = JSONTreeLSTM(hidden_size)
    output_layer = torch.nn.Linear(hidden_size*2, 2)
    model = torch.nn.Sequential(embedder, torch.nn.ReLU(), output_layer)
    criterion = torch.nn.CrossEntropyLoss()

    batch_size = 1
    learning_rate = 0.001

    #need to run a forward pass to initialise parameters for optimizer :(
    with open("../../Datasets/dota_matches/match_{0:06d}.json".format(0)) as f:
            json_decoded = json.load(f)
    model(json_decoded)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    running_loss = 0.0
    running_acc = 0.0


    for i in tqdm.trange(50000):

        if i % batch_size == 0:
            optimizer.zero_grad()

        with open("../../Datasets/dota_matches/match_{0:06d}.json".format(i)) as f:
            json_decoded = json.load(f)

        labs = torch.LongTensor([json_decoded['radiant_win']], )
        json_decoded['radiant_win'] = None

        #json_decoded['teamfights'] = None
        #json_decoded['players'] = None
        json_decoded['objectives'] = None
        json_decoded['radiant_gold_adv'] = None
        json_decoded['radiant_xp_adv'] = None
        json_decoded['chat'] = None
        json_decoded['tower_status_radiant'] = None
        json_decoded['tower_status_dire'] = None
        json_decoded['barracks_status_radiant'] = None
        json_decoded['barracks_status_dire'] = None
        inputs = json_decoded['teamfights']
        #print(json_decoded)
        
        # forward + backward + optimize
        outputs = model(inputs).view(1, -1)
        loss = criterion(outputs, labs) / batch_size
        loss.backward()

        if (i+1) % batch_size == 0:
            optimizer.step()

        # print statistics
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labs)
        running_acc += acc.item()

        running_loss += loss.item() * batch_size

        if (i+1) % 100 == 0:    # print every 1000 mini-batches
            tqdm.tqdm.write('[%d, %5d] loss: %.3f, acc: %.3f' %
                  (1, i+1, running_loss / 100, running_acc ))
            running_loss = 0.0
            running_acc = 0.0

