import json
import math
import numpy as np

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

import numbers
import string
from collections import defaultdict


all_characters = string.printable
n_characters = len(all_characters)


class CUDAableModule(nn.Module):
    def __init__(self, mem_dim=128, overrides={}):
        super(CUDAableModule, self).__init__()
        self.use_cuda = False
        self.device = None

    def cuda(self, device=None):
        return self._cuda(device)

    def _cuda(self, device=None):
        self.use_cuda = True
        self.device = device
        for c in self.children():
            c.cuda(device)
        return self

    def cpu(self):
        return self._cpu(device)

    def _cpu(self):
        self.use_cuda = False
        self.device = None
        for c in self.children():
            c.cpu(device)
        return self


# Base module
class JSONnnBase(CUDAableModule):
    def __init__(self, base_funcs={}, batch=False, cache_path_maps=False):
        super(JSONnnBase, self).__init__()
        self.pathwise_funcs = base_funcs
        self.compiled_path_regexprs = False # if regexprs have been compiled
        self._path_dict = {}
        self.use_batch = batch

        self.cache_path_maps = cache_path_maps

    def forward(self, data):

        if self.use_batch:
            data = [ self.embedJSON(x) for x in data ]
            return torch.cat( data, dim=0 )
        else:
            return self.embedJSON(data) 
            
    def embedJSON(self, somejson):
        if isinstance(somejson, str):
            try:
                somejson = json.loads(somejson)
            except Exception as e:
                raise Exception(somejson)
        
        result = self.embedNode(somejson, path=JSONPath(["___root___"]))
        if result is None: return self._empty()
        return result

    def embedNode(self, node, path):

        # Apply function based on type
        if node is None:
            return None
        elif isinstance(node, dict): # DICT
            nodetype = 'object'
        elif isinstance(node, list): # LIST
            nodetype = 'array'
        elif isinstance(node, str): # STRING
            nodetype = 'string'
        elif isinstance(node, numbers.Number): # NUMBER
            nodetype = 'number'  
        else:
            # not impl error
            raise Exception("Unknown node type! ", node, "at", path.path)
            return None

        #Check if override exists
        if not self.compiled_path_regexprs:
            self._compileRegExprs()
        fn = self._path_dict.get(str(path.path))
        if fn is not None and self.cache_path_maps:
            return getattr(self, fn)(node, path)
        else:
            overrides = list(self.pathwise_funcs.items())
            for match_string, (types, pth_match, fn) in reversed(overrides):
                if types == '*' or nodetype in types:
                    if pth_match.fullmatch(path._hash()) is not None:
                        #print("Matched", match_string)
                        #print("To", path._hash())
                        self._path_dict[str(path.path)] = fn
                        return getattr(self, fn)(node, path)

        # not impl error
        raise Exception("No mapping found for node path " + nodetype + str(path.path))
        return None


    def pathwise(self, name, module, *args, **kwargs):
        def _newmodule(key):
            layer = module(*args, **kwargs)
            if self.use_cuda:
                layer.cuda(self.device)
            self.add_module("path:" + key + "->" + name, layer)
            return layer
        return keydefaultdict(_newmodule)


    def embedChildren(self, node, path):
        child_states = []
        if isinstance(node, dict): # DICT
            for child_name, child in node.items():
                child_path = path.extend(child_name)
                child_state = self.embedNode(child, child_path)
                if child_state is None:
                    #print(name + "." + childname + " skipped...?")
                    #print(child)
                    continue # Skip any
                child_states.append(child_state)

        elif isinstance(node, list): # LIST
            for i, child in enumerate(node):
                child_path = path.extend(i)
                child_state = self.embedNode(child, child_path)
                if child_state is None:
                    #print(name + "." + childname + " skipped...?")
                    #print(child)
                    continue # Skip any
                child_states.append(child_state)

        if not child_states:
            return None

        #return child_states

        states = []
        for state in child_states:
            if isinstance(state, list):
                states.extend(state)
            else:
                states.append(state)

        return states


    def _compilePathRegExpr(self, match_str):
        separator = "\\."
        nonsep = '[^.]+'
        sequence =  '({0}{1})*{0}'.format(separator,nonsep)
        start = '^'
        end = '$'

        exact_exprs = match_str.split('..')
        def buildExactExpr(match_str):
            if not match_str: return ''
            match_str = match_str.replace('.', separator).replace('*', nonsep)
            return match_str

        seq = sequence.join([buildExactExpr(expr) for expr in exact_exprs])
        if not exact_exprs[-1]:
            seq = seq[:-2] #Remove trailing '\.'
        return re.compile(start + seq + end)

    def _compileRegExprs(self):
        for match_str, v in self.pathwise_funcs.items():
            splitstr = match_str.split('~')
            types = splitstr[0].split(',') if len(splitstr) == 2 else '*'
            path_match_str = splitstr[-1]
            regex = self._compilePathRegExpr(path_match_str)
            self.pathwise_funcs[match_str] = (types, regex, v)
        self.compiled_path_regexprs = True




class JSONTreeLSTM(JSONnnBase):
    def __init__(self, hidden_dim=128, overrides = {}, **kwargs):
        super(JSONTreeLSTM, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim

        self.pathwise_funcs = {
            'object~..': 'sumTreeLSTM',
            'array~..': 'LSTM',
            'string~..': 'stringRNN',
            'number~..': 'embedNumber',
        }
        self.pathwise_funcs.update(overrides)


        if 'sumTreeLSTM' in self.pathwise_funcs.values():
            self.TreeLSTM = SumTreeLSTM(hidden_dim//2)
            #self.TreeLSTM_hid_out = self.pathwise("obj_out",
            #                            nn.Linear, hidden_dim//2, hidden_dim//2)
            self.TreeLSTM_hid_out = moduledict(nn.Linear, hidden_dim//2, hidden_dim//2)
        if 'LSTM' in self.pathwise_funcs.values():
            self.LSTMs = moduledict(nn.LSTMCell, input_size = hidden_dim ,
                hidden_size = hidden_dim //2)
        if 'deepSets' in self.pathwise_funcs.values():
            self.DeepSets = DeepSetEncoder(hidden_dim)
            #self.DeepSetsOut = self.pathwise("arr_out", nn.Linear, hidden_dim, hidden_dim)
            self.DeepSetsOut = moduledict(nn.Linear, hidden_dim, hidden_dim)
        if 'deepSets2' in self.pathwise_funcs.values():
            self.DeepSets2 = DeepSetEncoder2(hidden_dim)
        if 'stringRNN' in self.pathwise_funcs.values():
            self.string_rnn = moduledict(StringRNN, hidden_dim // 2)
        if 'catEmbedding' in self.pathwise_funcs.values():
            #self.category_embeddings = keydefaultdict(
            #    lambda k: torch.rand((1,hidden_dim), requires_grad=True) )
            self.category_embeddings = infiniteEmbedding(hidden_dim)
        if 'embedNumber' in self.pathwise_funcs.values():
            self.numberTensor = torch.Tensor([[0.0]])
            #self.numberEmbeddings = keydefaultdict(
            #    lambda k: torch.rand((1,hidden_dim), requires_grad=True) )
            self.numberEmbeddings = infiniteEmbedding(hidden_dim)
            self.numberLin = nn.Linear(1, hidden_dim)
            self.numberStats = defaultdict(lambda: [])
            self.numberSum = defaultdict(lambda: 0.0)
            self.numberSumSq = defaultdict(lambda: 0.0)
            self.numberCount = defaultdict(lambda: 0.0)
            self.alpha = 0.8
            self.alpha_recip = 1 / (1-self.alpha)

        self.string_encoder = nn.Embedding(n_characters, self.hidden_dim // 2)
        #self.string_rnn_ = keydefaultdict(self._newStringRNN)
        self.string_rnn_ = moduledict(nn.LSTM, self.hidden_dim //2, self.hidden_dim // 2, 1)


    def _empty(self, dim=None):
        dim = self.hidden_dim if dim is None else dim
        empty = torch.empty(1, dim).fill_(0.).requires_grad_()
        if self.use_cuda:
            empty = empty.cuda(self.device)
        return empty


    def sumTreeLSTM(self, node, path):
        child_states = self.embedChildren(node, path)
        if not child_states: return None
        canon_path = hashPath(path)
        ch = self.TreeLSTM(child_states)
        c, h = torch.split(ch, ch.size(1) // 2, dim=1)
        return torch.cat([c, self.TreeLSTM_hid_out[canon_path](h)], dim=1)

    def LSTM(self, node, path):
        child_states = self.embedChildren(node, path)
        if not child_states: return None
        canon_path = hashPath(path)

        lstm = self.LSTMs[canon_path]
        c = self._empty(self.hidden_dim//2) ; h = self._empty(self.hidden_dim//2)
        for child_ch in child_states:
            #child_ch = torch.cat([child_c, child_h], dim=1)
            c, h = lstm(child_ch, (c, h))
        return torch.cat([c,h], dim=1)


    def deepSets(self, node, path):
        child_states = self.embedChildren(node, path)
        if not child_states: return None
        canon_path = hashPath(path)
        h = self.DeepSets(child_states)
        return self.DeepSetsOut[canon_path](h)

    def deepSets2(self, node, path):
        child_states = self.embedChildren(node, path)
        if not child_states: return None
        return self.DeepSets2(child_states)

    def catEmbedding(self, node, path):
        emb = self.category_embeddings[str(node).replace('.','#')]
        if self.use_cuda: emb = emb.cuda(self.device)
        return emb

    def stringRNN(self, node, path):
        canon_path = hashPath(path)
        return self.string_rnn[canon_path](str(node))


    def embedNumber(self, node, path):
        #return self.string_rnn(str(node))
        canon_path = hashPath(path)

        #return self.numberEmbeddings[canon_path] * node# / 10

        #self.numberSum[canon_path] = node + self.alpha*self.numberSum[canon_path]
        #self.numberSumSq[canon_path] = node**2 + self.alpha*self.numberSumSq[canon_path]
        if self.numberCount[canon_path] < 100:
            self.numberSum[canon_path] = node + self.numberSum[canon_path]
            self.numberSumSq[canon_path] = node**2 + self.numberSumSq[canon_path]
            self.numberCount[canon_path] = 1 + self.numberCount[canon_path]

        #mn = self.numberSum[canon_path] * self.alpha_recip
        mn = self.numberSum[canon_path] / self.numberCount[canon_path]

        node = node - mn

        #var =  self.numberSumSq[canon_path] * self.alpha_recip - mn **2
        var = self.numberSumSq[canon_path] / self.numberCount[canon_path] - mn **2

        if var > .0001:
            node = node / math.sqrt(var)

        return self.numberEmbeddings[canon_path] * node


        #node = torch.Tensor([[node]])
        #if self.use_cuda:
        #    node = node.cuda(self.device)
        #
        #encoded = self.numberEmbeddings[canon_path](node)
        #return encoded


    def flatten(self, node, path):
        return self.embedChildren(node, path)

    def sum(self, node, path):
        child_states = self.embedChildren(node, path)
        if not child_states: return None
        state = child_states[0]
        for s in child_states[1:]:
            state = state + s
        return torch.sum(torch.cat(child_states), dim=0, keepdim=True)


    def skipJunction(self, node, path):
        child_states = self.embedChildren(node, path)
        if not child_states: return None
        state = child_states[0]
        for s in child_states[1:]:
            state = state + s
        return torch.sum(torch.cat(child_states), dim=0, keepdim=True)

    def ignore(self, node, path):
        return None


    def _newStringRNN(self, key):
        lstm = nn.LSTM(self.hidden_dim //2, self.hidden_dim // 2, 1)
        self.add_module(str(key)+"_StringLSTM", lstm)
        return lstm

    def embedString(self, string, path):
        canon_path = hashPath(path)
        string = str(string)
        if string == "":
            return self._empty(), self._empty()

        tensor_input = torch.zeros(len(string)).long()
        for c in range(len(string)):
            try:
                tensor_input[c] = all_characters.index(string[c])
            except:
                continue
        encoded_input = self.string_encoder(tensor_input.view(1, -1)).view(-1, 1, self.hidden_dim // 2)
        output, hidden = self.string_rnn_[canon_path](encoded_input)
        return torch.cat([self._empty(self.hidden_dim // 2), hidden[0].mean(dim=1)], dim=1)



class JSONPath(object):
    debug = False
    def __init__(self, path=[]):
        self.path = path

    def extend(self, name):
        return JSONPath(self.path + [name])

    def _hash(self):
        path_str = ''
        for name in self.path:
            if isinstance(name, str):
                path_str += '.' + name
            else:
                path_str += '.' + str(name) #'['+ str(name) + ']'
        return path_str

    def match(self, match_str):
        exact_exprs = match_str.split('..')
        if self.debug:
            print("Found sub-expressions: ", ", ".join(exact_exprs))
        
        if len(exact_exprs) == 1:
            return self._match_exact(exact_exprs[0], strict=True)[0]

        offset = 0
        if self.debug:
            print("Matching sub-expression ", exact_exprs[0], " to path ", self.path)
        match, offset = self._match_exact(exact_exprs[0])
        if not match:
            if self.debug:
                print("Could not find a match!")
            return False

        for expr in exact_exprs[1:-1]:
            if self.debug:
                print("Matching sub-expression ", expr, " to path ", self.path[offset:])
            match, offset = self._find_match(expr, offset)
            if not match:
                if self.debug:
                    print("Could not find a match!")
                return False

        if self.debug:
            print("Matching sub-expression ", exact_exprs[-1], " to path ", self.path[offset:])
        match, _ = self._match_exact(exact_exprs[-1], offset, from_rhs=True)
        return match


    def _match_exact(self, match_str, offset=0, from_rhs=False, strict=False):
        path = self.path[offset:]
        if not match_str:
            return True, 0

        # Split by path
        exprs = [ expr for expr in match_str.split(".") if expr ]

        # Get bracketed
        names = []
        for expr in exprs:
            names.extend([name.strip(']') for name in expr.split('[') ])

        if strict:
            if not len(path) == len(names):
                return False, 0
        if from_rhs:
            path = path[-len(names):]

        for name, pathval in zip(names, path):
            if name == pathval:
                continue
            elif name == '*':
                continue
            else:
                return False, 0
        return True, len(names)

    def _find_match(self, match_str, offset=0):
        for i in range(offset, len(self.path)):
            if self.debug:
                sub_path = self.path[i:]
                print("-->Trying to match", match_str, "to", sub_path)
            match, num_elems = self._match_exact(match_str, i)
            if match:
                if self.debug:
                    print("Found match at offset", i)
                return True, i+num_elems
        if self.debug:
            print("Could not find a match!")
        return False, 0


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError( key )
        else:
            ret = self[key] = self.default_factory(key)
            return ret




def hashPath(path):
    return str([ e for e in path.path if isinstance(e, str)])

class SumTreeLSTM(nn.Module):
    def __init__(self, hidden_dim=128):
        super(SumTreeLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # from h to i, o, and u
        self.h_to_f = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.add_module("h_to_f", self.h_to_f)

        # from h to i, o, and u
        self.h_to_iou = nn.Linear(self.hidden_dim, self.hidden_dim * 3)
        self.add_module("h_to_iou", self.h_to_iou)

    def forward(self, states):
        hs = [] ; fcs = []
        for input_ch in states:
            input_c, input_h = torch.split(input_ch, input_ch.size(1) // 2, dim=1)
            hs.append(input_h)
            f = self.h_to_f(input_h)
            fc = torch.mul(torch.sigmoid(f), input_c)
            fcs.append(fc)

        if not hs:
            return None

        hs_bar = torch.sum(torch.cat(hs), dim=0, keepdim=True)
        iou = self.h_to_iou(hs_bar)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        c = torch.mul(i, u) + torch.sum(torch.cat(fcs), dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))
        return torch.cat([c, h], dim=1)


class DeepSetEncoder(nn.Module):
    def __init__(self, hidden_dim=128, num_embedding_layers=2):
        super(DeepSetEncoder, self).__init__()
        self.hidden_dim = hidden_dim

        layers = []
        for i in range(num_embedding_layers):
            layer = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.add_module("embedding_layer_"+str(i), layer)
            layers.append(layer)
            layers.append(torch.nn.ReLU())
        self.embedder = torch.nn.Sequential(*layers)

    def forward(self, states):
        if isinstance(states, list):
            states = torch.stack(states, dim=1)
        embedded_states = self.embedder(states)
        pooled = embedded_states.max(dim=1)[0]
        return pooled


class DeepSetEncoder2(nn.Module):
    def __init__(self, hidden_dim=128, num_embedding_layers=4, num_output_layers=3):
        super(DeepSetEncoder2, self).__init__()
        self.hidden_dim = hidden_dim

        layers = []
        for i in range(num_embedding_layers):
            layer = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.add_module("embedding_layer_"+str(i), layer)
            layers.append(layer)
        self.embedding_layers = layers

        layers = []
        for i in range(num_output_layers):
            layer = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.add_module("output_layer_"+str(i), layer)
            layers.append(layer)
        self.output_layers = layers

    def forward(self, states):
        #print(len(states))
        if isinstance(states, list):
            states = torch.stack(states, dim=1)
        
        for layer in self.embedding_layers:
            states = layer(states)
            states = states.max(dim=1,keepdim=True)[0] - states
            #states = states - states.mean(dim=1,keepdim=True)
            states = F.relu(states)

        pooled = states.max(dim=1)[0]
        #pooled = states.mean(dim=1)

        for layer in self.output_layers:
            pooled = layer(pooled)
            pooled = F.relu(pooled)

        return pooled


class StringRNN(CUDAableModule):
    def __init__(self, hidden_dim=128):
        super(StringRNN, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, 1)
        self.char_encoder = nn.Embedding(n_characters, self.hidden_dim)

    def _empty(self):
        empty = torch.empty(1, self.hidden_dim).fill_(0.).requires_grad_()
        if self.use_cuda:
            emt = emt.cuda(self.device)
        return empty

    def forward(self, string):
        if string == "":
            return self._empty(), self._empty()

        tensor_input = torch.zeros(len(string)).long()
        for c in range(len(string)):
            try:
                tensor_input[c] = all_characters.index(string[c])
            except:
                continue
        if self.use_cuda:
            tensor_input = tensor_input.cuda(self.device)
        encoded_input = self.char_encoder(tensor_input.view(1, -1))
        encoded_input = encoded_input.view(-1, 1, self.hidden_dim)
        output, hidden = self.lstm(encoded_input)
        out = torch.cat([hidden[1].mean(dim=1), hidden[0].mean(dim=1)], dim=1)
        return out


class moduledict(CUDAableModule):
    def __init__(self, module, *args, **kwargs):
        super(moduledict, self).__init__()
        self._path_dict = {}
        self._module = module
        self._args = args
        self._kwargs = kwargs

    def forward(self, x, path):
        module = self[path]
        return module.forward(x)

    def _newmodule(self, key):
        module = self._module(*self._args, **self._kwargs)
        if self.use_cuda:
            module.cuda(self.device)
        self.add_module("path:" + key + "->" + key, module)
        self._path_dict[key] = module
        return module

    def __getitem__(self, key):
        result = self._path_dict.get(key)
        if result is None:
            result = self._newmodule(key)
        return result


class infiniteEmbedding(CUDAableModule):
    def __init__(self, hidden_dim):
        super(infiniteEmbedding, self).__init__()
        self._emb_dict = {}
        self.hidden_dim = hidden_dim

    def forward(self, key):
        return self[key]

    def _newembedding(self, key):
        emb = torch.rand((1,self.hidden_dim), requires_grad=True)
        if self.use_cuda:
            emb = emb.cuda(self.device)
        emb = nn.Parameter(emb)

        self._emb_dict[key] = emb
        self.register_parameter(key, emb)

        return emb

    def __getitem__(self, key):
        result = self._emb_dict.get(key)
        if result is None:
            result = self._newembedding(key)
        return result


        
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


    model2 = JSONTreeLSTM(128)
    model2(some_json)
    #for name, param in model2.named_parameters():
    #    if param.requires_grad:
    #        print(name)

    print("Model's state_dict:")
    for param_tensor in model2.state_dict():
        print(param_tensor, "\t", model2.state_dict()[param_tensor].size())
    #torch.save(model2, 'tmp/model.pty')

    #model3 = torch.load('tmp/model.pty')
    #model3 = JSONTreeLSTM(128)
    #model3.load_state_dict(torch.load('tmp/model.pty'))
    #print("Model's state_dict:")
    #for param_tensor in model3.state_dict():
    #    print(param_tensor, "\t", model3.state_dict()[param_tensor].size())


    import re

    path = JSONPath(['a', 'b', 'c', 'd', 'e'])
    #path.debug = True
    match_str = '..a.b.c..'
    #print(path.match(match_str))
    debug=True
    def compileRegExpr(match_str):
        separator = "\\."
        nonsep = '[^.]+'
        sequence =  '({0}{1})*{0}'.format(separator,nonsep)
        start = '^'
        end = '$'

        exact_exprs = match_str.split('..')

        if debug:
            print("Found sub-expressions: ", ", ".join(exact_exprs))

        def buildExactExpr(match_str):
            if not match_str:
                return ''#sequence
            match_str = match_str.replace('.', separator).replace('*', nonsep)
            return match_str

        seq = sequence.join([buildExactExpr(expr) for expr in exact_exprs])
        if not exact_exprs[-1]:
            #Remove trailing '\.'
            seq = seq[:-2]
        regex = start + seq + end
        print(regex)
        return re.compile(regex)

    regex = compileRegExpr(match_str)
    print(path._hash())
    print(regex.fullmatch(path._hash()))
            


    import tqdm

    hidden_size = 128
    embedder = JSONTreeLSTM(hidden_size)
    output_layer = torch.nn.Linear(hidden_size, 2)
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

