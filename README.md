# json2vec

json2vec is a neural network 'architecture' for JSON data. It is based on the STRLA algorithm presented in this preprint [https://arxiv.org/abs/2002.05707].

Essentially, json2vec dynamically constructs a neural network for a given piece of input data by recursively applying a set of common neural network components. Each of these components have trainable parameters and json2vec can be trained as a conventional nerual network (so, despite the name, json2vec is not an unsupervised algorithm). For further details, see the link above.

The core code is given in two forms:
- `json2vec.py`, which contains two example architectures with fixed functions (with path-sepcific weights)
- `json2vec_v2.py`, which incorporates a function-override capability, as well as a number of helpful utilities (such as wrappers for path-specific function weights). In future the plan is to have a single `json2vec` class  where all components are user-specified.

The current iteration of this code is highly unoptimised, and hence is very slow for complex data structures. However, initial results on a variety of datasets (including a number beyond those covered the above paper) suggest that good results can be achieved on a variety of dataset with only CPU, hence don't be afraid to try the algorithm out.

For applying deep learning to JSON, the following repo may also be useful [https://github.com/pevnak/JsonGrinder.jl].

Comments and contributions appreciated.



## Installation

To install, first clone the repository to your local machine via
```
git clone https://github.com/EndingCredits/json2vec.git
cd json2vec
```

Install the required python librareies via pip 

```
pip install -r requirements.txt
```
or via your preferred python package manager.

If you don't already have it installed, install pytorch according to the instructions at [https://pytorch.org/].

For the cpu version on linux, the CPU-only version of pytorch can be installed via the following pip command
```
pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

You can test the installation by running
```
python uci_test.py
```

## Using json2vec

json2vec is very simple to apply, and can be treated like a standard pytorch module.

```
from json2vec import JSONTreeLSTM
import torch

embedder = JSONTreeLSTM(mem_dim=64)
output_layer = torch.nn.Linear(128, num_classes)
model = torch.nn.Sequential(embedder, output_layer)

some_json = """
{
    "glossary": {
        "title": "example glossary",
		"GlossDiv": {
            "title": "S",
			"GlossList": {
                "GlossEntry": {
                    "ID": "SGML",
					"SortAs": "SGML",
					"GlossTerm": "Standard Generalized Markup Language",
					"Acronym": "SGML",
					"Abbrev": "ISO 8879:1986",
					"GlossDef": {
                        "para": "A meta-markup language, used to create markup languages such as DocBook.",
						"GlossSeeAlso": ["GML", "XML"]
                    },
					"GlossSee": "markup"
                }
            }
        }
    }
}
"""

print(model(some_json))
```

HOne thing to be aware of is that json2vec dynamically instantiates parameters when new data paths are encountered. Hence, care should be taken to ensure all parameters are added to the in-built pytorch optimisers. In practice, one can usually get around this by constructing the optimiser after the initial test of the model, as all relevant data will have been seen by the network.


To tailor the architecture to a given dataset, `json2vec_v2.py` can be used. It allows setting of path-specific function overrides, so you can set different functions for different parts of the data. Paths are given in JSONPath style format, e.g.

```
from json2vec_v2 import JSONTreeLSTM

self.tree = JSONTreeLSTM(128, overrides={
        'array~..': 'LSTM',
        'object~..': 'sum',
        'string,number~..': 'catEmbedding',

        '.*': 'deepSets2',
        'number~..rect..': 'embedNumber',

        '..lastrect': 'ignore',
        '..physicstype': 'ignore',
        '..physics': 'ignore',
        '..img': 'ignore',
        '..color': 'ignore',
        '..image': 'ignore',
        '..scale_image': 'ignore',
        '..stypes': 'ignore',
    })
```




## Experiment code

Experiment code can be run using `hparam_tuning_uci.py` and `hparam_tuning_poker.py` for parameter tuning, or `experiments_uci.py` and `experiments_other.py` for actual experiments, uncommenting the relevant hyperparameters for the runs to be tested.

`uci_test.py` and `uci_test_v2.py` can be run with command line arguments for single runs.

The data in `uci_data` is the same data as downloaded from the UCI repository, however, some files have been reformatted to enable easy loading. All rebalancing, and conversion to both JSON and feature vector formats.

Experiment code for reinforcement learning code will be included at a later date.