import csv
import json
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_car_dataset(fix_schema=False):
    jsons = []
    vectors = []
    labels = []

    attributes = get_attributes('uci_data/car.data', verbose=False)
    label_col = len(attributes) - 1
    ignore_cols = [ label_col, ]

    with open('uci_data/car.data', 'r') as f:
        csvreader = csv.reader(f, delimiter=',', quotechar='|')
        for row in csvreader:

            vector = get_vector(row, attributes, ignore_cols)
            vectors.append(vector)
            
            try:
                row[2] = int(row[2])
            except:
                row[2] = "\""+row[2]+"\""
            
            try:
                row[3] = int(row[3])
            except:
                row[3] = "\""+row[3]+"\""
            
            json = """{{
  "PRICE": {{
    "buying": "{0}",
    "maint": "{1}"
  }},
  "TECH": {{
    "COMFORT": {{
      "doors": {2},
      "persons": {3},
      "lug_boot": "{4}"
    }},
    "safety": "{5}"
  }}
}}""".format(*row)
            
            jsons.append(json)
            labels.append(row[label_col])
    return jsons, vectors, labels




def load_nursery_dataset(fix_schema=False):
    jsons = []
    labels = []
    vectors = []

    attributes = get_attributes('uci_data/nursery.data', verbose=False)
    label_col = len(attributes) - 1
    ignore_cols = [ label_col, ]

    with open('uci_data/nursery.data', 'r') as f:
        csvreader = csv.reader(f, delimiter=',', quotechar='|')
        for row in csvreader:
            if not row: continue
                
            vector = get_vector(row, attributes, ignore_cols)
            vectors.append(vector)
            
            try:
                row[3] = int(row[3])
            except:
                row[3] = "\""+row[3]+"\""
                
            json = """{{
  "EMPLOY": {{
    "parents": "{0}",
    "has_nurs": "{1}"
  }},
  "STRUCT_FINAN": {{
    "STRUCTURE": {{
      "form": "{2}",
      "children": {3}
    }},
    "housing": "{4}",
    "finance": "{5}"
  }},
  "SOC_HEALTH": {{
    "social": "{6}",
    "health": "{7}"
  }}
}}""".format(*row)
            
            jsons.append(json)
            labels.append(row[label_col])
    return jsons, vectors, labels




def load_seismic_dataset(balance=True, fix_schema=False):
    jsons = []
    labels = []
    vectors = []

    attributes = get_attributes('uci_data/seismic-bumps.csv', verbose=False)
    label_col = len(attributes) - 1
    ignore_cols = [ label_col, ]


    with open('uci_data/seismic-bumps.csv', 'r') as f:
        csvreader = csv.reader(f, delimiter=',', quotechar='|')
        for row in csvreader:
            if not row: continue
                
            vector = get_vector(row, attributes, ignore_cols)
            vectors.append(vector)

            for colnum in range(len(row)):
                if colnum in [ 3,4,5,6,8,9,10,11,12,13,14,15,16,17]:
                    row[colnum] = int(row[colnum])
            
            json = """{{
    "work-shift": "{2}",
    "assessments": [
        {{
            "type": "seismic",
            "result": "{0}"
        }},
        {{
            "type": "acoustic",
            "result": "{1}"
        }},
        {{
            "type": "geophone",
            "readings": {{
                "total-energy": {3},
                "deviation-energy": {5},
                "number-pulses": {4},
                "deviation-pulses": {6}
            }},
            "result": "{7}"
        }}
    ],
    "readings": {{
        "total-energy": {16},
        "max-energy": {17},
        "bumps": [
            {{
                "range-start": 10e2,
                "range-end": 10e3,
                "total-bumps": {9}
            }},
            {{
                "range-start": 10e3,
                "range-end": 10e4,
                "total-bumps": {10}
            }},
            {{
                "range-start": 10e4,
                "range-end": 10e5,
                "total-bumps": {11}
            }},
            {{
                "range-start": 10e5,
                "range-end": 10e6,
                "total-bumps": {12}
            }},
            {{
                "range-start": 10e6,
                "range-end": 10e7,
                "total-bumps": {13}
            }},
            {{
                "range-start": 10e7,
                "range-end": 10e8,
                "total-bumps": {14}
            }},
            {{
                "range-start": 10e8,
                "range-end": 10e10,
                "total-bumps": {15}
            }}
        ],
        "total-bumps": {8}
    }}
}}""".format(*row)
            
            jsons.append(json)
            labels.append(row[label_col])

    if balance:
        jsons_ = []
        vectors_ = []
        labels_ = []
        np.random.seed(123)
        pos_inds = [ i for i, l in enumerate(labels) if l == "1"]
        neg_inds = [ i for i, l in enumerate(labels) if l == "0"]
        sample_inds = pos_inds + list(np.random.choice(neg_inds, size=len(pos_inds)*2))
        for i in sample_inds:
            jsons_.append(jsons[i])
            vectors_.append(vectors[i])
            labels_.append(labels[i])
        return jsons_, vectors_, labels_

    return jsons, vectors, labels




def load_poker_dataset(fix_schema=False):

    suits = [ "Hearts", "Spades", "Diamonds", "Clubs" ]
    ranks = [ "Ace", 2, 3, 4, 5, 6, 7, 8, 9, 10, "Jack", "Queen", "King" ]
    #ranks = [ "\"Ace\"", 2, 3, 4, 5, 6, 7, 8, 9, 10, "\"Jack\"", "\"Queen\"", "\"King\"" ]

    jsons = []
    labels = []
    vectors = []

    attributes = get_attributes('uci_data/poker-hand-training-true.data',
        verbose=False, force_discrete=True)
    label_col = len(attributes) - 1
    ignore_cols = [ label_col, ]

    with open('uci_data/poker-hand-training-true.data', 'r') as f:
        csvreader = csv.reader(f, delimiter=',', quotechar='|')
        for row in csvreader:
            if not row: continue
                
            vector = get_vector(row, attributes, ignore_cols)
            vectors.append(vector)
            
            card_jsons = []
            for i in range(5):
                suit = int(row[2*i])-1
                rank = int(row[2*i + 1])-1
                card_json = """    {{
        "suit": "{0}",
        "rank": "{1}"
    }}""".format(suits[suit], ranks[rank])
                card_jsons.append(card_json)
            json = "[\n{}\n]".format(',\n'.join(card_jsons))
            jsons.append(json)
            labels.append(row[label_col])
    return jsons, vectors, labels




def load_poker_dataset_train_test(fix_schema=False):

    suits = [ "Hearts", "Spades", "Diamonds", "Clubs" ]
    ranks = [ "Ace", 2, 3, 4, 5, 6, 7, 8, 9, 10, "Jack", "Queen", "King" ]

    jsons = []
    labels = []
    vectors = []

    attributes = get_attributes('uci_data/poker-hand-training-true.data',
        verbose=False, force_discrete=True)
    label_col = len(attributes) - 1
    ignore_cols = [ label_col, ]

    with open('uci_data/poker-hand-training-true.data', 'r') as f:
        csvreader = csv.reader(f, delimiter=',', quotechar='|')
        for row in csvreader:
            if not row: continue
                
            vector = get_vector(row, attributes, ignore_cols)
            vectors.append(vector)
            
            card_jsons = []
            for i in range(5):
                suit = int(row[2*i])-1
                rank = int(row[2*i + 1])-1
                card_json = """    {{
        "suit": "{0}",
        "rank": "{1}"
    }}""".format(suits[suit], ranks[rank])
                card_jsons.append(card_json)
            json = "[\n{}\n]".format(',\n'.join(card_jsons))
            jsons.append(json)
            labels.append(row[label_col])

    jsons_test = []
    labels_test = []
    vectors_test = []

    with open('uci_data/poker-hand-testing.data', 'r') as f:
        csvreader = csv.reader(f, delimiter=',', quotechar='|')
        for row in csvreader:
            if not row: continue
                
            vector = get_vector(row, attributes, ignore_cols)
            vectors_test.append(vector)
            
            card_jsons = []
            for i in range(5):
                suit = int(row[2*i])-1
                rank = int(row[2*i + 1])-1
                card_json = """    {{
        "suit": "{0}",
        "rank": "{1}"
    }}""".format(suits[suit], ranks[rank])
                card_jsons.append(card_json)
            json = "[\n{}\n]".format(',\n'.join(card_jsons))
            jsons_test.append(json)
            labels_test.append(row[label_col])

    return jsons, vectors, labels, jsons_test, vectors_test, labels_test




def load_mushroom_dataset(fix_schema=False):
    d = not fix_schema
    jsons = []
    labels = []
    vectors = []

    attributes = get_attributes('uci_data/agaricus-lepiota.data', verbose=False)
    label_col = 0
    ignore_cols = [ label_col, ]

    with open('uci_data/agaricus-lepiota.data', 'r') as f:
        csvreader = csv.reader(f, delimiter=',', quotechar='|')
        for row in csvreader:
            if not row: continue
                
            vector = get_vector(row, attributes, ignore_cols)
            vectors.append(vector)
            
            colors = {'n':'brown', 'b':'buff', 'c':'cinnamon', 'g':'gray', 'r':'green',
                      'p':'pink', 'u':'purple', 'e':'red', 'w':'white', 'y':'yellow',
                      'o':'orange', 'k':'black', 'h':'chocolate'}
            
            # cap
            row[1] = {'b':'bell','c':'conical','x':'convex','f':'flat','k':'knobbed',
                      's':'sunken'}[row[1]]
            row[2] = {'f':'fibrous','g':'grooves','y':'scaly','s':'smooth',}[row[2]]
            row[3] = colors[row[3]]
            
            # gill
            row[6] = {'a':'attached','d':'descending','f':'free','n':'notched'}[row[6]]
            row[7] = {'c':'close','w':'crowded','d':'distant'}[row[7]]
            row[8] = {'b':'broad','n':'narrow'}[row[8]]
            row[9] = colors[row[9]]

            # stalk
            row[10] = {'e':'enlarging','t':'tapering',}[row[10]]
            row[11] = '' if row[11] == '?' and d else "\n        \"root\": \"{}\",".format({
                'b':'bulbous','c':'club','u':'cup','e':'equal',
                'z':'rhizomorphs','r':'rooted', '?': '?'}[row[11]])
            row[12] = {'f':'fibrous','y':'scaly','k':'silky','s':'smooth',}[row[12]]
            row[13] = {'f':'fibrous','y':'scaly','k':'silky','s':'smooth',}[row[13]]
            row[14] = colors[row[14]]
            row[15] = colors[row[15]]
            
            # veil
            row[16] = {'p':'partial','u':'universal'}[row[16]]
            row[17] = colors[row[17]]
            
            # ring
            row[18] = '' if row[18] == 'n' and d else """
    "ring": {{
        "type": "{0}",
        "number": {1}
    }},""".format({'c':'cobwebby','e':'evanescent','f':'flaring','l':'large','p':'pendant',
                  's':'sheathing','z':'zone', 'n': 'none'}[row[19]],
                  {'n': 0, 'o': 1, 't': 2}[row[18]])
            
            # other
            row[4] = {'t':'true','f':'false'}[row[4]]
            row[5] = {'a':'almond','l':'anise','c':'creosote','y':'fishy','f':'foul',
                     'm':'musty','n':'none','p':'pungent','s':'spicy'}[row[5]]
            row[20] = colors[row[20]]
            row[21] = {'a':'abundant','c':'clustered','n':'numerous','s':'scattered',
                       'v':'several','y':'solitary'}[row[21]]
            row[22] = {'g':'grasses','l':'leaves','m':'meadows','p':'paths',
                       'u':'urban','w':'waste', 'd':'woods'}[row[22]]
            
            json = """{{
    "cap": {{
        "shape": "{1}",
        "surface": "{2}",
        "color": "{3}"
    }},
    "gill": {{
        "attachment": "{6}",
        "spacing": "{7}",
        "size": "{8}",
        "color": "{9}"
    }},
    "stalk": {{
        "shape": "{10}",{11}
        "surface": {{
            "above-ring": "{12}",
            "below-ring": "{13}"
        }},
        "color": {{
            "above-ring": "{14}",
            "below-ring": "{15}"
        }}
    }},
    "veil": {{
        "type": "{16}",
        "color": "{17}"
    }},{18}
    "bruising": {4},
    "odor": "{5}",
    "spore-print-color": "{20}",
    "population": "{21}",
    "habitat": "{22}"  
}}""".format(*row)
            jsons.append(json)
            labels.append(row[label_col])
    return jsons, vectors, labels




def load_contraceptive_dataset(fix_schema=False):
    jsons = []
    labels = []
    vectors = []

    attributes = get_attributes('uci_data/cmc.data', verbose=False, force_discrete=False)
    label_col = 9
    ignore_cols = [ label_col, ]

    with open('uci_data/cmc.data', 'r') as f:
        csvreader = csv.reader(f, delimiter=',', quotechar='|')
        for row in csvreader:
            if not row: continue
                
            vector = get_vector(row, attributes, ignore_cols)
            vectors.append(vector)

            row[4] = ['false', 'true'][int(row[4])]
            row[5] = ['false', 'true'][int(row[5])]
        
            json = """{{
    "wife": {{
        "age": {0},
        "education": {1},
        "religion-is-islam": {4},
        "now-working": {5}
    }},
    "husband": {{
        "education": {2},
        "occupation": {6}
    }},
    "children": {3},
    "standard-of-living": {7},
    "media-exposure": {8}
}}""".format(*row)
            jsons.append(json)
            labels.append(row[label_col])
    return jsons, vectors, labels



def load_automobile_dataset(fix_schema=False):
    d = not fix_schema
    jsons = []
    labels = []
    vectors = []

    attributes = get_attributes('uci_data/imports-85.data', verbose=False, force_discrete=False)
    label_col = 0
    ignore_cols = [ label_col, 1 ]


    with open('uci_data/imports-85.data', 'r') as f:
        csvreader = csv.reader(f, delimiter=',', quotechar='|')
        for row in csvreader:
            if not row: continue
                
            vector = get_vector(row, attributes, ignore_cols)
            vectors.append(vector)
            
            row[5] = '' if row[5] == '?' and d else ",\n        \"num-of-doors\": {}".format(
                {'four': 4, 'two':2, '?': '-1'}[row[5]])
            row[9] = str(float(row[9]))
            row[15] = {'eight': 8, 'five': 5, 'four': 4, 'six': 6,
                     'three': 3, 'twelve': 12, 'two': 2, '?': '-1'}[row[15]]
            row[18] = '' if row[18]=='?' else "\n            \"bore\": {},".format(row[18])
            row[19] = '' if row[19]=='?' else "\n            \"stroke\": {},".format(row[19])
            row[21] = '' if row[21]=='?' else ",\n            \"horsepower\": {}".format(row[21])
            row[22] = '' if row[22]=='?' else ",\n            \"peak-rpm\": {}".format(row[22])
            row[25] = '' if row[25]=='?' else "\n    \"price\": {},".format(row[25])
            
            json = """{{
    "make": "{2}",{25}
    "curb-weight": {13},
    "mpg": {{
        "city": {23},
        "highway": {24}
    }},
    "powertrain": {{
        "engine": {{
            "fuel-type": "{3}",
            "fuel-system": "{17}",
            "aspiration": "{4}",
            "engine-type": "{14}",
            "compression-ratio": {20},{18}{19}
            "num-of-cylinders": {15},
            "displacement": {16}{21}{22}
        }},
        "engine-location": "{8}",
        "drive-wheels": "{7}"
    }},
    "chassis": {{
        "dimensions": {{
            "length": {10},
            "width": {11},
            "height": {12}
        }},
        "wheel-base": {9},
        "body-style": "{6}"{5}
    }}
}}""".format(*row)
            
            jsons.append(json)
            labels.append(row[label_col])
    return jsons, vectors, labels




def load_bank_dataset(balance=True, fix_schema=False):
    d = not fix_schema
    jsons = []
    labels = []
    vectors = []

    attributes = get_attributes('uci_data/bank-additional-full.csv',
        separator=';', skip_header=True, verbose=False)
    label_col = 20
    ignore_cols = [ label_col, 10 ]


    with open('uci_data/bank-additional-full.csv', 'r') as f:
        csvreader = csv.reader(f, delimiter=';', quotechar='\"')
        next(csvreader) # skip header
        for row in csvreader:
            if not row: continue
                
            vector = get_vector(row, attributes, ignore_cols)
            vectors.append(vector)
            
            #replace row12 with row13 == 0
            row[12] = "-1" if row[12] == "999" else row[12]
            row[13] = '' if row[13] == "0" and d else """,
        "last-campaign": {{
            "number": {13},
            "days-since": {12},
            "outcome": "{14}"
        }}""".format(*row)
            
            json = """{{
    "age": {0},
    "job": "{1}",
    "marital-status": "{2}",
    "education": "{3}",
    "loan": {{
        "personal": "{6}",
        "mortgage": "{5}",
        "in-default": "{4}"
    }},
    "contact": {{
        "type": "{7}",
        "last-contact": {{
            "month": "{8}",
            "weekday": "{9}"
        }},
        "this-campaign": {{
            "number": {11}
        }}{13}
    }},
    "indicators": {{
        "emp-var-rate": {15},
        "cons-price-idx": {16},
        "cons-conf-idx": {17},
        "euribor3m": {18},
        "nr-employed": {19}
    }}
}}""".format(*row)
            jsons.append(json)
            labels.append(row[label_col])

    if balance:
        jsons_ = []
        vectors_ = []
        labels_ = []
        np.random.seed(123)
        pos_inds = [ i for i, l in enumerate(labels) if l == "yes"]
        neg_inds = [ i for i, l in enumerate(labels) if not l == "yes"]
        sample_inds = pos_inds + list(np.random.choice(neg_inds, size=len(pos_inds)*2))
        for i in sample_inds:
            jsons_.append(jsons[i])
            vectors_.append(vectors[i])
            labels_.append(labels[i])
        return jsons_, vectors_, labels_

    return jsons, vectors, labels




def load_student_dataset(fix_schema=False, regression=False):
    jsons = []
    vectors = []
    labels = []

    attributes = get_attributes('uci_data/student-por.csv', verbose=False, separator=';', skip_header=True)
    label_col = 32
    ignore_cols = [ label_col, 30, 31 ]

    with open('uci_data/student-por.csv', 'r') as f:
        csvreader = csv.reader(f, delimiter=';', quotechar='"')
        next(csvreader)
        for row in csvreader:
            vector = get_vector(row, attributes, ignore_cols)
            vectors.append(vector)
            
            row[0] = {"GP": "Gabriel Pereira", "MS": "Mousinho da Silveira"}[row[0]]
            row[1] = {"F": "Female", "M": "Male"}[row[1]]
            #row[2] age
            row[3] = {"U": "false", "R": "true"}[row[3]]
            row[4] = {"LE3": "=< 3", "GT3": "> 3"}[row[4]]
            
            row[5] = {"T": "false", "A": "true"}[row[5]]
            row[6] = {"0": "none",  "1": "primary education", "2": "5th to 9th grade",
                      "3": "secondary education", "4": "higher education"}[row[6]]
            row[7] = {"0": "none",  "1": "primary education", "2": "5th to 9th grade",
                      "3": "secondary education", "4": "higher education"}[row[7]]
            row[8] = {"teacher": "teacher", "health": "healthc-are", "services": "civil services",
                      "at_home": "stay-at-home", "other": "other"}[row[8]]
            row[9] = {"teacher": "teacher", "health": "healthc-are", "services": "civil services",
                      "at_home": "stay-at-home", "other": "other"}[row[9]]

            row[10] = {"home": "close to home", "reputation": "school reputation",
                       "course": "course preference", "other": "other"}[row[10]]
            #row[11] guardian
            row[12] = {"1": "<15min.", "2": "15 to 30 min.", "3": "30 min. to 1 hour",
                       "4": ">1 hour"}[row[12]]
            row[13] = {"1": "<2 hours", "2": "2 to 5 hours", "3": "5 to 10 hours",
                       "4": ">10 hours"}[row[13]]
            #row[14] failures
            
            row[15] = { "yes": "true", "no": "false"}[row[15]]
            row[16] = { "yes": "true", "no": "false"}[row[16]]
            row[17] = { "yes": "true", "no": "false"}[row[17]]
            row[18] = { "yes": "true", "no": "false"}[row[18]]
            row[19] = { "yes": "true", "no": "false"}[row[19]]

            row[20] = { "yes": "true", "no": "false"}[row[20]]
            row[21] = { "yes": "true", "no": "false"}[row[21]]  
            row[22] = { "yes": "true", "no": "false"}[row[22]]  
            #row[23] famrel 
            #row[24] freetime    
            
            #row[25] gout
            #row[26] Dalc
            #row[27] Walc
            #row[28] health
            #row[29] absences
            
            json = """{{
    "school": "{0}",
    "reason-for-chosing": "{10}",
    "sex": "{1}",
    "age": {2},
    "health": {28},
    
    "household": {{
        "rural": {3},
        "travel-time": "{12}",
        "internet": {21},
        "education-support": {16}, 
        "family": {{
            "size": "{4}",
            "relationship quality": {23},
            "parents": {{
                "separated": {5},
                "guardian": "{11}",
                "mother": {{ "education": "{6}", "job": "{8}" }},
                "father": {{ "education": "{7}", "job": "{9}" }}
            }} 
        }}
    }},
    "study": {{
        "hours-per-week": "{13}",
        "continue-to-higher": {20},
        "attended-nursery": {19},
        "extra-support": {15},
        "num-fails": {14},
        "tutored": {17},
        "absences": {29}
    }},
    "social": {{
        "free-time": {24},
        "socialising-external": {25},
        "alcohol-consumption": {{ "weekday": {26}, "weekend": {27} }},
        "extra-curricular": {18},
        "in-relationship": {22}
    }}
}}""".format(*row)
            jsons.append(json)

            score = int(row[label_col])
            if regression:
                labels.append(score/20) 
            else:
                labels.append({
                        0: "I", 1: "I", 2: "I", 3: "I", 4: "I",
                        5: "I", 6: "I", 7: "I", 8: "I", 9: "I",
                        10: "II", 11: "II",
                        12: "III", 13: "III",
                        14: "IV", 15: "IV",
                        16: "V", 17: "V", 18: "V", 19: "V", 20: "V"
                    }[score])

            #labels.append(str(int(row[label_col]) > 11))
    return jsons, vectors, labels




def load_census_dataset(balance=True, fix_schema=False):
    indent = '    '
    fields = [
        '"age": {}', # 0
        ',\n' + indent*2 + '"worker-class": "{}"', # 1
        ',\n' + indent*2 + '"industry-code": "{}"', # 2
        ',\n' + indent*2 + '"occupation-code": "{}"', # 3
        '"level": "{}"', # 4
        ',\n' + indent*2 + '"wage-per-hour": {}', # 5
        
        ',\n' + indent*2 + '"current": "{}"', # 6
        ',\n' + indent + '"maritial-status": "{}"', # 7
        ',\n' + indent*2 + '"industry": "{}"', # 8
        ',\n' + indent*2 + '"occupation": "{}"', # 9
        ',\n' + indent + '"race": "{}"', # 10
        
        
        ',\n' + indent + '"race-hispanic-origin": "{}"', #11
        ',\n' + indent + '"sex": "{}"', # 12
        ',\n' + indent*2 + '"labor-union-member": "{}"', # 13
        ',\n' + indent*2 + '"reason-for-unemployment": "{}"', # 14
        '"status": "{}"', # 15
        
        ',\n' + indent*2 + '"capital-gains": {}', # 16
        ',\n' + indent*2 + '"capital-losses": {}', # 17
        ',\n' + indent*2 + '"dividends": {}', # 18
        '"status": "{}"', # 19
        ',\n' + indent*2 + '"previous-residence-region": "{}"', # 20
        
        ',\n' + indent*2 + '"previous-residence-state": "{}"', # 21
        '"status": "{}"', # 22
        ',\n' + indent*2 + '"status-category": "{}"', # 23
        '', # 24
        ',\n' + indent*2 + '"msa-code-change": "{}"', # 25
        
        ',\n' + indent*2 + '"msa-code-reg": "{}"', # 26
        ',\n' + indent*2 + '"msa-move-reg": "{}"', # 27
        ',\n' + indent*2 + '"moved-in-last-year": "{}"', # 28
        ',\n' + indent*2 + '"previous-residence-in-sunbelt": "{}"', # 29
        ',\n' + indent*2 + '"workers-at-employer": {}', # 30
        
        ',\n' + indent*2 + '"parents-present": "{}"', # 31
        ',\n' + indent*2 + '"father-birthplace": "{}"', # 32
        ',\n' + indent*2 + '"mother-birthplace": "{}"', # 33
        ',\n' + indent*2 + '"birthplace": "{}"', # 34
        '"citizenship": "{}"', # 35
        
        '', # 36
        '', # 37
        ',\n' + indent*2 + '"veterans-benefits": {}', # 38
        ',\n' + indent*2 + '"weeks-worked": {}', # 39
        '{}', '{}', '{}', '{}'
    ]

    d = not fix_schema
    jsons = []
    labels = []
    vectors = []

    attributes = get_attributes('uci_data/census-income.data', verbose=False, force_discrete=False)
    label_col = 41
    ignore_cols = [ label_col, 24, 36, 37, 40 ]

    with open('uci_data/census-income.data', 'r') as f:
        csvreader = csv.reader(f, delimiter=',', quotechar='|')
        for row in csvreader:
            if not row: continue
                
            vector = get_vector(row, attributes, ignore_cols)
            vectors.append(vector)
            
            for i, col in enumerate(row):
                try:
                    row[i] = '' if ("Not in universe" in col or '?' in col) and d else fields[i].format(col.strip())
                except:
                    raise Exception(str(row))
            #print(row)
            json = """{{
        {0}{12}{10}{11}{7},
        "employment": {{
            {15}{14}{1}{8}{2}{9}{3}{13}{39}{30}{5}
        }},
        "education": {{
            {4}{6}
        }},
        "residence": {{
            {35}{34}{32}{33}{20}{21}{29}{25}{26}{27}{28}
        }},
        "household": {{
            {22}{23}{31}
        }},
        "taxes": {{
            {19}{16}{17}{18}{38}
        }}
    }}""".format(*row)
            jsons.append(json)
            labels.append(row[label_col])

    if balance:
        jsons_ = []
        vectors_ = []
        labels_ = []
        np.random.seed(123)
        pos_inds = [ i for i, l in enumerate(labels) if l == "50000+."]
        neg_inds = [ i for i, l in enumerate(labels) if not l == "50000+."]
        sample_inds = pos_inds + list(np.random.choice(neg_inds, size=len(pos_inds)*2))
        for i in sample_inds:
            jsons_.append(jsons[i])
            vectors_.append(vectors[i])
            labels_.append(labels[i])
        return jsons_, vectors_, labels_

    return jsons, vectors, labels





def load_covertype_dataset():
    wilderness_areas = {
        1: "Rawah Wilderness Area",
        2: "Neota Wilderness Area",
        3: "Comanche Peak Wilderness Area",
        4: "Cache la Poudre Wilderness Area"
    }

    soil_types = { 
        1: (2702, "Cathedral family - Rock outcrop complex, extremely stony."),
        2: (2703, "Vanet - Ratake families complex, very stony."),
        3: (2704, "Haploborolis - Rock outcrop complex, rubbly."),
        4: (2705, "Ratake family - Rock outcrop complex, rubbly."),
        5: (2706, "Vanet family - Rock outcrop complex complex, rubbly."),
        6: (2717, "Vanet - Wetmore families - Rock outcrop complex, stony."),
        7: (3501, "Gothic family."),
        8: (3502, "Supervisor - Limber families complex."),
        9: (4201, "Troutville family, very stony."),
        10: (4703, "Bullwark - Catamount families - Rock outcrop complex, rubbly."),
        11: (4704, "Bullwark - Catamount families - Rock land complex, rubbly."),
        12: (4744, "Legault family - Rock land complex, stony."),
        13: (4758, "Catamount family - Rock land - Bullwark family complex, rubbly."),
        14: (5101, "Pachic Argiborolis - Aquolis complex."),
        15: (5151, "unspecified in the USFS Soil and ELU Survey."),
        16: (6101, "Cryaquolis - Cryoborolis complex."),
        17: (6102, "Gateview family - Cryaquolis complex."),
        18: (6731, "Rogert family, very stony."),
        19: (7101, "Typic Cryaquolis - Borohemists complex."),
        20: (7102, "Typic Cryaquepts - Typic Cryaquolls complex."),
        21: (7103, "Typic Cryaquolls - Leighcan family, till substratum complex."),
        22: (7201, "Leighcan family, till substratum, extremely bouldery."),
        23: (7202, "Leighcan family, till substratum - Typic Cryaquolls complex."),
        24: (7700, "Leighcan family, extremely stony."),
        25: (7701, "Leighcan family, warm, extremely stony."),
        26: (7702, "Granile - Catamount families complex, very stony."),
        27: (7709, "Leighcan family, warm - Rock outcrop complex, extremely stony."),
        28: (7710, "Leighcan family - Rock outcrop complex, extremely stony."),
        29: (7745, "Como - Legault families complex, extremely stony."),
        30: (7746, "Como family - Rock land - Legault family complex, extremely stony."),
        31: (7755, "Leighcan - Catamount families complex, extremely stony."),
        32: (7756, "Catamount family - Rock outcrop - Leighcan family complex, extremely stony."),
        33: (7757, "Leighcan - Catamount families - Rock outcrop complex, extremely stony."),
        34: (7790, "Cryorthents - Rock land complex, extremely stony."),
        35: (8703, "Cryumbrepts - Rock outcrop - Cryaquepts complex."),
        36: (8707, "Bross family - Rock land - Cryumbrepts complex, extremely stony."),
        37: (8708, "Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony."),
        38: (8771, "Leighcan - Moran families - Cryaquolls complex, extremely stony."),
        39: (8772, "Moran family - Cryorthents - Leighcan family complex, extremely stony."),
        40: (8776, "Moran family - Cryorthents - Rock land complex, extremely stony.")
    }
        
    """
    Note:   First digit:  climatic zone             Second digit:  geologic zones
            1.  lower montane dry                   1.  alluvium
            2.  lower montane                       2.  glacial
            3.  montane dry                         3.  shale
            4.  montane                             4.  sandstone
            5.  montane dry and montane             5.  mixed sedimentary
            6.  montane and subalpine               6.  unspecified in the USFS ELU Survey
            7.  subalpine                           7.  igneous and metamorphic
            8.  alpine                              8.  volcanic
    """

    jsons = []
    vectors = []
    labels = []

    attributes = get_attributes('uci_data/covtype.data', verbose=False, separator=',')
    label_col = 54
    ignore_cols = [ label_col ]
    with open('uci_data/covtype.data', 'r') as f:
        csvreader = csv.reader(f, delimiter=',', quotechar='"')
        for row in csvreader:
            vector = get_vector(row, attributes, ignore_cols)
            vectors.append(vector)
            
            for i in range(4):
                if row[10+i] == "1":
                    row[10] = wilderness_areas[i+1]
            
            for i in range(40):
                if row[14+i] == "1":
                    code, text = soil_types[ i+1 ]
            row[14] = text
            row[15] = code
            
            json = """{{
        "area": "{10}",
        "elevation": {0},
        "aspect": {1},
        "slope": {2},
        "distance-to-hydrology": {3},
        "vert-distance-to-hydrology": {4},
        "distance-to-roadways": {5},
        "hillshade": {{
            "9am": {6},
            "noon": {7},
            "3pm": {8}
        }},
        "distance-to-fire-points": {9},
        "soil": {{
            "type": "{14}",
            "type-code": "{15}"
        }}
    }}""".format(*row)
            jsons.append(json)
            labels.append(row[label_col])
    return jsons, vectors, labels




def load_shopping_dataset(balance=True):
    jsons = []
    vectors = []
    labels = []

    attributes = get_attributes('uci_data/online_shoppers_intention.csv', verbose=False, separator=',', skip_header=True)
    label_col = 17
    ignore_cols = [ label_col ]
    with open('uci_data/online_shoppers_intention.csv', 'r') as f:
        csvreader = csv.reader(f, delimiter=',', quotechar='"')
        next(csvreader)
        for row in csvreader:
            vector = get_vector(row, attributes, ignore_cols)
            vectors.append(vector)
            
            row[16] = { "TRUE": "true", "FALSE": "false" }[row[16]]
            
            json = """{{
        "month": "{10}",
        "weekend": {16},
        "closest-holiday": {9},
        "visitor-type": "{15}",
        "visitor-details": {{
            "operating system": {11},
            "browser": {12},
            "traffic-type": {13}
        }},
        "page-visits": {{
            "administrative": {{ "visited": {0}, "duration": {1} }},
            "informational": {{ "visited": {2}, "duration": {3} }},
            "product-related": {{ "visited": {4}, "duration": {5} }}
        }},
        "google-analytics": {{
            "bounce-rates": {6},
            "exit-rates": {7},
            "page-value": {8}
        }}
    }}""".format(*row)
            jsons.append(json)
            labels.append(row[label_col])

    if balance:
        jsons_ = []
        vectors_ = []
        labels_ = []
        np.random.seed(123)
        pos_inds = [ i for i, l in enumerate(labels) if l == "TRUE"]
        neg_inds = [ i for i, l in enumerate(labels) if l == "FALSE"]
        sample_inds = pos_inds + list(np.random.choice(neg_inds, size=len(pos_inds)*2))
        for i in sample_inds:
            jsons_.append(jsons[i])
            vectors_.append(vectors[i])
            labels_.append(labels[i])
        return jsons_, vectors_, labels_

    return jsons, vectors, labels



def load_webpages_dataset(restrict=True, rebuild=False, retest=False):
    # Load CSV
    data = pd.read_csv("URL-categorization-DFE.csv", encoding="latin1")

    if rebuild:
        import os
        import subprocess
        from multiprocessing import Pool
        urls = data['url']

        def get_page(url):
            returned_value = subprocess.call("wget -O webpages/{} --timeout=10 --tries=2 {}".format(url, url), shell=True)
            return (url, returned_value)

        with Pool() as p:
            r = list(tqdm(p.imap_unordered(get_page, urls), total=len(urls)))
        
        with open('webpages/return_codes.pickle', 'wb') as file:
            pickle.dump(r, file, protocol=pickle.HIGHEST_PROTOCOL)
        
    if retest:
        from multiprocessing import Pool
        from XHTMLTreeParser import string2xml
        def test_file(url):
            try:
                with open('webpages/{}'.format(url), 'rb') as file:
                    inputs = file.read().decode("utf-8", errors="ignore")
                outputs = string2xml(inputs)
                return (url, 1)
            except Exception as error:
                return (url, 0)

        urls = data['url']
        with Pool() as p:
            r = list(tqdm(p.imap_unordered(test_file, urls), total=len(urls)))
        with open('webpages/valid_files.pickle', 'wb') as file:
            pickle.dump(r, file, protocol=pickle.HIGHEST_PROTOCOL)
            
        print("Encountered {} errors".format(len(errors)))

    # Add return codes
    with open('webpages/return_codes.pickle', 'rb') as file:
        return_codes = pickle.load(file)
    data['return_code'] = data['url'].map(dict(return_codes))

    # Add checked information
    with open('webpages/valid_files.pickle', 'rb') as file:
        valid_files = pickle.load(file)
    data['valid_xml'] = data['url'].map(dict(valid_files))

    if restrict:
        # Get only non-empty files
        data = data[data['return_code'] == 0]

        # Get only files which pass validation in our parser
        data = data[data['valid_xml'] == 1]#[:1000]

        # Get only pages labelled as working
        #df = df[df['main_category'] != 'Not_working']

    return data


from collections import defaultdict
    
def get_attributes(datafile, force_discrete=False, verbose=True, att_names=None,
    separator=',', skip_header=False):
    attributes = []

    with open(datafile, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=separator, quotechar='\"')
        if skip_header:
            next(csvreader)
        for row in csvreader:
            for colnum, col in enumerate(row):
                col = col.strip()
                if len(attributes) == colnum:
                    attributes.append({'values': [], 'counts': defaultdict(lambda: 0)})
                disc = False
                try:
                    val = float(col)
                except:
                    disc = True
                    
                if disc or force_discrete:
                    attributes[colnum]['type'] = 'discrete'
                    attributes[colnum]['counts'][col] += 1
                else:
                    attributes[colnum]['type'] = 'continuous'
                    attributes[colnum]['values'].append(val)

    att_names = [""]*len(attributes) if att_names is None else att_names

    
    i = 0
    for attribute, name in zip(attributes, att_names):
        if verbose:
            print(" ")
            print("Attribute {:d}: {}".format(i, name))
        i+=1
        if attribute['type'] == 'discrete' and verbose:
            if len(attribute['counts']) > 10:
                print("--> ({} unique values)".format(len(attribute['counts'])))
                continue
            for key, val in attribute['counts'].items():
                print("--> {}: {}".format(key, val))

        elif attribute['type'] == 'continuous':
            MEAN = np.mean(attribute['values'])
            STD = np.std(attribute['values'])
            attribute['mean'] = MEAN
            attribute['std'] = STD
            if verbose:
                print("--> mean: {:.2f}, std: {:.2f}".format(MEAN, STD))
    return attributes

def get_vector(row, attributes, ignore_cols):
    vector = []
    for colnum, col in enumerate(row):
        if colnum in ignore_cols:
            continue
        col = col.strip()

        attribute = attributes[colnum]
        if attribute['type'] == 'discrete':
            onehot = [ 0.0 ]*len(attribute['counts'])
            for i, key in enumerate(attribute['counts'].keys()):
                if col == key:
                    onehot[i] = 1.0
            vector = vector + onehot
        elif attribute['type'] == 'continuous':
            try:
                val = float(col)
                MEAN = attribute['mean']
                STD = attribute['std']
                val = (val - MEAN)
                val = val / STD if STD > 1e-6 else val
            except:
                #print(col)
                #print(colnum, attribute['type'])
                val = 0.
            vector = vector + [val]
    return vector