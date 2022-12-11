# -*- coding: utf-8 -*
import sys

sys.path.append('..')
import numpy as np
from src.config import *
from src.utils.utils import *
import os
import re
from pyltp import Parser
from pyltp import Postagger
from itertools import product

torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
torch.backends.cudnn.deterministic = True

LTP_DATA_DIR = './model/ltp_data_v3.4.0'  # ltp directory

FILE = '%s.json'
GRAPH = '%s.graph'


def read_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as fr:
        js = json.load(fr)
    return js


def build_graph(data_dir, data_type):
    if data_type == 'train':
        data_file = data_dir + '/train_data/' + FILE % data_type
        graph_out = data_dir + '/train_data/' + GRAPH % data_type
    elif data_type == 'test':
        data_file = data_dir + '/test_data/' + FILE % data_type
        graph_out = data_dir + '/test_data/' + GRAPH % data_type

    par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # Syntactic analysis model
    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # Part-of-speech tagging model

    data_list = read_json(data_file)  # read data
    cor_dic = read_json(data_dir + '/coreference_data/cor_data.json')  # read coreference data
    doc_graph = {}  # document graph
    hi = 0
    for doc in data_list:
        parser = Parser()
        parser.load(par_model_path)
        postagger = Postagger()
        postagger.load(pos_model_path)

        doc_id = doc['doc_id']
        print(doc_id)
        doc_len = doc['doc_len']
        doc_clauses = doc['clauses']
        doc_core_line = cor_dic[doc_id]  # read coreference data

        doc_words_m = sum([len(cla['clause'].split()) for cla in doc_clauses])
        hgraph_ws = np.zeros((doc_words_m + doc_len, doc_words_m)) # init graph
        graph_ss = np.ones((doc_len, doc_len))
        clause_p = 0
        doc_str = ''
        head_indexs = []

        for i in range(doc_len):
            clause = doc_clauses[i]
            doc_str += clause['clause'] + ' '

            word_list = clause['clause'].split()
            postags = postagger.postag(word_list)  # part-of-speech tagging
            postags = list((' '.join(postags)).split(" "))
            arcs = parser.parse(word_list, postags)  # syntactic analysis
            rely_id = [arc.head for arc in arcs]  # parent node id
            heads = ['Root' if id == 0 else word_list[id - 1] for id in rely_id]  # Matches the dependency parent node term
            head_indexs.append(clause_p + heads.index('Root') + 1)

            # Edge of dependence relation
            for widx in range(len(word_list)):
                if heads[widx] != 'Root':
                    hgraph_ws[clause_p + widx][clause_p + word_list.index(heads[widx])] = 1
                    hgraph_ws[clause_p + word_list.index(heads[widx])][clause_p + widx] = 1
                hgraph_ws[doc_words_m + i][clause_p + widx] = 1
            clause_p = clause_p + len(word_list)

        # Edge of coreference relation
        if len(doc_core_line) != 0:
            for cor_line in doc_core_line:
                cor_line = list(filter(None, re.split(r'[,|\n \']', str(cor_line))))
                temp, c_word_idx = 0, 0
                cor_word_list = []
                for c_w in cor_line:
                    if c_word_idx < len(doc_str.split()):
                        c_word_idx = doc_str.split().index(c_w, temp)
                        temp = c_word_idx + 1
                    cor_word_list.append(c_word_idx)
                for ind in list(product(cor_word_list, cor_word_list)):
                    hgraph_ws[ind[0]][ind[1]] = 1

        graph={
            'ED_graph_ws':hgraph_ws,
            'KA_graph_ss':graph_ss
        }
        doc_graph[hi] = graph
        hi += 1

        # release model
        postagger.release()
        parser.release()

    with open(graph_out, 'wb') as gout:
        pickle.dump(doc_graph, gout)
        gout.close()


if __name__ == '__main__':
    data_dir = './data'
    build_graph(data_dir, 'train')
    build_graph(data_dir, 'test')
