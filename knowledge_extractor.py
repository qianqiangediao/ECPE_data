# coding: utf-8
from src.sentence_transformers import SentenceTransformer,util
import pandas as pd
import os
import csv
import pickle
from tqdm import tqdm

# relation types
chosen_relation = set(['xNeed', 'xAttr', 'xReact', 'xEffect', 'xWant', 'xIntent', 'oEffect', 'oWant', 'oReact','isBefore','isAfter','HasSubEvent'])


def build_new_table(com_path,filenames):
    # Create the head -- >tail mapping table
    if not os.path.exists(com_path+'head2tail.csv'):
        head2tail = {}
        for filename in filenames:
            df = pd.read_csv(filename, sep='\t')
            for _, item in df.iterrows():
                head = item['head']
                relation = item['relation']
                tail = item['tail']
                if head not in head2tail.keys():
                    head2tail[head] = {}
                if relation not in head2tail[head].keys():
                    head2tail[head][relation] = []
                head2tail[head][relation].append(tail)
        print('Done')
        with open(com_path+'head2tail.csv','wb')as f:
            pickle.dump(head2tail,f)
        return head2tail
    else:
        with open(com_path+'head2tail.csv','rb')as f:
            head2tail = pickle.load(f)
            return head2tail


def append_tail(item, head2tail, i,event ,filter_relation=True):
    #input a head, return all the corresponding relation and tail
    ret = {}
    ret['index'] = i
    ret['clause'] = event
    for head_similarity in item:
        for head, similarity in head_similarity:
            if 'xReact' in head2tail[head].keys():
                ret[head] = {}
                ret[head]['similarity'] = similarity
                for relation in head2tail[head].keys():
                    if not filter_relation or relation in chosen_relation:
                        ret[head][relation] = head2tail[head][relation]
                break
            else:
                continue
    return ret


def match(head_embeddings, query_embeddings, trg):
    # The head of each clause with the highest similarity is returned based on semantic similarity
    all_matched = []
    for q_embedding in query_embeddings:
        matched = []
        cosine_scores =util.pytorch_cos_sim(q_embedding, head_embeddings).squeeze()
        values, indices = cosine_scores.topk(10, dim=0, largest=True, sorted=True)
        for value,indice in zip(values,indices):
            matched.append((trg[indice.item()], value.item()))
        all_matched.append(matched)
    return all_matched


if __name__ == '__main__':
    #comonsense knowledge download and embedding
    model_path='../src/model/SBert-ATOMIC'
    com_path = '../data/commonsense_data/'
    model = SentenceTransformer(model_path)
    head = pd.read_csv(com_path+'head_shortSentence.csv')
    head2tail = build_new_table(com_path,'ATOMIC_Chinese.tsv')
    trg = list(head['head_translated'])
    head_embeddings = model.encode(trg, convert_to_tensor=True)
    #data download
    filename = com_path+ 'ecpe_data.csv'
    data = csv.reader(
        open(filename, encoding="utf-8"),
        delimiter='\t', quoting=csv.QUOTE_NONE)
    all_event_list = [[uttr for uttr in row[0].split(' ') if uttr!=""] for row in data]
    all_match_result = []
    # match commonsense
    for i, event_list in enumerate(tqdm(all_event_list)):
        match_result = []
        for event in event_list:
            query_embeddings = model.encode(event, convert_to_tensor=True).unsqueeze_(0)
            match_result.append(append_tail(match(head_embeddings, query_embeddings, trg), head2tail,i,event))
        all_match_result.append(match_result)
    with open(filename[:-4] + '_commonsense1' + '.pkl', 'wb') as  datawriter:
         pickle.dump(all_match_result,datawriter)
