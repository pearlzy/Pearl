import pickle
import random
import torch
import numpy as np
from transformers import DistilBertModel, DistilBertTokenizer

from dataset.tcomplex import TComplEx

from tqdm import tqdm

def getQuestionEmbedding(lm_model,tokenizer ,text):
    b = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    output = lm_model(b['input_ids'], b['attention_mask'])
    return output

if __name__ == '__main__':
    pretrained_weights = './distil_bert/'
    lm_model = DistilBertModel.from_pretrained(pretrained_weights)
    tokenizer = DistilBertTokenizer.from_pretrained("./distil_bert/")
    output = getQuestionEmbedding(lm_model, tokenizer, "cat")
    print(output)

def loadTkbcModel(tkbc_model_file: object) -> object:
    print('Loading tkbc model from', tkbc_model_file)
    x = torch.load(tkbc_model_file, map_location=torch.device("cpu"))
    num_ent = x['embeddings.0.weight'].shape[0]
    num_rel = x['embeddings.1.weight'].shape[0]
    num_ts = x['embeddings.2.weight'].shape[0]
    print('Number ent,rel,ts from loaded model:', num_ent, num_rel, num_ts)
    sizes = [num_ent, num_rel, num_ent, num_ts]
    rank = x['embeddings.0.weight'].shape[1] // 2  # complex has 2*rank embedding size
    tkbc_model = TComplEx(sizes, rank, no_time_emb=False)
    tkbc_model.load_state_dict(x)
    tkbc_model.cuda()
    print('Loaded tkbc model')
    return tkbc_model

def loadTkbcModel_complex(tkbc_model_file):
    print('Loading complex tkbc model from', tkbc_model_file)
    tcomplex_file = 'models/wikidata_big/kg_embeddings/tcomplex.ckpt'  # TODO: hack
    tcomplex_params = torch.load(tcomplex_file)
    complex_params = torch.load(tkbc_model_file)
    num_ent = tcomplex_params['embeddings.0.weight'].shape[0]
    num_rel = tcomplex_params['embeddings.1.weight'].shape[0]
    num_ts = tcomplex_params['embeddings.2.weight'].shape[0]
    print('Number ent,rel,ts from loaded model:', num_ent, num_rel, num_ts)
    sizes = [num_ent, num_rel, num_ent, num_ts]
    rank = tcomplex_params['embeddings.0.weight'].shape[1] // 2  # complex has 2*rank embedding size

    # now put complex params in tcomplex model
    tcomplex_params['embeddings.0.weight'] = complex_params['embeddings.0.weight']
    tcomplex_params['embeddings.1.weight'] = complex_params['embeddings.1.weight']
    torch.nn.init.xavier_uniform_(tcomplex_params['embeddings.2.weight'])  # randomize time embeddings

    tkbc_model = TComplEx(sizes, rank, no_time_emb=False)
    tkbc_model.load_state_dict(tcomplex_params)
    tkbc_model.cuda()
    print('Loaded complex tkbc model')
    return tkbc_model
