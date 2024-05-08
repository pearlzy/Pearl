import pickle
import numpy as np
import torch
from tqdm import tqdm
from transformers import DistilBertTokenizer
# from nltk import word_tokenize
# warning: padding id 0 is being used, can have issue like in Tucker
# however since so many entities (and timestamps?), it may not pose problem
import random
from dataset.hard_supervision_functions import retrieve_times
# 导入Python中的collections模块的defaultdict类时，你实际上在引入一种特殊的字典数据结构。
# 这个字典不同于常规字典，因为它可以为键设置默认值，这意味着如果你尝试访问一个尚未存在的键，它不会引发错误，而是返回你指定的默认值。这在某些情况下非常有用，例如在统计计数时，你可以为每个键设置一个默认的初始计数值。
from collections import defaultdict
from torch.utils.data import Dataset

def getAllDicts(dataset_name):

    base_path = './data/{dataset_name}/kg/tkbc_processed_data/{dataset_name}/'.format(
        dataset_name=dataset_name
    )
    dicts = {}
    for f in ['ent_id', 'rel_id', 'ts_id']:
        in_file = open(str(base_path + f), 'rb')
        dicts[f] = pickle.load(in_file)
    rel2id = dicts['rel_id']
    ent2id = dicts['ent_id']
    ts2id = dicts['ts_id']
    file_ent = './data/{dataset_name}/kg/wd_id2entity_text.txt'.format(
        dataset_name=dataset_name
    )
    file_rel = './data/{dataset_name}/kg/wd_id2relation_text.txt'.format(
        dataset_name=dataset_name
    )

    def readDict(filename):
        f = open(filename, 'r', encoding="utf-8")
        d = {}
        for line in f:
            line = line.strip().split('\t')
            if len(line) == 1:
                line.append('')  # in case literal was blank or whitespace
            d[line[0]] = line[1]
        f.close()
        return d

    e = readDict(file_ent)
    r = readDict(file_rel)
    wd_id_to_text = dict(list(e.items()) + list(r.items()))

    def getReverseDict(d):
        return {value: key for key, value in d.items()}

    id2rel = getReverseDict(rel2id)
    id2ent = getReverseDict(ent2id)
    id2ts = getReverseDict(ts2id)

    all_dicts = {'rel2id': rel2id,
                 'id2rel': id2rel,
                 'ent2id': ent2id,
                 'id2ent': id2ent,
                 'ts2id': ts2id,
                 'id2ts': id2ts,
                 'wd_id_to_text': wd_id_to_text
                 }

    return all_dicts


class QA_Dataset(Dataset):
    def __init__(self, 
                split,
                dataset_name,
                tokenization_needed=True):
        filename = './data/{dataset_name}/questions/{split}.pickle'.format(
            dataset_name=dataset_name,
            split=split
        )
        questions = pickle.load(open(filename, 'rb'))
        
        #probably change for bert/roberta?
        self.tokenizer_class = DistilBertTokenizer 
        self.tokenizer = DistilBertTokenizer.from_pretrained("./distil_bert/")
        # self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        # self.tokenizer.save_pretrained("./distil_bert/")
        self.all_dicts = getAllDicts(dataset_name)
        print('Total questions = ', len(questions))
        # todo  对questions进行切片，可以控制数据大小 self.data = questions[:500]
        # if split == 'train':
        # self.data = questions[:1
        if split == 'train':
            self.data = questions[:900]
        else:
            self.data = questions
        # self.data = questions
        # else:
        #     # self.data = questions[:900]
        #     self.data = questions
        self.tokenization_needed = tokenization_needed

    # Q中的ent
    def getEntitiesLocations(self, question):
        question_text = question['question']
        entities = question['entities']
        ent2id = self.all_dicts['ent2id']
        loc_ent = []
        for e in entities:
            e_id = ent2id[e]
            location = question_text.find(e)
            loc_ent.append((location, e_id))
        return loc_ent

    # Q中的time
    def getTimesLocations(self, question):
        question_text = question['question']
        times = question['times']
        ts2id = self.all_dicts['ts2id']
        loc_time = []
        for t in times:
            t_id = ts2id[(t, 0, 0)] + len(self.all_dicts['ent2id']) # add num entities
            location = question_text.find(str(t))
            loc_time.append((location, t_id))
        return loc_time

    def isTimeString(self, s):
        # todo: cant do len == 4 since 3 digit times also there
        if 'Q' not in s:
            return True
        else:
            return False
    # Q中的ent_time id
    def textToEntTimeId(self, text):
        if self.isTimeString(text):
            t = int(text)
            ts2id = self.all_dicts['ts2id']
            t_id = ts2id[(t, 0, 0)] + len(self.all_dicts['ent2id'])
            return t_id
        else:
            ent2id = self.all_dicts['ent2id']
            e_id = ent2id[text]
            return e_id


    def getOrderedEntityTimeIds(self, question):
        loc_ent = self.getEntitiesLocations(question)
        loc_time = self.getTimesLocations(question)
        loc_all = loc_ent + loc_time
        loc_all.sort()
        ordered_ent_time = [x[1] for x in loc_all]
        return ordered_ent_time

    def entitiesToIds(self, entities):
        output = []
        ent2id = self.all_dicts['ent2id']
        for e in entities:
            output.append(ent2id[e])
        return output
    
    def getIdType(self, id):
        if id < len(self.all_dicts['ent2id']):
            return 'entity'
        else:
            return 'time'
    
    def getEntityToText(self, entity_wd_id):
        return self.all_dicts['wd_id_to_text'][entity_wd_id]
    
    def getEntityIdToText(self, id):
        ent = self.all_dicts['id2ent'][id]
        return self.getEntityToText(ent)
    
    def getEntityIdToWdId(self, id):
        return self.all_dicts['id2ent'][id]

    def timesToIds(self, times):
        output = []
        ts2id = self.all_dicts['ts2id']
        for t in times:
            output.append(ts2id[(t, 0, 0)])
        return output

    # 用于求hit@1
    def getAnswersFromScores(self, scores, largest=True, k=10):
        _, ind = torch.topk(scores, k, largest=largest)
        predict = ind
        answers = []

        for a_id in predict:
            a_id = a_id.item()
            type = self.getIdType(a_id)
            if type == 'entity':
                # answers.append(self.getEntityIdToText(a_id))
                answers.append(self.getEntityIdToWdId(a_id))
            else:
                time_id = a_id - len(self.all_dicts['ent2id'])
                time = self.all_dicts['id2ts'][time_id]
                answers.append(time[0])


        return answers
    # 用于求hit@k
    def getAnswersFromScoresWithScores(self, scores, largest=True, k=10):
        s, ind = torch.topk(scores, k, largest=largest)
        predict = ind
        # 创建一个空列表 answers，用于存储最高分数的答案。
        answers = []
        for a_id in predict:
            a_id = a_id.item()
            type = self.getIdType(a_id)
            if type == 'entity':
                # answers.append(self.getEntityIdToText(a_id))
                answers.append(self.getEntityIdToWdId(a_id))
            else:
                time_id = a_id - len(self.all_dicts['ent2id'])
                time = self.all_dicts['id2ts'][time_id]
                answers.append(time[0])
        #         返回包含最高分数和相应答案的元组 (s, answers)。
        return s, answers

    # from pytorch Transformer:
    # If a BoolTensor is provided, the positions with the value of True will be ignored 
    # while the position with the value of False will be unchanged.
    # 
    # so we want to pad with True
    def padding_tensor(self, sequences, max_len = -1):
        """
        :param sequences: list of tensors
        :return:
        """
        num = len(sequences)
        if max_len == -1:
            max_len = max([s.size(0) for s in sequences])
        out_dims = (num, max_len)
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)
        # mask = sequences[0].data.new(*out_dims).fill_(0)
        mask = torch.ones((num, max_len), dtype=torch.bool) # fills with True
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length] = tensor
            mask[i, :length] = False # fills good area with False
        return out_tensor, mask
    
    def toOneHot(self, indices, vec_len):
        indices = torch.LongTensor(indices)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def prepare_data(self, data):
        # we want to prepare answers lists for each question
        # then at batch prep time, we just stack these
        # and use scatter 
        question_text = []
        entity_time_ids = []
        num_total_entities = len(self.all_dicts['ent2id'])
        answers_arr = []
        for question in data:
            # first pp is question text
            # needs to be changed after making PD dataset
            # to randomly sample from list
            q_text = question['paraphrases'][0]
            question_text.append(q_text)
            et_id = self.getOrderedEntityTimeIds(question)
            entity_time_ids.append(torch.tensor(et_id, dtype=torch.long))
            if question['answer_type'] == 'entity':
                answers = self.entitiesToIds(question['answers'])
            else:
                # adding num_total_entities to each time id
                answers = [x + num_total_entities for x in self.timesToIds(question['answers'])]
            answers_arr.append(answers)
        # answers_arr = self.get_stacked_answers_long(answers_arr)
        return {'question_text': question_text, 
                'entity_time_ids': entity_time_ids, 
                'answers_arr': answers_arr}
    
    def is_template_keyword(self, word):
        if '{' in word and '}' in word:
            return True
        else:
            return False

    def get_keyword_dict(self, template, nl_question):
        template_tokenized = self.tokenize_template(template)
        keywords = []
        for word in template_tokenized:
            if not self.is_template_keyword(word):
                # replace only first occurence
                nl_question = nl_question.replace(word, '*', 1)
            else:
                keywords.append(word[1:-1]) # no brackets
        text_for_keywords = []
        for word in nl_question.split('*'):
            if word != '':
                text_for_keywords.append(word)
        keyword_dict = {}
        for keyword, text in zip(keywords, text_for_keywords):
            keyword_dict[keyword] = text
        return keyword_dict

    def addEntityAnnotation(self, data):
        for i in range(len(data)):
            question = data[i]
            keyword_dicts = []  # we want for each paraphrase
            template = question['template']
            #for nl_question in question['paraphrases']:
            nl_question = question['paraphrases'][0]
            keyword_dict = self.get_keyword_dict(template, nl_question)
            keyword_dicts.append(keyword_dict)
            data[i]['keyword_dicts'] = keyword_dicts
            #print(keyword_dicts)
            #print(template, nl_question)
        return data

    def tokenize_template(self, template):
        output = []
        buffer = ''
        i = 0
        while i < len(template):
            c = template[i]
            if c == '{':
                if buffer != '':
                    output.append(buffer)
                    buffer = ''
                while template[i] != '}':
                    buffer += template[i]
                    i += 1
                buffer += template[i]
                output.append(buffer)
                buffer = ''
            else:
                buffer += c
            i += 1
        if buffer != '':
            output.append(buffer)
        return output


class QA_Dataset_Baseline(QA_Dataset):
    def __init__(self, split, dataset_name,  tokenization_needed=True):
        super().__init__(split, dataset_name, tokenization_needed)
        print('Preparing data for split %s' % split)
        ents = self.all_dicts['ent2id'].keys()
        self.all_dicts['tsstr2id'] = {str(k[0]):v for k,v in self.all_dicts['ts2id'].items()}
        times = self.all_dicts['tsstr2id'].keys()
        rels = self.all_dicts['rel2id'].keys()

        self.prepared_data = self.prepare_data_(self.data)
        self.num_total_entities = len(self.all_dicts['ent2id'])
        self.num_total_times = len(self.all_dicts['ts2id'])
        self.answer_vec_size = self.num_total_entities + self.num_total_times

        
    def prepare_data_(self, data):
        # we want to prepare answers lists for each question
        # then at batch prep time, we just stack these
        # and use scatter 
        question_text = []
        heads = []
        tails = []
        times = []
        num_total_entities = len(self.all_dicts['ent2id'])
        answers_arr = []
        ent2id = self.all_dicts['ent2id']
        self.data_ids_filtered=[]
        # self.data=[]
        for i,question in enumerate(data):
            self.data_ids_filtered.append(i)

            # first pp is question text
            # needs to be changed after making PD dataset
            # to randomly sample from list
            q_text = question['paraphrases'][0]
            
            
            entities_list_with_locations = self.getEntitiesLocations(question)
            entities_list_with_locations.sort()
            entities = [id for location, id in entities_list_with_locations] # ordering necessary otherwise set->list conversion causes randomness
            head = entities[0] # take an entity
            if len(entities) > 1:
                tail = entities[1]
            else:
                tail = entities[0]
            times_in_question = question['times']
            if len(times_in_question) > 0:
                time = self.timesToIds(times_in_question)[0] # take a time. if no time then 0
                # exit(0)
            else:
                time = 0
                
            
            time += num_total_entities
            heads.append(head)
            times.append(time)
            tails.append(tail)
            question_text.append(q_text)
            
            if question['answer_type'] == 'entity':
                answers = self.entitiesToIds(question['answers'])
            else:
                # adding num_total_entities to each time id
                answers = [x + num_total_entities for x in self.timesToIds(question['answers'])]
            answers_arr.append(answers)
            
        # answers_arr = self.get_stacked_answers_long(answers_arr)
        self.data=[self.data[idx] for idx in self.data_ids_filtered]
        return {'question_text': question_text, 
                'head': heads, 
                'tail': tails,
                'time': times,
                'answers_arr': answers_arr}
    def print_prepared_data(self):
        for k, v in self.prepared_data.items():
            print(k, v)

    def __len__(self):
        return len(self.data)
        # return len(self.prepared_data['question_text'])

    def __getitem__(self, index):
        data = self.prepared_data
        question_text = data['question_text'][index]
        head = data['head'][index]
        tail = data['tail'][index]
        time = data['time'][index]
        answers_arr = data['answers_arr'][index]
        answers_single = random.choice(answers_arr)
        return question_text, head, tail, time, answers_single #,answers_khot

    def _collate_fn(self, items):
        batch_sentences = [item[0] for item in items]
        b = self.tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
        heads = torch.from_numpy(np.array([item[1] for item in items]))
        tails = torch.from_numpy(np.array([item[2] for item in items]))
        times = torch.from_numpy(np.array([item[3] for item in items]))
        answers_single = torch.from_numpy(np.array([item[4] for item in items]))
        return b['input_ids'], b['attention_mask'], heads, tails, times, answers_single
    def get_dataset_ques_info(self):
        type2num={}
        for question in self.data:
            if question["type"] not in type2num: type2num[question["type"]]=0
            type2num[question["type"]] += 1
        return {"type2num": type2num, "total_num": len(self.data_ids_filtered)}.__str__()


class QA_Dataset_SubGTR(QA_Dataset):
    def __init__(self, split, dataset_name, args, tokenization_needed=True):
        super().__init__(split, dataset_name, tokenization_needed)
        print('Preparing data for split %s' % split)
        ents = self.all_dicts['ent2id'].keys()
        self.all_dicts['tsstr2id'] = {str(k[0]): v for k, v in self.all_dicts['ts2id'].items()}
        times = self.all_dicts['tsstr2id'].keys()
        rels = self.all_dicts['rel2id'].keys()
        self.split = split

        # Aware module
        if args.aware_module:
            with open('./data/wikidata_big/saved_pkl/e2rt.pkl', 'rb') as f:
                self.e2rt = pickle.load(f)
            with open('./data/wikidata_big/saved_pkl/event2time.pkl', 'rb') as f:
                self.event2time = pickle.load(f)
            with open('./data/wikidata_big/saved_pkl/e2tr.pkl', 'rb') as f:
                self.e2tr = pickle.load(f)
            self.implicit_parsing(self.data)

        #args: given TKG, whether to corrupt hard, and how to use the reitrieved timestmaps  "在给定 TKG 的情况下，是否要损坏硬件，以及如何使用检索到的时间戳
        self.data = retrieve_times(args.tkg_file, args.dataset_name, self.data, args.corrupt_hard, args.fuse)
        
        
        self.data = self.addEntityAnnotation(self.data)
        self.num_total_entities = len(self.all_dicts['ent2id'])
        self.num_total_times = len(self.all_dicts['ts2id'])
        self.padding_idx = self.num_total_entities + self.num_total_times  # padding id for embedding of ent/time
        # self.answer_vec_size 可能用于表示一个向量的长度或特征的维度，其中包括了实体和时间。
        self.answer_vec_size = self.num_total_entities + self.num_total_times
          # todo 一轮预处理
        self.prepared_data = self.prepare_data2(self.data)

    def __len__(self):
        # return 100
        return len(self.data)

    def get_event_time(self, event):
        c = self.event2time[event]
        event_triples = list(c)
        time = event_triples[0][3]
        return time

    def check_triples(self, q1, q2):
        l = []
        if type(q2) == list:
            for i in q2:
                l += self.check_triples(q1,i)
        else:
            for i in self.e2rt[(q1,q2)]:
                l.append(i)
            for i in self.e2rt[(q2,q1)]:
                l.append(i)
        return l

    def get_neighbours(self, e):
        # todo e2tr: 表示与查询实体e 存在关系的
        tr = self.e2tr[e]
        neighbours = []
        for t in tr:
            neighbours.append(t[0])
            neighbours.append(t[2])
        neighbours = set(neighbours)
        neighbours.remove(e)
        return list(neighbours)

    def implicit_parsing(self, data):
        # general extraction
        for i in data:
            b = list(i['annotation'].keys())
            if 'event_head' in b:
                c = i['annotation']['event_head']
                if c[0]!='Q':
                    continue
                time = self.get_event_time(c)
                if i['type'] != 'before_after':
                    i['times'] = {int(time)}
                i['annotation']['time'] = time
                i['annotation']['event_head_bak'] = i['annotation']['event_head']
                i['annotation']['event_head'] = time
                i['paraphrases'][0] = i['paraphrases'][0].replace(self.all_dicts['wd_id_to_text'][c],time)

        # speific extraction
        for i in data:
            if i['type'] == 'before_after':
                if 'event_head' not in i['annotation'].keys():
                    head = i['annotation']['head']
                    tail = i['annotation']['tail']
                    related_triples = self.check_triples(head,tail)
                    if i['annotation']['type'] == 'before':
                        index = 0
                        time = related_triples[0][3]
                    else:
                        time = related_triples[-1][4]
                    # i['times'] = {int(time)}
                    #i['annotation']['time'] = time
                    # NL replace
                    text = self.all_dicts['wd_id_to_text'][head]
                    i['paraphrases'][0] = i['paraphrases'][0].replace(text,time)
    # 从问题中提取与实体或时间相关的关键词及其对应的标识符或时间信息。
    def getEntityTimeTextIds(self, question, pp_id=0):
        keyword_dict = question['keyword_dicts'][pp_id]
        keyword_id_dict = question['annotation']  # this does not depend on paraphrase
        # 与实体或时间相关的关键词添加到 output_text 列表中，
        output_text = []
        output_ids = []
        entity_time_keywords = set(['head', 'tail', 'time', 'event_head', 'time1', 'time2'])
        
        #print(keyword_dict, keyword_id_dict)
        for keyword, value in keyword_dict.items():
            if keyword in entity_time_keywords:
                wd_id_or_time = keyword_id_dict[keyword]
                output_text.append(value)
                output_ids.append(wd_id_or_time)
                
        #print(output_text, output_ids)
        return output_text, output_ids

    def get_entity_aware_tokenization(self, nl_question, ent_times, ent_times_ids):
        # what we want finally is that after proper tokenization
        # of nl question, we know which indices are beginning tokens
        # of entities and times in the question
        index_et_pairs = []
        index_et_text_pairs = []
        for e_text, e_id in zip(ent_times, ent_times_ids):
            location = nl_question.find(e_text)
            pair = (location, e_id)
            index_et_pairs.append(pair)
            pair = (location, e_text)
            index_et_text_pairs.append(pair)
        index_et_pairs.sort()
        index_et_text_pairs.sort()
        my_tokenized_question = []
        start_index = 0
        arr = []
        for pair, pair_id in zip(index_et_text_pairs, index_et_pairs):
            end_index = pair[0]
            if nl_question[start_index: end_index] != '':
                my_tokenized_question.append(nl_question[start_index: end_index])
                arr.append(self.padding_idx)
            start_index = end_index
            end_index = start_index + len(pair[1])
            # todo: assuming entity name can't be blank
            # my_tokenized_question.append(nl_question[start_index: end_index])
            my_tokenized_question.append(self.tokenizer.mask_token)
            matrix_id = self.textToEntTimeId(pair_id[1])  # get id in embedding matrix
            arr.append(matrix_id)
            start_index = end_index
        if nl_question[start_index:] != '':
            my_tokenized_question.append(nl_question[start_index:])
            arr.append(self.padding_idx)

        tokenized, valid_ids = self.tokenize(my_tokenized_question)
        entity_time_final = []
        index = 0
        for vid in valid_ids:
            if vid == 0:
                entity_time_final.append(self.padding_idx)
            else:
                entity_time_final.append(arr[index])
                index += 1
        entity_mask = []  # want 0 if entity, 1 if not, since will multiply this later with word embedding
        for x in entity_time_final:
            if x == self.padding_idx:
                entity_mask.append(1.)
            else:
                entity_mask.append(0.)
        #print(entity_time_final)
        return tokenized, entity_time_final, entity_mask
 #todo  一轮预处理代码
    def prepare_data2(self, data):
        # we want to prepare answers lists for each question
        # then at batch prep time, we just stack these
        # and use scatter
        heads = []
        times = []
        start_times = []
        end_times = []
        tails = []
        tails2 = []
        types = []
        rels = []
        question_text = []
        tokenized_question = []
        # entity_time_ids_tokenized_question 是一个列表，其中的每个元素对应于问题中已经被处理的标记化文本的一个部分。这些部分包括实体和时间信息，已经被处理成一系列的标识符。
        entity_time_ids_tokenized_question = []
        entity_mask_tokenized_question = []
        pp_id = 0
        num_total_entities = len(self.all_dicts['ent2id'])
        answers_arr = []
        answers_type = []
        for question in tqdm(data, desc=self.split):
            # randomly sample pp
            # in test there is only 1 pp, so always pp_id=0
            # TODO: this random is causing assertion bug later on
            # pp_id = random.randint(0, len(question['paraphrases']) - 1)
            pp_id = 0
            # 问题文本
            nl_question = question['paraphrases'][pp_id]
            q_text = nl_question
            et_text, et_ids = self.getEntityTimeTextIds(question, pp_id)
            entities_list_with_locations = self.getEntitiesLocations(question)
            entities_list_with_locations.sort()
            entities = [id for location, id in
                        entities_list_with_locations]  # ordering necessary otherwise set->list conversion causes randomness
            head = entities[0]  # take an entity
            if len(entities) > 1:
                tail = entities[1]
                if len(entities) > 2:
                    tail2 = entities[2]
                else:
                    tail2 = tail
            else:
                tail = entities[0]
                tail2 = tail
            times_in_question = question['times']

            if len(times_in_question) > 0:
                time = self.timesToIds(times_in_question)[0]  # take a time. if no time then 0
                #
                start_time = time
                end_time = time
            else:
                time = 0
                #check for retrieved timestmaps
                # 如果问题中没有明确的时间信息，但包含了一些事实信息（question['fact']），则从这些事实中提取时间戳，并将它们进行排序，
                # 然后取第一个时间戳作为起始时间（start_time），取最后一个时间戳作为结束时间（end_time），
                # 然后将它们的时间标识符分别赋给 time、start_time 和 end_time。
                if len(question['fact']) > 0:
                    ts = []
                    #add all timestmaps and sort
                    for f in question['fact']:
                        for t in range(int(f[0]), int(f[1])+1):
                            ts.append(t)
                    ts = sorted(ts)
                    sorted_times = self.timesToIds(ts)
                    
                    try:
                        start_time = sorted_times[0]   # for random random.choice(sorted_times)
                    except:
                        start_time = 0
                    try:
                        end_time = sorted_times[-1]
                    except:
                        end_time = 0
                # 如果问题中既没有明确的时间信息，也没有事实信息，那么将 time 设置为默认值 0，表示没有时间信息。
                else:
                    start_time = 0
                    end_time = 0

                # print('No time in qn!')

            # time += num_total_entities 的目的是为了在问题中使用一个唯一的时间标识符，并确保这个时间标识符与其他实体（entities）和关系（relations）的标识符不会冲突。
            # 这是因为在该代码中，问题中的不仅包含时间信息，还包含其他实体和关系信息，它们都需要被表示为标识符以便在后续处理中进行查询和分析。
            time += num_total_entities
            heads.append(head)
            times.append(time)
            start_times.append(start_time)
            end_times.append(end_time)
            tails.append(tail)
            tails2.append(tail2)
            types.append(question['type'])
            rel = self.all_dicts['rel2id'][list(question['relations'])[0]]
            rels.append(rel)
            tokenized, entity_time_final, entity_mask = self.get_entity_aware_tokenization(nl_question, et_text, et_ids)
            assert len(tokenized) == len(entity_time_final)
            question_text.append(nl_question)
            tokenized_question.append(self.tokenizer.convert_tokens_to_ids(tokenized))
            entity_mask_tokenized_question.append(entity_mask)
            entity_time_ids_tokenized_question.append(entity_time_final)
            answers_type.append(question['answer_type'])

            if question['answer_type'] == 'entity':
                answers = self.entitiesToIds(question['answers'])
            else:
                # adding num_total_entities to each time id
                answers = [x + num_total_entities for x in self.timesToIds(question['answers'])]
            answers_arr.append(answers)
        return {'question_text': question_text,
                'tokenized_question': tokenized_question,
                'entity_time_ids': entity_time_ids_tokenized_question,
                'entity_mask': entity_mask_tokenized_question,
                'head': heads,
                'tail': tails,
                'time': times,
                'start_time': start_times,
                'end_time': end_times,
                'tail2': tails2,
                'types': types,
                'rels': rels,
                'answers_arr': answers_arr,
                'answers_type': answers_type}
#当对questions有疑问时，可以对这段代码进行debug，就可以知道每个部分的含义
    # tokenization function taken from NER code
    def tokenize(self, words):
        """ tokenize input"""
        tokens = []
        valid_positions = []
        tokens.append(self.tokenizer.cls_token)
        valid_positions.append(0)
        for i, word in enumerate(words):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for i in range(len(token)):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)
        tokens.append(self.tokenizer.sep_token)
        valid_positions.append(0)
        return tokens, valid_positions
    # _getitem__(self, index) 方法的作用是根据给定的 index 参数，返回对象中对应索引位置的元素或属性。
    def __getitem__(self, index):
        #todo 一轮预处理
        data = self.prepared_data
        question_text = data['question_text'][index]
        entity_time_ids = np.array(data['entity_time_ids'][index], dtype=np.float64)
        answers_arr = data['answers_arr'][index]
        answers_single = random.choice(answers_arr)
        # answers_khot = self.toOneHot(answers_arr, self.answer_vec_size)
        tokenized_question = data['tokenized_question'][index]
        entity_mask = data['entity_mask'][index]
        head = data['head'][index]
        tail = data['tail'][index]
        tail2 = data['tail2'][index]
        time = data['time'][index]
        start_time = data['start_time'][index]
        end_time = data['end_time'][index]
        types = data['types'][index]
        rels = data['rels'][index]
        answers_arr = data['answers_arr'][index]
        answers_type = data['answers_type'][index]
        return question_text, tokenized_question, entity_time_ids, entity_mask, head, tail, time, start_time, end_time, tail2, types, rels, answers_single, answers_arr, answers_type

    def pad_for_batch(self, to_pad, padding_val, dtype=np.float64):
        padded = np.ones([len(to_pad), len(max(to_pad, key=lambda x: len(x)))], dtype=dtype) * padding_val
        for i, j in enumerate(to_pad):
            padded[i][0:len(j)] = j
        return padded

    # do this before padding for batch
    def get_attention_mask(self, tokenized):
        # first make zeros array of appropriate size
        mask = np.zeros([len(tokenized), len(max(tokenized, key=lambda x: len(x)))], dtype=np.float64)
        # now set ones everywhere needed
        for i, j in enumerate(tokenized):
            mask[i][0:len(j)] = np.ones(len(j), dtype=np.float64)
        return mask
    # _collate_fn 的方法，通常用于在数据集中的多个样本（items）组成一个批次（batch）时，对这些样本进行处理和整理.
    # 这个方法的主要作用是将来自数据集的多个样本整理成一个可以输入到神经网络模型的批次，并且将它们转换为PyTorch张量（tensors）的形式。
    def _collate_fn(self, items):  # todo items是一轮预处理数据
        # please don't tokenize again
        # b = self.tokenizer(batch_sentences, padding=True, truncation=False, return_tensors="pt")
        # 问题文本（tokenized_questions）
        tokenized_questions = [item[1] for item in items]
        # 注意力掩码（attention_mask）
        attention_mask = torch.from_numpy(self.get_attention_mask(tokenized_questions))
        input_ids = torch.from_numpy(self.pad_for_batch(tokenized_questions, self.tokenizer.pad_token_id, np.float64))

        entity_time_ids_list = [item[2] for item in items]
        # 实体时间标识符（entity_time_ids_padded）
        entity_time_ids_padded = self.pad_for_batch(entity_time_ids_list, self.padding_idx, np.float64)
        entity_time_ids_padded = torch.from_numpy(entity_time_ids_padded)

        entity_mask = [item[3] for item in items]  # 0 if entity, 1 if not
        # 实体掩码（entity_mask_padded）
        entity_mask_padded = self.pad_for_batch(entity_mask, 1.0,
                                                np.float32)  # doesnt matter probably cuz attention mask will be used. maybe pad with 1?
        entity_mask_padded = torch.from_numpy(entity_mask_padded)
        # can make foll mask in forward function using attention mask
        # entity_time_ids_padded_mask = ~(attention_mask.bool())

        heads = torch.from_numpy(np.array([item[4] for item in items]))
        tails = torch.from_numpy(np.array([item[5] for item in items]))
        times = torch.from_numpy(np.array([item[6] for item in items]))
        start_times = torch.from_numpy(np.array([item[7] for item in items]))
        end_times = torch.from_numpy(np.array([item[8] for item in items]))

        tails2 = torch.from_numpy(np.array([item[9] for item in items]))
        types = [item[10] for item in items]
        rels = torch.from_numpy(np.array([item[11] for item in items]))

        # answers_khot = np.array([item[14].numpy() for item in items], dtype=np.int32)
        # answers_khot = torch.from_numpy(answers_khot)


        answers_single = torch.from_numpy(np.array([item[12] for item in items]))
        # 生成答案列表
        answers_arr = np.array([item[13] for item in items], dtype=object)
        # 找到最长子列表的长度
        max_len = max(len(sublist) for sublist in answers_arr)
        # 填充子列表，使它们具有相同的长度
        padded_answer_arr = [sublist + [0] * (max_len - len(sublist)+1) for sublist in answers_arr]
        padded_answer_arr_np = np.array(padded_answer_arr, dtype=np.int32)
        answers_arr = torch.from_numpy(padded_answer_arr_np)
        answers_type = [item[14] for item in items]
        # 返回包含所有这些张量的元组，以构成一个批次。
        return input_ids, attention_mask, entity_time_ids_padded, entity_mask_padded, heads, tails, times, start_times, end_times, tails2, types, rels, answers_single, answers_arr, answers_type


class baseDataset(object):
    #类的构造函数，用于初始化基本数据集。
    # 它接受四个参数，分别是训练数据文件路径 (trainpath)、测试数据路径 (testpath)、实体和关系数量统计文件路径 (statpath) 以及验证数据文件路径 (validpath)。
    # 在初始化过程中，它执行以下操作：
    def __init__(self, trainpath, testpath, statpath, validpath):
        """base Dataset. Read data files and preprocess.
        Args:
            trainpath: File path of train Data;训练数据文件的路径;
            testpath: File path of test data;测试数据文件的路径;
            statpath: File path of entities num and relatioins num;实体数量和关系数量统计文件的路径;
            validpath: File path of valid data 验证数据文件的路径
        """
        # 加载数据文件并初始化基本数据集

        self.trainQuadruples = self.load_quadruples(trainpath)   # 加载训练数据
        self.testQuadruples = self.load_quadruples(testpath)      # 加载测试数据
        self.validQuadruples = self.load_quadruples(validpath)    # 加载验证数据
        self.allQuadruples = self.trainQuadruples + self.validQuadruples + self.testQuadruples
        self.num_e, self.num_r = self.get_total_number(statpath)  # number of entities, number of relations # 实体数量，关系数量
        # self.skip_dict = self.get_skipdict(self.allQuadruples)   # 时间依赖过滤度量
        # self.ent2id = getAllDicts(args.dataset_name)
        # self.timestamps = self.get_all_timestamps(self.allQuadruples)
        # 存储训练集中出现的实体
        # self.train_entities = set()  # Entities that have appeared in the training set
        # for query in self.trainQuadruples:
        #     self.train_entities.add(query[0])
        #     self.train_entities.add(query[2])
        # 调用 self.getRelEntCooccurrence() 方法，获取训练集中实体和关系的共现情况。
        # self.RelEntCooccurrence = self.getRelEntCooccurrence(self.trainQuadruples)  # -> dict


    # todo: 这里可能也有问题，之后记得debug 查看下这里的id  是否是异常的
    # 这个方法用于获取训练集中实体和关系的共现情况。它返回一个字典，包含了每个关系对应的共现实体集合。
    # def getRelEntCooccurrence(self, quadruples):
    #     """Used for Inductive-Mean. Get co-occurrence in the training set.
    #      用于Inductive-Mean。获取训练集中的共现情况。
    #     return:
    #         {'subject': a dict[key -> relation, values -> a set of co-occurrence subject entities],
    #          'object': a dict[key -> relation, values -> a set of co-occurrence object entities],}
    #     共现情况字典:
    #     {'subject': {'relation': {共现实体集合}},
    #      'object': {'relation': {共现实体集合}},}
    #     """
    #     # 用于存储主语实体和关系的共现情况
    #     relation_entities_s = {}
    #     # 用于存储宾语实体和关系的共现情况
    #     relation_entities_o = {}
    #     for ex in quadruples:
    #         # 从四元组中提取主语、关系和宾语
    #         s, r, o = ex[0], ex[1], ex[2]
    #         # 计算关系的反向版本
    #         reversed_r = r + self.num_r + 1
    #
    #         # 处理主语实体和关系的共现情况
    #         if r not in relation_entities_s.keys():
    #             relation_entities_s[r] = set()
    #         relation_entities_s[r].add(s)
    #         # 处理宾语实体和关系的共现情况
    #         if r not in relation_entities_o.keys():
    #             relation_entities_o[r] = set()
    #         relation_entities_o[r].add(o)
    #
    #         # 处理反向关系的情况，更新共现情况
    #         if reversed_r not in relation_entities_s.keys():
    #             relation_entities_s[reversed_r] = set()
    #         relation_entities_s[reversed_r].add(o)
    #         if reversed_r not in relation_entities_o.keys():
    #             relation_entities_o[reversed_r] = set()
    #         relation_entities_o[reversed_r].add(s)
    #     # 返回共现情况字典，其中包含了主语和宾语的关系共现情况
    #     return {'subject': relation_entities_s, 'object': relation_entities_o}

    # 这个方法用于获取数据集中所有的时间戳（timestamps），并返回一个时间戳的集合。
    # @staticmethod
    # def get_all_timestamps(examples):
    #     """Get all the timestamps in the dataset.
    #     return:
    #         timestamps: a set of timestamps.
    #     """
    #     timestamps = []
    #     for (head, rel, tail, t1, t2) in examples:
    #         timestamp = [t1, t2, head]
    #         timestamps.append(timestamp)
    #     timestamps.sort(key=lambda x:x[0])
    #     return timestamps

    # def find_nearest_timestamp_pair(self, query_times):
    #     # 创建一个结果张量，用于存储每个查询时间戳的最接近的时间对
    #     timestamps = torch.tensor(self.timestamps)
    #     nearest_pairs = torch.zeros_like(timestamps, dtype=torch.int)  # 用零初始化，你可以根据需要选择其他初始值
    #
    #     for i, query_time in enumerate(query_times):
    #         min_time_diff = float('inf')
    #         nearest_pair = None
    #
    #         for timestamp_pair in self.timestamps:
    #             time1, time2 = timestamp_pair
    #             time_diff1 = abs((query_time - 125726) - time1)
    #             time_diff2 = abs((query_time - 125726) - time2)
    #             total_time_diff = time_diff1 + time_diff2
    #
    #             if total_time_diff < min_time_diff:
    #                 min_time_diff = total_time_diff
    #                 nearest_pair = timestamp_pair
    #
    #         nearest_pairs[i] = nearest_pair
    #
    #     return nearest_pairs
    # 该方法用于执行时间依赖的过滤度量，并返回一个字典，其中键是元组 (实体, 关系, 时间戳)，值是与之相关的实体集合
    # def get_skipdict(self, quadruples):
    #     """Used for time-dependent filtered metrics.
    #     return: a dict [key -> (entity, relation, timestamp),  value -> a set of ground truth entities]
    #     """
    #     filters = defaultdict(set)
    #     for src, rel, dst, time1, time2 in quadruples:
    #         filters[(src, rel, time1, time2)].add(dst)
    #         filters[(dst, rel+self.num_r+1, time1, time2)].add(src)
    #     return filters

    # 这是一个静态方法，用于加载包含五元组数据（subject/headEntity, relation, object/tailEntity, timestamp）的文本文件。它接受一个文件路径 inpath 作为参数，返回一个包含所有四元组的列表。
    @staticmethod
    def load_quadruples(inpath):
        """train.txt/valid.txt/test.txt reader
        inpath: File path. train.txt, valid.txt or test.txt of a dataset;
        return:
            quadrupleList: A list
            containing all quadruples([subject/headEntity, relation, object/tailEntity, timestamp]) in the file.
        """
        with open(inpath, 'r') as f:
            fiveList = []   # 用于存储四元组的空列表
            for line in f:
                try:
                    line_split = line.split() # 将每一行分割成多个部分
                    # head = int(line_split[0])  # 提取主体并将其转换为整数
                    head = int(line_split[0][1:])  # 提取主体并将其转换为整数
                    # rel = int(line_split[1])    # 提取关系并将其转换为整数
                    rel = int(line_split[1][1:])    # 提取关系并将其转换为整数
                    # tail = int(line_split[2])   # 提取对象并将其转换为整数
                    tail = int(line_split[2][1:])   # 提取对象并将其转换为整数
                    time1 = int(line_split[3])   # 提取时间戳并将其转换为整数
                    time2 = int(line_split[4])
                    fiveList.append([head, rel, tail, time1, time2])  # 将五元组添加到列表中
                except Exception as e:
                    pass
                    # print(e)
                    # print(line_split)
                    # print(line)
        return fiveList # 返回包含所有四元组的列表

    # 这是一个静态方法，用于加载包含实体数量和关系数量统计信息的文本文件。它接受一个文件路径 statpath 作为参数，返回一个包含实体数量和关系数量的元组。
    @staticmethod
    def get_total_number(statpath):
        """stat.txt reader
        return:
            (number of entities -> int, number of relations -> int)
        """
        with open(statpath, 'r') as fr:
            for line in fr:
                line_split = line.split()
                return int(line_split[0]), int(line_split[1])


class QuadruplesDataset(Dataset):
    # __init__ 中，它接受一个四元组列表 examples 和关系数量 num_r 作为参数。
    # 在初始化过程中，它将输入的四元组列表复制一次，并对每个四元组进行修改，将主语和宾语互换，并更新关系。
    # 这样，数据集中将包含原始四元组和相应的反向四元组。
    def __init__(self, examples, num_r):
        """
        examples: a list of quadruples.
        num_r: number of relations
        """
        self.fivedruples = examples.copy()
        # 遍历传递给构造函数的四元组列表 examples 中的每个四元组。
        for ex in examples:
            # [ex[2], ex[1]+num_r+1, ex[0], ex[3]] 创建了一个新的四元组。在这个新四元组中，原四元组中的主语和宾语被互换了（ex[2] 变成了新四元组的宾语，ex[0] 变成了新四元组的主语），同时关系也被更新。
            # 关系的更新是通过将原关系 ex[1] 的值增加 num_r + 1 来实现的，这个操作使得原关系和反向关系之间有一个差值。
            self.fivedruples.append([ex[2], ex[1]+num_r+1, ex[0], ex[3], ex[4]])
    # __len__ 方法返回数据集的长度，即包含的四元组数量。
    def __len__(self):
        return len(self.fivedruples)
    # __getitem__ 方法根据索引获取数据集中的一个样本，以元组的形式返回四元组的主语、关系、宾语和时间戳。
    # 该方法接受一个参数 item，它是样本的索引，表示要获取数据集中的哪个样本。
    # 然后，该方法返回一个包含四元组数据的元组，其中四元组的元素按顺序分别是主语 (subject)、关系 (relation)、宾语 (object) 和时间戳 (timestamp)。
    def __getitem__(self, item):
        return self.fivedruples[item][0], \
               self.fivedruples[item][1], \
               self.fivedruples[item][2], \
               self.fivedruples[item][3], \
               self.fivedruples[item][4]


