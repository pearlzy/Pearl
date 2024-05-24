import pickle
import numpy as np
import torch
from tqdm import tqdm
from transformers import DistilBertTokenizer
import random
from dataset.hard_supervision_functions import retrieve_times
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
        self.all_dicts = getAllDicts(dataset_name)
        print('Total questions = ', len(questions))

        if split == 'train':
            self.data = questions[:100]
        else:
            self.data = questions[:100]
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
        for i, question in enumerate(data):
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

        self.data = retrieve_times(args.tkg_file, args.dataset_name, self.data, args.corrupt_hard, args.fuse)
        
        
        self.data = self.addEntityAnnotation(self.data)
        self.num_total_entities = len(self.all_dicts['ent2id'])
        self.num_total_times = len(self.all_dicts['ts2id'])
        self.padding_idx = self.num_total_entities + self.num_total_times  # padding id for embedding of ent/time
        self.answer_vec_size = self.num_total_entities + self.num_total_times

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
        tr = self.e2tr[e]
        neighbours = []
        for t in tr:
            neighbours.append(t[0])
            neighbours.append(t[2])
        neighbours = set(neighbours)
        neighbours.remove(e)
        return list(neighbours)

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
        # entity_time_ids_tokenized_question
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

                else:
                    start_time = 0
                    end_time = 0

                # print('No time in qn!')


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

    def _collate_fn(self, items):

        tokenized_questions = [item[1] for item in items]
        attention_mask = torch.from_numpy(self.get_attention_mask(tokenized_questions))
        input_ids = torch.from_numpy(self.pad_for_batch(tokenized_questions, self.tokenizer.pad_token_id, np.float64))

        entity_time_ids_list = [item[2] for item in items]
        entity_time_ids_padded = self.pad_for_batch(entity_time_ids_list, self.padding_idx, np.float64)
        entity_time_ids_padded = torch.from_numpy(entity_time_ids_padded)

        entity_mask = [item[3] for item in items]  # 0 if entity, 1 if not
        entity_mask_padded = self.pad_for_batch(entity_mask, 1.0,
                                                np.float32)  # doesnt matter probably cuz attention mask will be used. maybe pad with 1?
        entity_mask_padded = torch.from_numpy(entity_mask_padded)

        heads = torch.from_numpy(np.array([item[4] for item in items]))
        tails = torch.from_numpy(np.array([item[5] for item in items]))
        times = torch.from_numpy(np.array([item[6] for item in items]))
        start_times = torch.from_numpy(np.array([item[7] for item in items]))
        end_times = torch.from_numpy(np.array([item[8] for item in items]))

        tails2 = torch.from_numpy(np.array([item[9] for item in items]))
        types = [item[10] for item in items]
        rels = torch.from_numpy(np.array([item[11] for item in items]))

        answers_single = torch.from_numpy(np.array([item[12] for item in items]))
        answers_arr = np.array([item[13] for item in items], dtype=object)
        max_len = max(len(sublist) for sublist in answers_arr)
        padded_answer_arr = [sublist + [0] * (max_len - len(sublist)+1) for sublist in answers_arr]
        padded_answer_arr_np = np.array(padded_answer_arr, dtype=np.int32)
        answers_arr = torch.from_numpy(padded_answer_arr_np)
        answers_type = [item[14] for item in items]
        answers_type = np.array(answers_type)

        def map_strings_to_integers(strings):
            unique_strings = np.unique(strings)
            string_to_int = {key: val for val, key in enumerate(unique_strings)}
            return np.vectorize(string_to_int.get)(strings)
        answers_type = map_strings_to_integers(answers_type)
        answers_type = torch.from_numpy(answers_type)
        return input_ids, attention_mask, entity_time_ids_padded, entity_mask_padded, heads, tails, times, start_times, end_times, tails2, types, rels, answers_single, answers_arr, answers_type


class baseDataset(object):

    def __init__(self, trainpath, testpath, statpath, validpath):


        self.trainQuadruples = self.load_quadruples(trainpath)
        self.testQuadruples = self.load_quadruples(testpath)
        self.validQuadruples = self.load_quadruples(validpath)
        self.allQuadruples = self.trainQuadruples + self.validQuadruples + self.testQuadruples
        self.num_e, self.num_r = self.get_total_number(statpath)

    @staticmethod
    def load_quadruples(inpath):

        with open(inpath, 'r') as f:
            fiveList = []
            for line in f:
                try:
                    line_split = line.split()
                    head = int(line_split[0][1:])
                    rel = int(line_split[1][1:])
                    tail = int(line_split[2][1:])
                    time1 = int(line_split[3])
                    time2 = int(line_split[4])
                    fiveList.append([head, rel, tail, time1, time2])  # 将五元组添加到列表中
                except Exception as e:
                    pass

        return fiveList

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
    def __init__(self, examples, num_r):
        """
        examples: a list of quadruples.
        num_r: number of relations
        """
        self.fivedruples = examples.copy()

        for ex in examples:
            self.fivedruples.append([ex[2], ex[1]+num_r+1, ex[0], ex[3], ex[4]])

    def __len__(self):
        return len(self.fivedruples)

    def __getitem__(self, item):
        return self.fivedruples[item][0], \
               self.fivedruples[item][1], \
               self.fivedruples[item][2], \
               self.fivedruples[item][3], \
               self.fivedruples[item][4]


