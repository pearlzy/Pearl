import torch
import tqdm
import numpy as np
import random

class Tester(object):

    def __init__(self, model, args, train_entities, train_timestamps):
        self.model = model.cuda()
        self.args = args
        self.train_entities = train_entities
        self.train_timestamps = train_timestamps

    def get_rank(self, score, answer_arr, entities_space, num_ent):
        """Get the location of the answer, if the answer is not in the array,
        the ranking will be the total number of entities.
        Args:
            score: list, entity score
            answer_arr: int, the ground truth entity
            entities_space: corresponding entity with the score
            num_ent: the total number of entities
        Return: the rank of the ground truth.
        """

        is_matching = np.vectorize(lambda entity: entity in entities_space)(answer_arr)
        is_matching_tensor = torch.tensor(is_matching, dtype=torch.bool)
        is_matching = torch.any(is_matching_tensor, dim=0)
        if not is_matching.any():
            rank = num_ent
        else:
            answer_indices = []
            for i, answer_tensor in enumerate(answer_arr):
                for entity in entities_space:
                    if entity == answer_tensor:
                        answer_indices.append(entities_space.index(entity))
                if answer_tensor == -125726 or answer_tensor == 0:
                    break

            answer_prob = []
            prob = [score[answer_index] for answer_index in answer_indices]
            prob_max = max(prob)
            answer_prob.append(prob_max)
            score.sort(reverse=True)
            rank = score.index(answer_prob) + 1
        return rank

    def test(self, dataloader, ntriple, skip_dict, num_ent, num_ts):
        """Get time-aware filtered metrics(MRR, Hits@1/3/10).
        Args:
            ntriple: number of the test examples.
            skip_dict: time-aware filter. Get from baseDataset
            num_ent: number of the entities.
        Return: a dict (key -> MRR/HITS@1/HITS@3/HITS@10, values -> float)
        """
        self.model.eval()
        logs = []
        logs_answer_time = []
        logs_answer_entity = []
        logs_simple_entity = []
        logs_simple_time = []
        logs_time_join = []
        logs_first_last = []
        logs_before_after = []

        with torch.no_grad():
            with tqdm.tqdm(total=ntriple,  unit='it', leave=True) as bar:
                current_time = 0
                for input_ids, attention_mask, entity_time_ids_padded, entity_mask_padded, heads, tails, times, start_times, end_times, tails2, types, rels, answers_single, answers_arr, answers_type in dataloader:
                    batch_size = tails.size(0)

                    if self.args.cuda:
                        if type(input_ids) != list:
                            input_ids = input_ids.cuda()
                        if type(attention_mask) != list:
                            attention_mask = attention_mask.cuda()
                        if type(entity_time_ids_padded) != list:
                            entity_time_ids_padded = entity_time_ids_padded.cuda()
                        if type(entity_mask_padded) != list:
                            entity_mask_padded = entity_mask_padded.cuda()
                        if type(heads) != list:
                            heads = heads.cuda()
                        if type(tails) != list:
                            tails = tails.cuda()
                        if type(times) != list:
                            times = times.cuda()
                        if type(start_times) != list:
                            start_times = start_times.cuda()
                        if type(end_times) != list:
                            end_times = end_times.cuda()
                        if type(tails2) != list:
                            tails2 = tails2.cuda()
                        if type(types) != list:
                            types = types.cuda()
                        if type(rels) != list:
                            rels = rels.cuda()
                        if type(answers_single) != list:
                            answers_single = answers_single.cuda()
                        if type(answers_arr) != list:
                            answers_arr = answers_arr.cuda()
                        if type(answers_type) != list:
                            answers_type = answers_type.cuda()
                    current_entities, beam_prob, current_timestamps1, current_timestamps2 = \
                        self.model.beam_search(input_ids, attention_mask, entity_time_ids_padded, entity_mask_padded, heads, tails, times, start_times, end_times, tails2, types, rels, answers_single, answers_arr, answers_type)
                    #

                    if self.args.cuda:
                        current_entities = current_entities.cpu()
                        current_timestamps1 = current_timestamps1.cpu()
                        current_timestamps2 = current_timestamps2.cpu()
                        beam_prob = beam_prob.cpu()

                    current_entities = current_entities.numpy()
                    current_timestamps1 = current_timestamps1.numpy()
                    current_timestamps2 = current_timestamps2.numpy()

                    beam_prob = beam_prob.numpy()

                    HITS1 = 0
                    HITS10 = 0

                    count_answer_time = 0
                    count_answer_entity = 0
                    count_before_after = 0
                    count_first_last = 0
                    count_time_join = 0
                    count_simple_entity = 0
                    count_simple_time = 0

                    HITS1_count_answer_time = 0
                    HITS10_count_answer_time = 0
                    HITS1_count_answer_entity = 0
                    HITS10_count_answer_entity = 0
                    HITS1_count_before_after = 0
                    HITS10_count_before_after = 0
                    HITS1_count_first_last = 0
                    HITS10_count_first_last = 0
                    HITS1_count_time_join = 0
                    HITS10_count_time_join = 0
                    HITS1_count_simple_entity = 0
                    HITS10_count_simple_entity = 0
                    HITS1_count_simple_time = 0
                    HITS10_count_simple_time = 0

                    for i in range(batch_size):
                        candidate_entity_answers = current_entities[i]
                        candidate_entity_timestamps1 = current_timestamps1[i]
                        candidate_entity_timestamps2 = current_timestamps2[i]
                        candidate_score = beam_prob[i]
                        # sort by score from largest to smallest
                        idx = np.argsort(-candidate_score)
                        candidate_entity_answers = candidate_entity_answers[idx]
                        candidate_timestamps1_answers = candidate_entity_timestamps1[idx]
                        candidate_timestamps2_answers = candidate_entity_timestamps2[idx]
                        candidate_score = candidate_score[idx]

                        # remove duplicate entities
                        candidate_entity_answers, ent_idx = np.unique(candidate_entity_answers, return_index=True)
                        candidate_entity_answers = list(candidate_entity_answers)
                        candidate_entity_score = list(candidate_score[ent_idx])

                        # remove duplicate timestamps1,2
                        candidate_timestamps1_answers, t1_idx = np.unique(candidate_timestamps1_answers, return_index=True)
                        candidate_timestamps1_answers = list(candidate_timestamps1_answers)
                        # candidate_timestamps1_answers[0] = -1e10
                        candidate_timestamps1_score = list(candidate_score[t1_idx])

                        candidate_timestamps2_answers, t2_idx = np.unique(candidate_timestamps2_answers, return_index=True)
                        candidate_timestamps2_answers = list(candidate_timestamps2_answers)
                        # candidate_timestamps2_answers[0] = -1e10
                        candidate_timestamps2_score = list(candidate_score[t2_idx])

                        headsim = heads[i].item()
                        relsim = rels[i].item()
                        tailsim = tails[i].item()
                        ts1im = start_times[i].item()
                        ts2im = end_times[i].item()
                        answers_arrim = answers_arr[i].tolist()

                        filter = skip_dict[(headsim, relsim)]# a set of ground truth entities and timestamps
                        tmp_entities = candidate_entity_answers.copy()
                        tmp_entities_prob = candidate_entity_score.copy()
                        tmp_timestamps1 = candidate_timestamps1_answers.copy()
                        tmp_timestamps1_prob = candidate_timestamps1_score.copy()
                        tmp_timestamps2 = candidate_timestamps2_answers.copy()
                        tmp_timestamps2_prob = candidate_timestamps2_score.copy()

                        # time-aware filter     # a set of ground truth entities
                        if answers_type[i] == 'time':
                            answers_arrts = [x - 125726 for x in answers_arrim]
                            for j in range(len(tmp_timestamps1)):

                                if tmp_timestamps1[j] not in filter and tmp_timestamps1[j] not in answers_arrts:
                                    candidate_timestamps1_answers.remove(tmp_timestamps1[j])
                                    candidate_timestamps1_score.remove(tmp_timestamps1_prob[j])

                            ranking_raw1 = self.get_rank(candidate_timestamps1_score, answers_arrts, candidate_timestamps1_answers, num_ts)
                            for j in range(len(tmp_timestamps2)):
                               if tmp_timestamps2[j] not in filter and tmp_timestamps2[j] not in answers_arrts:
                                   candidate_timestamps2_answers.remove(tmp_timestamps2[j])
                                   candidate_timestamps2_score.remove(tmp_timestamps2_prob[j])
                            ranking_raw2 = self.get_rank(candidate_timestamps2_score, answers_arrts, candidate_timestamps2_answers, num_ts)
                            ranking_raw = min(ranking_raw1, ranking_raw2)
                            count_answer_time += 1
                            if ranking_raw <= 1:
                                HITS1_count_answer_time += 1
                            else:
                                HITS10_count_answer_time += 1
                            ranking_raw_time = ranking_raw
                        else:
                            for j in range(len(tmp_entities)):
                                if tmp_entities[j] not in filter and tmp_entities[j] not in answers_arrim:
                                    candidate_entity_answers.remove(tmp_entities[j])
                                    candidate_entity_score.remove(tmp_entities_prob[j])
                            ranking_raw = self.get_rank(candidate_entity_score, answers_arrim, candidate_entity_answers, num_ent)
                            count_answer_entity += 1
                            if ranking_raw <= 1:
                                HITS1_count_answer_entity += 1
                            else:
                                HITS10_count_answer_entity += 1
                            ranking_raw_entity = ranking_raw

                        if types[i] == 'simple_time':
                            count_simple_time += 1
                            if ranking_raw <= 1:
                                HITS1_count_simple_time += 1
                            else:
                                HITS10_count_simple_time += 1
                            ranking_simple_time = ranking_raw
                        elif types[i] == 'simple_entity':
                            count_simple_entity += 1
                            if ranking_raw <= 1:
                                HITS1_count_simple_entity += 1
                            else:
                                HITS10_count_simple_entity += 1
                            ranking_simple_entity = ranking_raw
                        elif types[i] == 'time_join':
                            count_time_join += 1
                            if ranking_raw <= 1:
                                HITS1_count_time_join += 1
                            else:
                                HITS10_count_time_join += 1
                            ranking_time_join = ranking_raw
                        elif types[i] == 'first_last':
                            count_first_last += 1
                            if ranking_raw <= 1:
                                HITS1_count_first_last += 1
                            else:
                                HITS10_count_first_last += 1
                            ranking_first_last = ranking_raw
                        elif types[i] == 'before_after':
                            count_before_after += 1
                            if ranking_raw <= 1:
                                HITS1_count_before_after += 1
                            else:
                                HITS10_count_before_after += 1
                            ranking_before_after = ranking_raw
                        logs.append({
                            'HITS@1': 1.0 if ranking_raw <= 1 else 0.0,
                            'HITS@10': 1.0 if ranking_raw <= 10 else 0.0,
                        })
                        if count_answer_time != 0:
                            logs_answer_time.append({
                                'HITS@1_answer_time': 1.0 if ranking_raw_time <= 1 else 0.0,
                                'HITS@10_answer_time': 1.0 if ranking_raw_time <= 10 else 0.0,
                            })
                        if count_answer_entity != 0:
                            logs_answer_entity.append({
                                'HITS@1_answer_entity': 1.0 if ranking_raw_entity <= 1 else 0.0,
                                'HITS@10_answer_entity': 1.0 if ranking_raw_entity <= 10 else 0.0,
                            })
                        if count_simple_time != 0:
                            logs_simple_time.append({
                                'HITS@1_simple_time': 1.0 if ranking_simple_time <= 1 else 0.0,
                                'HITS@10_simple_time': 1.0 if ranking_simple_time <= 10 else 0.0,
                            })
                        if count_simple_entity != 0:
                            logs_simple_entity.append({
                                'HITS@1_simple_entity': 1.0 if ranking_simple_entity <= 1 else 0.0,
                                'HITS@10_simple_entity': 1.0 if ranking_simple_entity <= 10 else 0.0,
                            })
                        if count_time_join != 0:
                            logs_time_join.append({
                                'HITS@1_time_join': 1.0 if ranking_time_join <= 1 else 0.0,
                                'HITS@10_time_join': 1.0 if ranking_time_join <= 10 else 0.0,
                            })
                        if count_before_after != 0:
                            logs_before_after.append({
                                'HITS@1_before_after': 1.0 if ranking_before_after <= 1 else 0.0,
                                'HITS@10_before_after': 1.0 if ranking_before_after <= 10 else 0.0,
                            })
                        if count_first_last != 0:
                            logs_first_last.append({
                                'HITS@1_first_last': 1.0 if ranking_first_last <= 1 else 0.0,
                                'HITS@10_first_last': 1.0 if ranking_first_last <= 10 else 0.0,
                            })

                    bar.update(batch_size)


                    bar.set_postfix(
                        HITS1='{}'.format(HITS1 / batch_size), HITS10='{}'.format(HITS10 / batch_size)
                    )
                    if count_answer_time != 0:
                        bar.set_postfix(HITS1_answer_time='{}'.format(HITS1_count_answer_time / count_answer_time),
                                        HITS10_answer_time='{}'.format(HITS10_count_answer_time / count_answer_time))

                    if count_answer_entity != 0:
                        bar.set_postfix(HITS1_answer_entity='{}'.format(HITS1_count_answer_entity / count_answer_entity),
                                        HITS10_answer_entity='{}'.format(HITS10_count_answer_entity / count_answer_entity))

                    if count_simple_time != 0:
                        bar.set_postfix(HITS1_simple_time='{}'.format(HITS1_count_simple_time / count_simple_time),
                                        HITS10_simple_time='{}'.format(HITS10_count_simple_time / count_simple_time))

                    if count_simple_entity != 0:
                        bar.set_postfix(HITS1_simple_entity='{}'.format(HITS1_count_simple_entity / count_simple_entity),
                                        HITS10_simple_entity='{}'.format(HITS10_count_simple_entity / count_simple_entity))

                    if count_time_join != 0:
                        bar.set_postfix(HITS1_time_join='{}'.format(HITS1_count_time_join / count_time_join),
                                        HITS10_time_join='{}'.format(HITS10_count_time_join / count_time_join))

                    if count_before_after != 0:
                        bar.set_postfix(HITS1_before_after='{}'.format(HITS1_count_before_after / count_before_after),
                                        HITS10_before_after='{}'.format(HITS10_count_before_after / count_before_after))

                    if count_first_last != 0:
                        bar.set_postfix(HITS1_first_last='{}'.format(HITS1_count_first_last / count_first_last),
                                        HITS10_first_last='{}'.format(HITS10_count_first_last / count_first_last))

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = round(sum([log[metric] for log in logs]) / len(logs),3)

            for metric in logs_answer_time[0].keys():
                metrics[metric] = round(sum([log[metric] for log in logs_answer_time]) / len(logs_answer_time),3)

            for metric in logs_answer_entity[0].keys():
                metrics[metric] = round(sum([log[metric] for log in logs_answer_entity]) / len(logs_answer_entity),3)

            if (len(logs_simple_time) != 0):
                for metric in logs_simple_time[0].keys():
                    metrics[metric] = round(sum([log[metric] for log in logs_simple_time]) / len(logs_simple_time),3)
            else:
                print(00000)

            if (len(logs_simple_entity) != 0):
                for metric in logs_simple_entity[0].keys():
                    metrics[metric] = round(sum([log[metric] for log in logs_simple_entity]) / len(logs_simple_entity),3)
            else:
                print(00000)

            if (len(logs_time_join) != 0):
                for metric in logs_time_join[0].keys():
                    metrics[metric] = round(sum([log[metric] for log in logs_time_join]) / len(logs_time_join),3)
            else:
                print(00000)

            if (len(logs_before_after) != 0):
                for metric in logs_before_after[0].keys():
                    metrics[metric] = round(sum([log[metric] for log in logs_before_after]) / len(logs_before_after),3)
            else:
                print(00000)

            if (len(logs_first_last) != 0):
                for metric in logs_first_last[0].keys():
                    metrics[metric] = round(sum([log[metric] for log in logs_first_last]) / len(logs_first_last),3)
            else:
                print(00000)

            return metrics
