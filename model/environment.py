import networkx as nx
from collections import defaultdict, OrderedDict
import numpy as np
import torch
class Env(object):
    def __init__(self, examples, trainexamples, config, dataset,  args, state_action_space=None):
        """Temporal Knowledge Graph Environment.
        examples: quadruples (subject, relation, object, timestamps);四元组（主体，关系，对象，时间戳）；
        config: config dict;
        state_action_space: Pre-processed action space;
        """

        self.config = config

        self.num_rel = config['num_rel']
        self.ent2id = dataset.all_dicts["ent2id"]
        self.rel2id = dataset.all_dicts["rel2id"]
        self.ts2id = dataset.all_dicts["ts2id"]
        self.newexamples = self.new_examples(examples)
        self.timestamps, self.rev_timestamps = self.get_all_timestamps(self.newexamples)
        self.skip_dict = self.get_skipdict(self.newexamples)
        self.newtrainexamples = self.new_trainexamples(trainexamples)
        self.train_entities = self.appear_entities(self.newtrainexamples)
        self.train_timestamps = self.appear_timestamps(self.newtrainexamples)
        self.graph, self.label2nodes = self.build_graph(self.newexamples)
        self.NO_OP = config['num_rel']# Stay in place; No Operation
        self.ePAD = config['num_ent']  # Padding entity
        self.rPAD = config['num_rel'] * 2 + 1  # Padding relation.
        self.tPAD = 9621  # Padding time
        self.state_action_space = state_action_space  # Pre-processed action space
        if state_action_space:
            self.state_action_space_key = self.state_action_space.keys()

    def new_examples(self, examples):
        newexamples = []
        for example in examples:
            src = self.ent2id["Q" + str(example[0])]
            rel = self.rel2id["P" + str(example[1])]
            dst = self.ent2id["Q" + str(example[2])]
            start_time = self.ts2id[(example[3], 0, 0)]
            end_time = self.ts2id[(example[4], 0, 0)]
            newexample = [src, rel, dst, start_time, end_time]
            newexamples.append(newexample)
        return newexamples
    def new_trainexamples(self, trainexamples):
        newtrainexamples = []
        for trainexample in trainexamples:
            src = self.ent2id["Q" + str(trainexample[0])]
            rel = self.rel2id["P" + str(trainexample[1])]
            dst = self.ent2id["Q" + str(trainexample[2])]
            start_time = self.ts2id[(trainexample[3], 0, 0)]
            end_time = self.ts2id[(trainexample[4], 0, 0)]
            trainexample = [src, rel, dst, start_time, end_time]
            newtrainexamples.append(trainexample)
        return newtrainexamples
    def build_graph(self, examples):
        graph = nx.MultiDiGraph()
        label2nodes = defaultdict(set)
        examples.sort(key=lambda x: x[3], reverse=True)  # Reverse chronological order
        for example in examples:
            src = example[0]
            rel = example[1]
            dst = example[2]
            start_time = example[3]
            end_time = example[4]


            src_node = (src, start_time, end_time)
            dst_node = (dst, start_time, end_time)

            if src_node not in label2nodes[src]:
                graph.add_node(src_node, label=src)
            if dst_node not in label2nodes[dst]:
                graph.add_node(dst_node, label=dst)

            graph.add_edge(src_node, dst_node, relation=rel, start_time=start_time, end_time=end_time)
            graph.add_edge(dst_node, src_node, relation=rel+self.num_rel+1, start_time=start_time, end_time=end_time)
            label2nodes[src].add(src_node)
            label2nodes[dst].add(dst_node)

        return graph, label2nodes

    def get_all_timestamps(self, examples):
        timestamps = defaultdict(set)
        timestamps_tail = defaultdict(set)
        for example in examples:
            head = example[0]
            rel = example[1]
            tail = example[2]
            start_time = example[3]
            end_time = example[4]

            timestamps[(start_time, end_time)].add(head)
            timestamps_tail[(start_time, end_time)].add(head)
        sorted_timestamps = sorted(timestamps.items(), key=lambda x: (x[0][0], x[0][1]))
        timestamps = OrderedDict(sorted_timestamps)
        rev_sorted_timestamps = sorted(timestamps.items(), key=lambda x: (x[0][0], -x[0][1]))
        rev_timestamps = OrderedDict(rev_sorted_timestamps)
        return timestamps,  rev_timestamps

    def get_state_actions_space_complete(self, entity, time1, time2,  max_action_num=None):
        """Get the action space of the current state.
        Args:
            entity: The entity of the current state;
            time1: Maximum timestamp for candidate actions;
            time2:
            max_action_num: Maximum number of events stored;
        Return:
            numpy array，shape: [number of events，3], (relation, dst, time)
        """
        if self.state_action_space:
           if (entity, time1, time2) in self.state_action_space_key:
               return self.state_action_space[(entity, time1, time2)]
        nodes = list(self.label2nodes[entity].copy())

        det1 = []
        det2 = []

        for node in nodes:
            dt1 = torch.abs(torch.tensor(time1, dtype=torch.float32) - torch.tensor(node[1], dtype=torch.float32))
            dt2 = (torch.tensor(node[2], dtype=torch.float32) - torch.tensor(time2, dtype=torch.float32))
            det1.append(dt1)
            det2.append(dt2)
        sorted_nodes = sorted(zip(det1, det2, nodes), key=lambda x: (x[0], x[1]))
        sorted_nodes = [node for _, _, node in sorted_nodes]

        actions_space = []
        i = 0
        for node in sorted_nodes:
            #    dst_node = (dst, start_time, end_time)
            for src, dst, rel in self.graph.out_edges(node, data=True):
                actions_space.append((rel['relation'], dst[0], dst[1], dst[2]))
                i += 1
                if max_action_num and i >= max_action_num:
                    break
            if max_action_num and i >= max_action_num:
                break

        return np.array(list(actions_space), dtype=np.dtype('int32'))


    def next_actions(self, entites, time1, time2, query_times, types, max_action_num=200, first_step=False):
        """Get the current action space. There must be an action that stays at the current position in the action space.
        Args:
            entites: torch.tensor, shape: [batch_size], the entity where the agent is currently located;
            time1: torch.tensor, shape: [batch_size], the timestamp of the current entity;
            time2: torch.tensor
            query_times: torch.tensor, shape: [batch_size], the timestamp of query;
            max_action_num: The size of the action space;
            first_step: Is it the first step for the agent.
        Return: torch.tensor, shape: [batch_size, max_action_num, 3], (relation, entity, time)
        """

        if self.config['cuda']:
            entites = entites.cpu()
            time1 = time1.cpu()
            time2 = time2.cpu()
            query_times = query_times.cpu()
        entites = entites.numpy()
        time1 = time1.numpy()
        time2 = time2.numpy()
        query_times = query_times.numpy()
        actions = self.get_padd_actions(entites, time1, time2, query_times, types, max_action_num, first_step)
        if self.config['cuda']:
            actions = torch.tensor(actions, dtype=torch.long, device='cuda')
        else:
            actions = torch.tensor(actions, dtype=torch.long)

        return actions

    def get_padd_actions(self, entites, time1,time2, query_times, types, max_action_num=200, first_step=False):
        """Construct the model input array.
        entites,
         time1
         time2,
         query_times,

         max_action_num=200,
         first_step=False
        If the optional actions are greater than the maximum number of actions, then sample,
        otherwise all are selected, and the insufficient part is pad.
        """
        actions = np.ones((entites.shape[0], max_action_num, 4), dtype=np.dtype('int32'))
        actions[:, :, 0] *= self.rPAD
        actions[:, :, 1] *= self.ePAD
        actions[:, :, 2] *= self.tPAD
        actions[:, :, 3] *= self.tPAD

        for i in range(entites.shape[0]):
            # NO OPERATION
            actions[i, 0, 0] = self.NO_OP
            actions[i, 0, 1] = entites[i]
            actions[i, 0, 2] = time1[i]
            actions[i, 0, 3] = time2[i]

            action_array = self.get_state_actions_space_complete(entites[i], time1[i], time2[i])
            if action_array.shape[0] == 0:
                continue
            start_idx = 1
            if first_step:
                start_idx = 0

            if action_array.shape[0] > (max_action_num - start_idx):
                # Sample. Take the latest events.
                actions[i, start_idx:] = action_array[:max_action_num-start_idx]

            else:
                actions[i, start_idx:action_array.shape[0]+start_idx] = action_array
        return actions

    def get_skipdict(self, examples):
        """Used for time-dependent filtered metrics.
        return: a dict [key -> (entity, relation, timestamp),  value -> a set of ground truth entities]
        """
        filters = defaultdict(set)
        for example in examples:
            head = example[0]
            rel = example[1]
            tail = example[2]
            start_time = example[3]
            end_time = example[4]
            filters[(head, rel)].add(tail)
            filters[(head, rel)].add(start_time)
            filters[(head, rel)].add(end_time)
            filters[(tail, rel+self.num_rel+1)].add(head)
            filters[(tail, rel + self.num_rel + 1)].add(start_time)
            filters[(tail, rel + self.num_rel + 1)].add(end_time)
        return filters

    def appear_entities(self, trainexamples):
        train_entities = set()
        for query in trainexamples:
            train_entities.add(query[0])
            train_entities.add(query[2])
        return train_entities

    def appear_timestamps(self, trainexamples):
        train_timestamps = set()
        for query in trainexamples:
            train_timestamps.add(query[3])
            train_timestamps.add(query[4])
        return train_timestamps
