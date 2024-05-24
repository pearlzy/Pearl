import torch
import torch.nn as nn
from model.agent import QA_SubGTR
import numpy as np

class PearsonCorrelation(nn.Module):
    def forward(self,tensor_1,tensor_2):
        x = tensor_1
        y = tensor_2

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return cost

class Episode(nn.Module):

    def __init__(self, env, agent, config, args):

        super(Episode, self).__init__()
        self.config = config
        self.env = env
        self.agent = agent
        self.path_length = config['path_length']
        self.num_rel = config['num_rel']
        self.max_action_num = config['max_action_num']
        self.qa_model = QA_SubGTR(args)
        self.pearson = PearsonCorrelation()
    def forward(self, input_ids, attention_mask, entity_time_ids_padded, entity_mask_padded, heads, tails, times, start_times, end_times, tails2, types, rels, answers_single, answers_arr, answers_type):
        """
        Args:
            query_entities: [batch_size]
            query_timestamps: [batch_size]
            query_relations: [batch_size]

        Return:
            all_loss: list
            all_logits: list
            all_actions_idx: list
            current_entities: torch.tensor, [batch_size]
            current_timestamps: torch.tensor, [batch_size]

        """
        a = [input_ids, attention_mask, entity_time_ids_padded, entity_mask_padded, heads, tails, times, start_times, end_times, tails2, types, rels, answers_single, answers_arr, answers_type]
        query_entities1, query_entities2, query_timestamps, query_relations = heads, tails, times, rels

        current_entites1 = query_entities1
        current_entites2 = query_entities2

        current_entites1_kg_embedding = self.qa_model.embedding(current_entites1)
        current_entites2_kg_embedding = self.qa_model.embedding(current_entites2)
        context_qa = self.qa_model.forward(a)[4]

        similarities1 = torch.zeros(current_entites1.shape[0], dtype=torch.float32).to('cuda')
        similarities2 = torch.zeros(current_entites1.shape[0], dtype=torch.float32).to('cuda')
        current_entites = torch.zeros(current_entites1.shape[0], dtype=torch.float32).to('cuda')
        for i in range(current_entites1.shape[0]):
            similarities1[i] = self.pearson(context_qa[i], current_entites1_kg_embedding[i])
            similarities2[i] = self.pearson(context_qa[i], current_entites2_kg_embedding[i])
            current_entites[i] = torch.where(similarities1[i] > similarities2[i],  current_entites1[i], current_entites2[i])

        query_entities_embeds = self.agent.dynamicEmbedding(current_entites, torch.zeros_like(query_timestamps)) # [batch_size, ent_dim]
        query_relations_embeds = self.agent.dynamicEmbedding(query_relations, torch.zeros_like(query_timestamps))  # [batch_size, rel_dim=100]

        current_timestamps = start_times
        current_timestamps2 = end_times
        prev_relations = torch.ones_like(query_relations) * self.num_rel  # NO_OP
        prev_relations = torch.ones_like(query_relations) * 407
        all_loss = []
        all_logits = []
        all_actions_idx = []
        all_actions_scores = []
        self.agent.policy_step.set_hidden(query_relations.shape[0])
        for t in range(self.path_length):
            if t == 0:
                first_step = True
            else:
                first_step = False


            action_space = self.env.next_actions(
                current_entites,
                current_timestamps,
                current_timestamps2,
                query_timestamps,
                types,
                self.max_action_num,
                first_step
            )

            loss, logits, action_id, action_prob_new, context_qa = self.agent(
                a,
                prev_relations,
                current_entites,
                current_timestamps,
                current_timestamps2,
                query_relations_embeds,
                query_entities_embeds,
                query_timestamps,
                action_space,
            )

            chosen_relation = torch.gather(action_space[:, :, 0], dim=1, index=action_id).reshape(action_space.shape[0])
            chosen_entity = torch.gather(action_space[:, :, 1], dim=1, index=action_id).reshape(action_space.shape[0])
            chosen_entity_timestamps1 = torch.gather(action_space[:, :, 2], dim=1, index=action_id).reshape(action_space.shape[0])
            chosen_entity_timestamps2 = torch.gather(action_space[:, :, 3], dim=1, index=action_id).reshape(action_space.shape[0])

            all_loss.append(loss)
            all_logits.append(logits)
            all_actions_idx.append(action_id)
            all_actions_scores.append(torch.gather(action_prob_new, dim=1, index=action_id).reshape(action_space.shape[0]))

            current_entites = chosen_entity
            current_timestamps = chosen_entity_timestamps1
            current_timestamps2 = chosen_entity_timestamps2

            prev_relations = chosen_relation


        return all_loss, all_logits, all_actions_idx, current_entites, current_timestamps, current_timestamps2, all_actions_scores[self.config['path_length']-1]


    def beam_search(self,input_ids, attention_mask, entity_time_ids_padded, entity_mask_padded, heads, tails, times, start_times, end_times, tails2, types, rels, answers_single, answers_arr, answers_type):
        """
        Args:
            query_entities: [batch_size]
            query_timestamps: [batch_size]
            query_relations: [batch_size]

        Return:
               current_entites: [batch_size, test_rollouts_num]
               beam_prob: [batch_size, test_rollouts_num]

        """
        a = [input_ids, attention_mask, entity_time_ids_padded, entity_mask_padded, heads, tails, times, start_times,
             end_times, tails2, types, rels, answers_single, answers_arr, answers_type]
        query_entities1, query_entities2, query_timestamps, query_relations = heads, tails, times, rels
        batch_size = heads.shape[0]

        current_entites1 = query_entities1
        current_entites2 = query_entities2
        current_entites1_kg_embedding = self.qa_model.embedding(current_entites1)
        current_entites2_kg_embedding = self.qa_model.embedding(current_entites2)
        context_qa = self.qa_model.forward(a)[4]
        similarities1 = torch.zeros(current_entites1.shape[0], dtype=torch.float32).to('cuda')
        similarities2 = torch.zeros(current_entites1.shape[0], dtype=torch.float32).to('cuda')
        current_entites = torch.zeros(current_entites1.shape[0], dtype=torch.float32).to('cuda')
        for i in range(current_entites1.shape[0]):
             similarities1[i] = self.pearson(context_qa[i], current_entites1_kg_embedding[i])
             similarities2[i] = self.pearson(context_qa[i], current_entites2_kg_embedding[i])
             current_entites[i] = torch.where(similarities1[i] > similarities2[i], current_entites1[i], current_entites2[i])

        self.agent.policy_step.set_hidden(batch_size)

        # todo 2024/5/16
        query_entities_embeds = self.agent.dynamicEmbedding(current_entites,
                                                            torch.zeros_like(query_timestamps))  # [batch_size, ent_dim]
        query_relations_embeds = self.agent.dynamicEmbedding(query_relations, torch.zeros_like(
            query_timestamps))  # [batch_size, rel_dim=100]

        current_timestamps = start_times
        current_timestamps2 = end_times
        prev_relations = torch.ones_like(query_relations) * self.num_rel  # NO_OP # 初始化为NO_OP操作

        prev_relations = torch.ones_like(query_relations) *408  # NO_OP # 初始化为NO_OP操作

        action_space = self.env.next_actions(
            current_entites,
            current_timestamps,
            current_timestamps2,
            query_timestamps,
            types,
            self.max_action_num)

        loss, logits, action_id, _, _ = self.agent(
            a,
            prev_relations,
            current_entites,
            current_timestamps,
            current_timestamps2,
            query_relations_embeds,
            query_entities_embeds,
            query_timestamps,
            action_space,
        ) # logits.shape: [batch_size, max_action_num]


        action_space_size = action_space.shape[1]


        if self.config['beam_size'] > action_space_size:
            beam_size = action_space_size
        else:
            beam_size = self.config['beam_size']


        beam_log_prob, top_k_action_id = torch.topk(logits, beam_size, dim=1)  # beam_log_prob.shape [batch_size, beam_size]

        beam_log_prob = beam_log_prob.reshape(-1)  # [batch_size * beam_size]

        current_entites = torch.gather(action_space[:, :, 1], dim=1, index=top_k_action_id).reshape(-1)  # [batch_size * beam_size]
        current_timestamps = torch.gather(action_space[:, :, 2], dim=1, index=top_k_action_id).reshape(-1)  # [batch_size * beam_size]
        current_timestamps2 = torch.gather(action_space[:, :, 3], dim=1, index=top_k_action_id).reshape(-1)
        prev_relations = torch.gather(action_space[:, :, 0], dim=1, index=top_k_action_id).reshape(-1)  # [batch_size * beam_size]
        self.agent.policy_step.hx = self.agent.policy_step.hx.repeat(1, 1, beam_size).reshape([batch_size * beam_size, -1])  # [batch_size * beam_size, state_dim]
        self.agent.policy_step.cx = self.agent.policy_step.cx.repeat(1, 1, beam_size).reshape([batch_size * beam_size, -1])  # [batch_size * beam_size, state_dim]

        beam_tmp = beam_log_prob.repeat([action_space_size, 1]).transpose(1, 0)  # [batch_size * beam_size, max_action_num]

        for t in range(1, self.path_length):

            query_timestamps_roll = query_timestamps.repeat(beam_size, 1).permute(1, 0).reshape(-1)
            query_entities_embeds_roll = query_entities_embeds.repeat(1, 1, beam_size)
            query_entities_embeds_roll = query_entities_embeds_roll.reshape([batch_size * beam_size, -1])  # [batch_size * beam_size, ent_dim]

            query_relations_embeds_roll = query_relations_embeds.repeat(1, 1, beam_size)
            query_relations_embeds_roll = query_relations_embeds_roll.reshape([batch_size * beam_size, -1])  # [batch_size * beam_size, rel_dim]

            types = np.array(types)
            query_types_roll = types.repeat(beam_size, axis=0)   # [batch_size * beam_size, rel_dim]

            action_space = self.env.next_actions(current_entites, current_timestamps, current_timestamps2, query_timestamps_roll, query_types_roll, self.max_action_num)

            loss, logits, action_id, action_prob_new,context_qa = self.agent(
                a,
                prev_relations,
                current_entites,
                current_timestamps,
                current_timestamps2,
                query_relations_embeds_roll,
                query_entities_embeds_roll,
                query_timestamps_roll,
                action_space
            )# logits.shape [bs * rollouts_num, max_action_num] # logits的形状 [bs * rollouts_num, max_action_num]

            hx_tmp = self.agent.policy_step.hx.reshape(batch_size, beam_size, -1)
            cx_tmp = self.agent.policy_step.cx.reshape(batch_size, beam_size, -1)

            beam_tmp = beam_log_prob.repeat([action_space_size, 1]).transpose(1, 0) # [batch_size * beam_size, max_action_num]
            beam_tmp += logits
            beam_tmp = beam_tmp.reshape(batch_size, -1)  # [batch_size, beam_size * max_actions_num]

            if action_space_size * beam_size >= self.config['beam_size']:
                beam_size = self.config['beam_size']
            else:
                beam_size = action_space_size * beam_size

            top_k_log_prob, top_k_action_id = torch.topk(beam_tmp, beam_size, dim=1)  # [batch_size, beam_size]
            offset = torch.div(top_k_action_id, action_space_size, rounding_mode='floor')
            offset = offset.unsqueeze(-1).repeat(1, 1, self.config['state_dim']*2)  # [batch_size, beam_size]
            self.agent.policy_step.hx = torch.gather(hx_tmp, dim=1, index=offset)
            self.agent.policy_step.hx = self.agent.policy_step.hx.reshape([batch_size * beam_size, -1])
            self.agent.policy_step.cx = torch.gather(cx_tmp, dim=1, index=offset)
            self.agent.policy_step.cx = self.agent.policy_step.cx.reshape([batch_size * beam_size, -1])

            current_entites = torch.gather(action_space[:, :, 1].reshape(batch_size, -1), dim=1, index=top_k_action_id).reshape(-1)
            current_timestamps = torch.gather(action_space[:, :, 2].reshape(batch_size, -1), dim=1, index=top_k_action_id).reshape(-1)
            current_timestamps2 = torch.gather(action_space[:, :, 3].reshape(batch_size, -1), dim=1, index=top_k_action_id).reshape(-1)
            prev_relations = torch.gather(action_space[:, :, 0].reshape(batch_size, -1), dim=1, index=top_k_action_id).reshape(-1)

            beam_log_prob = top_k_log_prob.reshape(-1)  # [batch_size * beam_size]

        return action_space[:, :, 1].reshape(batch_size, -1), beam_tmp, action_space[:, :, 2].reshape(batch_size, -1), action_space[:, :, 3].reshape(batch_size, -1)
