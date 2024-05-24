
import torch
from torch import nn
import numpy as np
from transformers import DistilBertModel
import torch.nn.functional as F
from dataset.utils import loadTkbcModel_complex

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.linear_query = nn.Linear(hidden_size, hidden_size)
        self.linear_key = nn.Linear(hidden_size, hidden_size)
        self.linear_value = nn.Linear(hidden_size, hidden_size)

    def forward(self, last_hidden_state):
        # 分割成多个头
        batch_size = last_hidden_state.shape[0]
        query = self.linear_query(last_hidden_state).view(batch_size, -1, self.num_heads, self.head_dim) #[128,20,8,96]
        key = self.linear_key(last_hidden_state).view(batch_size, -1, self.num_heads, self.head_dim) #[128,20,8,96]
        value = self.linear_value(last_hidden_state).view(batch_size, -1, self.num_heads, self.head_dim) #[128,20,8,96]

        attention_weights = torch.softmax(torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5), dim=-1) #[128,20,8,8]

        weighted_sum = torch.matmul(attention_weights, value) #[128,20,8,96]

        weighted_sum = weighted_sum.view(batch_size, -1, self.num_heads * self.head_dim)  #[128,20,768]

        return weighted_sum


class QA_SubGTR(nn.Module):
    def __init__(self, args):
        super().__init__()
        tkbc_model: object = loadTkbcModel_complex('./models/{dataset_name}/kg_embeddings/{tkbc_model_file}'.format(
                     dataset_name=args.dataset_name, tkbc_model_file=args.tkbc_model_file
                ))
        self.tkbc_embedding_dim = tkbc_model.embeddings[0].weight.shape[1]    #tkbc_embedding_dim =512
        self.sentence_embedding_dim = 768  # hardwired from
        self.pretrained_weights = './distil_bert/'
        self.lm_model = DistilBertModel.from_pretrained(self.pretrained_weights)

        if args.lm_frozen == 1:
            print('Freezing LM params')
            for param in self.lm_model.parameters():
                param.requires_grad = False
        else:
            print('Unfrozen LM params')
        self.project_sentence_to_transformer_dim = nn.Linear(self.sentence_embedding_dim, 512)
        # TKG embeddings
        self.tkbc_model = tkbc_model
        num_entities = tkbc_model.embeddings[0].weight.shape[0]
        num_times = tkbc_model.embeddings[2].weight.shape[0]
        ent_emb_matrix = tkbc_model.embeddings[0].weight.data
        time_emb_matrix = tkbc_model.embeddings[2].weight.data

        full_embed_matrix = torch.cat([ent_emb_matrix, time_emb_matrix], dim=0)
        # +1 is for padding idx
        self.entity_time_embedding = nn.Embedding(num_entities + num_times + 1,
                                                  self.tkbc_embedding_dim,
                                                  padding_idx=num_entities + num_times)
        self.entity_time_embedding.weight.data[:-1, :].copy_(full_embed_matrix)
        self.num_entities = num_entities
        self.num_times = num_times
        self.ent_emb_matrix = ent_emb_matrix
        self.time_emb_matrix = time_emb_matrix
        if args.frozen == 1:
            print('Freezing entity/time embeddings')
            self.entity_time_embedding.weight.requires_grad = False
            for param in self.tkbc_model.parameters():
                param.requires_grad = False
        else:
            print('Unfrozen entity/time embeddings')

        # position embedding for transformer
        self.max_seq_length = 100  # randomly defining max length of tokens for question
        self.position_embedding = nn.Embedding(self.max_seq_length, self.tkbc_embedding_dim)

        num_heads = 8
        self.multihead_attention = MultiHeadSelfAttention(self.tkbc_embedding_dim, num_heads)

        return

    def invert_binary_tensor(self, tensor):
        ones_tensor = torch.ones(tensor.shape, dtype=torch.long).cuda()
        inverted = ones_tensor - tensor
        return inverted

    def embedding(self, entities):
        entities = entities.cuda().long()
        entities_embedding = self.entity_time_embedding(entities).cuda()
        # entities_embedding = self.tkbc_model.embeddings[1](entities)
        return entities_embedding

    def time_embedding(self, times):   #query_times
        times = times.cuda().long()
        times_emb = self.entity_time_embedding(times)
        return times_emb

    def forward(self, a):
        # Tokenized questions, where entities are masked from the sentence to have TKG embeddings
        question_tokenized = a[0].cuda().long()  # torch.Size([1, 10])
        question_attention_mask = a[1].cuda().long()
        entities_times_padded = a[2].cuda().long()
        entity_mask_padded = a[3].cuda().long()

        # Annotated entities/timestamps

        t1 = a[7].cuda().long()
        t2 = a[8].cuda().long()

        # One extra entity for new before & after question type
        self.entity_time_embedding.to('cuda')

        # Hard Supervision
        t1_emb = self.tkbc_model.embeddings[2](t1)
        t2_emb = self.tkbc_model.embeddings[2](t2)

        # entity embeddings to replace in sentence
        entity_time_embedding = self.entity_time_embedding(entities_times_padded)  # torch.Size([1, 10, 512])

        # context-aware step
        outputs = self.lm_model(question_tokenized, attention_mask=question_attention_mask)
        last_hidden_states = outputs.last_hidden_state  # torch.Size([batchsize, 17, 768])

        # entity-aware step
        # 768->512
        question_embedding = self.project_sentence_to_transformer_dim(last_hidden_states)  # torch.Size([batchsize, 17, 512])

        entity_mask = entity_mask_padded.unsqueeze(-1).expand(question_embedding.shape)   # torch.Size([batchsize, 17, 512])
        masked_question_embedding = question_embedding * entity_mask  # set entity positions 0   # torch.Size([batchsize, 17, 512])

        entity_time_embedding_projected = entity_time_embedding
        # time-aware step
        time_pos_embeddings1 = t1_emb.unsqueeze(0).transpose(0, 1)  # torch.Size([1, 1, 512])
        time_pos_embeddings1 = time_pos_embeddings1.expand(entity_time_embedding_projected.shape)  # torch.Size([1, 10, 512])

        time_pos_embeddings2 = t2_emb.unsqueeze(0).transpose(0, 1)
        time_pos_embeddings2 = time_pos_embeddings2.expand(entity_time_embedding_projected.shape)

        entity_time_embedding_projected = entity_time_embedding_projected + time_pos_embeddings1 + time_pos_embeddings2  # torch.Size([1, 10, 512])


        masked_entity_time_embedding = entity_time_embedding_projected * self.invert_binary_tensor(entity_mask)
        combined_embed = masked_question_embedding + masked_entity_time_embedding   # torch.Size([batchsize, 17, 512])

        # also need to add position embedding
        sequence_length = combined_embed.shape[1]
        v = np.arange(0, sequence_length, dtype=np.float64)   # torch.Size(17,)
        indices_for_position_embedding = torch.from_numpy(v).cuda().long()   # torch.Size(17,)
        position_embedding = self.position_embedding(indices_for_position_embedding)   # torch.Size(17,512)
        position_embedding = position_embedding.unsqueeze(0).expand(combined_embed.shape)   # torch.Size(batchsize, 17,512)

        combined_embed = combined_embed + position_embedding   # torch.Size(batchsize, 17,512)
        weighted_representation = self.multihead_attention(combined_embed)   # torch.Size(17, batchsize, 512)
        question_mp = F.max_pool1d(weighted_representation.transpose(1, 2), combined_embed.shape[1]).squeeze(2)

        entity_time_embedding_projected = entity_time_embedding_projected.transpose(1, 2)
        entity_time_embedding_projected = F.max_pool1d(entity_time_embedding_projected, entity_time_embedding_projected.size(2)).squeeze(2)

        return t1_emb, t2_emb, masked_entity_time_embedding, entity_time_embedding_projected, question_mp

class HistoryEncoder(nn.Module):
    def __init__(self, config):
        super(HistoryEncoder, self).__init__()
        self.config = config
        self.lstm = torch.nn.LSTM(input_size=(config['action_dim'] + 512), hidden_size=config['state_dim'], num_layers=1,
                                  bidirectional=True, batch_first=True)
        self.set_hidden(batch_size=config['batch_size'])

    def set_hidden(self, batch_size):
        """Set hidden layer parameters. Initialize to 0"""
        if self.config['cuda']:
            self.hx = torch.zeros(batch_size, 2 * self.config['state_dim'],  device='cuda')  # Note: Doubled state_dim due to bidirectional LSTM
            self.cx = torch.zeros(batch_size, 2 * self.config['state_dim'], device='cuda')  # Note: Doubled state_dim due to bidirectional LSTM
        else:
            self.hx = torch.zeros(batch_size, 2 * self.config['state_dim'])  # Note: Doubled state_dim due to bidirectional LSTM
            self.cx = torch.zeros(batch_size, 2 * self.config['state_dim'])  # Note: Doubled state_dim due to bidirectional LSTM

    def forward(self, prev_actions, mask):
        """mask: True if NO_OP. ON_OP does not affect history coding results"""
        outputs, (self.hx_, self.cx_) = self.lstm(prev_actions)
        # Combine forward and backward hidden states
        self.hx_ = torch.cat((self.hx_[0], self.hx_[1]), dim=-1)
        self.cx_ = torch.cat((self.cx_[0], self.cx_[1]), dim=-1)
        if self.hx.shape[0] != mask.shape[0]:
            self.hx = self.hx.repeat(int(mask.shape[0]/self.hx.shape[0]), 1)
        if self.cx.shape[0] != mask.shape[0]:
            self.cx = self.cx.repeat(int(mask.shape[0]/self.cx.shape[0]), 1)
        mask = mask.unsqueeze(2).expand(-1, -1, 2).reshape(mask.size(0), -1)
        self.hx = torch.where(mask, self.hx, self.hx_)
        self.cx = torch.where(mask, self.cx, self.cx_)
        return self.hx

class PolicyMLP(nn.Module):
    def __init__(self, config):
        super(PolicyMLP, self).__init__()
        # todo: 6.0
        self.mlp_l1 = nn.Linear(config['mlp_input_dim'] , config['mlp_hidden_dim'], bias=True)   # 'mlp_input_dim': args.ent_dim + args.rel_dim + args.state_dim,
        self.mlp_l2 = nn.Linear(config['mlp_hidden_dim'], config['action_dim'], bias=True)

    def forward(self, state_query):
        hidden = torch.relu(self.mlp_l1(state_query))
        output = self.mlp_l2(hidden).unsqueeze(1)
        return output


class DynamicEmbedding(nn.Module):
    def __init__(self, n_ent, dim_ent, dim_t, args):
        super(DynamicEmbedding, self).__init__()
        self.w = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim_t))).float())
        self.b = torch.nn.Parameter(torch.zeros(dim_t).float())
        self.entity_embedding = nn.Embedding(125726 + 9621 + 1, 80)
        self.w = torch.nn.Parameter(self.w.cuda())
        self.b = torch.nn.Parameter(self.b.cuda())

        self.tkbc_linearlayer = nn.Linear(100, dim_ent - dim_t)
    def forward(self, entities, dt):

        dt = dt.unsqueeze(-1)
        dt = dt.to(torch.float)
        batch_size = dt.size(0)

        seq_len = dt.size(1)
        dt = dt.view(batch_size, seq_len, 1)
        t = torch.cos(self.w.view(1, 1, -1) * dt + self.b.view(1, 1, -1))
        t = t.squeeze(1)  # [batch_size, time_dim]   #t [batch,20]
        e = self.entity_embedding(entities.long())
        new_entity = torch.cat((e, t), -1)
        return new_entity

class StaticEmbedding(nn.Module):
    def __init__(self, n_ent, dim_ent):
        super(StaticEmbedding, self).__init__()
        self.ent_embs = nn.Embedding(n_ent, dim_ent)

    def forward(self, entities, timestamps=None):
        return self.ent_embs(entities)

class Agent(nn.Module):
    def __init__(self, config, args):
        super(Agent, self).__init__()
        self.num_rel = config['num_rel'] * 2 + 2
        self.config = config

        # [0, num_rel) -> normal relations; num_rel -> stay in place，(num_rel, num_rel * 2] reversed relations.
        self.NO_OP = config['num_rel'] * 2 + 2  # Stay in place; No Operation
        self.ePAD = config['num_ent']  # Padding entity
        self.rPAD = config['num_rel'] * 2 + 1  # Padding relation
        self.tPAD = 9621   # Padding time

        self.qa_subgtr = QA_SubGTR(args).cuda()
        # self.dropout = torch.nn.Dropout(0.3)
        # self.tkbc_embedding_dim = self.qa_subgtr.tkbc_embedding_dim

        # todo 5/20 简化参数
        self.bn1 = torch.nn.BatchNorm1d(config['ent_dim'])
        self.bn2 = torch.nn.BatchNorm1d(config['max_action_num'])
        self.bn3 = torch.nn.BatchNorm1d(config['time_dim'])
        self.entity_time_embedding_projected_linearlayer = nn.Linear(512, config['time_dim'])
        if self.config['entities_embeds_method'] == 'dynamic':
            self.dynamicEmbedding = DynamicEmbedding(config['num_ent']+1, config['ent_dim'], config['time_dim'], args)
        else:
            self.ent_embs = StaticEmbedding(config['num_ent']+1, config['ent_dim'])
        self.rel_embs = nn.Embedding(config['num_ent'], config['rel_dim'])
        self.action_state_linearlayer = nn.Linear(config['state_dim']+config['ent_dim']+config['rel_dim']+config['time_dim'],config['ent_dim']+config['rel_dim']+config['time_dim']+config['time_dim'])
        self.policy_step = HistoryEncoder(config)
        self.policy_mlp = PolicyMLP(config)

        self.score_weighted_fc = nn.Linear(config['ent_dim'] * 4 + config['time_dim'] * 4, 1, bias=True)
        self.lstm_layer = nn.Linear(config['ent_dim'] * 2, config['state_dim'])
    def forward(self, a, prev_relation, current_entities, start_timestamps1, end_timestamps2, query_relation, query_entity, query_timestamps, action_space):
        """
        Args:
            a
            prev_relation: [batch_size]
            current_entities: [batch_size]
            current_timestamps: [batch_size]
            query_relation: embeddings of query relation，[batch_size, rel_dim]
            query_entity: embeddings of query entity, [batch_size, ent_dim]
            query_timestamps:query_timestamps[batch_size]
            current_timestamps2
            action_space: [batch_size, max_actions_num, 4] (relations, entities, timestamps1，timestamps2)
        """
        # embeddings
        query_timestamps = (query_timestamps - 125726)

        current_delta_time = (end_timestamps2 - start_timestamps1)


        current_embds = self.bn1(self.dynamicEmbedding(current_entities, current_delta_time))   #DynamicEmbedding()
        prev_relation_embds = self.bn1(self.dynamicEmbedding(prev_relation, current_delta_time))  # [batch_size, rel_dim]

        values_to_set_zero = torch.tensor([407, 125726, 9621, 9621]).to('cuda')

        condition = torch.all(action_space == values_to_set_zero.view(1, 1, -1), dim=-1)
        action_space[condition, :] = 0
        # Pad Mask
        pad_mask = torch.zeros_like(action_space[:, :, 0]) # [batch_size, action_number]  self.rPAD = config['num_rel'] * 2 + 1  # Padding relation
        pad_mask = torch.eq(action_space[:, :, 0], pad_mask)  # [batch_size, action_number]

        # History Encode
        NO_OP_mask = torch.eq(prev_relation, torch.ones_like(prev_relation) * self.NO_OP)  # [batch_size]
        NO_OP_mask = NO_OP_mask.repeat(self.config['state_dim'], 1).transpose(1, 0)  # [batch_size, state_dim]
        mask = NO_OP_mask
        #action_num
        action_num = action_space.size(1)
        match_query_relations_entities = torch.matmul(prev_relation_embds, current_embds.transpose(0, 1))
        context_qa = self.qa_subgtr(a)[4]
        if context_qa.shape[0] != current_embds.shape[0]:
            context_qa = context_qa.repeat(int(current_embds.shape[0]/context_qa.shape[0]), 1)

        match_vecter = torch.matmul(match_query_relations_entities, context_qa)
        forward_prev_action_embedding_new = torch.cat([prev_relation_embds, current_embds, match_vecter], dim=-1)
        all_action_embedding_new = forward_prev_action_embedding_new
        lstm_output_new = self.lstm_layer(self.policy_step(all_action_embedding_new, mask))


        # Neighbor/condidate_actions embeddings
        neighbors_delta_time1 = torch.abs(query_timestamps.unsqueeze(-1).repeat(1, action_num) - action_space[:, :, 2])
        neighbors_delta_time2 = torch.abs(query_timestamps.unsqueeze(-1).repeat(1, action_num) - action_space[:, :, 3])
        neighbors_delta_time = torch.abs(neighbors_delta_time2 + neighbors_delta_time1)/2

        neighbors_entities = self.bn2(self.dynamicEmbedding(action_space[:, :, 1], neighbors_delta_time))  # [batch_size, action_num, ent_dim]  action[关系（relation）、实体（entity）和时间戳1（time1）时间戳2（time2）]
        neighbors_relations = self.bn2(self.dynamicEmbedding(action_space[:, :, 0], neighbors_delta_time))  # [batch_size, action_num, rel_dim]
        neighbors_timestamps1 = self.qa_subgtr.time_embedding(action_space[:, :, 2])
        neighbors_timestamps1 = self.bn2(self.entity_time_embedding_projected_linearlayer(neighbors_timestamps1))
        neighbors_timestamps2 = self.qa_subgtr.time_embedding(action_space[:, :, 3])
        neighbors_timestamps2 = self.bn2(self.entity_time_embedding_projected_linearlayer(neighbors_timestamps2))

        # agent state representation
        entity_time_embedding_projected = self.qa_subgtr(a)[3]  # [64,19,512] #qa_subgtr中得到的对times， start_time,end_time 与问句中其他信息进行融合后的 时间张量
        if(entity_time_embedding_projected.shape[0] != current_embds.shape[0]):
            entity_time_embedding_projected = entity_time_embedding_projected.repeat(int(current_embds.shape[0]/entity_time_embedding_projected.shape[0]), 1)
        entity_time_embedding_projected = self.entity_time_embedding_projected_linearlayer(entity_time_embedding_projected)

        agent_state_new = torch.cat([lstm_output_new, query_entity, query_relation, entity_time_embedding_projected], dim=-1)  # [batch_size, state_dim + ent_dim + rel_dim + time_dim]
        output_new = torch.layer_norm(self.policy_mlp(agent_state_new), normalized_shape=[self.config['action_dim']])

        entitis_output_new = output_new[:, :, self.config['rel_dim']:]
        relation_ouput_new = output_new[:, :, :self.config['rel_dim']]
        ts_output = output_new[:, :, -self.config['time_dim']:]

        relation_score = torch.sum(torch.mul(neighbors_relations, relation_ouput_new), dim=2)
        entities_score = torch.sum(torch.mul(neighbors_entities, entitis_output_new), dim=2)

        ts_score1 = torch.sum(torch.mul(neighbors_timestamps1, ts_output), dim=2)
        ts_score2 = torch.sum(torch.mul(neighbors_timestamps2, ts_output), dim=2)
        # ts_score = torch.max(ts_score1, ts_score2)
        ts_score = torch.mean(torch.stack([ts_score1, ts_score2], dim=2), dim=2)
        actions = torch.cat([neighbors_relations, neighbors_entities, neighbors_timestamps1, neighbors_timestamps2], dim=-1)
        agent_state_repeats_new = agent_state_new.unsqueeze(1).repeat(1, actions.shape[1], 1)
        agent_state_repeats_new = self.action_state_linearlayer(agent_state_repeats_new)

        score_attention_input_new = torch.cat([actions, agent_state_repeats_new], dim=-1)

        a_new = self.score_weighted_fc(score_attention_input_new).squeeze(-1)
        a = torch.sigmoid(torch.layer_norm(a_new, normalized_shape=[action_num]))

        scores_new = 0.4 * a * relation_score + 0.4*entities_score * a + ts_score * a*0.2  # [batchsize, 30]

        # Padding mask
        scores_new = scores_new.masked_fill(pad_mask, -1e10)
        action_prob_new = torch.softmax(scores_new, dim=1)

        action_id_new = torch.multinomial(action_prob_new, 1)

        logits_new = torch.nn.functional.log_softmax(scores_new, dim=1) # [batch_size, action_number]

        one_hot_new = torch.zeros_like(logits_new).scatter_(1, action_id_new, 1)
        loss_new = - torch.sum(torch.mul(logits_new, one_hot_new), dim=1)
        # return loss_new, logits_new, action_id_new, action_prob_new
        return loss_new, logits_new, action_id_new, action_prob_new, context_qa

