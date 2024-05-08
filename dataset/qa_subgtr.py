import math
import torch
from torch import nn
import numpy as np
from dataset.tcomplex import TComplEx
from transformers import DistilBertModel
from torch.nn import LayerNorm, TransformerEncoder, TransformerEncoderLayer
# todo:（zy）最大池化层求问题上下文向量  import torch.nn.functional as F
import torch.nn.functional as F
from dataset.Transformer import TransformerModel


class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.LayerNorm(in_dim),
        nn.GELU(),
        nn.Linear(in_dim, 1),
        )

        self.attention_weights = None

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask == 0] = float('-inf')
        w = torch.softmax(w, 1)
        # 在训练或推断完成后，你可以访问你模型的 attention_pooling_layer.attention_weights 属性，将这些权重用于可视化。
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # import numpy as np
        #
        # # 注意力权重矩阵，这里假设是一个2D矩阵，实际上应该是你模型输出的注意力权重
        # attention_weights = np.random.rand(10, 10)
        #
        # # 使用 seaborn 的热图绘制   attention_weights 是一个注意力权重矩阵，你应该用你模型的注意力权重替换这个变量。cmap 参数用于设置颜色映射，annot=True 参数表示在热图上显示数值。
        # sns.heatmap(attention_weights, cmap="viridis", annot=True)
        # plt.show()
        self.attention_weights = w.detach().cpu().numpy()
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings

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

        # 计算注意力权重
        attention_weights = torch.softmax(torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5), dim=-1) #[128,20,8,8]

        # 加权求和
        weighted_sum = torch.matmul(attention_weights, value) #[128,20,8,96]

        # 合并多个头
        weighted_sum = weighted_sum.view(batch_size, -1, self.num_heads * self.head_dim)  #[128,20,768]

        return weighted_sum


class QA_SubGTR(nn.Module):
    def __init__(self, tkbc_model, args):
        super().__init__()
        self.model = args.model
        self.time_sensitivity = args.time_sensitivity
        self.supervision = args.supervision
        self.extra_entities = args.extra_entities
        self.fuse = args.fuse
        self.tkbc_embedding_dim = tkbc_model.embeddings[0].weight.shape[1]    #tkbc_embedding_dim =512
        self.sentence_embedding_dim = 768  # hardwired from

        self.pretrained_weights = './distil_bert/'
        self.lm_model = DistilBertModel.from_pretrained(self.pretrained_weights)
        self.attention_pooling = AttentionPooling(self.sentence_embedding_dim)

        if args.lm_frozen == 1:
            print('Freezing LM params')
            for param in self.lm_model.parameters():
                param.requires_grad = False
        else:
            print('Unfrozen LM params')

        # transformer
        # self.transformer_dim 表示 Layer Normalization 层的输入特征维度。具体来说，对于输入数据张量的每个样本，encoder_norm 将对其特征维度（即 self.transformer_dim 维度）进行归一化处理。
        # d_model（self.transformer_dim）：表示输入特征的维度，通常称为模型的隐藏状态维度。它定义了输入和输出的特征维度
        self.transformer_dim = self.tkbc_embedding_dim  # keeping same so no need to project embeddings
        # nhead（self.nhead）：表示注意力机制中的头数。Transformer 中的多头自注意力机制允许模型在不同位置关注输入序列中的不同部分。nhead 定义了使用的注意力头的数量。
        self.nhead = 8
        # self.num_layers 是 Transformer 模型中编码器层的数量。在深度学习中，通常会将多个编码器层叠加在一起，以构建更深层次的模型。这是为了允许模型捕获输入数据的更复杂的特征和关系，从而提高模型性能。
        self.num_layers = 6
        # dropout（self.transformer_dropout）：表示在层输出之前应用的 dropout 概率。它有助于模型的正则化，防止过拟合。
        self.transformer_dropout = 0.1
        # self.encoder_layer 是一个 Transformer 编码器层（nn.TransformerEncoderLayer）。在深度学习中，Transformer 编码器通常用于处理序列数据，如自然语言文本。
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.transformer_dim, nhead=self.nhead,
                                                        dropout=self.transformer_dropout)
        # encoder_norm 是一个 LayerNorm 层，它用于对输入数据进行 Layer Normalization（层归一化）处理。这种处理在神经网络中常用于提高训练稳定性和加速训练过程。
        # encoder_norm 是一个 LayerNorm 层，用于对 Transformer 编码器的输出进行标准化。标准化可以帮助提高模型的训练稳定性，确保不同层的输出具有相似的均值和方差。
        encoder_norm = LayerNorm(self.transformer_dim)
        # self.transformer_encoder 是一个 nn.TransformerEncoder 模块，它使用多个编码器层来处理输入数据。在这里，self.encoder_layer 表示每个编码器层的配置，num_layers 表示要叠加的编码器层的数量，norm 表示用于层规范化的层。
        # 这个编码器模块的主要目标是将输入数据进行编码，提取输入数据的特征，并产生编码后的表示。T
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers,
                                                         norm=encoder_norm)

        self.project_sentence_to_transformer_dim = nn.Linear(self.sentence_embedding_dim, self.transformer_dim)
        self.project_entity = nn.Linear(self.tkbc_embedding_dim, self.tkbc_embedding_dim)

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
        # print('Random starting embedding')
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.layer_norm = nn.LayerNorm(self.transformer_dim)

        self.linear = nn.Linear(768, self.tkbc_embedding_dim)  # to project question embedding
        self.linearT = nn.Linear(768, self.tkbc_embedding_dim)  # to project question embedding
        self.lin_cat = nn.Linear(3 * self.transformer_dim, self.transformer_dim)

        self.linear1 = nn.Linear(self.tkbc_embedding_dim, self.tkbc_embedding_dim)
        self.linear2 = nn.Linear(self.tkbc_embedding_dim, self.tkbc_embedding_dim)


        # todo：这三个参数  在接下来没有用到
        self.dropout = torch.nn.Dropout(0.3)
        self.bn1 = torch.nn.BatchNorm1d(self.tkbc_embedding_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.tkbc_embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(self.tkbc_embedding_dim)

        # todo：（zy）
        self.batch_size = args.batch_size



        # 使用 MultiHeadSelfAttention
        # hidden_size = 768  # 你的隐藏状态维度
        # num_heads = 8  # 多头注意力的头数
        self.multihead_attention = MultiHeadSelfAttention(self.tkbc_embedding_dim, self.nhead)


        return

    def invert_binary_tensor(self, tensor):
        ones_tensor = torch.ones(tensor.shape, dtype=torch.long).cuda()
        inverted = ones_tensor - tensor
        return inverted


    # scoring function from TComplEx
    def score_time(self, head_embedding, tail_embedding, relation_embedding):
        lhs = head_embedding
        rhs = tail_embedding
        rel = relation_embedding

        time = self.tkbc_model.embeddings[2].weight

        lhs = lhs[:, :self.tkbc_model.rank], lhs[:, self.tkbc_model.rank:]
        rel = rel[:, :self.tkbc_model.rank], rel[:, self.tkbc_model.rank:]
        rhs = rhs[:, :self.tkbc_model.rank], rhs[:, self.tkbc_model.rank:]
        time = time[:, :self.tkbc_model.rank], time[:, self.tkbc_model.rank:]

        return (
                (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
                 lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
                (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
                 lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )

    def score_entity(self, head_embedding, tail_embedding, relation_embedding, time_embedding):

        lhs = head_embedding[:, :self.tkbc_model.rank], head_embedding[:, self.tkbc_model.rank:]
        rel = relation_embedding
        time = time_embedding

        rel = rel[:, :self.tkbc_model.rank], rel[:, self.tkbc_model.rank:]
        time = time[:, :self.tkbc_model.rank], time[:, self.tkbc_model.rank:]

        right = self.tkbc_model.embeddings[0].weight
        # right = self.entity_time_embedding.weight
        right = right[:, :self.tkbc_model.rank], right[:, self.tkbc_model.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        return (
                (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
                (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
        )
    def embedding(self,entities):
        entities = entities.cuda().long()
        # print(torch.max(entities), torch.min(entities))
        entities_embedding = self.entity_time_embedding(entities)
        return entities_embedding

    def time_embedding(self, times):   #query_times
        times = times.cuda().long()
        times_emb = self.entity_time_embedding(times)
        return times_emb

    # def tkg_embedding(self,t1,t2):
    #     self
    #
    def forward(self, a):
        # Tokenized questions, where entities are masked from the sentence to have TKG embeddings
        question_tokenized = a[0].cuda().long()  # torch.Size([1, 10])
        question_attention_mask = a[1].cuda().long()
        entities_times_padded = a[2].cuda().long()
        entity_mask_padded = a[3].cuda().long()

        # Annotated entities/timestamps
        heads = a[4].cuda().long()
        tails = a[5].cuda().long()
        times = a[6].cuda().long()

        t1 = a[7].cuda().long()
        t2 = a[8].cuda().long()

        # One extra entity for new before & after question type
        tails2 = a[9].cuda().long()
        self.entity_time_embedding.to('cuda')
        # TKG embeddings
        head_embedding = self.entity_time_embedding(heads)
        tail_embedding = self.entity_time_embedding(tails.long())
        tail_embedding2 = self.entity_time_embedding(tails2.long())
        time_embedding = self.entity_time_embedding(times.long())

        # Hard Supervision
        t1_emb = self.tkbc_model.embeddings[2](t1)
        t2_emb = self.tkbc_model.embeddings[2](t2)

        # entity embeddings to replace in sentence
        entity_time_embedding = self.entity_time_embedding(entities_times_padded)  # torch.Size([1, 10, 512])

        # context-aware step
        outputs = self.lm_model(question_tokenized, attention_mask=question_attention_mask)
        last_hidden_states = outputs.last_hidden_state  # torch.Size([batchsize, 17, 768])
        # outputs_attentioning_pooling = self.attention_pooling(last_hidden_states, question_attention_mask)
        # outputs_attentioning_pooling = outputs_attentioning_pooling.unsqueeze(1).repeat(1, last_hidden_states.shape[1], 1)

        # 得到每个 token 对应的加权表示
        #weighted_representation = self.multihead_attention(last_hidden_states)


        # entity-aware step
        # 768->512
        question_embedding = self.project_sentence_to_transformer_dim(last_hidden_states)  # torch.Size([batchsize, 17, 512])
        # todo 008
        # question_embedding = self.project_sentence_to_transformer_dim(outputs_attentioning_pooling)
        # question_embedding = self.project_sentence_to_transformer_dim(weighted_representation)
        entity_mask = entity_mask_padded.unsqueeze(-1).expand(question_embedding.shape)   # torch.Size([batchsize, 17, 512])
        masked_question_embedding = question_embedding * entity_mask  # set entity positions 0   # torch.Size([batchsize, 17, 512])
        # 512->512
        entity_time_embedding_projected = self.project_entity(entity_time_embedding)  # torch.Size([1, 10, 512])

        # time-aware step
        time_pos_embeddings1 = t1_emb.unsqueeze(0).transpose(0, 1)  # torch.Size([1, 1, 512])
        time_pos_embeddings1 = time_pos_embeddings1.expand(entity_time_embedding_projected.shape)  # torch.Size([1, 10, 512])

        time_pos_embeddings2 = t2_emb.unsqueeze(0).transpose(0, 1)
        time_pos_embeddings2 = time_pos_embeddings2.expand(entity_time_embedding_projected.shape)
        if self.fuse == 'cat':
            entity_time_embedding_projected = self.lin_cat(
                torch.cat((entity_time_embedding_projected, time_pos_embeddings1, time_pos_embeddings2), dim=-1))
        else:
            entity_time_embedding_projected = entity_time_embedding_projected + time_pos_embeddings1 + time_pos_embeddings2  # torch.Size([1, 10, 512])

        #TODO:Transformer information fusion layer
        masked_entity_time_embedding = entity_time_embedding_projected * self.invert_binary_tensor(entity_mask)
        # entity_time_embedding_projected 是一个实体时间嵌入的张量，它包含了实体和时间的嵌入表示。
        # self.invert_binary_tensor(entity_mask) 是一个方法调用，它的目的是获取 entity_mask 的逆掩码,将 entity_mask 中的零值变为一，将非零值变为零。逆掩码的目的是在乘法操作中，将 entity_mask 中的零对应的元素设置为零，而将非零对应的元素保持不变。
        # 通过执行按元素的乘法操作，将 entity_time_embedding_projected 中的元素与逆掩码相乘。这将导致实体时间嵌入中与 entity_mask 中的零对应的元素变为零，而与非零对应的元素保持不变。
        # masked_entity_time_embedding 包含了根据 entity_mask 控制的实体时间嵌入，其中某些元素被设置为零，而其他元素保持不变。
        combined_embed = masked_question_embedding + masked_entity_time_embedding   # torch.Size([batchsize, 17, 512])

        # also need to add position embedding
        sequence_length = combined_embed.shape[1]
        v = np.arange(0, sequence_length, dtype=np.float64)   # torch.Size(17,)
        indices_for_position_embedding = torch.from_numpy(v).cuda().long()   # torch.Size(17,)
        position_embedding = self.position_embedding(indices_for_position_embedding)   # torch.Size(17,512)
        position_embedding = position_embedding.unsqueeze(0).expand(combined_embed.shape)   # torch.Size(batchsize, 17,512)

        combined_embed = combined_embed + position_embedding   # torch.Size(batchsize, 17,512)

        combined_embed = self.layer_norm(combined_embed) # torch.Size(batchsize, 17,512)
        # combined_embed = torch.transpose(combined_embed, 0, 1)   # torch.Size(17, batchsize, 512)

        # mask2 = ~(question_attention_mask.bool()).cuda()

        #todo: 在进入transformer重新编码后 使用最大池化层获得具有上下文感知的问题表示

        # output = self.transformer_encoder(combined_embed, src_key_padding_mask=mask2)  # torch.Size([10, 1, 512])
        # transformer_out = output.transpose(0, 1)
        # _, _, output_dim = transformer_out.shape
        # question_mask = mask2.unsqueeze(-1).repeat(1, 1, output_dim)
        # transformer_out_masked = transformer_out.masked_fill(question_mask, float('-inf'))
        #
        # question_transformer_masked = transformer_out_masked.transpose(1, 2)

        # 得到每个 token 对应的加权表示
        weighted_representation = self.multihead_attention(combined_embed)   # torch.Size(17, batchsize, 512)
        question_mp = F.max_pool1d(weighted_representation.transpose(1, 2), combined_embed.shape[1]).squeeze(2)
        # question_mp = F.max_pool1d(question_transformer_masked, question_transformer_masked.size(2)).squeeze(2)
        output = question_mp.unsqueeze(0)    #[50,512]-->[1,50,512]     .unsqueeze(0)  增加第0维
        # todo
        # return output

        # Answer Predictions
        # relation_embedding 是从模型输出 output 中取得的关系嵌入，通常是模型针对输入数据计算得出的关系表示。这个嵌入的形状是 [1, 512]，表示一个大小为 512 的向量。
        # self.linear1 和 self.linear2 是两个线性层，用于对 relation_embedding 进行线性变换（例如，将维度从 512 改变为其他值）。
        #         self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)   self.linear1 是一个线性层，它将输入数据的维度从 d_model（输入维度）变换为 dim_feedforward（输出维度）。这个操作通常用于降低维度或进行特征映射，以便在网络中进行进一步的计算。
        #         self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)   self.linear2 是另一个线性层，它将输入数据的维度从 dim_feedforward 变换回到 d_model。这个操作通常用于将数据映射回原始维度，或者进行其他形式的维度调整。
        relation_embedding = output[0]  # self.linear(output[0]) #cls token embedding # torch.Size([1, 512])

        relation_embedding1 = self.dropout(self.bn1(self.linear1(relation_embedding)))
        relation_embedding2 = self.dropout(self.bn1(self.linear2(relation_embedding)))

        # Time sensitivity layer
        # 这段代码的目的是根据时间敏感性的设置来计算时间相关的得分。如果启用了时间敏感性，它会执行两种不同的时间得分计算，并选择其中较大的一个作为最终的时间得分。这个时间得分通常用于模型中与时间相关的任务或推断。否则，它会直接计算时间得分，不考虑时间敏感性。
        #if self.time_sensitivity:
            # 计算 scores_time1：通过调用 self.score_time 函数，传入 head_embedding、tail_embedding 和 relation_embedding1 来计算时间敏感性分数。这个分数通常用于衡量关系和实体之间的时间关联性。
          #  scores_time1 = self.score_time(head_embedding, tail_embedding, relation_embedding1)
            # 计算 scores_time2：通过将 relation_embedding1 与 self.entity_time_embedding 中的一部分（去除了 padding idx 部分）进行矩阵乘法计算来获得额外的时间敏感性分数。
        #    scores_time2 = torch.matmul(relation_embedding1, self.entity_time_embedding.weight.data[self.num_entities:-1, :].T) # cuz padding idx
            # 通过 torch.maximum 函数将 scores_time1 和 scores_time2 中的较大值作为最终的时间敏感性分数 scores_time。
       #     scores_time = torch.maximum(scores_time1, scores_time2)
     #   else:
         #   scores_time = self.score_time(head_embedding, tail_embedding, relation_embedding1)



    #    scores_entity1 = self.score_entity(head_embedding, tail_embedding, relation_embedding2, ime_embedding)
     #   scores_entity2 = self.score_entity(tail_embedding, head_embedding, relation_embedding2, time_embedding)
    #    scores_entity = torch.maximum(scores_entity1, scores_entity2)

        # 通过 torch.maximum 函数将这两部分中的较大值合并在一起，形成了一个包含所有候选实体的得分的张量 scores_entity。
        # 这意味着对于每一对头实体和尾实体的组合，都会计算一个得分，并且这些得分被合并成一个张量，其中每个元素对应一个候选实体对的得分。
     #   scores = torch.cat((scores_entity, scores_time), dim=1)
        # todo： 通过加权平均将候选实体向量进行加权平均得到batch_pred_vector  作为策略网络动作选择的额外输入
     #   batch_pred_vector = torch.matmul(scores, self.entity_time_embedding.weight.data[:-1, :]) / torch.sum(scores, dim=1, keepdim=True)

        entity_time_embedding_projected = entity_time_embedding_projected.transpose(1, 2)
        entity_time_embedding_projected = F.max_pool1d(entity_time_embedding_projected, entity_time_embedding_projected.size(2)).squeeze(2)

               # batch_pred_vetor [256,512]  entity_time_embedding_projected [256,512]
            # scores, batch_pred_vector, masked_entity_time_embedding, entity_time_embedding_projected, question_mp
        return t1_emb, t2_emb, masked_entity_time_embedding, entity_time_embedding_projected, question_mp



