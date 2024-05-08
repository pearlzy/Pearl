import argparse
#argparse是python用于解析命令行参数和选项的标准模块，用于代替已经过时的optparse模块。argparse模块的作用是用于解析命令行参数。
import pickle
# pickle是python序列化的一个工具!可以用来把对象来以文件的形式存储起来，用的时候再加载！
# 注意：pickle模块只能在python中使用，python中的几乎所有的数据类型都可以序列化！但是序列化以后得到的文件人看不懂！
# pickle模块内部实现了用于序列化和反序列化Python对象结构的二进制协议
from collections import defaultdict
from datetime import datetime
import tcomplex
# todo: tensorboard的可视化代码（很重要哦！！！）
# from torch.utils.tensorboard import SummaryWriter

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from urllib3.filepost import writer

import utils
# 以下zy注释掉的  qa_baselines 里的 QA_lm, QA_embedkgqa， QA_cronkgqa被注释掉了
#from qa_baselines import QA_cronkgqa
# from qa_baselines import QA_lm, QA_embedkgqa, QA_cronkgqa
#你只想导入某个模块中的特定函数或变量，可以使用 from…import 语句。这种方式不需要在使用函数时写出完整的模块名 后续使用时 直接调用函数名 不用再模块.函数名
from qa_datasets import QA_Dataset_SubGTR, QA_Dataset_Baseline
from qa_subgtr import QA_SubGTR
from utils import loadTkbcModel_complex, loadTkbcModel_complex, print_info, rerank_ba, rerank_fl, rerank_st, rerank_tj
# 我们常常可以把argparse的使用简化成下面四个步骤
# 1：import argparse 1.首先导入该模块；
# 2：parser = argparse.ArgumentParser() 2.然后创建一个解析对象；然后向该对象中添加你要关注的命令行参数和选项，
# 3：parser.add_argument() 3.每一个add_argument方法对应一个你要关注的参数或选项；
# 4：parser.parse_args() 4.最后调用parse_args()方法进行解析；解析成功之后即可使用。
parser = argparse.ArgumentParser(
    description="Temporal KGQA"
)
parser.add_argument(
    '--tkbc_model_file', default='tcomplex.ckpt', type=str,
    help="Pretrained tkbc model checkpoint"
)
parser.add_argument(
    '--tkg_file', default='full.txt', type=str,
    help="TKG to use for hard-supervision"
)

parser.add_argument(
    '--model', default='subgtr', type=str,
    help="Which model to use."
)


parser.add_argument(
    '--subgraph_reasoning',
    help="whether use subgraph reasoning module",
    action="store_true"
)

parser.add_argument(
    '--time_sensitivity',
    help="whether use time sensitivity module",
    action="store_true"
)

parser.add_argument(
    '--aware_module',
    help="whether use aware module",
    action="store_true"
)

parser.add_argument(
    '--dataset_name', default='wikidata_big', type=str,
    help="Which dataset."
)

parser.add_argument(
    '--supervision', default='hard', type=str,
    help="Which supervision to use."
)

parser.add_argument(
    '--load_from', default='', type=str,
    help="Pretrained qa model checkpoint"
)

parser.add_argument(
    '--save_to', default='', type=str,
    help="Where to save checkpoint."
)

parser.add_argument(
    '--max_epochs', default=10, type=int,
    help="Number of epochs."
)

parser.add_argument(
    '--eval_k', default=1, type=int,
    help="Hits@k used for eval. Default 10."
)

parser.add_argument(
    '--valid_freq', default=1, type=int,
    help="Number of epochs between each valid."
)

parser.add_argument(
    '--batch_size', default=150, type=int,
    help="Batch size."
)

parser.add_argument(
    '--valid_batch_size', default=50, type=int,
    help="Valid batch size."
)

parser.add_argument(
    '--frozen', default=1, type=int,
    help="Whether entity/time embeddings are frozen or not. Default frozen."
)

parser.add_argument(
    '--lm_frozen', default=1, type=int,
    help="Whether language model params are frozen or not. Default frozen."
)

parser.add_argument(
    '--lr', default=2e-4, type=float,
    help="Learning rate"
)

parser.add_argument(
    '--mode', default='train', type=str,
    help="Whether train or eval."
)

parser.add_argument(
    '--eval_split', default='valid', type=str,
    help="Which split to validate on"
)

parser.add_argument(
    '--lm', default='distill_bert', type=str,
    help="Lm to use."
)
parser.add_argument(
    '--fuse', default='add', type=str,
    help="For fusing time embeddings."
)
parser.add_argument(
    '--extra_entities', default=False, type=bool,
    help="For some question types."
)
parser.add_argument(
    '--corrupt_hard', default=0., type=float,
    help="For some question types."
)

parser.add_argument(
    '--test', default="test", type=str,
    help="Test data."
)

args = parser.parse_args()
print_info(args)

# # pkl是预处理之后的文件
# with open('../data/wikidata_big/saved_pkl/e2rt.pkl', 'rb') as f:
#     # pickle.load()函数作用：用于反序列化，将序列化的对象重新恢复成python对象
#     e2rt = pickle.load(f)
# with open('../data/wikidata_big/saved_pkl/event2time.pkl', 'rb') as f:
#     event2time = pickle.load(f)
# with open('../data/wikidata_big/saved_pkl/e2tr.pkl', 'rb') as f:
#     e2tr = pickle.load(f)


# def eval(qa_model, dataset, batch_size=128, split='valid', k=200, subgraph_reasoning=False):
#     num_workers = 0
#     qa_model.eval()
#     eval_log = []
#     print_numbers_only = False
#     k_for_reporting = k  # not change name in fn signature since named param used in places
#     k_list = [1, 10]
#     # max num of subgraph candidate answers
#     max_k = 100
#     eval_log.append("Split %s" % (split))
#     print('Evaluating split', split)
#
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
#                              num_workers=num_workers, collate_fn=dataset._collate_fn)
#     topk_answers = []
#     total_loss = 0
#     loader = tqdm(data_loader, total=len(data_loader), unit="batches")
#     for i_batch, a in enumerate(loader):
#         # if size of split is multiple of batch size, we need this
#         # todo: is there a more elegant way?
#         if i_batch * batch_size == len(dataset.data):
#             break
#         answers_khot = a[-1]  # last one assumed to be target
#         scores, _ = qa_model.forward(a)
#         for s in scores:
#             pred = dataset.getAnswersFromScores(s, k=max_k)
#             topk_answers.append(pred)
#         loss = qa_model.loss(scores, answers_khot.cuda().long())
#         total_loss += loss.item()
#     eval_log.append('Loss %f' % total_loss)
#     eval_log.append('Eval batch size %d' % batch_size)
#
#     # do eval for each k in k_list
#     # want multiple hit@k
#     eval_accuracy_for_reporting = 0
#     for k in k_list:
#         hits_at_k = 0
#         total = 0
#         question_types_count = defaultdict(list)
#         simple_complex_count = defaultdict(list)
#         entity_time_count = defaultdict(list)
#
#         for i, question in enumerate(dataset.data):
#             actual_answers = question['answers']
#             question_type = question['type']
#             if 'simple' in question_type:
#                 simple_complex_type = 'simple'
#             else:
#                 simple_complex_type = 'complex'
#             entity_time_type = question['answer_type']
#             predicted = topk_answers[i]
#
#             if subgraph_reasoning:
#                 neighbours = []
#                 # ADD subgraph neighbours
#                 # for e in question['entities']:
#                 #     neighbours += get_neighbours(e)
#                 # for neighbour in neighbours:
#                 #     predicted.append(neighbour)
#                 # predicted = list(set(predicted))
#                 if question['type'] == 'before_after':
#                     if 'event_head' in question['annotation'].keys():
#                         event = question['annotation']['event_head']
#                         if event[0] !='Q':
#                             t = int(event)
#                         else:
#                             t = int(list(event2time[event])[0][3])
#                         d = list(dataset[i])
#                         d[7] = t
#                         d[8] = t
#                         predicted = rerank_ba(predicted, question['entities'], question['annotation']['type'], d)[:k]
#                     else:
#                         predicted = rerank_ba(predicted, question['entities'], question['annotation']['type'],
#                                               dataset[i])[:k]
#
#                 if question['type'] == 'first_last':
#                     if question['answer_type'] == 'entity':
#                         predicted = rerank_fl(predicted, question['entities'], question['annotation']['adj'],
#                                               dataset[i])[:k]
#
#                 if question['type'] == 'simple_time':
#                     predicted = rerank_st(predicted, question['entities'], None,
#                                           dataset[i])[:k]
#
#                 if question['type'] == 'time_join':
#                     if len(question['entities']) == 2:
#                         if 'event_head' in question['annotation'].keys():
#                             event = question['annotation']['event_head']
#                             if event[0] !='Q':
#                                 t = int(event)
#                             else:
#                                 t = int(list(event2time[event])[0][3])
#                             d = list(dataset[i])
#                             d[7] = t
#                             d[8] = t
#                             predicted = rerank_tj(predicted, question['entities'], None, d)[:k]
#                         else:
#                             predicted = rerank_tj(predicted, question['entities'], None,
#                                                   dataset[i])[:k]
#             predicted = predicted[:k]
#             if len(set(actual_answers).intersection(set(predicted))) > 0:
#                 val_to_append = 1
#                 hits_at_k += 1
#             else:
#                 val_to_append = 0
#             question_types_count[question_type].append(val_to_append)
#             simple_complex_count[simple_complex_type].append(val_to_append)
#             entity_time_count[entity_time_type].append(val_to_append)
#             total += 1
#
#         eval_accuracy = hits_at_k / total
#         if k == k_for_reporting:
#             eval_accuracy_for_reporting = eval_accuracy
#         if not print_numbers_only:
#             eval_log.append('Hits at %d: %f' % (k, round(eval_accuracy, 3)))
#         else:
#             eval_log.append(str(round(eval_accuracy, 3)))
#
#         question_types_count = dict(sorted(question_types_count.items(), key=lambda x: x[0].lower()))
#         simple_complex_count = dict(sorted(simple_complex_count.items(), key=lambda x: x[0].lower()))
#         entity_time_count = dict(sorted(entity_time_count.items(), key=lambda x: x[0].lower()))
#         # for dictionary in [question_types_count]:
#         for dictionary in [question_types_count, simple_complex_count, entity_time_count]:
#             # for dictionary in [simple_complex_count, entity_time_count]:
#             for key, value in dictionary.items():
#                 hits_at_k = sum(value) / len(value)
#                 s = '{q_type} \t {hits_at_k} \t total questions: {num_questions}'.format(
#                     q_type=key,
#                     hits_at_k=round(hits_at_k, 3),
#                     num_questions=len(value)
#                 )
#                 if print_numbers_only:
#                     s = str(round(hits_at_k, 3))
#                 eval_log.append(s)
#             eval_log.append('')
#
#     # print eval log as well as return it
#     for s in eval_log:
#         print(s)
#
#     return eval_accuracy_for_reporting, eval_log
#
#
#
# def append_log_to_file(eval_log, epoch, filename):
#     f = open(filename, 'a+')
#     now = datetime.now()
#     dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
#     f.write('Log time: %s\n' % dt_string)
#     f.write('Epoch %d\n' % epoch)
#     for line in eval_log:
#         f.write('%s\n' % line)
#     f.write('\n')
#     f.close()

# # #
# def train(qa_model, dataset, valid_dataset, args, result_filename=None):
#     num_workers = 0
#     optimizer = torch.optim.Adam(qa_model.parameters(), lr=args.lr)
#     optimizer.zero_grad()
#     batch_size = args.batch_size
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
#                              collate_fn=dataset._collate_fn)
#     max_eval_score = 0
#     if args.save_to == '':
#         args.save_to = 'temp'
#     if result_filename is None:
#         result_filename = '../results/{dataset_name}/{model_file}.log'.format(
#             dataset_name=args.dataset_name,
#             model_file=args.save_to
#         )
#     checkpoint_file_name = '../models/{dataset_name}/qa_models/{model_file}.ckpt'.format(
#         dataset_name=args.dataset_name,
#         model_file=args.save_to
#     )
# #
# #     # if not loading from any previous file
# #     # we want to make new log file
# #     # also log the config ie. args to the file
#     if args.load_from == '':
#         print('Creating new log file')
#         f = open(result_filename, 'a+')
#         now = datetime.now()
#         dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
#         f.write('Log time: %s\n' % dt_string)
#         f.write('Config: \n')
#         for key, value in vars(args).items():
#             key = str(key)
#             value = str(value)
#             f.write('%s:\t%s\n' % (key, value))
#         f.write('\n')
#         f.close()
#
#     max_eval_score = 0.
#
#     print('Starting training')
#     for epoch in range(args.max_epochs):
#         qa_model.train()
#         epoch_loss = 0
#         loader = tqdm(data_loader, total=len(data_loader), unit="batches") #tqdm：进度条可视化
#         running_loss = 0
#         for i_batch, a in enumerate(loader):
#             qa_model.zero_grad()
#             # so that don't need 'if condition' here
#             scores = qa_model.forward(a)
#
#             answers_khot = a[-1]  # last one assumed to be target
#             scores, _ = qa_model.forward(a)
#
#             loss = qa_model.loss(scores, answers_khot.cuda().long())
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
#             running_loss += loss.item()
#             # 这行代码是使用 tqdm 这个库为数据加载过程添加一个进度条，并在进度条的右侧显示额外的信息，例如损失（Loss）和当前的训练周期（Epoch）。
#             # loader 是你的数据加载器对象，通常使用 tqdm(data_loader, total=len(data_loader), unit="batches") 来创建。
#             # set_postfix 是 tqdm 进度条的方法，用于设置右侧显示的信息。
#             # Loss=running_loss / ((i_batch + 1) * batch_size) 这部分是设置进度条右侧显示的损失值，其中 running_loss 是在训练过程中不断更新的损失值，i_batch 表示当前批次的索引，batch_size 是每个批次的大小。这个表达式计算了平均损失。
#             # Epoch=epoch 这部分是设置显示当前的训练周期，epoch 表示当前训练周期的索引。
#             loader.set_postfix(Loss=running_loss / ((i_batch + 1) * batch_size), Epoch=epoch)
#             # 这行代码是使用 tqdm 这个库为数据加载过程添加一个进度条，并在进度条的左侧显示训练周期（epoch）和最大训练周期（max_epochs）的信息。
#             # loader 是你的数据加载器对象，通常使用 tqdm(data_loader, total=len(data_loader), unit="batches") 来创建。
#             # set_description 是 tqdm 进度条的方法，用于设置左侧显示的描述信息。
#             # '{}' 和 format(epoch, args.max_epochs) 这部分是用来生成左侧显示的文本。epoch 表示当前的训练周期，args.max_epochs 表示总的最大训练周期。
#             # 通过这个代码，你可以在训练过程中实时了解当前的训练周期和总的最大训练周期，从而知道训练进展到了哪个阶段。这对于追踪训练进度非常有用。
#             loader.set_description('{}/{}'.format(epoch, args.max_epochs))
#             loader.update()
#
#         print('Epoch loss = ', epoch_loss)
#
#         # todo: tensorboard的可视化代码（很重要哦！！！）
#         writer.add_scalars(tag="/loss", scalar_value=running_loss / ((i_batch + 1) * batch_size), global_step=epoch)
#
#
#         if (epoch + 1) % args.valid_freq == 0:
#             print('Starting eval')
#             eval_score, eval_log = eval(qa_model, valid_dataset, batch_size=args.valid_batch_size,
#                                         split=args.eval_split, k=args.eval_k, subgraph_reasoning=args.subgraph_reasoning)
#             if eval_score > max_eval_score:
#                 print('Valid score increased')
#                 save_model(qa_model, checkpoint_file_name)
#                 max_eval_score = eval_score
#
#
#             # log each time, not max
#             # can interpret max score from logs later
#             append_log_to_file(eval_log, epoch, result_filename)
#
#
# def save_model(qa_model, filename):
#     print('Saving model to', filename)
#     torch.save(qa_model.state_dict(), filename)
#     print('Saved model to ', filename)
#     return
# # 在 NLP 领域，可以使用 Pytorch 的 torch.nn.Embeding() 类对数据进行词嵌入预处理
# if args.model != 'embedkgqa':  # TODO this is a hack
#     tkbc_model: object = loadTkbcModel('../models/{dataset_name}/kg_embeddings/{tkbc_model_file}'.format(
#         dataset_name=args.dataset_name, tkbc_model_file=args.tkbc_model_file
#     ))
#     print('../models/{dataset_name}/kg_embeddings/{tkbc_model_file}'.format(
#         dataset_name=args.dataset_name, tkbc_model_file=args.tkbc_model_file))
# else:
#     tkbc_model : object = loadTkbcModel_complex('../models/{dataset_name}/kg_embeddings/{tkbc_model_file}'.format(
#         dataset_name=args.dataset_name, tkbc_model_file=args.tkbc_model_file
#     ))
#
# if args.mode == 'test_kge':
#     utils.checkIfTkbcEmbeddingsTrained(tkbc_model, args.dataset_name, args.eval_split)
#     exit(0)
#
# train_split = 'train'
# test = args.test
# # train_split = 'train_aware3'
# # test = 'test_aware3'
# test = 'test'
#
# # 以下zy注释掉的
# # if args.model == 'bert' or args.model == 'roberta':
# #     qa_model = QA_lm(tkbc_model, args)
# #     dataset = QA_Dataset_Baseline(split=train_split, dataset_name=args.dataset_name)
# #     # valid_dataset = QA_Dataset_baseline(split=args.eval_split, dataset_name=args.dataset_name)
# #     test_dataset = QA_Dataset_Baseline(split=test, dataset_name=args.dataset_name)
# # elif args.model == 'embedkgqa':
# #     qa_model = QA_embedkgqa(tkbc_model, args)
# #     dataset = QA_Dataset_Baseline(split=train_split, dataset_name=args.dataset_name)
# #     # valid_dataset = QA_Dataset_baseline(split=args.eval_split, dataset_name=args.dataset_name)
# #     test_dataset = QA_Dataset_Baseline(split=test, dataset_name=args.dataset_name)
# # if args.model == 'cronkgqa' and args.supervision != 'hard':
# #     qa_model = QA_cronkgqa(tkbc_model, args)
# #     dataset = QA_Dataset_Baseline(split=train_split, dataset_name=args.dataset_name)
# #     # valid_dataset = QA_Dataset_baseline(split=args.eval_split, dataset_name=args.dataset_name)
# #     test_dataset = QA_Dataset_Baseline(split=test, dataset_name=args.dataset_name)
# if args.model == 'subgtr':  # supervised models
#     qa_model = QA_SubGTR(tkbc_model, args)
#     if args.mode == 'train':
#         dataset = QA_Dataset_SubGTR(split=train_split, dataset_name=args.dataset_name, args=args )
#     # valid_dataset = QA_Dataset_TempoQR(split=args.eval_split, dataset_name=args.dataset_name, args=args)
#     # test_dataset = QA_Dataset_SubGTR(split=test, dataset_name=args.dataset_name, args=args)
#     # filepath = '../saved_pkl/dataset.pkl'
#     # with open(filepath, 'wb') as f:
#     #     pickle.dump(dataset,f)
#     # filepath = '../saved_pkl/test_dataset.pkl'
#     # with open(filepath, 'wb') as f:
#     #     pickle.dump(test_dataset,f)
#     # exit(0)
# else:
#     print('Model %s not implemented!' % args.model)
#     exit(0)
#
# print('Model is', args.model)
# if args.load_from != '':
#
#     filename = '../models/{dataset_name}/qa_models/{model_file}.ckpt'.format(
#         dataset_name=args.dataset_name,
#         model_file=args.load_from
#     )
#     print('Loading model from', filename)
#     qa_model.load_state_dict(torch.load(filename))
#     print('Loaded qa model from ', filename)
#     # TKG embeddings
#     tkbc_model = loadTkbcModel('../models/{dataset_name}/kg_embeddings/{tkbc_model_file}'.format(
#     dataset_name=args.dataset_name, tkbc_model_file=args.tkbc_model_file
#     ))
#
#
#     qa_model.tkbc_model = tkbc_model
#     num_entities = tkbc_model.embeddings[0].weight.shape[0]
#     num_times = tkbc_model.embeddings[2].weight.shape[0]
#     ent_emb_matrix = tkbc_model.embeddings[0].weight.data
#     time_emb_matrix = tkbc_model.embeddings[2].weight.data
#
#     full_embed_matrix = torch.cat([ent_emb_matrix, time_emb_matrix], dim=0)
#     # +1 is for padding idx
#     qa_model.entity_time_embedding = torch.nn.Embedding(num_entities + num_times + 1,
#                                               qa_model.tkbc_embedding_dim,
#                                               padding_idx=num_entities + num_times)
#     qa_model.entity_time_embedding.weight.data[:-1, :].copy_(full_embed_matrix)
#
#     for param in tkbc_model.parameters():
#         param.requires_grad = False
#
# else:
#     print('Not loading from checkpoint. Starting fresh!')

# qa_model = qa_model.cuda()
#
# # if args.mode == 'eval':
# #     score, log = eval(qa_model, test_dataset, batch_size=args.valid_batch_size, split=args.eval_split, k=args.eval_k,subgraph_reasoning = args.subgraph_reasoning)
# #     exit(0)
#
# result_filename = '../results/{dataset_name}/{model_file}.log'.format(
#     dataset_name=args.dataset_name,
#     model_file=args.save_to
# )
#
# train(qa_model, dataset, test_dataset, args, result_filename=result_filename)

# score, log = eval(qa_model, test_dataset, batch_size=args.valid_batch_size, split="test", k=args.eval_k)
# log=["######## TEST EVALUATION FINAL (BEST) #########"]+log
# append_log_to_file(log, 0, result_filename)
#
# print('Training finished')
