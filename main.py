import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset.qa_datasets import QA_Dataset_SubGTR
from utils.logger import *
from utils.trainer import Trainer
from utils.tester import Tester
from dataset.qa_datasets import baseDataset, QuadruplesDataset
from model.agent import Agent
from model.environment import Env
from model.episode import Episode
from model.policyGradient import PG
from model.dirichlet import Dirichlet
import os
import pickle
from datetime import datetime

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Temporal Knowledge Graph Forecasting Models',
        usage='main.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='whether to use GPU or not.')

    parser.add_argument('--data_path', type=str, default='./data/wikidata_big/kg', help='Path to data.')
    parser.add_argument('--do_train', action='store_true', help='whether to train.')
    parser.add_argument('--do_test', action='store_true', help='whether to test.')
    parser.add_argument('--save_path', default='./logs', type=str, help='log and model save path.')
    parser.add_argument('--load_model_path', default='./logs', type=str, help='trained model checkpoint path.')

    # Train Params
    parser.add_argument('--batch_size', default=256, type=int, help='training batch size.')
    parser.add_argument('--max_epochs', default=20, type=int, help='max training epochs.')
    parser.add_argument('--num_workers', default=0, type=int, help='workers number used for dataloader.')
    parser.add_argument('--valid_epoch', default=10, type=int, help='validation frequency.')
    parser.add_argument('--lr', default=0.00001, type=float, help='learning rate.')
    parser.add_argument('--save_epoch', default=50, type=int, help='model saving frequency.')
    parser.add_argument('--clip_gradient', default=10.0, type=float, help='for gradient crop.')

    # Test Params
    parser.add_argument('--test_batch_size', default=32, type=int, help='test batch size, it needs to be set to 1 when using IM module.')
    parser.add_argument('--beam_size', default=40, type=int, help='the beam number of the beam search.')
    parser.add_argument('--test_inductive', action='store_true', help='whether to verify inductive inference performance.')

    # Agent Params
    parser.add_argument('--ent_dim', default=100, type=int, help='Embedding dimension of the entities')
    parser.add_argument('--rel_dim', default=100, type=int, help='Embedding dimension of the relations')
    parser.add_argument('--state_dim', default=100, type=int, help='dimension of the LSTM hidden state')
    parser.add_argument('--hidden_dim', default=100, type=int, help='dimension of the MLP hidden layer')
    parser.add_argument('--time_dim', default=20, type=int, help='Embedding dimension of the timestamps')
    parser.add_argument('--entities_embeds_method', default='dynamic', type=str,
                        help='representation method of the entities, dynamic or static')

    # Environment Params
    parser.add_argument('--state_actions_path', default='state_actions_space.pkl', type=str,
                        help='the file stores preprocessed candidate action array.')

    # Episode Params
    parser.add_argument('--path_length', default=4, type=int, help='the agent search path length.')
    parser.add_argument('--max_action_num', default=30, type=int, help='the max candidate actions number.')

    # Policy Gradient Params
    parser.add_argument('--Lambda', default=0.01, type=float, help='update rate of baseline.')
    parser.add_argument('--Gamma', default=0.95, type=float, help='discount factor of Bellman Eq.')
    parser.add_argument('--Ita', default=0.01, type=float, help='regular proportionality constant.')
    parser.add_argument('--Zita', default=0.9, type=float, help='attenuation factor of entropy regular term.')
    # reward shaping params
    parser.add_argument('--reward_shaping', action='store_true', help='whether to use reward shaping.')
    parser.add_argument('--time_span', default=36, type=int, help='24 for ICEWS, 1 for WIKI and YAGO')
    parser.add_argument('--alphas_pkl', default='dirchlet_alphas.pkl', type=str, help='the file storing the alpha parameters of the Dirichlet distribution.')
    parser.add_argument('--k', default=300, type=int, help='statistics recent K historical snapshots.')

    # subgtr
    parser.add_argument('--tkbc_model_file', default='tcomplex.ckpt', type=str, help="Pretrained tkbc model checkpoint")
    parser.add_argument('--tkg_file', default='full.txt', type=str, help="TKG to use for hard-supervision")
    parser.add_argument('--dataset_name', default='wikidata_big', type=str, help="Which dataset.")
    parser.add_argument('--supervision', default='hard', type=str, help="Which supervision to use.")
    parser.add_argument('--load_from', default='', type=str, help="Pretrained qa model checkpoint")
    parser.add_argument( '--frozen', default=1, type=int, help="Whether entity/time embeddings are frozen or not. Default frozen.")
    parser.add_argument('--lm_frozen', default=1, type=int, help="Whether language model params are frozen or not. Default frozen.")

    parser.add_argument('--eval_split', default='valid', type=str, help="Which split to validate on")
    parser.add_argument('--lm', default='distill_bert', type=str, help="Lm to use.")
    parser.add_argument('--fuse', default='add', type=str, help="For fusing time embeddings.")

    parser.add_argument('--corrupt_hard', default=0., type=float, help="For some question types.")
    parser.add_argument('--test', default="test", type=str, help="Test data.")

    return parser.parse_args(args)


def get_model_config(args, num_ent, num_rel, num_ts):

    config = {
        'cuda': args.cuda,  # whether to use GPU or not.  # 是否使用 GPU
        'batch_size': args.batch_size,  # training batch size.   # 训练批次大小
        'num_ent': num_ent,  # number of entities   # 实体数量
        'num_rel': num_rel,  # number of relations   # 关系数量
        'num_ts': num_ts,
        'ent_dim': args.ent_dim,  # Embedding dimension of the entities # 实体嵌入的维度   100
        'rel_dim': args.rel_dim,  # Embedding dimension of the relations # 关系嵌入的维度  100
        'time_dim': args.time_dim,  # Embedding dimension of the timestamps  # 时间戳嵌入的维度
        'state_dim': args.state_dim,  # dimension of the LSTM hidden state # LSTM隐藏状态的维度
        'action_dim': args.ent_dim + args.rel_dim,  # dimension of the actions # 动作的维度
        # 'mlp_input_dim': args.ent_dim + args.rel_dim + args.state_dim, # dimension of the input of the MLP  # MLP输入层的维度
        'mlp_input_dim': args.ent_dim + args.rel_dim + args.state_dim + args.time_dim,
        # 'mlp_hidden_dim': args.hidden_dim,  # dimension of the MLP hidden layer # MLP隐藏层的维度
        'mlp_hidden_dim': args.hidden_dim + args.time_dim,  # dimension of the MLP hidden layer # MLP隐藏层的维度
        'path_length': args.path_length,  # agent search path length  # 代理搜索路径的长度
        'max_action_num': args.max_action_num,  # max candidate action number  # 最大候选动作数
        'lambda': args.Lambda,  # update rate of baseline # 基线更新速率
        'gamma': args.Gamma,  # discount factor of Bellman Eq. # Bellman方程的折现因子
        'ita': args.Ita,  # regular proportionality constant  # 正则化比例常数
        'zita': args.Zita,  # attenuation factor of entropy regular term # 熵正则项的衰减因子
        'beam_size': args.beam_size,  # beam size for beam search # Beam搜索的束大小
        'entities_embeds_method': args.entities_embeds_method,  # default: 'dynamic', otherwise static encoder will be used  # 实体嵌入的方法（动态或静态）
    }
    return config

def main(args):
    # ###################### Set Logger #################################

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.cuda and torch.cuda.is_available():
        args.cuda = True
    else:
        args.cuda = False
    set_logger(args)

    #######################Create DataLoader#################################
    train_path_new = os.path.join('data/{dataset_name}/kg/'.format(dataset_name=args.dataset_name), 'train.txt')
    test_path_new = os.path.join('data/{dataset_name}/kg/'.format(dataset_name=args.dataset_name), 'test.txt')
    stat_path_new = os.path.join('data/{dataset_name}/kg/'.format(dataset_name=args.dataset_name), 'stat.txt')
    valid_path_new = os.path.join('data/{dataset_name}/kg/'.format(dataset_name=args.dataset_name), 'valid.txt')

    baseData_new = baseDataset(train_path_new, test_path_new, stat_path_new, valid_path_new)

    dataset = QA_Dataset_SubGTR(split="train", dataset_name=args.dataset_name, args=args)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset. _collate_fn)

    dataset2 = QA_Dataset_SubGTR(split="valid", dataset_name=args.dataset_name, args=args)
    valid_dataloader = DataLoader(dataset2, batch_size=args.test_batch_size, shuffle=True, num_workers=0, collate_fn=dataset2._collate_fn)

    dataset3 = QA_Dataset_SubGTR(split="test", dataset_name=args.dataset_name, args=args)
    test_dataloader = DataLoader(dataset3, batch_size=args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=dataset3._collate_fn)


    ######################Creat the agent and the environment###########################
    config_new = get_model_config(args, len(dataset.all_dicts["ent2id"]), len(dataset.all_dicts["rel2id"]), len(dataset.all_dicts["ts2id"]))
    logging.info(args)
    agent_new = Agent(config_new, args)

    # creat the environment
    state_actions_path = os.path.join(args.data_path, args.state_actions_path)
    if not os.path.exists(state_actions_path):
        state_action_space = None
    else:
        state_action_space = pickle.load(open(os.path.join(args.data_path, args.state_actions_path), 'rb'))
    env_new = Env(baseData_new.allQuadruples, baseData_new.trainQuadruples, config_new, dataset, args, state_action_space)
    # Create episode controller
    episode_new = Episode(env_new, agent_new, config_new, args)
    if args.cuda:
        episode_new = episode_new.cuda()
    pg_new = PG(config_new)

    optimizer_new = torch.optim.AdamW(episode_new.parameters(), lr=args.lr, weight_decay=0.000001)

    if os.path.isdir(args.load_model_path):
         model_dir = os.path.abspath(args.load_model_path)
         files = os.listdir(model_dir)
         checkpoint_files = [file for file in files if file.startswith('checkpoint_')]
         if checkpoint_files:
             latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
             latest_checkpoint_path = os.path.join(model_dir, latest_checkpoint)
             params = torch.load(latest_checkpoint_path)
             episode_new.load_state_dict(params['model_state_dict'])
             optimizer_new.load_state_dict(params['optimizer_state_dict'])
             logging.info('Load the latest pretrain model: {}'.format(latest_checkpoint_path))
         else:
             logging.info('No checkpoint files found in the model directory.')
    else:
         logging.info('The  model path is not a checkpoint files.')

    ######################Training and Testing###########################
    if args.reward_shaping:
        alphas = pickle.load(open(os.path.join(args.data_path, args.alphas_pkl), 'rb'))
        distributions = Dirichlet(alphas, args.k)
    else:
        distributions = None
    trainer_new = Trainer(episode_new, pg_new, optimizer_new, args, dataset, config_new, distributions)

    tester_new = Tester(episode_new, args, env_new.train_entities, env_new.train_timestamps)
    if args.do_train:
        logging.info('Start Training......')
        for i in range(args.max_epochs):

            loss_new, reward_new = trainer_new.train_epoch(data_loader, dataset.__len__(), args)
            logging.info('Epoch {}/{} Loss: {}, reward: {}'.format(i+1, args.max_epochs, loss_new, reward_new))
            index = {'loss': loss_new, 'rewards': reward_new}
            # #
            log_dir = f'./logs/new'
            writer = SummaryWriter(log_dir)
            writer.add_scalars("loss", {'loss': index['loss']}, i+1)
            writer.add_scalars("reward", {'rewards': index['rewards']}, i+1)


            if (i+1) % args.save_epoch == 0 and (i+1) != 0:
                trainer_new.save_model('checkpoint_{}.pth'.format(i+1))
                logging.info('Save Model in {}'.format(args.save_path))

            if (i+1) % args.valid_epoch == 0 and (i+1) != 0:
                logging.info('Start Val......')

                metrics_new = tester_new.test(valid_dataloader,
                                              dataset2.__len__(),
                                              env_new.skip_dict,
                                              config_new['num_ent'],
                                              config_new['num_ts'])

                for mode in metrics_new.keys():
                    logging.info('{} at epoch {}: {}'.format(mode, i+1, metrics_new[mode]))
                writer = SummaryWriter(log_dir)
                writer.add_scalars("HITS@1", {'HITS@1': metrics_new['HITS@1']}, i+1)
                writer.add_scalars("HITS@10", {'HITS@10': metrics_new['HITS@10']}, i+1)
                writer.add_scalars("HITS@1_answer_time", {'HITS@1_answer_time': metrics_new['HITS@1_answer_time']}, i + 1)
                writer.add_scalars("HITS@10_answer_time", {'HITS@10_answer_time': metrics_new['HITS@10_answer_time']}, i + 1)
                writer.add_scalars("HITS@1_answer_entity", {'HITS@1_answer_entity': metrics_new['HITS@1_answer_entity']}, i + 1)
                writer.add_scalars("HITS@10_answer_entity", {'HITS@10_answer_entity': metrics_new['HITS@10_answer_entity']}, i + 1)
                writer.add_scalars("HITS@1_simple_time", {'HITS@1_simple_time': metrics_new['HITS@1_simple_time']}, i + 1)
                writer.add_scalars("HITS@10_simple_time", {'HITS@10_simple_time': metrics_new['HITS@10_simple_time']}, i + 1)
                writer.add_scalars("HITS@1_simple_entity", {'HITS@1_simple_entity': metrics_new['HITS@1_simple_entity']}, i + 1)
                writer.add_scalars("HITS@10_simple_entity", {'HITS@10_simple_entity': metrics_new['HITS@10_simple_entity']}, i + 1)
                writer.add_scalars("HITS@1_time_join", {'HITS@1_time_join': metrics_new['HITS@1_time_join']}, i + 1)
                writer.add_scalars("HITS@10_time_join", {'HITS@10_time_join': metrics_new['HITS@10_time_join']}, i + 1)
                writer.add_scalars("HITS@1_before_after", {'HITS@1_before_after': metrics_new['HITS@1_before_after']}, i + 1)
                writer.add_scalars("HITS@10_before_after", {'HITS@10_before_after': metrics_new['HITS@10_before_after']}, i + 1)
                writer.add_scalars("HITS@1_first_last", {'HITS@1_first_last': metrics_new['HITS@1_first_last']}, i + 1)
                writer.add_scalars("HITS@10_first_last", {'HITS@10_first_last': metrics_new['HITS@10_first_last']}, i + 1)
        trainer_new.save_model()
        logging.info('Save Model in {}'.format(args.save_path))

    if args.do_test:
        logging.info('Start Testing......')
        metrics_new = tester_new.test(test_dataloader,
                                      dataset3.__len__(),
                                      env_new.skip_dict,
                                      config_new['num_ent'],
                                      config_new['num_ts'])
        for mode in metrics_new.keys():
            logging.info('Test {} : {}'.format(mode, metrics_new[mode]))
        # 获取当前时间
        current_time = datetime.now().strftime("%Y-%m-%d")
        log_dir = f'./logs/log_{current_time}'
        writer = SummaryWriter(log_dir)
        writer.add_scalars("HITS@1", {'HITS@1': metrics_new['HITS@1']}, global_step=args.test_batch_size)
        writer.add_scalars("HITS@10", {'HITS@10': metrics_new['HITS@10']}, global_step=args.test_batch_size)
        writer.add_scalars("HITS@1_answer_time", {'HITS@1_answer_time': metrics_new['HITS@1_answer_time']}, args.test_batch_size)
        writer.add_scalars("HITS@10_answer_time", {'HITS@10_answer_time': metrics_new['HITS@10_answer_time']}, args.test_batch_size)
        writer.add_scalars("HITS@1_answer_entity", {'HITS@1_answer_entity': metrics_new['HITS@1_answer_entity']},args.test_batch_size)
        writer.add_scalars("HITS@10_answer_entity", {'HITS@10_answer_entity': metrics_new['HITS@10_answer_entity']}, args.test_batch_size)
        writer.add_scalars("HITS@1_simple_time", {'HITS@1_simple_time': metrics_new['HITS@1_simple_time']},args.test_batch_size)
        writer.add_scalars("HITS@10_simple_time", {'HITS@10_simple_time': metrics_new['HITS@10_simple_time']}, args.test_batch_size)
        writer.add_scalars("HITS@1_simple_entity", {'HITS@1_simple_entity': metrics_new['HITS@1_simple_entity']}, args.test_batch_size)
        writer.add_scalars("HITS@10_simple_entity", {'HITS@10_simple_entity': metrics_new['HITS@10_simple_entity']}, args.test_batch_size)
        writer.add_scalars("HITS@1_time_join", {'HITS@1_time_join': metrics_new['HITS@1_time_join']}, args.test_batch_size)
        writer.add_scalars("HITS@10_time_join", {'HITS@10_time_join': metrics_new['HITS@10_time_join']}, args.test_batch_size)
        writer.add_scalars("HITS@1_before_after", {'HITS@1_before_after': metrics_new['HITS@1_before_after']}, args.test_batch_size)
        writer.add_scalars("HITS@10_before_after", {'HITS@10_before_after': metrics_new['HITS@10_before_after']}, args.test_batch_size)
        writer.add_scalars("HITS@1_first_last", {'HITS@1_first_last': metrics_new['HITS@1_first_last']}, args.test_batch_size)
        writer.add_scalars("HITS@10_first_last", {'HITS@10_first_last': metrics_new['HITS@10_first_last']}, args.test_batch_size)
if __name__ == '__main__':
    args = parse_args()
    main(args)
