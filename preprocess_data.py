import pickle
import os
import argparse
from model.environment import Env
from dataset.qa_datasets import baseDataset, QA_Dataset_SubGTR
from tqdm import tqdm
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocess', usage='preprocess_data.py [<args>] [-h | --help]')
    parser.add_argument('--data_dir', default='D:\Pearl\RTA\data\wikidata_big\kg', type=str)

    parser.add_argument('--outfile', default='state_actions_space.pkl', type=str, help='file to save the preprocessed data.')
    parser.add_argument('--store_actions_num', default=50, type=int, help='maximum number of stored neighbors, 0 means store all.')

    parser.add_argument('--dataset_name', default='wikidata_big', type=str, help="Which dataset.")
    parser.add_argument('--aware_module', help="whether use aware module", action="store_true")
    parser.add_argument('--tkg_file', default='full.txt', type=str, help="TKG to use for hard-supervision")
    parser.add_argument('--corrupt_hard', default=0., type=float, help="For some question types.")
    parser.add_argument('--fuse', default='add', type=str, help="For fusing time embeddings.")
    args = parser.parse_args()

    trainF = os.path.join(args.data_dir, 'train.txt')
    testF = os.path.join(args.data_dir, 'test.txt')
    statF = os.path.join(args.data_dir, 'stat.txt')
    validF = os.path.join(args.data_dir, 'valid.txt')
    if not os.path.exists(validF):
        validF = None
    dataset = baseDataset(trainF, testF, statF, validF)
    config = {
        'num_rel': dataset.num_r,
        'num_ent': dataset.num_e,

    }
    state_actions_path = os.path.join(args.data_path, args.state_actions_path)
    if not os.path.exists(state_actions_path):
        state_action_space = None
    else:
        state_action_space = pickle.load(open(os.path.join(args.data_path, args.state_actions_path), 'rb'))

    qa_dataset = QA_Dataset_SubGTR(split="train", dataset_name=args.dataset_name, args=args)
    env = Env(dataset.allQuadruples, dataset.trainQuadruples, config, args, state_action_space)
    state_actions_space = {}

    with tqdm(total=len(dataset.allQuadruples)) as bar:
        for example in dataset.allQuadruples:

            head = example[0]
            rel = example[1]
            tail = example[2]
            start_time = example[3]
            end_time = example[4]
            if (head, start_time, end_time) not in state_actions_space.keys():
                state_actions_space[(head, start_time, end_time)] = env.get_state_actions_space_complete(head, start_time, end_time, args.store_actions_num)

            if (tail, start_time, end_time) not in state_actions_space.keys():
                state_actions_space[(tail, start_time, end_time)] = env.get_state_actions_space_complete(tail, start_time,end_time, args.store_actions_num)

            bar.update(1)
    pickle.dump(state_actions_space, open(os.path.join(args.data_dir, args.outfile), 'wb'))
