import pickle
import os
import argparse
from model.dirichlet import MLE_Dirchlet
from dataset.qa_datasets import baseDataset, QA_Dataset_SubGTR
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dirichlet MLE', usage='mle_dirichlet.py [<args>] [-h | --help]')

    parser.add_argument('--data_dir', default='D:\Pearl\RTA\data\wikidata_big\kg', type=str)

    parser.add_argument('--outfile', default='dirchlet_alphas.pkl', type=str)
    parser.add_argument('--k', default=300, type=int)
    parser.add_argument('--time_span', default=36, type=int, help='36 for wikidata_big, 1 for WIKI and YAGO')  #这里default=24，（zy）改成了36，，1，24，12都显示索引错误
    parser.add_argument('--tol', default=1e-7, type=float)
    parser.add_argument('--method', default='meanprecision', type=str)
    parser.add_argument('--maxiter', default=100, type=int)
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
    qa_dataset = QA_Dataset_SubGTR(split="train", dataset_name=args.dataset_name, args=args)
    train_examples = []
    for example in dataset.trainQuadruples:
        head = qa_dataset.all_dicts["ent2id"]["Q" + str(example[0])]
        rel = qa_dataset.all_dicts["rel2id"]["P" + str(example[1])]
        tail = qa_dataset.all_dicts["ent2id"]["Q" + str(example[2])]
        start_time = qa_dataset.all_dicts["ts2id"][(example[3], 0, 0)]
        end_time = qa_dataset.all_dicts["ts2id"][(example[4], 0, 0)]
        train_example = [head, rel, tail, start_time, end_time]
        train_examples.append(train_example)

    mle_d = MLE_Dirchlet(train_examples, dataset.num_r, args.k, args.time_span,
                         args.tol, args.method, args.maxiter)
    pickle.dump(mle_d.alphas, open(os.path.join(args.data_dir, args.outfile), 'wb'))
