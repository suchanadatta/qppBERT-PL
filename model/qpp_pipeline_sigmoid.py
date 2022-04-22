import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_pisa import PisaIndex
import ir_datasets
import random
import torch
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import logging
import itertools
import torch.nn.functional as F
import argparse
import more_itertools
import numpy as np

torch.manual_seed(0)
logger = ir_datasets.log.easy()


def position_encoding_init(max_pos, emb_dim):
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(max_pos)])
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


class BertQppModel(torch.nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.emb_dim = 768
        self.max_pos = 1000
        self.position_enc = torch.nn.Embedding(self.max_pos, self.emb_dim, padding_idx=0)
        self.position_enc.weight.data = position_encoding_init(self.max_pos, self.emb_dim)
        self.bert = BertModel.from_pretrained(model_name)
        self.lstm = torch.nn.LSTM(input_size=self.emb_dim, hidden_size=self.bert.config.hidden_size,
                                  num_layers=1, bias=True, batch_first=False, dropout=0.2)
        self.utility = torch.nn.Sequential(
            torch.nn.Linear(self.bert.config.hidden_size, 100),
            torch.nn.Linear(100, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, pos_list, input_ids, attention_mask, token_type_ids):
        res = self.bert(input_ids, attention_mask, token_type_ids).last_hidden_state  # [BATCH, LEN, DIM]
        res = res[:, 0]  # get CLS token rep [BATCH, DIM]
        # combine with position
        res = res + self.position_enc(torch.tensor([pos for pos in pos_list], dtype=torch.long))  # [BATCH, DIM]
        res = res.unsqueeze(1)  # [BATCH, 1, DIM]
        lstm_output = self.lstm(res)[0].squeeze(1)  # [BATCH, DIM]
        return self.utility(lstm_output).reshape(input_ids.shape[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default='/store/index/msmarco-passage.pisa')
    # parser.add_argument('--query', default='/home/suchana/PycharmProjects/ltr-qpp/qpp/train-10k_test-200/exp_sample.query.sorted')
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--train-its', default=100000, type=int)
    parser.add_argument('--chunk-per-query', default=25, type=int)
    parser.add_argument('--docs-per-query', default=100, type=int)
    parser.add_argument('--outfile', default='/home/suchana/PycharmProjects/ltr-qpp/qpp/models/test_softmax.res')
    parser.add_argument('--skip-utility', action='store_true')
    parser.add_argument('--skip-norel', action='store_true')
    args = parser.parse_args()

    index = PisaIndex(args.index)
    bm25 = index.bm25()
    dataset_train = ir_datasets.load('msmarco-passage/train')
    dataset_eval = ir_datasets.load('msmarco-passage/dev/small')
    # https://github.com/allenai/ir_datasets
    # dataset = ir_datasets.load("msmarco-passage/train/split200-valid")
    queries_train = list(dataset_train.queries)
    dev_size = int(0.01 * len(queries_train))
    train_size = len(queries_train) - dev_size
    train_queries, dev_queries = torch.utils.data.random_split(list(dataset_train.queries), [train_size, dev_size])
    print('Train queries : ', len(list(train_queries)))
    print('Dev queries : ', len(list(dev_queries)))
    eval_queries = list(dataset_eval.queries)
    print('Test queries : ', len(list(eval_queries)))
    qrels = dataset_train.qrels.asdict()
    print('Total qrels : ', len(qrels))
    rng = random.Random(43)


    def _build_input(queries):
        while True:
            for query in list(queries):
                print('Current query : ', query.text)
                res = [r.docno for r in bm25.search(query.text).itertuples(index=False)]  # retrieve a list of documents; store docids
                judged_dids = set(qrels.get(query.query_id, []))
                print('JUDGED : ', judged_dids)
                judged_res = [did for did in res if did in judged_dids]
                if args.skip_norel or len(judged_res) == 0:
                    continue
                res_window = list(more_itertools.windowed(res, n=args.batch_size, step=args.batch_size))
                win_count = 0
                for curr_window in res_window:
                    if win_count < args.chunk_per_query:
                        pos_hit = [win_count * args.batch_size + i for i in range(0, args.batch_size)]
                        texts = {d.doc_id: d.text for d in dataset_train.docs.lookup(list(curr_window)).values()}
                        win_count += 1
                        yield query.text, [texts[did] for did in list(curr_window)], [did in judged_dids for did in
                                                                            list(curr_window)], pos_hit

    def _test_iter():
        while True:
            for query in rng.sample(list(test_queries), k=len(list(test_queries))):
                print('Current query : ', query.text)
                res = [r.docno for r in bm25.search(query.text).itertuples(index=False)]
                res_window = list(more_itertools.windowed(res, n=args.batch_size, step=args.batch_size))
                win_count = 0
                for curr_window in res_window:
                    if win_count < len(res)/args.batch_size:
                        pos_hit = [win_count * args.batch_size + i for i in range(0, args.batch_size)]
                        texts = {d.doc_id: d.text for d in dataset_train.docs.lookup(list(curr_window)).values()}
                        win_count += 1
                        yield query, list(curr_window), [texts[did] for did in list(curr_window)], pos_hit

    tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertQppModel()
    train_iter = _build_input(train_queries)
    dev_iter = _build_input(dev_queries)
    optim = torch.optim.Adam(model.parameters(), lr=2e-5)
    mse = torch.nn.MSELoss()
    suffixes = []
    model_no = 0
    u_loss = 0
    count = 0
    utl_loss = []
    min_valid_loss = np.inf
    if args.skip_utility:
        suffixes.append('noutil')
    if args.skip_norel:
        suffixes.append('norel')
    suffixes.append('{}')
    model_name = f'models/model-{"-".join(suffixes)}.pt'
    with logger.pbar_raw(total=args.train_its, ncols=200) as pbar:
        # train the model
        for train_i in range(len(train_queries) * args.chunk_per_query):
            query, docs, labels, poshit = next(train_iter)
            count += 1
            print('**** : ', query)
            # print('#### : ', docs)
            print('%%%% : ', labels)
            print('@@@@ : ', poshit)
            inputs_train = tokeniser([query for _ in docs], [doc for doc in docs], padding=True,
                                     truncation='only_second',
                                     return_tensors='pt')
            utility = model(poshit, **{k: v for k, v in inputs_train.items()})
            print('UTILITY : ', utility)
            u_loss += mse(utility, torch.tensor([1. if l else 0. for l in labels]))
            print('LOSS : ', u_loss)
            if count == args.chunk_per_query:
                print('============== in here ============')
                u_loss /= args.chunk_per_query
                print('average u loss :::: ', u_loss)
                u_loss.backward()
                optim.step()
                optim.zero_grad()
                print('=========== back propagated ===========')
                if u_loss.cpu().detach().item() != 0:
                    utl_loss.append(u_loss.cpu().detach().item())
                    print('UTL LoSS : ', utl_loss)
                u_loss = 0
                count = 0
                pbar.set_postfix(
                    {'avg_utl_loss': sum(utl_loss) / len(utl_loss),
                     'recent_utl_loss': sum(utl_loss[-100:]) / len(utl_loss[-100:])})
                pbar.update(1)

            # validation after no. of iteration
            if train_i % 20000 == 0:
                model_no = train_i
                valid_loss = 0.0
                model.eval()
                for valid_i in range(len(dev_queries) * args.chunk_per_query):
                    query, docs, labels, poshit = next(dev_iter)
                    print('**** : ', query)
                    # print('#### : ', docs)
                    print('%%%% : ', labels)
                    print('@@@@ : ', poshit)
                    inputs_train = tokeniser([query for _ in docs], [doc for doc in docs], padding=True,
                                             truncation='only_second',
                                             return_tensors='pt')
                    utility = model(poshit, **{k: v for k, v in inputs_train.items()})
                    print('UTILITY : ', utility)
                    loss = mse(utility, torch.tensor([1. if l else 0. for l in labels]))
                    print('LOSS : ', loss)
                    # Calculate total valid Loss
                    valid_loss += loss.item()
                print(f'Training Loss: {sum(utl_loss) / len(utl_loss)} \t\t '
                      f'Validation Loss: {valid_loss / len(dev_queries) * args.chunk_per_query}')
                if min_valid_loss > valid_loss:
                    print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                    min_valid_loss = valid_loss
                    # Saving State Dict
                    torch.save(model.state_dict(), model_name.format(train_i))

    # evaluation
    model.load_state_dict(torch.load('./models/model-'+model_no+'.pt'))
    model.eval()
    test_iter = _test_iter()
    res_out = ''
    log = 0
    for test_i in range(len(test_queries) * args.chunk_per_query):
        log += 1
        query, res, docs, docpos = next(test_iter)
        inputs_test = tokeniser([query.text for _ in docs], [doc for doc in docs], padding=True, truncation='only_second',
                                return_tensors='pt')
        utility = model(docpos, **{k: v for k, v in inputs_test.items()})
        predicted = utility.data.numpy()
        c = 0
        for x in res:
            res_out += str(query.query_id) + '\tQ0\t' + str(res[c]) + '\t' + str(c + 1) + '\t' + str(
                predicted[c]) + '\tltr-qpp' + '\n'
            c += 1

    # compute qpp scores
    qpp_file = open(args.outfile, 'a')
    full_res = res_out.split('\n')
    res_win = list(more_itertools.windowed(full_res, n=args.docs_per_query, step=args.docs_per_query))
    res = ''
    for curr_win in res_win:
        qpp = 0
        weights = 0
        values = [chunk[0].split('\t')[4] for chunk in curr_win]
        print(values)
        for i in range(1, args.docs_per_query+1):
            print(1 / i, '\t', values[i - 1])
            weights += 1 / i
            qpp += int(values[i - 1]) * (1 / i)
        print(curr_win[0][0].split('\t')[0], '\t', round(qpp / weights, 4))
        res += str(curr_win[0][0].split('\t')[0]) + '\t' + str(round(qpp / weights, 4)) + '\n'
    qpp_file.write(res)


if __name__ == '__main__':
    main()