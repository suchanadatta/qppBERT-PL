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
import os

torch.manual_seed(0)
logger = ir_datasets.log.easy()


def position_encoding_init(max_pos, emb_dim):
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
        self.bert = BertModel.from_pretrained(model_name)
        self.lstm = torch.nn.LSTM(input_size=self.emb_dim, hidden_size=self.bert.config.hidden_size,
                                  num_layers=1, bias=True, batch_first=False, dropout=0.2)
        self.utility = torch.nn.Sequential(
            torch.nn.Linear(self.bert.config.hidden_size, 100),
            torch.nn.Linear(100, 5),
            torch.nn.LogSoftmax(dim=1)
        )

    def forward(self, pos_list, input_ids, attention_mask, token_type_ids):
        res = self.bert(input_ids, attention_mask, token_type_ids).last_hidden_state  # [BATCH, LEN, DIM]
        res = res[:, 0]  # get CLS token rep [BATCH, DIM]
        res = res + self.position_enc(torch.tensor([pos for pos in pos_list], dtype=torch.long))  # [BATCH, DIM]
        res = res.unsqueeze(1)  # [BATCH, 1, DIM]
        lstm_output, recent_hidden = self.lstm(res)  # [BATCH, DIM]
        return self.utility(recent_hidden[0].squeeze(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default='/store/index/msmarco-passage.pisa')
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--train-its', default=250000, type=int)
    parser.add_argument('--chunk-per-query', default=25, type=int)
    parser.add_argument('--docs-per-query', default=1000, type=int)
    parser.add_argument('--skip-utility', action='store_true')
    parser.add_argument('--skip-norel', action='store_true')
    args = parser.parse_args()

    index = PisaIndex(args.index)
    bm25 = index.bm25()
    dataset = ir_datasets.load('msmarco-passage/train')
    print('Total no. of docs : ', len(dataset.docs))
    queries = list(dataset.queries)
    dev_size = int(0.0005 * len(queries))
    train_size = len(queries) - dev_size
    train_queries, dev_queries = torch.utils.data.random_split(queries, [train_size, dev_size])
    print('Total train queries : ', len(train_queries))
    print('Total dev queries : ', len(dev_queries))
    qrels = dataset.qrels.asdict()
    print('Total qrels : ', len(qrels))
    rng = random.Random(43)


    def _build_input(queries):
        while True:
            for query in list(queries):
                print('Current query : ', query.text)
                res = [r.docno for r in bm25.search(query.text).itertuples(index=False)]
                judged_dids = set(qrels.get(query.query_id, []))
                judged_res = [did for did in res if did in judged_dids]
                if args.skip_norel or len(judged_res) == 0:
                    continue
                res_window = list(more_itertools.windowed(res, n=args.batch_size, step=args.batch_size))
                win_count = 0
                for curr_window in res_window:
                    if win_count < args.chunk_per_query:
                        pos_hit = [win_count * args.batch_size + i for i in range(0, args.batch_size)]
                        numrel = [judge for judge in judged_dids if judge in curr_window]
                        texts = {d.doc_id: d.text for d in dataset.docs.lookup(list(curr_window)).values()}
                        win_count += 1
                        if texts is not None:
                            yield query.text, [texts[did] for did in list(curr_window)], len(numrel), pos_hit


    tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertQppModel()
    train_iter = _build_input(train_queries)
    dev_iter = _build_input(dev_queries)
    optim = torch.optim.Adam(model.parameters(), lr=2e-5)
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
    model_name = f'../models/model-{"-".join(suffixes)}.pt'
    with logger.pbar_raw(total=args.train_its, ncols=200) as pbar:
        # train the model
        for train_i in range(len(train_queries) * args.chunk_per_query * args.batch_size):
            query, docs, numrel, poshit = next(train_iter)
            count += 1
            inputs_train = tokeniser([query for _ in docs], [doc for doc in docs], padding=True,
                                     truncation='only_second',
                                     return_tensors='pt')
            utility = model(poshit, **{k: v for k, v in inputs_train.items()})
            # print('STEP LOSS : ', F.nll_loss(utility, torch.tensor([numrel])).item())
            u_loss += F.nll_loss(utility, torch.tensor([numrel])).item()
            if count == args.chunk_per_query:
                u_loss /= args.chunk_per_query
                u_loss = torch.tensor([u_loss], requires_grad=True)
                u_loss.backward()
                optim.step()
                optim.zero_grad()
                if u_loss.cpu().detach().item() != 0:
                    utl_loss.append(u_loss.cpu().detach().item())
                u_loss = 0
                count = 0
                pbar.set_postfix(
                    {'avg_utl_loss': sum(utl_loss) / len(utl_loss),
                     'recent_utl_loss': sum(utl_loss[-100:]) / len(utl_loss[-100:])})
                pbar.update(1)

            # validation after no. of iteration
            if (train_i+1) % 20000 == 0:
                # model_no = train_i
                valid_loss = 0.0
                model.eval()
                print('\n============ i am in validation ===========')
                for valid_i in range(len(dev_queries) * args.chunk_per_query * args.batch_size):
                    query, docs, numrel, poshit = next(dev_iter)
                    inputs_train = tokeniser([query for _ in docs], [doc for doc in docs], padding=True,
                                             truncation='only_second',
                                             return_tensors='pt')
                    utility = model(poshit, **{k: v for k, v in inputs_train.items()})
                    loss = F.nll_loss(utility, torch.tensor([numrel]))
                    # Calculate total valid Loss
                    valid_loss += loss.item()
                print(f'Training Loss: {sum(utl_loss) / len(utl_loss)} \t\t '
                      f'Validation Loss: {valid_loss / len(dev_queries) * args.chunk_per_query}')
                if min_valid_loss > valid_loss:
                    print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                    min_valid_loss = valid_loss
                    # delete previous models and save the new one
                    for f in os.listdir('../models/'):
                        os.remove(os.path.join('../models/', f))
                    print('\n\n=========writing the model ==========')
                    torch.save(model.state_dict(), model_name.format(train_i))

if __name__ == '__main__':
    main()
