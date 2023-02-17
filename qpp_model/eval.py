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
import argparse, csv
import more_itertools
import numpy as np
import os, pickle

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
    parser.add_argument('--index', default='/store/index/msmarco-v2.pisa')
    parser.add_argument('--dataset', default='irdataset')
    parser.add_argument('--collection', default='/store/collection/ms-marco/msmarco-v2.pickle')
    parser.add_argument('--query', default='../data/trec-dl-2021/queries.tsv')
    parser.add_argument('--checkpoint', default='/home/suchana/PycharmProjects/ltr-qpp/qpp/msmarco/models/model-499999.pt')
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--test-its', default=50000, type=int)
    parser.add_argument('--chunk-per-query', default=25, type=int)
    parser.add_argument('--docs-per-query', default=1000, type=int)
    parser.add_argument('--outfile', default='/home/suchana/PycharmProjects/qppBERT-PL/dl21.out')
    parser.add_argument('--per-chunk-pred', default='/home/suchana/PycharmProjects/qppBERT-PL/dl21.pred')
    parser.add_argument('--pred-ap', default='/home/suchana/PycharmProjects/qppBERT-PL/dl21.pred.ap')
    parser.add_argument('--skip-utility', action='store_true')
    parser.add_argument('--skip-norel', action='store_true')
    args = parser.parse_args()

    # if use irdatasets
    def _build_data_irdataset(queries):
        while True:
            for query in list(queries):
                print('Current query : ', query.text)
                res = [r.docno for r in bm25.search(query.text).itertuples(index=False)]
                res_window = list(more_itertools.windowed(res, n=args.batch_size, step=args.batch_size))
                win_count = 0
                for curr_window in res_window:
                    if win_count < args.chunk_per_query:
                        pos_hit = [win_count * args.batch_size + i for i in range(0, args.batch_size)]
                        texts = {d.doc_id: d.text for d in dataset.docs.lookup(list(curr_window)).values()}
                        win_count += 1
                        yield query, list(curr_window), [texts[did] for did in list(curr_window)], pos_hit, win_count

    # if use jsonl collection
    def _build_data_pickle(queries, data):
        while True:
            for qid, qtext in queries.items():
                print('Current query : ', qtext)
                res = set([r.docno for r in bm25.search(qtext).itertuples(index=False)])
                res_window = list(more_itertools.windowed(res, n=args.batch_size, step=args.batch_size))
                win_count = 0
                for curr_window in res_window:
                    if win_count < args.chunk_per_query:
                        pos_hit = [win_count * args.batch_size + i for i in range(0, args.batch_size)]
                        texts = {id: text for id, text in data.items()}
                        win_count += 1
                        yield qid, qtext, list(curr_window), [texts[did] for did in list(curr_window)], pos_hit, win_count

    # predict AP to measure QPP
    def predict_ap():
        pred_ap = {}  # qid, weighted ap
        pred_file = csv.reader(open(args.per_chunk_pred, 'r'), delimiter='\t')
        ap_file = open(args.pred_ap, 'a')
        qid = ''
        weighted_ap = 0
        rel_doc = 0
        rel_doc_count = ''
        for line in pred_file:
            if qid == '' or line[0] == qid:
                qid = line[0]
                weighted_ap += float(1/float(line[1])) * float(line[2])
                if int(line[2]) > 0:
                    rel_doc += 1
            else:
                pred_ap[qid] = weighted_ap
                rel_doc_count += qid + '\t' + str(rel_doc) + '\n'
                weighted_ap = 0
                rel_doc = 0
                qid = line[0]
                weighted_ap += float(1 / float(line[1])) * float(line[2])
        pred_ap[qid] = weighted_ap
        rel_doc_count += qid + '\t' + str(rel_doc) + '\n'
        # print(rel_doc_count)

        # min-max normalize
        ap_norm = ''
        for qid, ap in pred_ap.items():
            norm = round((float(ap) - min(pred_ap.values())) /
                                 (max(pred_ap.values()) - min(pred_ap.values())), 5)
            ap_norm += qid + '\t' + str(norm) + '\n'
        ap_file.write(ap_norm)

    if args.dataset == 'irdataset':
        index = PisaIndex(args.index)
        bm25 = index.bm25()
        dataset = ir_datasets.load('msmarco-passage/trec-dl-2019/judged')
        print('Total no. of docs : ', len(dataset.docs))
        queries = list(dataset.queries)
        print('Total test queries : ', len(queries))
        test_iter = _build_data_irdataset(queries)

        tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertQppModel()
        model.load_state_dict(torch.load(args.checkpoint))
        model.eval()
        res_out = ''
        rel_out = ''
        chunk_pred = open(args.outfile, 'a')
        rel_pred = open(args.per_chunk_pred, 'a')
        for test_i in range(len(queries) * args.chunk_per_query):
            query, res, docs, docpos, chunk_id = next(test_iter)
            inputs_test = tokeniser([query.text for _ in docs], [doc for doc in docs], padding=True,
                                    truncation='only_second',
                                    return_tensors='pt')
            utility = model(docpos, **{k: v for k, v in inputs_test.items()})
            reldocs = torch.argmax(utility).item()
            c = 0
            for x in res:
                res_out += str(query.query_id) + '\t' + str(res[c]) + '\t' + str(c + 1)
                if c == 0:
                    res_out += '\t' + str(chunk_id) + '\t' + str(reldocs) + '\n'
                else:
                    res_out += '\n'
                c += 1
            rel_out += str(query.query_id) + '\t' + str(chunk_id) + '\t' + str(reldocs) + '\n'
            print(res_out)
            print(rel_out)
            if test_i % 10 == 0:
                chunk_pred.write(res_out)
                rel_pred.write(rel_out)
                res_out = ''
                rel_out = ''
        chunk_pred.write(res_out)
        rel_pred.write(rel_out)
        chunk_pred.close()
        rel_pred.close()
        predict_ap()

    else:
        index = PisaIndex(args.index)
        bm25 = index.bm25()
        q_read = csv.reader(open(args.query, 'r'), delimiter='\t')
        q_dict = {line[0]: line[1] for line in q_read}
        print(len(q_dict))
        data_dict = pickle.load(open(args.collection, 'rb'))
        print('Data loaded .....')
        print(len(data_dict))
        test_iter = _build_data_pickle(q_dict, data_dict)

        tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertQppModel()
        model.load_state_dict(torch.load(args.checkpoint))
        model.eval()
        res_out = ''
        rel_out = ''
        chunk_pred = open(args.outfile, 'a')
        rel_pred = open(args.per_chunk_pred, 'a')
        for test_i in range(len(q_dict) * args.chunk_per_query):
            qid, qtext, res, docs, docpos, chunk_id = next(test_iter)
            inputs_test = tokeniser([qtext for _ in docs], [doc for doc in docs], padding=True,
                                    truncation='only_second',
                                    return_tensors='pt')
            utility = model(docpos, **{k: v for k, v in inputs_test.items()})
            reldocs = torch.argmax(utility).item()
            c = 0
            for x in res:
                res_out += str(qid) + '\t' + str(res[c]) + '\t' + str(c + 1)
                if c == 0:
                    res_out += '\t' + str(chunk_id) + '\t' + str(reldocs) + '\n'
                else:
                    res_out += '\n'
                c += 1
            rel_out += str(qid) + '\t' + str(chunk_id) + '\t' + str(reldocs) + '\n'
            print(res_out)
            print(rel_out)
            if test_i % 100 == 0:
                chunk_pred.write(res_out)
                rel_pred.write(rel_out)
                res_out = ''
                rel_out = ''
        chunk_pred.write(res_out)
        rel_pred.write(rel_out)
        chunk_pred.close()
        rel_pred.close()
        predict_ap()

if __name__ == '__main__':
    main()
