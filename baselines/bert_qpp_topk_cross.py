import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_pisa import PisaIndex
import ir_datasets
import random
import torch
from transformers import BertModel, BertTokenizer, BertConfig
from sentence_transformers import SentenceTransformer, InputExample, losses, util,evaluation
from torch.utils.data import DataLoader
import pickle
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
import math
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
        self.bert = BertModel.from_pretrained(model_name)
        self.lstm = torch.nn.LSTM(input_size=self.emb_dim, hidden_size=self.bert.config.hidden_size,
                                  num_layers=1, bias=True, batch_first=False, dropout=0.2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        res = self.bert(input_ids, attention_mask, token_type_ids).last_hidden_state  # [BATCH, LEN, DIM]
        res = res.unsqueeze(1)  # [BATCH, 1, DIM]
        lstm_output, recent_hidden = self.lstm(res)  # [BATCH, DIM]
        return lstm_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default='/store/index/msmarco-passage.pisa')
    # parser.add_argument('--query', default='/home/suchana/PycharmProjects/ltr-qpp/qpp/train-10k_test-200/exp_sample.query.sorted')
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--train-its', default=100000, type=int)
    parser.add_argument('--chunk-per-query', default=100, type=int)
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
    train_queries, dev_queries = torch.utils.data.random_split(list(dataset.queries), [train_size, dev_size])
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
                res = res[:args.docs_per_query]
                numrel = [judge for judge in judged_dids if judge in res]
                texts = {d.doc_id: d.text for d in dataset.docs.lookup(res).values()}
                yield query.text, [texts[did] for did in res], [did in judged_dids for did in res], numrel

    def _test_iter():
        while True:
            for query in rng.sample(list(test_queries), k=len(list(test_queries))):
                print('Current query : ', query.text)
                res = [r.docno for r in
                       bm25.search(query.text).itertuples(index=False)]  # retrieve a list of documents; store docids
                judged_dids = set(qrels.get(query.query_id, []))
                print('JUDGED : ', judged_dids)
                judged_res = [did for did in res if did in judged_dids]
                if args.skip_norel or len(judged_res) == 0:
                    continue
                res = res[:args.docs_per_query]
                texts = {d.doc_id: d.text for d in dataset.docs.lookup(res).values()}
                yield query.text, [texts[did] for did in res], [did in judged_dids for did in res]

    tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertQppModel()
    model_cross = CrossEncoder(model_name, num_labels=1)
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
    model_name = f'models/model-{"-".join(suffixes)}.pt'
    with logger.pbar_raw(total=args.train_its, ncols=200) as pbar:
        # train the model
        for train_i in range(len(train_queries) * args.chunk_per_query):
            query, docs, labels = next(train_iter)
            input_query = model_cross.encode(query, convert_to_tensor=True)
            input_doc = tokeniser([doc for doc in docs], padding=True, truncation='only_second',
                                  return_tensors='pt')
            res_doc = model(**{k: v for k, v in input_doc.items()})
            u_loss += F.cosine_embedding_loss(input_query, res_doc, labels, reduction='mean')
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
                    query, docs, numrel = next(dev_iter)
                    input_query = model_cross.encode(query, convert_to_tensor=True)
                    input_doc = tokeniser([doc for doc in docs], padding=True, truncation='only_second',
                                          return_tensors='pt')
                    res_doc = model(**{k: v for k, v in input_doc.items()})
                    u_loss += F.cosine_embedding_loss(input_query, res_doc, labels, reduction='mean')
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
        query, res, docs = next(test_iter)
        input_query = model_cross.encode(query, convert_to_tensor=True)
        input_doc = tokeniser([doc for doc in docs], padding=True, truncation='only_second',
                              return_tensors='pt')
        res_doc = model(**{k: v for k, v in input_doc.items()})
        cosine_scores = util.pytorch_cos_sim(input_query, res_doc)
        for i in range(len(sentences1)):
            out.write(qs[i] + '\t' + str(float(cosine_scores[i][i])) + '\n')
        out.close()

if __name__ == '__main__':
    main()