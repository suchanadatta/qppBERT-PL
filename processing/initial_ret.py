import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_pisa import PisaIndex
import ir_datasets
import argparse, csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default='/store/index/msmarco-passage.pisa')
    parser.add_argument('--outfile', default='/home/suchana/PycharmProjects/ltr-qpp/data/bertqpp_ablation/trecdl_lmdir_norm.res')
    parser.add_argument('--query-file', default='/home/suchana/PycharmProjects/ColBERT/data/trecDL_data/trecDL_q97.tsv')
    args = parser.parse_args()

    index = PisaIndex(args.index)
    bm25 = index.bm25(k1=1.2, b=0.4)
    lmdir = index.qld(mu=1000.)
    # from irdatasets
    # dataset = ir_datasets.load('msmarco-passage/train/split200-valid')
    # queries = list(dataset.queries)
    # print('Total no. of queries : ', len(queries))

    # from tsv file
    q_file = open(args.query_file, 'r')
    q_read = csv.reader(q_file, delimiter='\t')
    q_dict = {line[0] : line[1] for line in q_read}
    print(q_dict)

    # for irdataset
    # res_out = ''
    # with open(args.outfile, 'a') as outFile:
    #     for query in list(queries):
    #         print('Current query : ', query.text)
    #         res = lmdir.search(query.text)
    #         print('total hit : ', len(res))
    #         docid = res['docno'].values
    #         # print(docid)
    #         scores = res['score'].values
    #         # print(scores)
    #         rank = 0
    #         while rank < min(len(res), 1000):
    #             res_out += query.query_id + '\tQ0\t' + str(docid[rank]) + '\t' + str(rank+1) + '\t' +\
    #                        str(scores[rank]) + '\trerank-bert' + '\n'
    #             rank += 1
    #         outFile.write(res_out)
    #         res_out = ''
    # outFile.close()

    # for tsv query file
    # res_out = ''
    # with open(args.outfile, 'a') as outFile:
    #     for q_id, q_text in q_dict.items():
    #         print('Current query : ', q_id, '\t', q_text)
    #         res = lmdir.search(q_text)
    #         print('total hit : ', len(res))
    #         docid = res['docno'].values
    #         # print(docid)
    #         scores = res['score'].values
    #         # print(scores)
    #         rank = 0
    #         while rank < min(len(res), 100):
    #             res_out += str(q_id) + '\tQ0\t' + str(docid[rank]) + '\t' + str(rank+1) + '\t' +\
    #                        str(scores[rank]) + '\tlmdir' + '\n'
    #             rank += 1
    #         outFile.write(res_out)
    #         res_out = ''
    # outFile.close()

    # for tsv query file (normalize score)
    res_out = ''
    with open(args.outfile, 'a') as outFile:
        for q_id, q_text in q_dict.items():
            print('Current query : ', q_id, '\t', q_text)
            res = lmdir.search(q_text)
            print('total hit : ', len(res))
            docid = res['docno'].values
            # print(docid)
            scores = res['score'].values
            print(sum(scores))
            rank = 0
            while rank < min(len(res), 100):
                res_out += str(q_id) + '\tQ0\t' + str(docid[rank]) + '\t' + str(rank + 1) + '\t' + \
                           str(scores[rank]/100) + '\tltr-sigmoid' + '\n'
                rank += 1
            outFile.write(res_out)
            res_out = ''
    outFile.close()

if __name__ == '__main__':
  main()


