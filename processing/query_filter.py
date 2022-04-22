import pyterrier as pt
if not pt.started():
  pt.init()
from pyterrier_pisa import PisaIndex
import ir_datasets
import argparse

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--index', default='/store/index/msmarco-passage.pisa')
  args = parser.parse_args()

  hit_query = ''
  no_hit_query = ''
  c = 0
  index = PisaIndex(args.index)
  bm25 = index.bm25()
  # dataset = ir_datasets.load('msmarco-passage/train')
  # https://github.com/allenai/ir_datasets
  dataset = ir_datasets.load("msmarco-passage/dev/small")
  queries = list(dataset.queries)
  print('Total no. of queries : ', len(queries))
  qrels = dataset.qrels.asdict()
  print('Total qrels : ', len(qrels))
  for query in queries:
    c += 1
    print('Current query :', query)
    res = [r.docno for r in bm25.search(query.text).itertuples(index=False)]
    # print(len(res))
    judged_dids = set(qrels.get(query.query_id, []))
    print('JUDGED : ', judged_dids)
    # if len(judged_dids) == 0 or len(res) == 0:
    if len(res) == 0:
      print('======= problem here ======')
      no_hit_query += str(query.query_id) + '\t' + query.text + '\n'
      print(no_hit_query)
    else:
      hit_query += str(query.query_id) + '\t' + query.text + '\n'
      print(hit_query)
    if c % 100 == 0:
      with open('../eval_no_hit.query', 'a') as no_hit:
        no_hit.write(no_hit_query)
        no_hit_query = ''
      with open('../eval_hit.query', 'a') as hit:
        hit.write(hit_query)
        hit_query = ''

if __name__ == '__main__':
  main()

