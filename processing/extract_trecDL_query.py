import csv

qrels = open('/store/collection/ms-marco/trec_DL/pass_2020.qrels', 'r')
queries = open('/home/suchana/.ir_datasets/msmarco-passage/queries.eval.tsv', 'r')
qrel_file = csv.reader(qrels, delimiter='\t')
query_file = csv.reader(queries, delimiter='\t')
res = open('/store/collection/ms-marco/trec_DL/pass_2020.queries', 'w')

qrel_set = set(line[0] for line in qrel_file)
print(len(qrel_set))
q_dict = {line[0] : line[1] for line in query_file}
print(len(q_dict))
for qid in qrel_set:
    res.writelines(qid + '\t' + q_dict.get(qid) + '\n')
res.close()

