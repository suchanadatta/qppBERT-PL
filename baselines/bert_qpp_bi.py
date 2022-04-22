from collections import defaultdict
from sentence_transformers import SentenceTransformer, InputExample, losses, util,evaluation
from torch.utils.data import DataLoader
import collections,pickle
from typing import runtime_checkable

col_dic=defaultdict(list)

# replace msmarco collection path
collection_file=open('/store/collection/msmarco/collection.tsv','r').readlines()

for line in collection_file:
    docid,doctext= line.rstrip().split('\t')
    col_dic[docid]=doctext
q_map_dic_train={}
q_map_dic_test={}

# path to evaluation metric per query
# qid<\t>metric_value
query_map_file=open('ltr_qpp/data/train_full/train_full_map_100.tsv','r').readlines()
for line in query_map_file:
    qid,qtext,qmap=line.split('\t')
    q_map_dic_train[qid]={}
    q_map_dic_train[qid] ["qtext"]=qtext
    q_map_dic_train[qid] ["label"]=float(qmap)

# run file including first retrieved documents per query
run_file=open('ltr_qpp/data/msmarco_bm25_100d.tsv','r').readlines()
for line in run_file:
    qid,docid,rank=line.split('\t')
    if qid in q_map_dic_train.keys():
        q_map_dic_train[qid]["doc_text"]=col_dic[docid]

with open('pklfiles/train_bertqpp_bi.pkl', 'wb') as f:
    pickle.dump(q_map_dic_train, f, pickle.HIGHEST_PROTOCOL)

# load data and training
with open('pklfiles/train_bertqpp_bi.pkl', 'rb') as f:
    q_map_first_doc_train=pickle.load(f)
train_examples=[]
for key in q_map_first_doc_train:
    qtext=q_map_first_doc_train[key]["qtext"]
    firstdoctext=q_map_first_doc_train[key]["doc_text"]
    map_value=q_map_first_doc_train[key]["label"]
    train_examples.append( InputExample(texts=[qtext,firstdoctext],label=map_value ))

batch_size=8
num_epoch=1
model = SentenceTransformer('bert-base-uncased')
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
train_loss = losses.CosineSimilarityLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epoch, warmup_steps=100)
model.save("models/biencoder_ap_e_"+str(num_epoch)+'_b_'+str(batch_size))

# load test data
q_file = open('/store/collection/msmarco/queries.dev.small.tsv', 'r').readlines()
q_map_dic = {}
for line in q_file:
    qid, qtext = line.rstrip().split('\t')
    q_map_dic[qid] = {}
    q_map_dic[qid]["qtext"] = qtext

run_file = open('ltr_qpp/data/train_full/bm25_dev.tsv', 'r').readlines()
for line in run_file:
    qid, docid, rank = line.split('\t')
    if qid in q_map_dic.keys():
        q_map_dic[qid]["doc_text"] = col_dic[docid]

with open('pklfiles/test_dev_map.pkl', 'wb') as f:
    pickle.dump(q_map_dic, f, pickle.HIGHEST_PROTOCOL)

# eval
with open('pklfiles/train_bertqpp_bi.pkl', 'rb') as f:
    q_map_first_doc_test=pickle.load(f)

model_name="biencoder_ap_e_1_b_8"
sentences1 = []
sentences2 = []
map_value_test=[]
qs=[]

for key in q_map_first_doc_test:
    sentences1.append(q_map_first_doc_test[key]["qtext"])
    sentences2.append(q_map_first_doc_test[key]["doc_text"])
    qs.append(key)

model=SentenceTransformer("bertqpp/"+model_name)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

actual=[]
predicted=[]
out=open('results/QPP-bi_'+model_name,'w')

for i in range(len(sentences1)):
    out.write(qs[i]+'\t'+str(float(cosine_scores[i][i]))+'\n')
out.close()




