import ir_datasets
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re, string

# Preprocessing and tokenizing
def preprocessing(line):
    line = line.lower()
    line = re.sub(r"[{}]".format(string.punctuation), " ", line)
    print('LINE : ', line)
    return line

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', default='msmarco-passage/train')
    parser.add_argument('--output', default='exp_sample.query')
    args = parser.parse_args()

    dataset = ir_datasets.load(args.query)
    queries = list(dataset.queries)
    print('total train queries : ', len(list(queries)))

    all_query = []
    for query in list(queries):
        all_query.append(query.text)
    print('query list : ', len(all_query))

    tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocessing)
    tfidf = tfidf_vectorizer.fit_transform(all_query)
    # print(tfidf)
    kmeans = KMeans(n_clusters=20).fit(tfidf)
    count = 0
    res = ''
    with open(args.output, 'a') as outfile:
        for q in all_query:
            count += 1
            cluster = kmeans.predict(tfidf_vectorizer.transform([q]))
            print(q + '\t' + str(cluster))
            res += q + '\t' + str(cluster[0]) + '\n'
            if count % 100 == 0:
                outfile.write(res)
                res = ''
    outfile.close()

if __name__ == '__main__':
    main()