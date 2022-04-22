import argparse, csv
import ir_datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queryfile', default='/home/suchana/PycharmProjects/ColBERT/data/trecDL_data/trecDL_q97.tsv')
    parser.add_argument('--outfile', default='/home/suchana/PycharmProjects/ColBERT/data/trecDL_data/trecDL_q97.xml')
    args = parser.parse_args()

    def from_irdataset():
        dataset = ir_datasets.load(args.queryfile)
        queries = list(dataset.queries)
        print('Total queries : ', len(queries))
        query_file = open(args.outfile, 'w')
        res = ''
        res += '<topics>\n'
        for query in queries:
            res += '<top>\n<num>' + query.query_id + '</num>\n' + '<title>' + query.text + '</title>\n' + \
                   '<desc>' + query.text + '</desc>\n' + '<narr>' + query.text + '</narr>\n</top>\n'
        res += '</topics>'
        query_file.write(res)

    def from_tsvqfile():
        q_file = open(args.queryfile, 'r')
        q_read = csv.reader(q_file, delimiter='\t')
        q_dict = {line[0] : line[1] for line in q_read}
        print('Total queries : ', len(q_dict))
        trec_file = open(args.outfile, 'w')
        res = ''
        res += '<topics>\n'
        for q_id, q_text in q_dict.items():
            res += '<top>\n<num>' + q_id + '</num>\n' + '<title>' + q_text + '</title>\n' + \
                   '<desc>' + q_text + '</desc>\n' + '<narr>' + q_text + '</narr>\n</top>\n'
        res += '</topics>'
        trec_file.write(res)

    # from_irdataset()
    from_tsvqfile()

if __name__ == '__main__':
    main()