import csv
import argparse
import more_itertools

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-res-file', default='/home/suchana/PycharmProjects/ltr-qpp/qpp/test_softmax.res')
    parser.add_argument('--qpp-file', default='/home/suchana/PycharmProjects/ltr-qpp/qpp/test_softmax_qpp.res')
    parser.add_argument('--docs-per-query', default=20, type=int)
    parser.add_argument('--chunk-size', default=4, type=int)
    args = parser.parse_args()

    res_file = open(args.model_res_file, 'r')
    qpp_file = open(args.qpp_file, 'a')
    full_res = [line for line in res_file]
    res_win = list(more_itertools.windowed(full_res, n=args.docs_per_query, step=args.docs_per_query))
    res = ''
    for curr_win in res_win:
        qpp = 0
        weights = 0
        query_win = list(more_itertools.windowed(curr_win, n=args.chunk_size, step=args.chunk_size))
        values = [chunk[0].split('\t')[4] for chunk in query_win]
        print(values)
        for i in range(1,6):
            print(1/i, '\t', values[i-1])
            weights += 1/i
            qpp += int(values[i-1]) * (1/i)
        print(query_win[0][0].split('\t')[0], '\t', round(qpp/weights, 4))
        res += str(query_win[0][0].split('\t')[0]) + '\t' + str(round(qpp/weights, 4)) + '\n'
    qpp_file.write(res)

if __name__ == '__main__':
  main()
