import os
import argparse

import pandas as pd


def read_fasta(path):
    seqs = []
    seq_ids = []
    dataY = []
    with open(path) as f:
        for line in f:
            if '>' in line:
                label = line.strip().split('_')[-1]
                if not label.isdigit():     # The label is not marked at the end of the seq id
                    label = '1'     # Default label = 1
                    seq_id = line.strip().replace(" ", "|").replace("\t", "|")
                else:
                    seq_id = line.strip().replace(" ", "|").replace("\t", "|").rsplit('_', 1)[0]
                seq_id = seq_id[0]+seq_id[1:].replace('>', 'to')
                seq_ids.append(seq_id)
                dataY.append(int(label))
            if '>' not in line:
                seqs.append(line.strip().upper().replace('U', 'T'))
    return seqs, seq_ids, dataY


def save_to_fasta(ids, seqs, filename):
    with open(filename, 'w') as fasta_file:
        for rna_id, sequence in zip(ids, seqs):
            fasta_file.write(f"{rna_id}\n")
            fasta_file.write(f"{sequence}\n")


def save_to_csv(ids, labels, csv_path):
    dataframe = pd.DataFrame({'Seqname': ids, 'Label': labels})
    dataframe.to_csv(csv_path, index=False)


def process(args):
    base_path = args.base_path
    set_path = args.set_path

    seqs, ids, labels = read_fasta(base_path)
    opath = os.path.join(set_path, 'sequential')
    if not os.path.exists(opath):
        os.makedirs(opath)
    save_to_fasta(ids, seqs, os.path.join(opath, 'test.fa'))
    save_to_csv(ids, labels, os.path.join(opath, 'test.csv'))


def run_instinct_fea(args):
    set_path = args.set_path
    out_path = args.out_path
    method_path = args.method_path
    num = args.num

    fastapath_test = os.path.join(set_path, 'sequential/test.fa')
    label_csvpath_test = os.path.join(set_path, 'sequential/test.csv')
    outpath_test = os.path.join(out_path, 'sequential_feat')
    os.system(f"python {method_path}/corain.py \
                                   -t='RNAonly' \
                                   -a=" + fastapath_test + " \
                                   -l=" + label_csvpath_test + " \
                                   -o=" + outpath_test + " \
                                   -s='csvnpy' \
                                   -d=1 \
                                   -n=" + str(num))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        default="",
        help="Path to the fasta files"
    )
    parser.add_argument(
        "--set_path",
        type=str,
        default="",
        help="Path to the fasta and csv files"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="",
        help="Output path"
    )
    parser.add_argument(
        "--method_path",
        type=str,
        default="../sequential_feat/",
        help="Path to the method files"
    )
    parser.add_argument(
        "--num",
        type=int,
        choices=[2, 3, 5, 7, 10],
        help="Selection range of additional features"
    )
    args = parser.parse_args()

    process(args)
    run_instinct_fea(args)


if __name__ == '__main__':
    main()
