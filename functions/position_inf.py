import os
import argparse
import numpy as np


base_positions = {
    'A': 0,
    'U': 1,
    'C': 2,
    'G': 3,
    0: 'A',
    1: 'U',
    2: 'C',
    3: 'G',
}


def mk_dir(dir):
    try:
        os.makedirs(dir)
    except OSError:
        print('Can not make directory:', dir)


def one_hot_encode_sequence(seq, pad_to_len=-1):
    output_len = len(seq)

    if pad_to_len > 0:
        assert pad_to_len >= output_len
        output_len = pad_to_len
    encoded_seq = np.zeros((output_len, 4), dtype=np.float32)
    for i, base in enumerate(seq):
        if base == 'N':
            print('N in sequence')
            encoded_seq[i][:] = 0.25
        else:
            encoded_seq[i][base_positions[base]] = 1.
    return encoded_seq


def dealwithdata(fastapath):
    dataX = []
    dataY = []
    print(fastapath, ' dealing')
    with open(fastapath) as f:

        for line in f:
            if '>' in line:
                label = line.strip().split('_')[-1]
                if not label.isdigit():  # The label is not marked at the end of the seq id
                    label = '1'  # Default label = 1
                dataY.append(int(label))
            if '>' not in line:
                dataX.append(one_hot_encode_sequence(line.strip().upper().replace('T', 'U')))
    return np.array(dataX), np.array(dataY)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--set_path",
        type=str,
        default="",
        help="Path to the fasta files"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="",
        help="Output path"
    )
    args = parser.parse_args()

    set_path = args.set_path
    out_path = args.out_path

    fastapath = set_path
    pos_matrix, label_matrix = dealwithdata(fastapath)
    outpath_test = os.path.join(out_path, 'onekey')
    mk_dir(outpath_test)
    np.save(os.path.join(outpath_test, 'pos_inf.npy'), pos_matrix)
    np.save(os.path.join(outpath_test, 'label.npy'), label_matrix)
    
    
if __name__ == '__main__':
    main()
                