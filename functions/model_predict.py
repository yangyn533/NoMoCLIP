import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from spektral.layers import GCNConv, GlobalMaxPool
from tensorflow.keras.initializers import GlorotUniform


def make_comcod():
    combination_methods = ['Entropy density of transcript (1D)', 'Global descriptor (1D)', 'Codon related (1D)',
                           'Guanine-cytosine related (1D)', 'EIIP based spectrum (1D)']
    return combination_methods


def mknpy_RNAonly(combin2, datapath, indices):
    for numc in indices:
        file_tempA = np.load(os.path.join(datapath, combin2[numc] + '.npy'))
        if numc == 0:
            combnpysA = file_tempA
        else:
            combnpysA = np.concatenate((combnpysA, file_tempA), axis=1)
    return combnpysA


def con_seq_1D(datapath):
    combination_methods = make_comcod()
    combin_num = [0, 1, 2, 3, 4]
    combnpysA = mknpy_RNAonly(combination_methods, datapath, combin_num)
    return combnpysA


def read_fasta(filename):
    ids = []
    sequences = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('>'):
                ids.append(line.strip())
                continue
            else:
                sequences.append(line.strip().upper().replace('T', 'U'))
    return ids, sequences


def predict(args):
    model_path = args.model_path
    set_path = args.set_path
    out_path = args.out_path

    seq_path = os.path.join(set_path, f'ACGU/test.fa')
    ids, seqs = read_fasta(seq_path)
    pos_test = np.load(os.path.join(set_path, 'onekey/pos_inf.npy')).astype(np.float32)
    ss_test = np.load(os.path.join(set_path, 'ss/ss.npy')).astype(np.float32)
    ss_onehot_test = np.load(os.path.join(set_path, 'ss/ss_onehot.npy')).astype(np.float32)
    nlp_test_adj = np.load(os.path.join(set_path, 'nlp/adj.npy')).astype(np.float32)
    nlp_test_node = np.load(os.path.join(set_path, 'nlp/node_embedding_inf.npy')).astype(np.float32)
    seq_test_dimone = con_seq_1D(os.path.join(set_path, 'sequential_feat', 'RNAonly', 'encoding_features/'))
    y_test = np.load(os.path.join(set_path, 'onekey/label.npy'))
    print('y_test:', y_test.shape)

    gcn = GCNConv(256)
    globalMaxPool = GlobalMaxPool()
    layer_dic = {'GCNConv': gcn, 'GlobalMaxPool': globalMaxPool, 'GlorotUniform': GlorotUniform}
    data_dic = [pos_test, ss_onehot_test, nlp_test_node, nlp_test_adj, seq_test_dimone, ss_test]

    model = load_model(model_path,
                       custom_objects=layer_dic,
                       compile=False)

    ss_y_hat_test = model.predict(data_dic)
    yprob = ss_y_hat_test[:, 1]
    ypred = np.argmax(ss_y_hat_test, axis=-1)
    data = {
        "ID": ids,
        "Sequence": seqs,
        "Label": y_test,
        "pred": ypred,
        "Score": yprob
    }
    df = pd.DataFrame(data)

    # Writing to a CSV File
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    output_csv = os.path.join(out_path, f"pred_results.csv")
    df.to_csv(output_csv, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--set_path",
        type=str,
        default="",
        help="Path to the feature file"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="",
        help="Output path of prediction results"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Model path for prediction"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=1
    )
    args = parser.parse_args()

    gpu_id = args.gpu_id
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.system('echo $CUDA_VISIBLE_DEVICES')

    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=tf_config)
    gpus = tf.config.experimental.list_physical_devices('GPU')

    predict(args)


if __name__ == '__main__':
    main()
