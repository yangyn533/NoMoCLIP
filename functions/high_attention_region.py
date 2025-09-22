import argparse
import os
import re

import pandas as pd
import tensorflow as tf
import numpy as np
from scipy import special
from spektral.layers import GCNConv, GlobalMaxPool
from visualization import plot_saliency
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import GlorotUniform


def seq2kmer_bert(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    seq_length = len(seq)
    kmer = [seq[x:x + k] for x in range(seq_length - k + 1)]
    kmers = " ".join(kmer)
    return kmers


def convert_one_hot(sequence, max_length=None):
    """convert DNA/RNA sequences to a one-hot representation"""
    one_hot_seq = []
    for seq in sequence:
        seq = seq.upper()
        seq_length = len(seq)
        one_hot = np.zeros((4, seq_length))
        index = [j for j in range(seq_length) if seq[j] == 'A']
        # print(index)
        one_hot[0, index] = 1
        index = [j for j in range(seq_length) if seq[j] == 'C']
        one_hot[1, index] = 1
        index = [j for j in range(seq_length) if seq[j] == 'G']
        one_hot[2, index] = 1
        index = [j for j in range(seq_length) if (seq[j] == 'U') | (seq[j] == 'T')]
        one_hot[3, index] = 1

        # handle boundary conditions with zero-padding
        if max_length:
            offset1 = int((max_length - seq_length) / 2)
            offset2 = max_length - seq_length - offset1

            if offset1:
                one_hot = np.hstack([np.zeros((4, offset1)), one_hot])
            if offset2:
                one_hot = np.hstack([one_hot, np.zeros((4, offset2))])

        one_hot_seq.append(one_hot)

    # convert to numpy array
    one_hot_seq = np.array(one_hot_seq)
    return one_hot_seq


def convert_one_hot2(sequence, attention, max_length=None):
    """convert DNA/RNA sequences to a one-hot representation"""
    one_hot_seq = []
    for seq in sequence:
        seq = seq.upper()
        seq_length = len(seq)
        one_hot = np.zeros((4, seq_length))
        index = [j for j in range(seq_length) if seq[j] == 'A']
        for i in index:
            one_hot[0, i] = attention[i]
        index = [j for j in range(seq_length) if seq[j] == 'C']
        for i in index:
            one_hot[1, i] = attention[i]
        index = [j for j in range(seq_length) if seq[j] == 'G']
        for i in index:
            one_hot[2, i] = attention[i]
        index = [j for j in range(seq_length) if (seq[j] == 'U') | (seq[j] == 'T')]
        for i in index:
            one_hot[3, i] = attention[i]

        # handle boundary conditions with zero-padding
        if max_length:
            offset1 = int((max_length - seq_length) / 2)
            offset2 = max_length - seq_length - offset1

            if offset1:
                one_hot = np.hstack([np.zeros((4, offset1)), one_hot])
            if offset2:
                one_hot = np.hstack([one_hot, np.zeros((4, offset2))])

        one_hot_seq.append(one_hot)

    # convert to numpy array
    one_hot_seq = np.array(one_hot_seq)
    return one_hot_seq


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


def normalize_distribution(dist):
    # Ensure the distributions are normalized to prob distribution and make sure (0,1)
    epsilon = 1e-10
    dist += epsilon
    # Adjust axis to sum across the sequence length for normalization
    sum_dist = np.sum(dist, axis=2, keepdims=True)
    normalized_dist = dist / sum_dist
    return normalized_dist


def compute_kl_divergence(p, q):
    # p is the reference allele distribution and q is the alternative allele
    # p: (N,1,101)
    p = normalize_distribution(p)
    q = normalize_distribution(q)

    # Compute KL divergence for each element in the batch
    kl_elementwise = special.rel_entr(p, q)
    kl_div = np.sum(kl_elementwise, axis=2)
    kl_div = np.squeeze(kl_div)
    return kl_div


class MultiInputSmoothGrad:
    def __init__(self, model, x_stddev=0.015, nsamples=20, magnitude=2):
        """
        :param model:
        :param x_stddev: The standard deviation of the added noise
        :param nsamples: SmoothGrad sampling times
        :param magnitude: 0: Directly accumulate gradient;
                          1: Accumulate the absolute value of the gradient;
                          2: Accumulate the square value of the gradient
        """
        self.model = model
        self.x_stddev = x_stddev
        self.nsamples = nsamples
        self.magnitude = magnitude

    def get_gradients(self, inputs, target_class_idx=None):
        inputs_tf = [tf.convert_to_tensor(inp) for inp in inputs]
        with tf.GradientTape() as tape:
            tape.watch(inputs_tf)
            preds = self.model(inputs_tf)
            if target_class_idx is not None:
                preds = preds[:, target_class_idx]
            else:
                preds = tf.reduce_max(preds, axis=-1)
        grads = tape.gradient(preds, inputs_tf)
        return [g.numpy() for g in grads]

    def get_smooth_gradients(self, inputs, target_class_idx=None):
        np.random.seed(42)
        smooth_grads = [np.zeros_like(inp) for inp in inputs]

        for _ in range(self.nsamples):
            noisy_inputs = []
            for inp in inputs:
                inp_range = np.max(inp) - np.min(inp)
                stddev = self.x_stddev * inp_range
                noise = np.random.normal(0, stddev, inp.shape)
                noisy_inputs.append(inp + noise)

            grads = self.get_gradients(noisy_inputs, target_class_idx)

            for i, grad in enumerate(grads):
                if self.magnitude == 1:
                    smooth_grads[i] += np.abs(grad)
                elif self.magnitude == 2:
                    smooth_grads[i] += grad ** 2
                else:
                    smooth_grads[i] += grad

        smooth_grads = [g / self.nsamples for g in smooth_grads]
        return smooth_grads


def saliency_smoothGrad(data, data_mut, model, smooth=True, nsamples=20, stddev=0.015, batch_size=64):
    """
    Computes SmoothGrad saliency maps and performs KLD difference calculations on wild/mutant samples
    """
    saliency_generator = MultiInputSmoothGrad(model, x_stddev=stddev, nsamples=nsamples, magnitude=2)

    def batch_gradient(data_inputs):
        """Calculate smooth gradient by batch"""
        total_grads = None
        n = data_inputs[0].shape[0]
        for i in range(0, n, batch_size):
            batch = [x[i:i + batch_size] for x in data_inputs]
            grad = (
                saliency_generator.get_smooth_gradients(batch)
                if smooth else
                saliency_generator.get_gradients(batch)
            )
            if total_grads is None:
                total_grads = [g for g in grad]
            else:
                for j in range(len(grad)):
                    total_grads[j] = np.concatenate([total_grads[j], grad[j]], axis=0)
        return total_grads

    grad = batch_gradient(data)
    grad_mut = batch_gradient(data_mut)

    grad_seq = grad[2]  # shape: (batch, 101, 512)
    grad_seq_mut = grad_mut[2]

    grad_seq = np.max(grad_seq, axis=-1)  # (batch, 101)
    grad_seq = np.expand_dims(grad_seq, axis=1)  # (batch, 1, 101)

    grad_seq_mut = np.max(grad_seq_mut, axis=-1)
    grad_seq_mut = np.expand_dims(grad_seq_mut, axis=1)

    sal_kld = compute_kl_divergence(grad_seq, grad_seq_mut)
    return sal_kld


def saliency(i, smooth_grad, pos, ss_onehot, nlp_node, nlp_adj, seq_dimone, ss, input_seq, out_path, ref, alt, type=''):
    saliency_maps = smooth_grad.get_smooth_gradients(
        [pos, ss_onehot, nlp_node, nlp_adj, seq_dimone, ss],
        target_class_idx=1
    )
    seq_attention = saliency_maps[2]
    seq_attention = np.max(seq_attention, axis=2)  # (1, 101)
    seq_attention = convert_one_hot2([input_seq], seq_attention[0, :]).transpose([1, 0, 2]).squeeze()   # (4, 101)

    struct_attention = saliency_maps[1]
    struct_attention = np.max(struct_attention, axis=2)  # (1, 101)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    W = np.concatenate([seq_attention, struct_attention], axis=0)  # (5, 101)
    np.save(out_path + '../W.npy', W)
    onehot = convert_one_hot([input_seq]).squeeze()

    input_str = np.max(ss, axis=2)  # (1, 101)

    x_str = input_str.reshape(101, 1)
    str_null = np.zeros_like(x_str)
    ind = np.where(x_str == -1)[0]
    str_null[ind, 0] = 1
    str_null = np.squeeze(str_null).T

    X = np.concatenate([onehot, input_str], axis=0)  # (5, 101)
    np.save(out_path + '../X.npy', X)

    plot_saliency(X, W, nt_width=101, norm_factor=3, str_null=str_null,
                  outdir=out_path + "/{}_{}_{}-{}.png".format(i, type, ref, alt))


def high_attention(args):
    model_path = args.model_path
    out_path = args.out_path
    set_path = args.set_path

    seq_path = os.path.join(set_path, f'before/ACGU', 'test.fa')
    ids, seqs = read_fasta(seq_path)
    pos_test = np.load(os.path.join(set_path, 'before/onekey', 'pos_inf.npy')).astype(np.float32)
    ss_test = np.load(os.path.join(set_path, 'before/ss', 'ss.npy')).astype(np.float32)
    nlp_test_adj = np.load(os.path.join(set_path, 'before/nlp', 'adj.npy')).astype(np.float32)
    nlp_test_node = np.load(os.path.join(set_path, 'before/nlp', 'node_embedding_inf.npy')).astype(np.float32)
    seq_test_dimone = con_seq_1D(os.path.join(set_path, 'before/sequential_feat', 'RNAonly', 'encoding_features/'))
    ss_onehot_test = np.load(os.path.join(set_path, 'before/ss', 'ss_onehot.npy')).astype(np.float32)

    seq_path_mut = os.path.join(set_path, f'after/ACGU', 'test.fa')
    ids_mut, seqs_mut = read_fasta(seq_path_mut)
    pos_test_mut = np.load(os.path.join(set_path, 'after/onekey', 'pos_inf.npy')).astype(np.float32)
    ss_test_mut = np.load(os.path.join(set_path, 'after/ss', 'ss.npy')).astype(np.float32)
    nlp_test_adj_mut = np.load(os.path.join(set_path, 'after/nlp', 'adj.npy')).astype(np.float32)
    nlp_test_node_mut = np.load(os.path.join(set_path, 'after/nlp', 'node_embedding_inf.npy')).astype(np.float32)
    seq_test_dimone_mut = con_seq_1D(os.path.join(set_path, 'after/sequential_feat', 'RNAonly', 'encoding_features/'))
    ss_onehot_test_mut = np.load(os.path.join(set_path, 'after/ss', 'ss_onehot.npy')).astype(np.float32)

    gcn = GCNConv(256)
    globalMaxPool = GlobalMaxPool()
    layer_dic = {'GCNConv': gcn, 'GlobalMaxPool': globalMaxPool, 'GlorotUniform': GlorotUniform}
    model = load_model(model_path,
                       custom_objects=layer_dic,
                       compile=False)

    data_dic = [pos_test, ss_onehot_test, nlp_test_node, nlp_test_adj, seq_test_dimone, ss_test]
    data_dic_mut = [pos_test_mut, ss_onehot_test_mut, nlp_test_node_mut, nlp_test_adj_mut, seq_test_dimone_mut,
                    ss_test_mut]

    # saliency kld
    sgs = saliency_smoothGrad(data_dic, data_dic_mut, model, smooth=True, batch_size=128)
    data = {
        "ID": ids,
        "Sequence": seqs,
        "Sequence_mut": seqs_mut,
        "sg": sgs
    }
    df = pd.DataFrame(data)
    os.makedirs(out_path, exist_ok=True)
    output_csv = os.path.join(out_path, "saliency_kld.csv")
    df.to_csv(output_csv, index=False)

    # saliency map
    smooth_grad = MultiInputSmoothGrad(model, x_stddev=0.015, nsamples=20, magnitude=2)

    for i in range(pos_test.shape[0]):
        match = re.search(r"([XY\d]+):(\d+)-(\d+)", ids[i])
        if not match:
            print(i, '\n', f"{ids[i]}")
            continue
        ref, alt = ids[i].split()[4:6]
        print(ref, alt)

        saliency(i, smooth_grad, pos_test[i:i + 1], ss_onehot_test[i:i + 1], nlp_test_node[i:i + 1],
                 nlp_test_adj[i:i + 1],
                 seq_test_dimone[i:i + 1], ss_test[i:i + 1], seqs[i], out_path, ref=ref, alt=alt, type='before')
        saliency(i, smooth_grad, pos_test_mut[i:i + 1], ss_onehot_test_mut[i:i + 1],
                 nlp_test_node_mut[i:i + 1], nlp_test_adj_mut[i:i + 1], seq_test_dimone_mut[i:i + 1],
                 ss_test_mut[i:i + 1],
                 seqs_mut[i], out_path, ref=ref, alt=alt, type='after')

    return sgs


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
        help="Output path"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Model path for Saliency map"
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

    sgs = high_attention(args)
    return sgs


if __name__ == '__main__':
    sgs = main()
    print(sgs)
