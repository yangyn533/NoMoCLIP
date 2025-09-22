import argparse
import os
import subprocess
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tensorflow.keras.models import load_model
from spektral.layers import GCNConv, GlobalMaxPool
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import GlorotUniform


weblogo_opts = '-X NO -Y NO --fineprint "" --size large'
weblogo_opts += ' -C "#000000" E E'
weblogo_opts += ' -C "#0D8163" H H'
weblogo_opts += ' -C "#7064AC" I I'
weblogo_opts += ' -C "#33BFC4" M M'
weblogo_opts += ' -C "#ED1A6D" P P'


def meme_intro(meme_file, seqs):
    """ Open MEME motif format file and print intro

    Attrs:
        meme_file (str) : filename
        seqs [str] : list of strings for obtaining background freqs

    Returns:
        mem_out : open MEME file
    """
    nts = {'E': 0, 'H': 1, 'I': 2, 'M': 3, 'P': 4}

    # count
    nt_counts = [1] * 5
    for i in range(len(seqs)):
        for nt in seqs[i]:
            try:
                nt_counts[nts[nt]] += 1
            except KeyError:
                pass

    # normalize
    nt_sum = float(sum(nt_counts))
    nt_freqs = [nt_counts[i] / nt_sum for i in range(5)]

    # open file for writing
    meme_out = open(meme_file, 'w')
    freq = [0.2, 0.2, 0.2, 0.2, 0.2]
    # print intro material
    meme_out.write('MEME version 5')
    meme_out.write('\n')
    meme_out.write('ALPHABET "EHIMP"' + '\n')
    meme_out.write('E "External" 000000' + '\n')
    meme_out.write('H "Hairpin" 0D8163' + '\n')
    meme_out.write('I "Internal" 7064AC' + '\n')
    meme_out.write('M "Multi" 33BFC4' + '\n')
    meme_out.write('P "Paired" ED1A6D' + '\n')
    meme_out.write('END ALPHABET' + '\n')
    meme_out.write('Background letter frequencies:')
    meme_out.write('\n')
    meme_out.write('E %.4f H %.4f I %.4f M %.4f P %.4f \n' % tuple(nt_freqs))
    meme_out.write('\n')

    return meme_out


def plot_filter_logo(filter_outs, filter_size, seqs, out_prefix, raw_t=0, maxpct_t=None):
    if maxpct_t:
        all_outs = np.ravel(filter_outs)
        all_outs_mean = all_outs.mean()
        all_outs_norm = all_outs - all_outs_mean
        raw_t = 0.75 * all_outs_norm.max() + all_outs_mean

    # print fasta file of positive outputs
    filter_fasta_out = open('%s.fa' % out_prefix, 'w')
    filter_count = 0
    for i in range(filter_outs.shape[0]):
        for j in range(filter_outs.shape[1]):
            if filter_outs[i, j] > raw_t:
                kmer = seqs[i][j - filter_size // 2: j - filter_size // 2 + filter_size]
                incl_kmer = len(kmer) - kmer.count('N')
                if incl_kmer < filter_size:
                    continue
                filter_fasta_out.write('>%d_%d' % (i, j) + '\n')
                filter_fasta_out.write(str(kmer) + '\n')
                filter_count += 1
    filter_fasta_out.close()
    print('filter_count:', filter_count)
    # make weblogo
    if filter_count > 0:
        weblogo_cmd = 'weblogo %s < %s.fa > %s.eps' % (weblogo_opts, out_prefix, out_prefix)
        os.system(weblogo_cmd)


def make_filter_pwm(filter_fasta):
    """ Make a PWM for this filter from its top hits """

    nts = {'E': 0, 'H': 1, 'I': 2, 'M': 3, 'P': 4}
    pwm_counts = []
    nsites = 5  # pseudocounts
    for line in open(filter_fasta):
        if line[0] != '>':
            seq = line.rstrip()
            nsites += 1
            if len(pwm_counts) == 0:
                # initialize with the length
                for i in range(len(seq)):
                    pwm_counts.append(np.array([1.0] * 5))

            # count
            for i in range(len(seq)):
                try:
                    pwm_counts[i][nts[seq[i]]] += 1
                except KeyError:
                    pwm_counts[i] += np.array([0.2] * 5)

    # normalize
    pwm_freqs = []
    for i in range(len(pwm_counts)):
        pwm_freqs.append([pwm_counts[i][j] / float(nsites) for j in range(5)])

    return np.array(pwm_freqs), nsites - 5


def info_content(pwm, transpose=False, bg_gc=0.415):
    """ Compute PWM information content.

    In the original analysis, I used a bg_gc=0.5. For any
    future analysis, I ought to switch to the true hg19
    value of 0.415.
    """
    pseudoc = 1e-9

    if transpose:
        pwm = np.transpose(pwm)

    bg_pwm = [1 - bg_gc, bg_gc, bg_gc, 1 - bg_gc]

    ic = 0
    for i in range(pwm.shape[0]):
        for j in range(4):
            ic += -bg_pwm[j] * np.log2(bg_pwm[j]) + pwm[i][j] * np.log2(pseudoc + pwm[i][j])

    return ic


def meme_add(meme_out, f, filter_pwm, nsites, trim_filters=False):
    """ Print a filter to the growing MEME file

    Attrs:
        meme_out : open file
        f (int) : filter index #
        filter_pwm (array) : filter PWM array
        nsites (int) : number of filter sites
    """
    if not trim_filters:
        ic_start = 0
        ic_end = filter_pwm.shape[0] - 1
    else:
        ic_t = 0.2

        # trim PWM of uninformative prefix
        ic_start = 0
        while ic_start < filter_pwm.shape[0] and info_content(filter_pwm[ic_start:ic_start + 1]) < ic_t:
            ic_start += 1

        # trim PWM of uninformative suffix
        ic_end = filter_pwm.shape[0] - 1
        while ic_end >= 0 and info_content(filter_pwm[ic_end:ic_end + 1]) < ic_t:
            ic_end -= 1

    if ic_start < ic_end:
        meme_out.write('MOTIF filter%d' % f + '\n')
        meme_out.write(
            'letter-probability matrix: alength= 5 w= %d nsites= %d' % (ic_end - ic_start + 1, nsites) + '\n')
        for i in range(ic_start, ic_end + 1):
            meme_out.write('%.4f %.4f %.4f %.4f %.4f' % tuple(filter_pwm[i]) + '\n')
        meme_out.write(' ' + '\n')


def get_motif_fig_new(filter_outs, out_dir, structs, str_path, motif_size, filter_weights):
    print('plot motif fig', out_dir)

    num_filters = filter_outs.shape[2]
    print('num_filters: ', num_filters)
    filter_size = motif_size

    # pdb.set_trace()
    #################################################################
    # individual filter plots
    #################################################################
    # also save information contents
    filters_ic = []
    meme_out = meme_intro('%s/filters_meme.txt' % out_dir, structs)

    for f in range(num_filters):
        print('Filter %d' % f)

        # plot filter parameters as a heatmap
        plot_filter_heat(filter_weights[:, :, f], '%s/filter%d_heat.pdf' % (out_dir, f))

        # plot weblogo of high scoring outputs
        plot_filter_logo(filter_outs[:, :, f], filter_size, structs, '%s/filter%d_logo' % (out_dir, f), maxpct_t=0.75)

        # make a PWM for the filter
        filter_pwm, nsites = make_filter_pwm('%s/filter%d_logo.fa' % (out_dir, f))

        if nsites < 10:
            # no information
            filters_ic.append(0)
        else:
            # compute and save information content
            filters_ic.append(info_content(filter_pwm))

            # add to the meme motif file
            meme_add(meme_out, f, filter_pwm, nsites, False)

    meme_out.close()

    result = subprocess.call(
        'ame --control --shuffle-- -oc {}/ame {}/positive_ss.fa {}/filters_meme.txt'.format(out_dir, str_path, out_dir), shell=True)


def get_motif_s(model, testing, structs, str_path, motif_size, layer='', out_dir=''):
    layer = model.get_layer(layer)
    get_output = K.function([model.input], [layer.output])
    # get output
    filter_outs = get_output(testing)[0]

    filter_weights = layer.get_weights()[0]
    print('filter_weights', filter_weights.shape)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    get_motif_fig_new(filter_outs, out_dir, structs, str_path, motif_size, filter_weights)


def read_fasta(fastapath):
    seqs = []
    with open(fastapath) as f:
        for line in f:
            if '>' not in line:
                seqs.append(line.strip())
    return seqs


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


################################################################################
# plot_filter_heat
#
# Plot a heatmap of the filter's parameters.
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:ftihms
################################################################################
def plot_filter_heat(param_matrix, out_pdf):
    param_matrix = param_matrix.T   # (5, 7)
    param_range = abs(param_matrix).max()
    threshold = 0.8 * param_range
    annot_matrix = np.where(param_matrix > threshold, '*', '')
    sns.set(font_scale=2)
    plt.figure(figsize=(param_matrix.shape[1], 4))
    colors = ["#6BB018", "white", "#D60000"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    ax = sns.heatmap(
        param_matrix,
        cmap=cmap,
        linewidths=0.2, vmin=-param_range, vmax=param_range,
        annot=annot_matrix,
        fmt='',
        annot_kws={"size": 40, "weight": 'bold', "color": 'white', 'va': 'center_baseline', 'ha': 'center'},
        cbar_kws={'label': None}
    )

    ax = plt.gca()
    ax.set_xticklabels(range(1, param_matrix.shape[1] + 1))
    ax.set_yticklabels('PHIME', rotation='horizontal')
    plt.savefig(out_pdf)
    plt.close()


def structure_motif(args):
    layer = args.layer
    set_path = args.set_path
    model_path = args.model_path
    out_path = args.out_path
    motif_size = args.motif_size

    str_path = os.path.join(set_path, 'ACGU')
    pos_test = np.load(os.path.join(set_path, 'onekey/pos_inf.npy')).astype(np.float32)
    ss_test = np.load(os.path.join(set_path, 'ss/ss.npy')).astype(np.float32)
    ss_onehot_test = np.load(os.path.join(set_path, 'ss/ss_onehot.npy')).astype(np.float32)
    nlp_test_adj = np.load(os.path.join(set_path, 'nlp/adj.npy')).astype(np.float32)
    nlp_test_node = np.load(os.path.join(set_path, 'nlp/node_embedding_inf.npy')).astype(np.float32)
    seq_test_dimone = con_seq_1D(os.path.join(set_path, 'sequential_feat/RNAonly', 'encoding_features/'))
    y_test = np.load(os.path.join(set_path, 'onekey/label.npy'))
    print('y_test:', y_test.shape)
    pos_ind = np.where(y_test == 1)[0]
    print('pos_ind.shape', pos_ind.shape)

    pos_test = pos_test[pos_ind]
    ss_test = ss_test[pos_ind]
    ss_onehot_test = ss_onehot_test[pos_ind]
    nlp_test_adj = nlp_test_adj[pos_ind]
    nlp_test_node = nlp_test_node[pos_ind]
    seq_test_dimone = seq_test_dimone[pos_ind]

    structs = read_fasta(os.path.join(str_path, f'positive_ss.fa'))
    structs = np.array(structs)

    gcn = GCNConv(256)
    globalMaxPool = GlobalMaxPool()
    layer_dic = {'GCNConv': gcn, 'GlobalMaxPool': globalMaxPool, 'GlorotUniform': GlorotUniform}
    model = load_model(model_path,
                       custom_objects=layer_dic,
                       compile=False)

    if not os.path.exists(out_path + f'/size_{motif_size}/'):
        os.makedirs(out_path + f'/size_{motif_size}/')
    data_dic = [pos_test, ss_onehot_test, nlp_test_node, nlp_test_adj, seq_test_dimone, ss_test]
    get_motif_s(model, data_dic, structs, str_path, motif_size, layer=layer,
                out_dir=out_path + f'/size_{motif_size}/{layer}/')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layer",
        type=str,
        default="str_conv_7",
        help="Model layer"
    )
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
        help="Model path for structure motif"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=1
    )
    parser.add_argument(
        "--motif_size",
        type=int,
        default=7
    )
    args = parser.parse_args()

    gpu_id = args.gpu_id
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.system('echo $CUDA_VISIBLE_DEVICES')

    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=tf_config)
    gpus = tf.config.experimental.list_physical_devices('GPU')

    structure_motif(args)


if __name__ == '__main__':
    main()