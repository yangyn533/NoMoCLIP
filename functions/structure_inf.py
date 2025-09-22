import os
import argparse
import linecache
import re

import numpy as np


def mk_dir(dir):
    try:
        os.makedirs(dir)
    except OSError:
        print('Can not make directory:', dir)


def defineExperimentPaths(output_path):
    E_path = output_path + '/E/'
    H_path = output_path + '/H/'
    I_path = output_path + '/I/'
    M_path = output_path + '/M/'
    mk_dir(E_path)
    mk_dir(H_path)
    mk_dir(I_path)
    mk_dir(M_path)
    return E_path, H_path, I_path, M_path


def run_RNA(filepath, E_path, H_path, I_path, M_path):
    os.system('./E_RNAplfold -W 240 -L 160 -u 1 <' + filepath + ' ' + '>' + E_path + 'E_profile.txt')
    os.system('./H_RNAplfold -W 240 -L 160 -u 1 <' + filepath + ' ' + '>' + H_path + 'H_profile.txt')
    os.system('./I_RNAplfold -W 240 -L 160 -u 1 <' + filepath + ' ' + '>' + I_path + 'I_profile.txt')
    os.system('./M_RNAplfold -W 240 -L 160 -u 1 <' + filepath + ' ' + '>' + M_path + 'M_profile.txt')


def concatenate(pairedness, hairpin_loop, internal_loop, multi_loop, external_region):
    combine_list = [pairedness.split(), hairpin_loop.split(), internal_loop.split(), multi_loop.split(),
                    external_region.split()]
    return np.array(combine_list).T


def read_fasta(file_path):
    i = 0
    secondary_structure_list = []
    filelines = linecache.getlines(file_path)
    file_length = len(filelines)

    while i <= file_length - 1:
        pairedness = re.sub('[\s+]', ' ', filelines[i + 1].strip())
        hairpin_loop = re.sub('[\s+]', ' ', filelines[i + 2].strip())
        internal_loop = re.sub('[\s+]', ' ', filelines[i + 3].strip())
        multi_loop = re.sub('[\s+]', ' ', filelines[i + 4].strip())
        external_region = re.sub('[\s+]', ' ', filelines[i + 5].strip())
        combine_array = concatenate(pairedness, hairpin_loop, internal_loop, multi_loop, external_region)
        secondary_structure_list.append(combine_array)
        i = i + 6

    return np.array(secondary_structure_list)


def run_RNAplfold(args):
    set_path = args.set_path
    out_path = args.out_path

    fastapath = set_path
    outpath_test = os.path.join(out_path, 'ss')
    E_path, H_path, I_path, M_path = defineExperimentPaths(outpath_test)
    run_RNA(fastapath, E_path, H_path, I_path, M_path)
    cmd = 'python combine_letter_profiles.py' + ' ' + E_path + 'E_profile.txt' + ' ' + H_path + 'H_profile.txt' + ' ' + I_path + 'I_profile.txt' + ' ' + M_path + 'M_profile.txt 1 ' + ' ' + outpath_test + '/' + 'combined_profile.txt'
    os.system(cmd)


def parser_secondary(args):
    out_path = args.out_path
    outpath_test = os.path.join(out_path, 'ss')

    fpath = os.path.join(outpath_test, 'combined_profile.txt')
    secondary = read_fasta(fpath)
    save_path = os.path.join(outpath_test, 'ss.npy').format('test')
    np.save(save_path, secondary)


def ss2onehot(args):
    out_path = args.out_path
    outpath_test = os.path.join(out_path, 'ss')

    data = np.load(os.path.join(outpath_test, 'ss.npy'))
    max_indices = np.argmax(data, axis=-1)
    one_hot = np.eye(5)[max_indices]
    np.save(os.path.join(outpath_test, 'ss_onehot.npy'), one_hot)


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

    run_RNAplfold(args)
    parser_secondary(args)
    ss2onehot(args)


if __name__ == '__main__':
    main()
