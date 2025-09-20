import argparse
import os
import torch
import numpy as np

from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer


def mk_dir(dir):
    try:
        os.makedirs(dir)
    except OSError:
        print('Can not make directory:', dir)


def seq2kmer(seq, k):
    kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmers = " ".join(kmer)
    return kmers


def read_fasta(file_path):
    seq_list = []
    f = open(file_path, 'r')
    for line in f:
        if '>' not in line:
            line = line.strip().upper().replace('U', 'T')
            seq_list.append(line)

    return seq_list


def preprocess_features(features):
    max_length = max([len(f) for f in features])
    for i in tqdm(range(features.shape[0])):
        feature = np.array(features[i])
        pad = max_length - feature.shape[0] 
        feature = np.pad(feature, ((0, pad), (0, 0)), mode='constant')
        features[i] = feature
    return np.array(list(features))


def generate_graph_inf(dataloader, args):
    hidden_embedding_gpu3 = []
    attention_head_avg_gpu3 = []

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    device1 = torch.device(args.device1 if torch.cuda.is_available() else 'cpu')
    device2 = torch.device(args.device2 if torch.cuda.is_available() else 'cpu')

    maxlen = args.maxlen

    model_type = args.model_type

    config = BertConfig.from_pretrained(model_type)
    config.output_attentions = True
    config.output_hidden_states = True
    model = BertModel.from_pretrained(model_type, config=config).to(device)
    tokenizer = BertTokenizer.from_pretrained(model_type)

    model = model.to(device)
    model = model.eval()

    for sequences in dataloader:
        ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=False, max_length=maxlen, pad_to_max_length=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embedding = torch.stack(outputs[2])
            selected_indices = [0, 3, 4, 5, 6]
            selected_tensors = [embedding[i] for i in selected_indices]
            embedding = torch.sum(torch.stack(selected_tensors), dim=0, keepdim=True).squeeze(dim=0)
            attention_weights = outputs[-1]

        summed_last_weights = attention_weights[-1].mean(1)

        hidden_embedding_gpu3.append(embedding.to(device1))
        attention_head_avg_gpu3.append(summed_last_weights.to(device2))

        del ids
        del input_ids
        del attention_mask
        del outputs
        del attention_weights
        del embedding
        del selected_tensors

    hidden_embedding_gpu3 = torch.cat(hidden_embedding_gpu3, dim=0).to(device1)
    attention_head_avg_gpu3 = torch.cat(attention_head_avg_gpu3, dim=0)

    return hidden_embedding_gpu3, attention_head_avg_gpu3


def GraphData(filepath, args):
    # load raw text
    doc_content_list = []
    final_matrix = []

    seq = read_fasta(filepath)
    for each_item in seq:
        
        sub_seq = seq2kmer(each_item.strip(), args.kmer)
        doc_content_list.append(sub_seq)

    dataloader = torch.utils.data.DataLoader(doc_content_list, batch_size=1, shuffle=False)
    node_inf, edge_inf = generate_graph_inf(dataloader, args)
    adjacency_matrix = edge_inf

    adjacency_matrix_inf_np = adjacency_matrix.detach().cpu().numpy()
    for each_item in range(adjacency_matrix_inf_np.shape[0]):
        ori_matrix = adjacency_matrix_inf_np[each_item]
        zomatrix = np.double(ori_matrix < np.mean(ori_matrix))
        diag_indices = np.diag_indices_from(zomatrix)
        zomatrix[diag_indices] = 0        
        final_matrix.append(zomatrix)
    return np.array(final_matrix), node_inf.detach().cpu().numpy()


def attention_graph(filepath, out_path, args):
    mk_dir(out_path)

    adjacency_matrix_inf, node_embedding_inf = GraphData(filepath, args)

    print('adjacency_matrix_inf: ', adjacency_matrix_inf.shape)
    print('node_embedding_inf: ', node_embedding_inf.shape)

    path = os.path.join(out_path, '{}')
    filepath = path.format('adj.npy')
    
    if os.path.exists(filepath) is True:
        os.remove(filepath)
    np.save(filepath, adjacency_matrix_inf)

    filepath_node = path.format('node_embedding_inf.npy')
    if os.path.exists(filepath_node) is True:
        os.remove(filepath_node)
    np.save(filepath_node, node_embedding_inf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kmer",
        type=int,
        default="1",
        help="Please give a k_mer"
    )
    parser.add_argument(
        "--set_path",
        type=str,
        default="",
        help="Path to the fasta file"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="",
        help="Output path"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="../SpliceBERT.1024nt",
        help="Path to the NLP model"
    )
    parser.add_argument(
        "--maxlen",
        type=int,
        default=101,
        help="Maximum length of a fasta sequence"
    )
    parser.add_argument(
        "--device",
        default="cuda:1"
    )
    parser.add_argument(
        "--device1",
        default="cuda:1"
    )
    parser.add_argument(
        "--device2",
        default="cuda:1"
    )
    args = parser.parse_args()

    set_path = args.set_path
    out_path = args.out_path

    fastapath = set_path
    outpath_test = os.path.join(out_path, 'nlp')
    attention_graph(fastapath, outpath_test, args)


if __name__ == '__main__':
    main()
