import os
import math
import keras
import random
import logging
import argparse
import numpy as np
import tensorflow as tf

from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, \
    precision_recall_curve
from sklearn.metrics import matthews_corrcoef, average_precision_score

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Input, Convolution1D, Activation, Dropout, Dense, Bidirectional, Multiply, \
    GRU, Softmax, Lambda, BatchNormalization, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from spektral.layers import GCNConv, GlobalMaxPool
from pytorch_lightning import seed_everything


seed_everything(42)


def fix_seed(seed):
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def mk_dir(dir):
    try:
        os.makedirs(dir)
    except OSError:
        print('Can not make directory:', dir)


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


def defineExperimentPaths(basic_path, methodName, experimentID):
    experiment_name = methodName + '/' + experimentID
    MODEL_PATH = basic_path + experiment_name + '/model/'
    CHECKPOINT_PATH = basic_path + experiment_name + '/checkpoints/'
    mk_dir(MODEL_PATH)
    mk_dir(CHECKPOINT_PATH)
    return [MODEL_PATH, CHECKPOINT_PATH]


def createModel():
    sequence_input = Input(shape=(101, 4), name='sequence_input', dtype='float32')
    sequence = Convolution1D(filters=256, kernel_size=7, padding='same', name='seq_conv_7')(sequence_input)
    sequence = BatchNormalization(axis=-1, name='seq_bn_7')(sequence)
    sequence = Activation('relu', name='seq_relu_7')(sequence)
    sequence = Convolution1D(filters=256, kernel_size=9, padding='same', name='seq_conv_9')(sequence)
    sequence = BatchNormalization(axis=-1, name='seq_bn_9')(sequence)
    sequence = Activation('relu', name='seq_relu_9')(sequence)
    sequence = Dropout(0.2, name='seq_drop')(sequence)
    ##########################################################################################################################
    profile_input = Input(shape=(101, 5), name='profile_input', dtype='float32')
    profile = Convolution1D(filters=256, kernel_size=7, padding='same', name='str_conv_7')(profile_input)
    profile = BatchNormalization(axis=-1, name='str_bn_7')(profile)
    profile = Activation('relu', name='str_relu_7')(profile)
    profile = Convolution1D(filters=256, kernel_size=9, padding='same', name='str_conv_9')(profile)
    profile = BatchNormalization(axis=-1, name='str_bn_9')(profile)
    profile = Activation('relu', name='str_relu_9')(profile)
    profile = Dropout(0.2, name='str_drop')(profile)
    ##########################################################################################################################
    concatenated_features = Concatenate(axis=-1, name='merge_output')([sequence, profile])
    attention_weighted = Dense(512, name='dense_att')(concatenated_features)
    attention_weighted = Activation('relu', name='str_relu_att')(attention_weighted)
    attention_weighted = Softmax()(attention_weighted)
    attention_scaled = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(attention_weighted)
    local_output = Multiply(name='Multiply_att')([concatenated_features, attention_scaled])
    ##########################################################################################################################
    additional_input = Input(shape=(70,), name='additional_input', dtype='float32')
    edge_input = Input(shape=(101, 101), name='edge_input', dtype='float32')
    node_input = Input(shape=(101, 512), name='node_input', dtype='float32')
    ##########################################################################################################################
    profile_prob = Input(shape=(101, 5), name='profile_prob', dtype='float32')
    gru_input = Concatenate(axis=-1, name='gru_input')([profile_prob, node_input])
    backward_gru_1st = Bidirectional(GRU(256, dropout=0.25, return_sequences=True, name='gru_1'))(gru_input)
    backward_gru_2nd = Bidirectional(GRU(256, dropout=0.25, return_sequences=True, name='gru_2'))(backward_gru_1st)
    ##########################################################################################################################
    node_update = Multiply(name='Multiply_1')([node_input, backward_gru_2nd])

    # Graph Convolution layer
    gc_1 = GCNConv(256, use_bias=False, activation='relu', dropout_rate=0.2, name='GraphConv_1')([node_update, edge_input])
    gc_2 = GCNConv(256, use_bias=False, activation='relu', dropout_rate=0.2, name='GraphConv_2')([gc_1, edge_input])

    gcnn_concat = [gc_1, gc_2]

    global_output = Concatenate(name='GCNN_concatenate')(gcnn_concat)
    overallResult = Multiply(name='multiply_2')([local_output, global_output])
    overallResult = GlobalMaxPool()(overallResult)
    Addallresult = Concatenate(axis=-1, name='Addall_concatenate')([overallResult, additional_input])
    overallResult = Dense(256, activation='relu', name='dense_1')(Addallresult)
    overallResult = Dense(128, activation='relu', name='dense_2')(overallResult)
    ss_output = Dense(2, activation='softmax', name='ss_output')(overallResult)
    return Model(inputs=[sequence_input, profile_input, node_input, edge_input, additional_input, profile_prob], outputs=[ss_output])


def parse_arguments(parser):
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
        help="Path to the feature file"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="",
        help="Output path"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=5,
        help="Number of folds of data"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=1
    )
    args = parser.parse_args()

    return args


def main(parser):
    gpu_id = parser.gpu_id
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.system('echo $CUDA_VISIBLE_DEVICES')

    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=tf_config)
    gpus = tf.config.experimental.list_physical_devices('GPU')

    fix_seed(42)

    base_path = parser.base_path
    set_path = parser.set_path
    out_path = parser.out_path
    fold = parser.fold

    rbps = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    print(rbps)

    for rbp in rbps:
        methodName = rbp

        Auc = []
        Acc = []
        Mcc = []
        Aupr = []
        Recall = []
        Fscore = []
        Precision = []

        batchSize = 20
        maxEpochs = 100

        for i in range(0, fold):
            # Features for training and validation
            pos_train = np.load(os.path.join(set_path, 'onekey', rbp, '{}', 'train', 'pos_inf.npy').format(i)).astype(
                np.float32)
            pos_val = np.load(os.path.join(set_path, 'onekey', rbp, '{}', 'val', 'pos_inf.npy').format(i)).astype(
                np.float32)

            ss_train = np.load(os.path.join(set_path, 'ss', rbp, '{}', 'train', 'ss.npy').format(i)).astype(np.float32)
            ss_val = np.load(os.path.join(set_path, 'ss', rbp, '{}', 'val', 'ss.npy').format(i)).astype(np.float32)

            ss_onehot_train = np.load(os.path.join(set_path, 'ss', rbp, '{}', 'train', 'ss_onehot.npy').format(i)).astype(np.float32)
            ss_onehot_val = np.load(os.path.join(set_path, 'ss', rbp, '{}', 'val', 'ss_onehot.npy').format(i)).astype(np.float32)

            nlp_train_adj = np.load(
                os.path.join(set_path, 'nlp', rbp, '{}', 'train', 'adj.npy').format(i)).astype(np.float32)
            nlp_val_adj = np.load(
                os.path.join(set_path, 'nlp', rbp, '{}', 'val', 'adj.npy').format(i)).astype(np.float32)

            nlp_train_node = np.load(
                os.path.join(set_path, 'nlp', rbp, '{}', 'train', 'node_embedding_inf.npy').format(i)).astype(np.float32)
            nlp_val_node = np.load(
                os.path.join(set_path, 'nlp', rbp, '{}', 'val', 'node_embedding_inf.npy').format(i)).astype(np.float32)
            seq_train_dimone = con_seq_1D(os.path.join(set_path, 'sequential_feat', rbp, '{}', 'train',
                                                    'RNAonly', 'encoding_features/').format(i))
            seq_val_dimone = con_seq_1D(os.path.join(set_path, 'sequential_feat', rbp, '{}', 'val', 'RNAonly',
                                                     'encoding_features/').format(i))
            y_train = np.load(os.path.join(set_path, 'onekey', rbp, '{}', 'train', 'label.npy').format(i))
            y_train = to_categorical(y_train)
            y_val = np.load(os.path.join(set_path, 'onekey', rbp, '{}', 'val', 'label.npy').format(i))
            y_val = to_categorical(y_val)

            # Features for testing
            pos_test = np.load(os.path.join(set_path, 'onekey', rbp, 'test', 'pos_inf.npy')).astype(np.float32)
            ss_onehot_test = np.load(os.path.join(set_path, 'ss', rbp, 'test', 'ss_onehot.npy')).astype(np.float32)

            ss_test = np.load(os.path.join(set_path, 'ss', rbp, 'test', 'ss.npy')).astype(np.float32)
            nlp_test_adj = np.load(os.path.join(set_path, 'nlp', rbp, 'test', 'adj.npy')).astype(np.float32)
            nlp_test_node = np.load(os.path.join(set_path, 'nlp', rbp, 'test', 'node_embedding_inf.npy')).astype(np.float32)
            seq_test_dimone = con_seq_1D(os.path.join(set_path, 'sequential_feat', rbp, 'test', 'RNAonly',
                                                      'encoding_features/'))
            y_test = np.load(os.path.join(set_path, 'onekey', rbp, 'test', 'label.npy'))
            y_test = to_categorical(y_test)

            test_y = y_test[:, 1]

            [MODEL_PATH, CHECKPOINT_PATH] = defineExperimentPaths(out_path, methodName, str(i))

            logging.debug("Loading network/training configuration...")
            model = createModel()
            logging.debug("Model summary ... ")
            model.count_params()
            model.summary()

            checkpoint_weight = CHECKPOINT_PATH + "weights.best.hdf5"
            if (os.path.exists(checkpoint_weight)):
                print("load previous best weights")
                model.load_weights(checkpoint_weight)

            model.compile(optimizer='adam',
                          loss=keras.losses.BinaryCrossentropy(),
                          metrics=['accuracy'])
            logging.debug("Running training...")

            def step_decay(epoch):
                initial_lrate = 0.0005
                drop = 0.8
                epochs_drop = 5.0
                lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
                print(lrate)
                return lrate

            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='min'),
                ModelCheckpoint(checkpoint_weight,
                                monitor='val_loss',
                                verbose=1,
                                save_best_only=True,
                                mode='min',
                                period=1),
                LearningRateScheduler(step_decay),
            ]

            history = model.fit(
                [pos_train, ss_onehot_train, nlp_train_node, nlp_train_adj, seq_train_dimone, ss_train],
                {'ss_output': y_train},
                epochs=maxEpochs,
                batch_size=batchSize,
                callbacks=callbacks,
                verbose=1,
                validation_data=(
                    [pos_val, ss_onehot_val, nlp_val_node, nlp_val_adj, seq_val_dimone, ss_val],
                    {'ss_output': y_val}),
                shuffle=False)

            logging.debug("Saving final model...")
            model.save(os.path.join(MODEL_PATH, 'model.h5'), overwrite=True)
            json_string = model.to_json()
            with open(os.path.join(MODEL_PATH, 'model.json'), 'w') as f:
                f.write(json_string)
            logging.debug("make prediction")

            ss_y_hat_test = model.predict(
                [pos_test, ss_onehot_test, nlp_test_node, nlp_test_adj, seq_test_dimone, ss_test]
            )

            ytrue = test_y
            yprob = ss_y_hat_test[:, 1]
            ypred = np.argmax(ss_y_hat_test, axis=-1)

            auc = roc_auc_score(ytrue, yprob)
            acc = accuracy_score(ytrue, ypred)
            fpr, tpr, thresholds = roc_curve(ytrue, yprob)

            precision = precision_score(ytrue, ypred)
            recall = recall_score(ytrue, ypred)
            fscore = f1_score(ytrue, ypred)
            mcc = matthews_corrcoef(ytrue, ypred)
            aupr = average_precision_score(ytrue, yprob)
            precision_auprc, recall_auprc, thresholds_auprc = precision_recall_curve(ytrue, yprob)

            np.save(out_path + methodName + '/' + str(i) + '/' + 'fpr.npy', np.array(fpr))
            np.save(out_path + methodName + '/' + str(i) + '/' + 'tpr.npy', np.array(tpr))
            np.save(out_path + methodName + '/' + str(i) + '/' + 'thresholds.npy', np.array(thresholds))
            np.save(out_path + methodName + '/' + str(i) + '/' + 'auc.npy', np.array(auc))
            np.save(out_path + methodName + '/' + str(i) + '/' + 'acc.npy', np.array(acc))
            np.save(out_path + methodName + '/' + str(i) + '/' + 'mcc.npy', np.array(mcc))
            np.save(out_path + methodName + '/' + str(i) + '/' + 'aupr.npy', np.array(aupr))
            np.save(out_path + methodName + '/' + str(i) + '/' + 'precision.npy', np.array(precision))
            np.save(out_path + methodName + '/' + str(i) + '/' + 'recall.npy', np.array(recall))
            np.save(out_path + methodName + '/' + str(i) + '/' + 'fscore.npy', np.array(fscore))
            np.save(out_path + methodName + '/' + str(i) + '/' + 'precision_auprc.npy', np.array(precision_auprc))
            np.save(out_path + methodName + '/' + str(i) + '/' + 'recall_auprc.npy', np.array(recall_auprc))
            np.save(out_path + methodName + '/' + str(i) + '/' + 'thresholds_auprc.npy', np.array(thresholds_auprc))

            Auc.append(auc)
            Acc.append(acc)
            Mcc.append(mcc)
            Aupr.append(aupr)
            Precision.append(precision)
            Recall.append(recall)
            Fscore.append(fscore)

            print("acid AUC: %.4f " % np.mean(Auc))
            print("acid ACC: %.4f " % np.mean(Acc))

        np.save(out_path + methodName + '/' + 'mean_auc.npy', np.mean(np.array(Auc)))
        np.save(out_path + methodName + '/' + 'mean_acc.npy', np.mean(np.array(Acc)))
        np.save(out_path + methodName + '/' + 'mean_mcc.npy', np.mean(np.array(Mcc)))
        np.save(out_path + methodName + '/' + 'mean_precision.npy', np.mean(np.array(Precision)))
        np.save(out_path + methodName + '/' + 'mean_recall.npy', np.mean(np.array(Recall)))
        np.save(out_path + methodName + '/' + 'mean_fscore.npy', np.mean(np.array(Fscore)))
        np.save(out_path + methodName + '/' + 'mean_aupr.npy', np.mean(np.array(Aupr)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    main(args)
