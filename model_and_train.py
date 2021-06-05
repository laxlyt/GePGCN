import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib
matplotlib.use('agg') 

import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv

import argparse
from collections import defaultdict
from collections import Counter
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_score

parser = argparse.ArgumentParser()
parser.add_argument('--input_feature', type=str, default='input/features_all.pkl')
parser.add_argument('--hidden_1', type=int, default=256)
parser.add_argument('--hidden_2', type=int, default=128)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--iter', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.01)
args = parser.parse_args()
#args = parser.parse_args(args=[])

def reshape(features):
    return np.hstack(features).reshape((len(features), len(features[0])))

def load_node2vec_embedding(file):
    with open(file) as f:
        data = f.readlines()
    embed = np.zeros(shape=tuple(map(int, data[0].split(" "))))
    for line in data[1:]:
        temp = list(map(float, line.split(" ")))
        embed[int(temp[0]), :] = temp[1:]
    return embed

class my_GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(my_GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, args.hidden_1, cached=False)
        self.conv2 = GCNConv(args.hidden_1, args.hidden_2, cached=False)
        self.conv3 = GCNConv(args.hidden_2, out_channels, cached=False)
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return x
    
def calculate_accuracy(final_preds, final_labels):
    # evaluate model performance # 
    # input: final_preds: validation set predicted label
    # input: final_labels: validation set true label
    # output: weighted_prediction_rate
    # output: prediction_rate
    # output: recall_rate 
    
    total_predicted_w = 0   
    total_labels = np.sum(final_labels)
    predict_num = 0
    total_predicted = 0
    right_list = []
    right_label = []
    for line in range(len(final_labels)):
        predicted = 0
        predicted_w = 0
        pred_num = int(np.sum(final_labels[line]))    #真实标签数量
        if pred_num == 0:
            continue
        # setting weight
        w_sum = sum([i for i in range(pred_num+1)])
        w_list = [i/w_sum for i in range(1, pred_num+1)][::-1] 
        # choose label number
        pred_index = np.argsort(-final_preds[line])[:pred_num]    
        # compute prediction rate
        for i in range(len(pred_index)):
            if final_labels[line][pred_index[i]] == 1:  
                predicted += 1
                predicted_w += w_list[i]
                right_list.append(line)
                right_label.append(pred_index[i])
        if predicted >= 1:   
            predict_num += 1
        total_predicted += predicted
        total_predicted_w += predicted_w        
    perf = [total_predicted_w/final_preds.shape[0], predict_num, predict_num/final_preds.shape[0], total_predicted, total_predicted/total_labels]
    
    return perf, right_list, right_label

def calculate_performance(y_pred, y_label):
    # evaluate model performance # 
    # input: final_preds: validation set predicted label
    # input: final_labels: validation set true label
    # output: kappa, 
    # output: ham_distance
    # output: jaccrd_score  
    
    y_new_pred = np.zeros_like(y_label)
    for i in range(y_label.shape[0]):
        alpha = int(np.sum(y_label[i]))
        top_alpha = np.argsort(y_pred[i, :])[-alpha:]
        y_new_pred[i, top_alpha] = np.array(alpha*[1])
        
    y_label_list = [int(j) for i in list(y_label) for j in i]
    y_pred_list = [int(j) for i in list(y_new_pred) for j in i]
    
    kappa = cohen_kappa_score(y_label_list,y_pred_list)
    ham_distance = hamming_loss(y_label,y_new_pred)
    jaccrd_score = jaccard_score(y_label, y_new_pred, average='micro')
    return kappa, ham_distance, jaccrd_score

def train(feature_df, id_node2vec_file, prob_node2vec_file, adj_distance_file, self_iter):
    # input: feature_matrix 
    # input: id_node2vec_file: node_id node2vec embedding file
    # input: prob_node2vec_file: prob_matrix node2vec embedding file
    # input: adj_distance_file: edge adjacent file  
    # output: res_file: record validation set predict performance
    # output: label_pred_file: record validation set predict right label
    # output: sample_pred_file: record validation set  
    # output: sample_label_file: record validation set  predict label
    # output: test_pred_file: record validation set test label
    # output: model: record model
    
    res = open('output/res_output_iter_' + str(self_iter) + '.txt', 'w+')
    label_pred_file = open('others/label_pred_output_iter_' + str(self_iter)  + '.txt', 'w+')
    res.write('sample_num\tweighted prediction rate\tprediction right nodes\tprediction rate\tprediction right labels\trecall rate\tkappa\tham_distance\tjaccrd_score\tloss\n')
    sample_file = open('others/sample_output_iter_' + str(self_iter)  + '.txt', 'w+')
    
    for epoch in range(args.epoch):
        print(epoch)
        res.write('epoch: ' + str(epoch) + '\n')
        label_pred_file.write('epoch: ' + str(epoch) + '\n')
        sample_file.write('epoch: ' + str(epoch) + '\n')
        dist_res = []
        
        sample = feature_df[(feature_df['type'] == 0)].sample(frac=0.2) 
        sample_list = []
        for i in sample['id']:
            sample_list.append(i)
        sample_file.write('\t'.join([str(i) for i in sample_list]) + '\n')
        feature_df['sample'] = feature_df['id'].apply(lambda x: 1 if x in sample_list else 0)    
        
        x_scale = reshape(feature_df['scale_encoding'].values)
        x_idn2v = load_node2vec_embedding(id_node2vec_file)
        x_pro = load_node2vec_embedding(prob_node2vec_file)
        x = np.hstack((x_scale, x_idn2v, x_pro))
        
        data_x = torch.tensor(x, dtype=torch.float)
        edges = sp.load_npz(adj_distance_file)
        data_y = torch.tensor(reshape(feature_df['ko_encoding']), dtype=torch.float)
        edge_index = edges.tocoo()
        row, col = edge_index.row, edge_index.col
        data_edge_index = torch.tensor([row, col], dtype=torch.long)
        data = Data(x=data_x, edge_index=data_edge_index, y=data_y)
        data.train_mask = torch.tensor((feature_df['type'] == 0) & (feature_df['sample'] == 0), dtype=torch.uint8)
        data.sample_mask = torch.tensor((feature_df['type'] == 0) & (feature_df['sample'] == 1), dtype=torch.uint8)
        data.test_mask = torch.tensor((feature_df['type'] != 0), dtype=torch.uint8)
        print("number of train cases", sum((feature_df['type'] == 0) & (feature_df['sample'] == 0)))
        print("number of sample cases", sum((feature_df['type'] == 0) & (feature_df['sample'] == 1)))
        print("number of test cases", sum((feature_df['type'] != 0)))
        
        
        model = my_GCN(data.num_features, data.y.size(1))
        loss_op = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for iter_time in range(args.iter):
            model.train()
            optimizer.zero_grad()
            loss = loss_op(model(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask])  
            print(iter_time, loss.item())
            loss.backward()
            optimizer.step()    
        model.eval()
        
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            final_preds = out[data.sample_mask].cpu().numpy()
            final_labels = data.y[data.sample_mask].cpu().numpy()
            test_preds = out[data.test_mask].cpu().numpy()
            perf, right_list, right_label = calculate_accuracy(final_preds,final_labels)
            right_list_id =  [sample_list[i] for i in right_list]
            kappa, ham_distance, jaccrd_score = calculate_performance(final_preds,final_labels)
            
            sample_pred_name = 'others/Output_sample_iter_' + str(self_iter) + '_' + str(epoch) + '.npy'
            np.save(sample_pred_name, final_preds)
            sample_labels_name = 'others/Output_label_iter_' + str(self_iter) + '_' + str(epoch) + '.npy'
            np.save(sample_labels_name, final_labels)
            test_pred_name = 'others/Output_test_iter_' + str(self_iter) + '_' + str(epoch) + '.npy'
            np.save(test_pred_name, test_preds)
            model_save_name = 'others/Output_model_iter_' + str(self_iter) + '_' + str(epoch) + '.model'
            torch.save(model.state_dict(), model_save_name)
        
        for i in range(len(right_list_id)):
            label_pred_file.write(str(right_list_id[i]) + '\t' + str(right_label[i]) + '\n')
        
        write_content =  str(len(sample_list)) + '\t' + '\t'.join([str(i) for i in perf]) +  '\t'+ str(kappa) + '\t'+ str(ham_distance) + '\t' + str(jaccrd_score) + '\t' + str(float(loss)) +'\n'
        res.write(write_content)  
        
    res.close()
    label_pred_file.close()
    sample_file.close()        
    print('\t'.join([str(i) for i in perf]) + '\t'+ str(kappa) + '\t'+ str(ham_distance) + '\t' + str(jaccrd_score) + '\t' + str(float(loss))) 
    print('iter '+ str(self_iter) + ' complete')
    return feature_df     

def test_to_train(feature_df, test_file_list, sample_file_list, self_iter):
    # choose stable test with predicted labels to train #
    # input：feature_df: origin feature matrix of this iter 
    # input：test_file_list: test prediction file
    # input: sample_file_list: train prediction file 
    # output: feature_df_iter: feature matrix after add test 
    
    ko_num = len(set([j for i  in feature_df.ko_label for j in i]))
    test_num = len(feature_df[feature_df['type'] != 0])
    
    # stat test label #
    ko_sec_encoding = [[0] * ko_num for i in range(test_num)] 
    for i in range(len(test_file_list)):
        file_label = np.load(read_dir + test_file_list[i])
        pred_index_f = []
        for test_id in range(len(file_label)):
            pred_index = list(np.argsort(-file_label[test_id])[:3])
            pred_index_f.append(pred_index)
    
        for test_id in range(len(pred_index_f)):
            for sort_index in pred_index_f[test_id]:
                ko_sec_encoding[test_id][sort_index] += 1
                
    test_ko_max_dict = {}
    for i in range(len(ko_sec_encoding)):
        test_ko_max_dict[i] = 0
    for test_id in range(len(ko_sec_encoding)):
        test_ko_max_dict[test_id] = max(ko_sec_encoding[test_id])
        
    # stat test max label distribution
    plt.figure(figsize=(10,10))
    x, y = zip(*sorted(dict(Counter(test_ko_max_dict.values())).items()))
    plt.bar(x, y, color='royalblue')
    plt.xlabel('label_id', fontdict={'size' : 15})
    plt.ylabel('count', fontdict={'size' : 15})
    plt.yticks(size = 12) 
    plt.xticks(size = 12) 
    plt.title('test max label distribution',fontsize='large')
    plt.savefig('output/test_max_label_distribution_iter_' + str(self_iter) + '.jpg')
    
    test_ko_ex_dict = {}
    for i in range(len(ko_sec_encoding)):
        test_ko_ex_dict[i] = []
    for test_id in range(len(ko_sec_encoding)):
        for ko_label in range(len(ko_sec_encoding[test_id])):
            if ko_sec_encoding[test_id][ko_label] > int(args.epoch*0.8-1):
                test_ko_ex_dict[test_id].append(ko_label)
    
    ko_filter_45_dict = {}
    for k, v in test_ko_ex_dict.items():
        if len(v) > 0:
            ko_filter_45_dict[k] = v
    
    stat_file = open('output/stat.txt', 'a+')
    stat_file.write("test add: " + str(len(ko_filter_45_dict)) + '\n' ) 
    stat_file.close()   
    
    # write to feature_df
    feature_train = feature_df[(feature_df['type'] == 0)]
    feature_test = feature_df[(feature_df['type'] != 0)].reset_index(drop=True)
    
    test_id = list(feature_df[(feature_df['type'] != 0)].id)
    ko_encoding = [[0] * ko_num for i in range(len(test_id))]
    for id, ko_list in ko_filter_45_dict.items():
        for kolabel in ko_list:
            ko_encoding[id][kolabel] = 1
    feature_test['ko_encoding'] = ko_encoding
    
    for i in ko_filter_45_dict.keys():
        feature_test.loc[i,'type'] = 0
    feature_new_df = pd.concat([feature_test, feature_train])
    feature_new_df['id'] = feature_new_df['id'].apply(lambda x:int(x))
    feature_new_df = feature_new_df.sort_values(by='id').reset_index(drop=True)
    
    feature_new_df.to_pickle('others/features_' + str(self_iter) + '_0_df.pkl')
    print( 'iter: ' + str(self_iter) + ' choose stable test complete')
    return feature_new_df, ko_sec_encoding

def correct_num_dist(label_pred_file):
    # stat prediction number from validation set #
    # input: label_pred_file: prediction right label and node file
    # output: right list: predict right number list
    
    right_dict = {}
    right_ko_dict = {}
    with open(label_pred_file, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if line.startswith('epoch'):
                key = line.split(':')[1]
                right_dict[key] = []
                right_ko_dict[key] = []
            else:
                right_dict[key].append(line.split('\t')[0])
                right_ko_dict[key].append(int(line.split('\t')[1]))
                
    right_list = []
    right_ko_list = []
    for k, v in right_dict.items():
        right_list += list(set(v))
    for k, v in right_ko_dict.items():
        right_ko_list += v

    return right_list  

def train_to_test(right_list, sample_list_file, feature_df, self_iter):
    # choose train with unstable predicted  to test #
    # input：right_list: predict right number list
    # input: sample_list_file: sample list file
    # input: feature_df: feature matrix after add test 
    # input: self_iter: number of iter
    # output: feature_df: feature matrix after add test and delete train
    # output: train_node: update train node
    # output: sample_ratio_dict: prediction distribution from train set of this iter
    
    # stat sample prediction rate
    sample_true_dict = dict(Counter(right_list))
    
    sample_list = []
    with open(sample_list_file, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if line.startswith('epoch'):
                continue
            sample_list += line.split('\t')
    sample_dict = dict(Counter(sample_list))
    
    sample_ratio_dict ={}
    for k,v in sample_dict.items():
        if k in sample_true_dict.keys():
            sample_ratio_dict[k] = round(sample_true_dict[k]/v, 1)
        else:
            sample_ratio_dict[k] = 0
            
    plt.figure(figsize=(10,10))
    x, y = zip(*sorted(dict(Counter(sample_ratio_dict.values())).items()))
    x = [i for i in x]
    sns.barplot(x,y, palette=sns.cubehelix_palette(10, start=.5, rot=-.75))
    plt.xlabel('Prediction rate', fontdict={'size' : 15})
    plt.ylabel('Number', fontdict={'size' : 15})
    plt.yticks(size = 12) 
    plt.xticks(size = 12) 
    plt.title('train prediction rate distribution',fontsize='large')
    plt.savefig('output/train_prediction_rate_distribution_iter_' + str(self_iter)  + '.jpg')
    
    # remove pradiction rate less than 40%  node
    del_3_list = []
    for k, v in sample_ratio_dict.items():
        if v < 0.4:
            del_3_list.append(k)  
    
    # write to feature_df
    feature_df = pd.read_pickle(feature_df)
            
    feature_df.loc[feature_df['id'].isin(del_3_list),'type'] = 2
    feature_df.to_pickle('others/features_' + str(self_iter) + '_1_df.pkl')
    
    stat_file = open('output/stat.txt', 'a+')
    stat_file.write("train del: " + str(len(del_3_list)) + '\n' ) 
    train_node = len(feature_df[feature_df['type'] == 0])
    stat_file.write("total train: " + str(train_node) + '\n' ) 
    stat_file.close()   
    
    return feature_df, train_node, sample_ratio_dict

def predict_output(origin_train_node, origin_test_node , test_node, sample_ratio_dict, ko_test_sec_encoding, feature_df, ko_hash_file, self_iter):    
    acc_train_node = []
    high_acc_train_node = []
    low_train_node = []

    for node, rate in sample_ratio_dict.items():
        if node in origin_test_node:
            if 1 > rate >= 0.9:
                acc_train_node.append(int(node))
            elif rate == 1:
                high_acc_train_node.append(int(node))
            else:
                low_train_node.append(int(node))
        
            
    # high prediction rate test node
    test_acc_dict = {}
    test_high_acc_dict = {}
    for i in range(len(ko_test_sec_encoding)):
        test_acc_dict[i] = []
        test_high_acc_dict[i] = []
    for test_id in range(len(ko_test_sec_encoding)):
        for ko_label in range(len(ko_test_sec_encoding[test_id])):
            if int(args.epoch) >ko_test_sec_encoding[test_id][ko_label] > int(args.epoch*0.8-1):
                test_acc_dict[test_id].append(ko_label)
            if ko_test_sec_encoding[test_id][ko_label] == int(args.epoch):
                test_high_acc_dict[test_id].append(ko_label)
                
    acc_test_list = []
    acc_test_node = []
    high_acc_test_node = []
    for k, v in test_acc_dict.items():
        if len(v) > 0 and (str(test_node[k]) in origin_test_node):
            acc_test_list.append(int(k))
            acc_test_node.append(int(test_node[k]))
    for k, v in test_high_acc_dict.items():
        if len(v) > 0 and (str(test_node[k]) in origin_test_node):
            acc_test_list.append(int(k))
            high_acc_test_node.append(int(test_node[k]))
    acc_node = acc_train_node + acc_test_node
    high_acc_node = high_acc_train_node + high_acc_test_node
    
    feature_df = pd.read_pickle(feature_df)
    low_feature_df = feature_df[feature_df['id'].isin(low_train_node)]
    acc_feature_df = feature_df[feature_df['id'].isin(acc_node)]
    high_acc_feature_df = feature_df[feature_df['id'].isin(high_acc_node)]
    acc_label_dict = {}
    for i in acc_node: 
        ko_em = acc_feature_df[acc_feature_df['id'] == i].ko_encoding
        ko_em = list(acc_feature_df[acc_feature_df['id'] == i].ko_encoding)[0]
        label_index = [index for index, value in enumerate(ko_em) if value != 0]
        acc_label_dict[i] = label_index
    high_acc_label_dict = {}
    for i in high_acc_node: 
        ko_em = high_acc_feature_df [high_acc_feature_df ['id'] == i].ko_encoding
        ko_em = list(high_acc_feature_df [high_acc_feature_df ['id'] == i].ko_encoding)[0]
        label_index = [index for index, value in enumerate(ko_em) if value != 0]
        high_acc_label_dict[i] = label_index
    low_acc_train_label_dict = {}
    for i in low_train_node:
        ko_em = low_feature_df[low_feature_df['id'] == i].ko_encoding
        ko_em = list(low_feature_df[low_feature_df['id'] == i].ko_encoding)[0]
        label_index = [index for index, value in enumerate(ko_em) if value != 0]
        low_acc_train_label_dict[i] = label_index
    
    # other test node
    low_acc_test = [ i for i in range(len(test_node)) if (i not in acc_test_list) and (str(test_node[i]) in origin_test_node)]
    test_low_dict = {}
    for i in low_acc_test:
        label = ko_test_sec_encoding[i].index(max(ko_test_sec_encoding[i]))
        test_low_dict[int(test_node[i])] = label
        
    print('rank ** acc number: ' + str(len(high_acc_label_dict)))      
    print('rank * acc number: ' + str(len(acc_label_dict)))   
    print('rank  acc number: ' + str(len(low_acc_test) + len(low_train_node)))
    print('total: ' + str(len(set(list(low_acc_train_label_dict.keys()) + 
                                  list(acc_label_dict.keys()) + 
                                  list(high_acc_label_dict.keys()))) + len(low_acc_test)))
    
    # write to output file #
    ko_hash_dict = {}
    with open(ko_hash_file, 'r') as f:
        for line in f:
            line = line.strip('\n')
            ko_hash_dict[int(line.split('\t')[0])] = line.split('\t')[2]    
            
    
    test_file_name = 'output/test_output_iter_' + str(self_iter) + '.txt'
    with open(test_file_name, 'w+') as f:
        f.write('node\tlabel\trank\n')
        for node, label_list in high_acc_label_dict.items():
            for label in label_list:
                write_content = str(node) +'\t'+ ko_hash_dict[label] + '\t**\n'
                f.write(write_content)
        for node, label_list in acc_label_dict.items():
            for label in label_list:
                write_content = str(node) +'\t'+ ko_hash_dict[label] + '\t*\n'
                f.write(write_content)       
        for node, label_list in low_acc_train_label_dict.items():
            for label in label_list:
                write_content = str(node) +'\t'+ ko_hash_dict[label] + '\t*\n'
                f.write(write_content)
        for node, label in test_low_dict.items():
            write_content = str(node) +'\t'+ ko_hash_dict[label] + '\t\t\n'
            f.write(write_content)
        f.close()
    print('iter: ' + str(self_iter) + ' finished')
    return feature_df

feature_df = pd.read_pickle(args.input_feature)
scale_encoding = [[0] for i in range(len(feature_df))]
for i, row in feature_df.iterrows():
    scale_encoding[i][0] = feature_df.scale[i]
feature_df['scale_encoding'] = scale_encoding
id_node2vec_file = 'input/node2vec_idOH.txt'
prob_node2vec_file = 'input/node2vec_prob.txt'
adj_distance_file = 'input/adj_distance.npz'

train_node = len(feature_df[feature_df['type'] == 0])
origin_train_node = list(feature_df[feature_df['type'] == 0].id)
origin_test_node = list(feature_df[feature_df['type'] == 1].id)
total_node = len(feature_df)
self_iter = 0
while train_node/total_node < 0.95:
    # train #
    print('iter: ', self_iter)
    test_node = list(feature_df[feature_df['type'] != 0].id)
    feature_df = train(feature_df, id_node2vec_file, prob_node2vec_file, adj_distance_file,self_iter)
    
    stat_file = open('output/stat.txt', 'a+')
    stat_file.write("iter: " + str(self_iter) + '\n' ) 
    stat_file.write("number of train cases: " + str(sum((feature_df['type'] == 0))) + '\n')
    stat_file.write("number of test cases: " + str(sum((feature_df['type'] != 0))) + '\n')
    stat_file.close()   
    
    # like—self-supervised learning # 
    read_dir = 'others/result/'
    test_path = 'Output_test_iter_' + str(self_iter)
    sample_path = 'Output_sample_iter_' + str(self_iter)
        
    test_file_list = [i for i in os.listdir(read_dir) if i.startswith(test_path)]
    sample_file_list = [i for i in os.listdir(read_dir)  if i.startswith(sample_path)]   
    
    feature_df, ko_test_sec_encoding = test_to_train(feature_df, test_file_list, sample_file_list, self_iter)
    label_pred_file = 'others/result/label_pred_output_iter_' + str(self_iter) + '.txt'
    right_list = correct_num_dist(label_pred_file)
    sample_list_file = 'others/result/sample_output_iter_' + str(self_iter) + '.txt'
    feature_df = 'others/features_' + str(self_iter) + '_0_df.pkl'
    feature_df, train_node, sample_ratio_dict = train_to_test(right_list, sample_list_file, feature_df, self_iter)
    feature_df = 'others/features_' + str(self_iter) + '_1_df.pkl'
    ko_hash_file = 'input/ko_hash.txt'
    feature_df = predict_output(origin_train_node, origin_test_node , test_node, sample_ratio_dict, 
                    ko_test_sec_encoding, feature_df, ko_hash_file, self_iter)
    print('train_node/total_node: ', str(train_node/total_node))
    self_iter += 1

srcFile = 'output/test_output_iter_'+ str(self_iter) + '.txt'
dstFile = 'output/test_output_iter_final.txt'
try:
    os.rename(srcFile,dstFile)
except Exception as e:
    print(e)
    print('predict fail\n')
else:
    print('predict success and output the final prediction\n')