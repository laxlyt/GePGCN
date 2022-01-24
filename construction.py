import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import argparse
from collections import Counter
from scipy import sparse
import os
import numpy as np
from scipy.special import comb
from node2vec import Node2Vec
matplotlib.use('agg') 

parser = argparse.ArgumentParser()
parser.add_argument('--input_blast', type=str, default='./test_data/similar_protein_ex.out')
parser.add_argument('--input_fasta', type=str, default='./test_data/whole.faa')
parser.add_argument('--input_pro_id', type=str, default='./input/pro_id.txt')
parser.add_argument('--input_pro_K', type=str, default='./test_data/pro_K.txt')
parser.add_argument('--input_cds_dir', type=str, default='./test_data/CDS/')
args = parser.parse_args()
#args = parser.parse_args([])

def blast_preprocess(blastp_result_file, fasta_file):
    # fliter the blast result #
    # note: blastp --outfmt 6,7 #
    pro = {}
    with open(fasta_file) as f:
        for line in f:
            line = line.strip('\n')
            if line.startswith('>'):
                k = line.split(' ')[0][1: ]
                pro[k] = ''
            else:
                pro[k] += line
    pro_len_dict = {}
    for k, v in pro.items():
        pro_len_dict[k] = len(v)

    read_file =  open(blastp_result_file, 'r', encoding='utf-8')
    print('read blast file successfully')
    
    write_file = open('./input/similar_protein.txt', 'w+', encoding='utf-8')
    for line in read_file:
        line = line.strip('\n')
        line = line.replace(':', '|')
        if line.startswith('#'):
            continue
        q_pro_length = pro_len_dict[line.split('\t')[0]]
        s_pro_length = pro_len_dict[line.split('\t')[1]]
        match_length = int(line.split('\t')[3])
        gap = int(line.split('\t')[4])
        cover_ratio = (match_length*2+gap)/(q_pro_length+s_pro_length)
        if float(line.split('\t')[2]) < 65.0:
            continue
        if float(cover_ratio) < 0.65:
            continue
        write_content = line.split('\t')[0]+'\t'+line.split('\t')[1] + '\t'+line.split('\t')[2]+'\n'
        write_file.write(write_content)
    write_file.close()
    print('write filter file successfully')
    print('step 1: blast result complete')


def generate_stat(fliter_input_path):
    # stat species and pro sequence #
    df_similar_pro = pd.read_csv(fliter_input_path, sep='\t', header=None)
    df_similar_pro = df_similar_pro.rename(columns={0:'pro_1', 1:'pro_2', 2:'similar'})
    df_similar_pro = df_similar_pro[df_similar_pro['pro_1'] != df_similar_pro['pro_2']]
    print('read filter file successfully')

    df_filter = df_similar_pro.iloc[:, 0:2]
    df_filter['A'] = df_filter.min(axis=1)
    df_filter['B'] = df_filter.max(axis=1)
    df_filter = df_filter.iloc[:,2:] 
    df_filter = df_filter.drop_duplicates().sort_values(by= ['A', 'B'], ascending=True).reset_index(drop=True)

    df_filter['A'] = df_filter['A'].apply(lambda x:x.replace(':', '|'))
    df_filter['B'] = df_filter['B'].apply(lambda x:x.replace(':', '|'))

    df_filter['sp_A'] = df_filter['A'].apply(lambda x:x.split('|')[0])
    df_filter['sp_B'] = df_filter['B'].apply(lambda x:x.split('|')[0])
    df_filter['A_p'] = df_filter['A'].apply(lambda x:x.split('|')[1])
    df_filter['B_p'] = df_filter['B'].apply(lambda x:x.split('|')[1])

    df_filter.to_csv('./input/unique_unique.csv', sep='\t', index=None, header = None)
    print('write protein stat file successfully')
    print('step 2: generate stat completed')

def generate_node(stat_input_path, fasta_file):
    # generate node and pro sequence hash #
    # need whole genome fasta file #
    read_file = open(stat_input_path, 'r', encoding='utf-8')
    stat_file = open('output/stat.txt', 'w+')
    edge_list = []
    for line in read_file:
        line = line.strip('\n')
        edge_list.append([line.split('\t')[4],line.split('\t')[5]])
    print('read pro file successfully')
    
    G = nx.Graph()
    G.add_edges_from(edge_list)

    connected_node = list()
    for c in nx.connected_components(G):
        nodeSet = G.subgraph(c).nodes()
        connected_node.append(nodeSet)
    
    id_dict = dict()
    with open(fasta_file, 'r') as f:
    # need whole genome fasta file #
        for line in f:
            line = line.strip('\n')
            if line.startswith('>'):
                id_dict[line.split(' ')[0].split('|')[1]] = ''
    stat_file.write('Pro total: ' + str(len(id_dict)) + '\n')
            
    for i in range(len(connected_node)):
        for j in connected_node[i]:
            id_dict[j] = i
        
    id_num = len(connected_node)
    unique_count = 0
    for key, val in id_dict.items():
        if id_dict[key] == '':
            unique_count += 1
            id_dict[key] = id_num
            id_num += 1
            
    stat_file.write('Nodes total: ' + str(len(connected_node) + unique_count) + '\n')        
    stat_file.write('Integrated nodes: ' + str(len(connected_node)) + '\n')
    stat_file.write('Single nodes: ' + str(unique_count) + '\n')
    stat_file.close()
    print('write node stat file successfully')

    write_file = open('input/pro_id.txt', 'w+', encoding='utf-8')
    for key, val in id_dict.items():
        write_content = key + '\t' + str(val) + '\n'
        write_file.write(write_content)
    write_file.close()
    print('write protein node map file successfully')
    print('step 3: generate node complete')

def node_scale_generate(pro_id_file):
    # generate pro_id_dict  node_scale_dict # 
    # input: pro_id file #
    # output: id_pro_dict, pro_id_dict, id_scale_dict #
    id_pro_dict = {}
    with open(pro_id_file, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if line.split('\t')[1] not in id_pro_dict.keys():
                id_pro_dict[line.split('\t')[1]] = []
            id_pro_dict[line.split('\t')[1]].append(line.split('\t')[0])
    print('create id_pro_dict complete')
    
    pro_id_dict = {}
    with open(pro_id_file, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if line.split('\t')[0] not in pro_id_dict.keys():
                pro_id_dict[line.split('\t')[0]] = []
            pro_id_dict[line.split('\t')[0]].append(line.split('\t')[1])
    print('create pro_id_dict complete')
            
    id_scale_dict = {}
    for k, v in id_pro_dict.items():
        id_scale_dict[k] = len(v)
    print('create id_scale_dict complete')
    

    plt.figure(figsize=(10,10))
    x, y = zip(*sorted(Counter(dict(Counter(id_scale_dict.values())).values()).items()))
    plt.bar(x, y, color='royalblue')
    plt.xlabel('node scale', fontdict={'size' : 15})
    plt.ylabel('count', fontdict={'size' : 15})
    plt.yticks(size = 12) 
    plt.xticks(size = 12) 
    plt.title('node scale distribution',fontsize='large')
    plt.savefig('output/node_scale_distribution.jpg')
    
    print('step 4: generate node scale successfully')
    return id_pro_dict, pro_id_dict, id_scale_dict

def feature_matrix_generate(pro_K_file, id_pro_dict, pro_id_dict, id_scale_dict, KO_ko_file, pathway_info_file):
    # generate feature matrix #
    # input: pro_K file, KO_ko_file, pathway_info_file #
    # output: feature.pkl #
    pro_K_dict = {}
    with open(pro_K_file, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if line.split('\t')[0].split('|')[1] not in pro_K_dict.keys():
                pro_K_dict[line.split('\t')[0].split('|')[1]] = []
            pro_K_dict[line.split('\t')[0].split('|')[1]].append(line.split('\t')[1])
    print('generate pro_K dict complete')
    
    node_df = pd.DataFrame.from_dict(id_scale_dict, orient='index',columns=['scale'])
    node_df = node_df.reset_index().rename(columns = {'index':'id'})
    node_df['pro'] = node_df['id'].apply(lambda x:id_pro_dict[str(x)])
    
    id_K_num_dict = {}
    for k, v in id_pro_dict.items():
        id_K_num_dict[k] = []
        for pro in v:
            if pro in pro_K_dict.keys():
                id_K_num_dict[k] += pro_K_dict[pro]
                
    id_K_dict = {}
    for k, v in id_K_num_dict.items():
        id_K_dict[k] = list(set(v))
        
    K_ko_dict = {}
    with open(KO_ko_file, 'r') as f:
        for line in f:
            line = line.strip('\n')
            K_ko_dict[line.split('\t')[0]] = line.split('\t')[1].split(' ')
                
    id_ko_tmp_dict = {}
    for k, v in id_K_dict.items():
        if len(v) != 0:
            id_ko_tmp_dict[k] = [K_ko_dict[K] for K in v if K in K_ko_dict.keys()]
        else:
            id_ko_tmp_dict[k] = []
            
    id_ko_dict = {}
    for k , v in id_ko_tmp_dict.items():
        if len(v) != 0:
            id_ko_dict[k] = list(set([ko for ko_list in v for ko in ko_list]))
        else:
            id_ko_dict[k] = []
            
    # stat node with labels #
    node_labeled_count = 0
    for v in id_ko_dict.values():
        if v != []:
            node_labeled_count += 1
    
    # stat pro with labels #
    pro_ko_dict = {}
    for k, v in pro_K_dict.items():
        if v[0] in K_ko_dict.keys():
            pro_ko_dict[k] = K_ko_dict[v[0]]
    pro_labeled_count = len(pro_ko_dict)
    
    stat_file = open('output/stat.txt', 'a+')
    stat_file.write('pro with labels: ' + str(pro_labeled_count) + '\n')
    stat_file.write('node with labels: ' + str(node_labeled_count) + '\n')
    stat_file.close()
    print('assign labels complete')
    
    # create feature matrix
    node_df['K'] = node_df['id'].apply(lambda x:id_K_dict[str(x)])
    node_df['ko'] = node_df['id'].apply(lambda x:id_ko_dict[str(x)])
        
    ko_list = list(set([j for i in id_ko_dict.values() for j in i]))
    ko_kod_dict = {}
    ko_sec_list = []
    with open(pathway_info_file, 'r') as f:
        for line in f:
            if line.split('\t')[0] in ko_list:
                ko_kod_dict[line.split('\t')[0]] = line.split('\t')[5]
                ko_sec_list.append(line.split('\t')[5])
    ko_sec_list = sorted(list(set(ko_sec_list)))
    
    ko_info_dict = {}
    with open(pathway_info_file, 'r') as f:
        for line in f:
            if line.split('\t')[5] not in ko_info_dict.keys():
                ko_info_dict[line.split('\t')[5]] = line.split('\t')[2]
    
    kod_ko_dict = {}
    for k, v in ko_kod_dict.items():
        if v not in kod_ko_dict.keys():
            kod_ko_dict[v] = []
        kod_ko_dict[v].append(k)
        
    id_kod_dict = {}
    for id, ko_list in id_ko_dict.items():
        if len(ko_list) > 0:
            id_kod_dict[id] = list(set([ko_kod_dict[i] for i in ko_list if i in ko_kod_dict.keys()]))
        else:
            id_kod_dict[id] = []
    
    node_df['ko_label'] = node_df['id'].apply(lambda x:id_kod_dict[str(x)])
    
    ko_sec_items = [item for sublist in node_df['ko_label'] for item in sublist]
    ko_sec_unique_elements, ko_sec_counts_elements = np.unique(ko_sec_items, return_counts=True)
    
    # stat label distribution #
    label_list = [j for i in list(node_df.ko_label) for j in i]
    label_list_new = [ko_info_dict[i] for i in label_list] 
    x, y = zip(*sorted(dict(Counter(label_list_new)).items()))
    x = [i for i in x]
    y = [i for i in y]
    plt.figure(figsize=(10,20))
    sns.barplot(y,x, palette=sns.cubehelix_palette(45, start=.5, rot=-.75), orient='h')
    plt.xlabel('Count', fontdict={'size' : 16})
    plt.xticks(size = 10) 
    plt.title('label distribution',fontsize='large')
    plt.savefig('output/label_distribution.jpg')
    
    # ko_label encoding #
    ko_sec_dict = dict(zip(list(ko_sec_list), range(len(ko_sec_list))))
    ko_sec_encoding = [[0] * len(ko_sec_list) for i in range(len(node_df))]
    for i, row in node_df.iterrows():
        for x in row['ko_label']:
            ko_sec_encoding[i][ko_sec_dict[x]] = 1
    node_df['ko_encoding'] = ko_sec_encoding
    
    # write ko encoding  hash information #
    ko_hash = open('input/ko_hash.txt', 'w+')
    for koid, encoding_id in ko_sec_dict.items():
        write_content = str(encoding_id) + '\t' + koid + '\t' +  ko_info_dict[koid] + '\n'
        ko_hash.write(write_content)
    ko_hash.close()
    print('write ko_hash.txt complete')
    
    # classified nodes between labels and unlabels
    id_ko_type = {}
    for k, v in id_ko_dict.items():
        if len(v) == 0:
            id_ko_type[k] = 1
        else:
            id_ko_type[k] = 0
    node_df['type'] = node_df['id'].apply(lambda x:id_ko_type[str(x)])       
    node_df.to_pickle('input/features_all.pkl')
    print('step 5: generate feature matrix successfully ')

def extract_CDS(cds_read_dir):
    # input: CDS dir #
    # output: distance file #
    
    read_dir = cds_read_dir
    read_file_list = [file for file in os.listdir(read_dir) if file.endswith('fna')]
    print('load cds dir successfully')
    
    for file in read_file_list:
        read_file = open(cds_read_dir + file, 'r', encoding='utf-8')
        loc_dict = {}
        for line in read_file:
            if line.startswith('>'):
                if line.split(' ')[0][5:].split('_')[0] not in loc_dict.keys():
                    loc_dict[line.split(' ')[0][5:].split('_')[0]] = []
                if line.split(' ')[0][5:].split('_')[-2:-1][0] != 'cds':
                    loc_dict[line.split(' ')[0][5:].split('_')[0]].append(line.split(' ')[0][5:].split('_')[-2:])
    
        file_name = 'others/distance/' + file.split('.')[0] + '_loc.txt'
        write_file = open(file_name, 'w+', encoding='utf-8')
        contig_num = 0
        for contig, protein_list in loc_dict.items():
            contig_num += 1
            write_file.write('>' + str(contig_num) + '\n')
            for protein in protein_list:
                write_content = '\t'.join(protein) + '\n'
                write_file.write(write_content)
        write_file.close()
    print('step 6: extract location order successfully')
    return read_file_list

def extract_location_pairs(read_file_list, loc_file_path):
    # extract location pairs from location orders #  
    # input: loc_file #
    # output: dis_file #
    write_single = open(loc_file_path + 'single.txt', 'w+')
    for file in read_file_list:
        loc_file_name = loc_file_path + file.split('.')[0] + '_loc.txt'
        dis_file_name = loc_file_path + file.split('.')[0] + '_dis.txt'
        write_file = open(dis_file_name, 'w+')
        tar_pro = {}
        with open(loc_file_name , 'r') as f: 
            for line in f:
                if line.startswith('>'):
                    line = line.strip('\n')
                    contig = line[1:] 
                    tar_pro[contig] = []            
                else:
                    tar_pro[contig].append(line.split('\t')[0])

        for contig in tar_pro.keys():
            if len(tar_pro[contig]) > 1:
                for i in range(len(tar_pro[contig])-1):
                     write_file.write(tar_pro[contig][i] + '\t' + tar_pro[contig][i+1] + '\n')
            elif len(tar_pro[contig]) == 1:
                write_single.write(tar_pro[contig][0] + '\n')
        write_file.close()
    write_single.close()
    print('step 7: extract location pairs successfully')

def concat_distance(loc_file_path):
    # concat dis file #
    # input: dis_file #
    # output: all_distance_file #
    
    dis_file_list = [i for i in os.listdir(loc_file_path) if i.endswith('_dis.txt')]
    all_dis_file = open('input/all_distance.txt', 'w+')
    for dis_file in dis_file_list:
        dis_file_path = loc_file_path + dis_file
        with open(dis_file_path, 'r') as f:
            for line in f:
                all_dis_file.write(line)
    all_dis_file.close()
    return 'step 8: concat distance file successfully'

def construct_edge(distance_file_path, pro_id_dict):
    # from distance file assign node id to form edgec #
    # input: all_distance.txt, pro_id_dict #
    # output: edge.txt #
    
    # assign node id to location pairs
    dis_df = pd.read_csv(distance_file_path,  sep='\t', header=None)
    dis_df = dis_df.rename(columns={0:'A', 1:'B'})
    dis_df['label_A'] = dis_df['A'].apply(lambda x:int(pro_id_dict[x][0]))
    dis_df['label_B'] = dis_df['B'].apply(lambda x:int(pro_id_dict[x][0]))
    dis = dis_df.iloc[:, 2:]
    dis['A_l'] = dis.min(axis=1)
    dis['B_l'] = dis.max(axis=1)
    dis = dis.drop(columns = ['label_A', 'label_B'])
    dis = dis.sort_values(by=['A_l', 'B_l']).reset_index(drop=True)
    dis.drop_duplicates().to_csv('others/edge_tmp.txt',sep='\t', header=None)
    print('assign node id to location pairs complete')
    
    # construct edge file #
    edge_list = []
    weight = []
    with open('others/edge_tmp.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')
            w, node_1, node_2 = line.split('\t')
            edge_list.append([node_1, node_2])
            weight.append(w)
    
    stat_file = open('output/stat.txt', 'a+')
    stat_file.write('edge: ' + str(len(edge_list)) + '\n')
    stat_file.close()
            
    w_list = []
    for i in range(len(weight)):
        if i != len(weight)-1:
            w_list.append(int(weight[i+1])-int(weight[i]))
        else:
            w_list.append(1) 
            
    with open('input/edge.txt', 'w+') as f:
        for i in range(len(w_list)):
            write_content = edge_list[i][0] + '\t' + edge_list[i][1] + '\t' + str(w_list[i]) + '\n'
            f.write(write_content)
        f.close()
    print('step 9: construct edge successfully')

def construct_adj_distacne_matrix(edge_file_path, node_number):
    # construct adj_distacne_matrix #
    # input: edge file, node_number #
    # output: adj_distance_matrix #
    edge_dic = {}
    with open(edge_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            edge_dic[line.split('\t')[0] + '-' + line.split('\t')[1]] = int(line.split('\t')[2])
    print('load edge file complete')
    
    adj_distance_w = np.zeros((node_number, node_number))
    for k, v in edge_dic.items():
        adj_distance_w[int(k.split('-')[0])][int(k.split('-')[1])] = int(v)
        adj_distance_w[int(k.split('-')[1])][int(k.split('-')[0])] = int(v)
    print('contruct adj distance matrix complete')
    
    sadj_distance_w = sparse.csc_matrix(adj_distance_w)
    sparse.save_npz('input/adj_distance.npz', sadj_distance_w)
    print('step 10: construct adj distance matrix successfully')
    return  adj_distance_w  

def construct_prob_matrix(adj_distance_w, node_number):
    # construct and embeding prob matrix #
    # input: adj_distance_w, node_number #
    # output: node2vec_prob.txt #
    
    # compute prob matrix
    degree_dict = {}
    for node in range(len(adj_distance_w)):
        degree_dict[node] = sum(adj_distance_w[node])    
    degree_total = sum(degree_dict.values())

    degree_p = {}
    for id, degree in degree_dict.items():
        degree_p[id] = degree/degree_total
    
    prob_matrix = np.zeros((node_number, node_number))
    for node_1 in range(node_number):
        for node_2 in range(node_number):
            if adj_distance_w[node_1][node_2] >= 1:
                n = adj_distance_w[node_1][node_2]
                prob = degree_p[node_2]
                degree = degree_dict[node_1]
                prob_matrix[node_1][node_2] = comb(degree, n) * (prob**n) * (1-prob)**(degree-n)
            else:
                prob_matrix[node_1][node_2] = 0
    prob_matrix[np.isnan(prob_matrix)] = 0
    prob_matrix[np.isinf(prob_matrix)] = 0
    print('compute prob matrix complete')
            
    # node2vec embedding prob_matrix
    sprob_matrix = sparse.csc_matrix(prob_matrix)
    sparse.save_npz('others/prob_matrix.npz', sprob_matrix)
    prob_m = nx.from_scipy_sparse_matrix(sprob_matrix)
    node2vec_prob_m = Node2Vec(prob_m, dimensions=256, walk_length=15, num_walks=150, workers=30)
    model_prob_m = node2vec_prob_m.fit(window=10, min_count=1, batch_words=5)
    model_prob_m.wv.save_word2vec_format('input/node2vec_prob.txt')
    print('embedding prob matrix complete')
    
    print('step 11: construct prob matrix successfully')   

def construct_id_node2vec(node_number):
    # node2vec embedding one-hot vector #
    # input: node_number #
    # output: node2vec_idOH.txt #
    
    id_encoding = [[0]*node_number for i in range(node_number)]
    for i in range(node_number):
        id_encoding[i][i] = 1
    print('construct one_hot embedding')
        
    sAdj_idOH = sparse.csr_matrix(id_encoding)
    sparse.save_npz('others/id_OH.npz', sAdj_idOH)
    adj_idOH = nx.from_scipy_sparse_matrix(sAdj_idOH)
    node2vec_idOH = Node2Vec(adj_idOH, dimensions=256, walk_length=15, num_walks=150, workers=30)
    model_idOH = node2vec_idOH.fit(window=10, min_count=1, batch_words=5)
    model_idOH.wv.save_word2vec_format('input/node2vec_idOH.txt')
    print('embedding node_id complete')
    
    print('step 12: embedding node id successfully')  
'''
blast_preprocess(args.input_blast, , args.input_fasta)
fliter_input_path = './input/similar_protein.txt'
generate_stat(fliter_input_path)
'''
stat_input_path = './input/unique_unique.csv'
generate_node(stat_input_path, args.input_fasta)
id_pro_dict, pro_id_dict, id_scale_dict = node_scale_generate(args.input_pro_id)
KO_ko_input_path = 'input/KO_ko.txt'
pathway_info_path = 'input/pathway_info.txt'
feature_matrix_generate(args.input_pro_K, id_pro_dict, pro_id_dict, id_scale_dict, KO_ko_input_path, pathway_info_path)
read_file_list = extract_CDS(args.input_cds_dir)
loc_file_path = 'others/distance/'
extract_location_pairs(read_file_list, loc_file_path)
concat_distance(loc_file_path)
distance_file_path = 'input/all_distance.txt'
construct_edge(distance_file_path, pro_id_dict)
edge_file_path = 'input/edge.txt'
adj_distance_w = construct_adj_distacne_matrix(edge_file_path, len(id_pro_dict))
construct_prob_matrix(adj_distance_w, len(id_pro_dict))
construct_id_node2vec(len(id_pro_dict))
