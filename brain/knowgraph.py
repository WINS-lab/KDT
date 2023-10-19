# coding: utf-8
"""
KnowledgeGraph
"""
import os
import brain.config as config
import pkuseg
import numpy as np


class KnowledgeGraph(object):
    """
    spo_files - list of Path of *.spo files, or default kg name. e.g., ['HowNet']
    """

    def __init__(self, spo_files, predicate=False):
        self.predicate = predicate
        self.spo_file_paths = [config.KGS.get(f, f) for f in spo_files]
        self.lookup_table = self._create_lookup_table()
        self.segment_vocab = list(self.lookup_table.keys()) + config.NEVER_SPLIT_TAG
        self.tokenizer = pkuseg.pkuseg(model_name="default", postag=False, user_dict=self.segment_vocab)
        self.special_tags = set(config.NEVER_SPLIT_TAG)

    def _create_lookup_table(self): #這邊要建置每個病症對應到的疾病有哪些
        lookup_table = {}
        for spo_path in self.spo_file_paths:
            print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
            with open(spo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        subj, pred, obje = line.strip().split("\t")    #移除檔案中Tab空格
                    except:
                        print("[KnowledgeGraph] Bad spo:", line)
                    if self.predicate: #目前建置的KG都符合此條件
                        value = pred + obje
                    else:
                        value = pred
                    if subj in lookup_table.keys(): #lookup_table已經有紀錄這個病症
                        lookup_table[subj].add(value) 
                        #print("lookup_table_true:",lookup_table) #lookup_table_true: {'乏力': {'單純性肺嗜酸粒細胞浸潤症症狀', '肺泡蛋白質沉積症症狀'}, '發燒': {'大葉性肺炎症狀'}}
                    else:
                        lookup_table[subj] = set([value]) #lookup_table沒有紀錄的病症，加入在後方
                        #print("lookup_table_false:",lookup_table) #lookup_table_false: {'乏力': {'單純性肺嗜酸粒細胞浸潤症症狀'}, '發燒': {'大葉性肺炎症狀'}, '鼻塞': {'大樓病綜合徵症狀'}}
                #print(lookup_table)
        return lookup_table

    def add_knowledge_with_vm(self, sent_batch, max_entities=config.MAX_ENTITIES, add_pad=True, max_length=128):
        """
        input: sent_batch - list of sentences, e.g., ["abcd", "efgh"] 句子列表
        return: know_sent_batch - list of sentences with entites embedding 嵌入實體的句子列表
                position_batch - list of position index of each character. 每個字符的位置索引列表
                visible_matrix_batch - list of visible matrixs 可見矩陣列表
                seg_batch - list of segment tags 段標籤列表
        """
        split_sent_batch = [self.tokenizer.cut(sent) for sent in sent_batch] #原始句子導入並做分詞切割
        know_sent_batch = [] #導入知識庫entity後的句子串列
        position_batch = []
        visible_matrix_batch = []
        seg_batch = []        
        for split_sent in split_sent_batch:
            # create tree
            sent_tree = []            
            pos_idx_tree = []
            abs_idx_tree = []
            pos_idx = -1 # 因為[CLS]佔一個位置
            abs_idx = -1
            abs_idx_src = []
            kg_temp = {}
            kg_final = {}
            
            for token in split_sent: # 開始檢查原始輸入句子的每個字詞
                entities_original = list(self.lookup_table.get(token, {}))[:max_entities] #會根據當下的字詞去對應知識庫的entity                
                if len(entities_original) != 0: #此段是將提取的entity暫存到一個新字典內
                    for temp_entities in entities_original:
                        if token in kg_temp.keys():
                            kg_temp[token].add(temp_entities)
                        else:
                            kg_temp[token] = set([temp_entities])
                    #print("kg_temp: ",kg_temp)
            
            values = list(list(kg_temp.values()))
            new_values = []
            
            for i in range(len(values)): #有8個
                for j in range(len(values[i])): #有10個
                    for x in range(i+1, len(values)): #跟2到8個比較
                        for y in range(len(values[x])):
                            if list(values[i])[j]==list(values[x])[y]:
                                new_values.append(list(values[i])[j])
            #print("new_values: ", new_values)
                ###可以逐一抓出病症對應的疾病###
            for key, value in kg_temp.items():
                for disease in kg_temp[key]:
                    if disease in new_values: #lookup_table已經有紀錄這個病症
                        if key in kg_final.keys():
                            kg_final[key].add(disease)
                            #print("lookup_table_true:",kg_final) #lookup_table_true: {'乏力': {'單純性肺嗜酸粒細胞浸潤症症狀', '肺泡蛋白質沉積症症狀'}, '發燒': {'大葉性肺炎症狀'}}
                        else:
                            kg_final[key] = set([disease]) #lookup_table沒有紀錄的病症，加入在後方
                            #print("lookup_table_false:",kg_final) #lookup_table_false: {'乏力': {'單純性肺嗜酸粒細胞浸潤症症狀'}, '發燒': {'大葉性肺炎症狀'}, '鼻塞': {'大樓病綜合徵症狀'}}
            #print("kg_final: ", kg_final)
                
                #print(entities)
                # if len(entities) != 0:
                #     for element in entities:
                #         kg_temp.append(element)
                #     print(kg_temp)

                # kg_temp = {}
                # kg_temp[token].add(entities)
                # print("kg_temp: ",kg_temp)
                
                
                # #kg_temp[token].add(temp_entities)
                
                # #kg_temp.add(entities)
                # print("Knowledge: ",kg_temp)
                
                #print("Knowledge: ", kg_temp)
                #if(kg_temp)
            for token in split_sent:                
                entities = list(kg_final.get(token, {})) #會根據當下的字詞去對應知識庫的entity
                sent_tree.append((token, entities))
                #print("sent_tree: ", sent_tree)
                                               
                if token in self.special_tags: # 檢查句子的每個字當中有特殊字元[CLS]、[SEP]等
                    token_pos_idx = [pos_idx+1] # 整個句子的soft-position從0開始計算
                    token_abs_idx = [abs_idx+1] # 整個句子的hard-position從0開始計算
                    # print("token_pos_idx: ", token_pos_idx)
                    # print("token_abs_idx: ", token_abs_idx)
                else: #去數分割的詞有幾個字
                    token_pos_idx = [pos_idx+i for i in range(1, len(token)+1)]
                    token_abs_idx = [abs_idx+i for i in range(1, len(token)+1)]
                    # print("token_pos_idx: ", token_pos_idx)
                    # print("token_abs_idx: ", token_abs_idx)
                abs_idx = token_abs_idx[-1] # 用最後一個字來做下面的Entity延伸
                pos_idx = token_pos_idx[-1] # 用最後一個字來做下面的Entity延伸

                entities_pos_idx = []
                entities_abs_idx = []
                for ent in entities:
                    ent_pos_idx = [pos_idx + i for i in range(1, len(ent)+1)] #Paper中hard-position index
                    entities_pos_idx.append(ent_pos_idx)
                    ent_abs_idx = [abs_idx + i for i in range(1, len(ent)+1)] #Paper中soft-position index
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)
                    # print("entities_abs_idx:", entities_abs_idx)

                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                # print("pos_idx_tree: ", pos_idx_tree)
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                # print("abs_idx_tree: ", abs_idx_tree)
                abs_idx_src += token_abs_idx
                
                # for a in range(len(sent_tree)):
                #     check = sent_tree[a][0]
                #     for b in range(len(sent_tree[a][1])):
                        

            # Get know_sent and pos
            know_sent = []
            pos = []
            seg = []

            # 從頭開始檢查句子中的每個字與special_tags(CLS、SPE等等)配對
            for i in range(len(sent_tree)):
                word = sent_tree[i][0]
                if word in self.special_tags: #如果檢查的第一個字是special_tags
                    know_sent += [word]
                    seg += [0]
                else:
                    add_word = list(word)
                    know_sent += add_word 
                    seg += [0] * len(add_word)
                pos += pos_idx_tree[i][0]
                for j in range(len(sent_tree[i][1])):
                    add_word = list(sent_tree[i][1][j])
                    know_sent += add_word
                    seg += [1] * len(add_word)
                    pos += list(pos_idx_tree[i][1][j])
            # print("know_sent: ", know_sent)
            # print("pos: ", pos)
            # print("seg: ", seg)
            token_num = len(know_sent)

            # Calculate visible matrix
            visible_matrix = np.zeros((token_num, token_num))
            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids:
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                    visible_matrix[id, visible_abs_idx] = 1
                for ent in item[1]:
                    for id in ent:
                        visible_abs_idx = ent + src_ids
                        visible_matrix[id, visible_abs_idx] = 1

            src_length = len(know_sent)
            if len(know_sent) < max_length:
                pad_num = max_length - src_length
                know_sent += [config.PAD_TOKEN] * pad_num
                seg += [0] * pad_num
                pos += [max_length - 1] * pad_num
                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
            else:
                know_sent = know_sent[:max_length]
                seg = seg[:max_length]
                pos = pos[:max_length]
                visible_matrix = visible_matrix[:max_length, :max_length]
            
            know_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)
            seg_batch.append(seg)
            # print("know_sent_batch: ", know_sent_batch)
            # print("position_batch: ", position_batch)
            # print("visible_matrix_batch: ", visible_matrix_batch)
            # print("seg_batch: ", seg_batch)
        
        return know_sent_batch, position_batch, visible_matrix_batch, seg_batch

