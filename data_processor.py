import tensorflow as tf
import pickle
import json
import os
import math
import numpy as np
import random

class DataProcessor():
    def __init__(self, path="./data/"):
        self.path = path
        self.train_file = self.path + "example.train"
        self.test_file = self.path + "example.test"
        self.map_file = self.path + "map_file.pkl"
        self.doc_file = self.path + "doc_dict.utf8"
        # 数据集中单词的数量
        self.n_words = None

    def load_sentences(self, path):
        sents = list()
        sent = list()
        with open(path, "r") as fr:
            for line in fr:
                # 删除右侧空格
                line = line.rstrip()
                if len(line) == 0:
                    if len(sent) > 0:
                        sents.append(sent)
                        sent = list()
                else:
                    # 每行开头的 一个或多个空格 替换为 $
                    # 经过验证ACE数据集并没有该情况
                    #if line[0] == " ":
                    #    line = "$" + line[1:]
                    #    print("True")
                    #    word = line.split()
                    #else:
                    #    word = line.split()
                    sent.append(line.split())
            if len(sent) > 0:
                sents.append(sent)
        return sents

    # 返回 word_index, index_word, tag_index, index_tag
    def word_tag_map(self, sents):

        # 获取列表每个元素出现频率字典
        def get_times(item_list):
            ret = {}
            for item in item_list:
                if item not in ret:
                    ret[item] = 1
                else:
                    ret[item] += 1
            return ret

        # 获取字符到文本的映射
        def get_mapping(word_nums):
            sorted_items = sorted(word_nums.items(), key=lambda x: (-x[1], x[0]))
            word_index = {word[0]: index for index, word in enumerate(sorted_items)}
            index_word = {index: word for word, index in word_index.items()}
            return word_index, index_word

        word_list = list()
        tag_list = list()
        for s in sents:
            for w in s:
                word_list.append(w[0])
                tag_list.append(w[-1])

        word_nums = get_times(word_list)
        word_nums['<PAD>'] = len(word_list) + 2
        word_nums['<UNK>'] = len(word_list) + 1
        word_index, index_word = get_mapping(word_nums)

        tag_nums = get_times(tag_list)
        tag_index, index_tag = get_mapping(tag_nums)

        return word_index, index_word, tag_index, index_tag


    # 获取一个句子周围的句子
    def get_doc_features(self, doc_id, word_index, doc_dict, word_as_num):
        sents_num = 8
        doc_sents = doc_dict[doc_id[0]]
        doc_as_num = list()
        # 文档文本数字化
        for sent in doc_sents:
            item = [word_index[w if w in word_index else '<UNK>'] for w in sent]
            doc_as_num.append(item)
        # 获取句子在文档中的序号
        sent_order = doc_as_num.index(word_as_num)
        # 文档中句子数量
        doc_sent_num = len(doc_as_num)
        # 如果文档中句子数量少于需要的数量
        doc_around_sent = None
        if doc_sent_num <= sents_num:
            doc_around_sent = doc_as_num
        else:
            if sent_order <= sents_num / 2:
                doc_around_sent = doc_as_num[:sents_num]
            elif doc_sent_num - sent_order <= sents_num / 4:
                doc_around_sent = doc_as_num[-sents_num:0]
            else:
                doc_around_sent = doc_as_num[int(sent_order - sents_num / 2): int(sent_order + sents_num / 2)]
        return doc_around_sent

    def get_sub_features(self, entity_subtypes):
        entity_subtype_dict = {'O': 0, '2_Individual': 1, '2_Time': 2, '2_Group': 3, '2_Nation': 4, '2_Indeterminate': 5, '2_Population_Center': 6, '2_Government': 7, '2_Commercial': 8, '2_Non_Governmental': 9, '2_Media': 10, '2_Building_Grounds': 11, '2_Numeric': 12, '2_State_or_Province': 13, '2_Region_General': 14, '2_Sports': 15, '2_Crime': 16, '2_Land': 17, '2_Air': 18, '2_Water': 19, '2_Airport': 20, '2_Sentence': 21, '2_Educational': 22, '2_Celestial': 23, '2_Underspecified': 24, '2_Shooting': 25, '2_Special': 26, '2_Subarea_Facility': 27, '2_Path': 28, '2_GPE_Cluster': 29, '2_Exploding': 30, '2_Water_Body': 31, '2_Land_Region_Natural': 32, '2_Nuclear': 33, '2_Projectile': 34, '2_Region_International': 35, '2_Medical_Science': 36, '2_Continent': 37, '2_Job_Title': 38, '2_County_or_District': 39, '2_Religious': 40, '2_Contact_Info': 41, '2_Chemical': 42, '2_Subarea_Vehicle': 43, '2_Entertainment': 44, '2_Biological': 45, '2_Boundary': 46, '2_Plant': 47, '2_Address': 48, '2_Sharp': 49, '2_Blunt': 50}
        subtype_features = list()
        for w in entity_subtypes:
            if w == "O":
                subtype_feature = 0
            else:
                subtype_feature = entity_subtype_dict[w.split('-')[1]]
            subtype_features.append(subtype_feature)
        return subtype_features

    def get_seg_features(self, tags):
        tags_dict = {'O': 0, '1_PER': 1, '1_Time': 2, '1_GPE': 3, '1_ORG': 4, '1_FAC': 5, '1_LOC': 6, '1_VEH': 7, '1_Numeric': 8, '1_WEA': 9, '1_Crime': 10, '1_Sentence': 11, '1_Job_Title': 12, '1_Contact_Info': 13}
        seg_feature = []
        for tag in tags:
            if "1_PER" in tag:
                entity_tag = 1
            elif "1_GPE" in tag:
                entity_tag = 2
            elif "1_Time" in tag:
                entity_tag = 3
            elif "1_ORG" in tag:
                entity_tag = 4
            elif "1_FAC" in tag:
                entity_tag = 5
            elif "1_VEH" in tag:
                entity_tag = 6
            elif "1_GPE" in tag:
                entity_tag = 7
            elif "1_Numeric" in tag:
                entity_tag = 8
            elif "1_Crime" in tag:
                entity_tag = 9
            elif "1_Sentence" in tag:
                entity_tag = 10
            elif "1_Contact_Info" in tag:
                entity_tag = 11
            elif "1_Job_Title" in tag:
                entity_tag = 12
            elif "1_WEA" in tag:
                entity_tag = 13
            else:
                entity_tag = 0
            seg_feature.append(entity_tag)
        return seg_feature


    def process_data(self, sents, word_index, tag_index, is_train=True):
        non_index = tag_index["O"]
        # 载入文档文本
        with open(self.doc_file, "r") as fr:
            data_doc = fr.readlines()[0]
            doc_dict = json.loads(data_doc)


        data = []
        for s in sents:
            # 句子的单词列表
            string = list()
            # 句子所属文档id
            doc_id = list()
            # 句子每个单词的 entity_types
            entity_types = list()
            # 句子每个单词的 entity_subtypes
            entity_subtypes = list()
            # 句子每个单词所属 事件标签
            tags = list()
            for w in s:
                if w[0] != "...":
                    string.append(w[0])
                    doc_id.append(w[-4])
                    entity_types.append(w[-3])
                    entity_subtypes.append(w[-2])
                    tags.append(w[-1])

            # 对于句子长度大于4的句子
            if len(string) > 4:
                # 对于每个句子, 单词转变为数字
                word_as_num = [word_index[w if w in word_index else '<UNK>'] for w in string]
                # 选取句子周围的 句子
                doc_around_sents = self.get_doc_features(doc_id, word_index, doc_dict, word_as_num)
                # 将实体标签转变为 数字
                types = self.get_seg_features(entity_types)
                # 将实体子标签转变为 数字
                subtypes = self.get_sub_features(entity_subtypes)

                if is_train:
                    tags = [tag_index[t] for t in tags]
                else:
                    tags = [none_index for _ in tags]
                data.append([string, doc_around_sents, word_as_num, types, subtypes, tags])
        return data

    def sort_and_pad(self, data, batch_size, num_steps):
        num_batch = int(math.ceil(len(data) / batch_size))

        # 根据句子长度排序
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i * batch_size: (i + 1) * batch_size], num_steps))
        return batch_data

    def pad_data(self, data, max_length):
        strings_list = list()
        doc_around_sents_list = list()
        word_as_num_list = list()
        types_list = list()
        subtypes_list = list()
        tags_list = list()

        for item in data:
            string, doc_around_sents, word_as_num, types, subtypes, tags = item
            seqs = list()
            for seq in doc_around_sents:
                if len(seq) <= max_length:
                    padding = [0] * (max_length - len(seq))
                    seqs.append(seq + padding)
                else:
                    seqs.append(seq[0: max_length])

            n_sents = len(doc_around_sents)
            n_sents_max = 8
            if n_sents <= n_sents_max:
                doc_padding = [[0] * max_length] * (n_sents_max - n_sents)
                doc_around_sents_list.append(seqs + doc_padding)
            else:
                doc_around_sents_list.append(seqs[0: n_sents_max])

            if len(string) <= max_length:
                padding = [0] * (max_length - len(string))
                strings_list.append(string + padding)
                word_as_num_list.append(word_as_num + padding)
                types_list.append(types + padding)
                subtypes_list.append(subtypes + padding)
                tags_list.append(tags + padding)
            else:
                strings_list.append(string[: max_length])
                word_as_num_list.append(word_as_num[: max_length])
                types_list.append(types[: max_length])
                subtypes_list.append(subtypes[: max_length])
                tags_list.append(tags[: max_length])

        strings_list = np.asarray(strings_list)
        doc_around_sents_list = np.asarray(doc_around_sents_list)
        word_as_num_list = np.asarray(word_as_num_list)
        types_list = np.asarray(types_list)
        subtypes_list = np.asarray(subtypes_list)
        tags_list = np.asarray(tags_list)

        return [strings_list, doc_around_sents_list, word_as_num_list, types_list, subtypes_list, tags_list]

    def load_dataset(self, batch_size, num_steps):
        # len(train_sents)为句子数量
        # train_sents = [w0, w1, w2, ..., wx, ..., wN]
        # wx = [单词本身, 所属文档ID, 实体类别，实体类别子类， 事件类型标签]
        train_sents = self.load_sentences(self.train_file)
        test_sents = self.load_sentences(self.test_file)

        word_index, index_word, tag_index, index_tag = self.word_tag_map(train_sents)

        self.n_words = len(word_index)
        self.num_tags = len(tag_index)

        with open(self.map_file, "wb") as fw:
            pickle.dump([word_index, index_word, tag_index, index_tag], fw)

        train_data = self.process_data(train_sents, word_index, tag_index)
        test_data = self.process_data(test_sents, word_index, tag_index)



        train_data = self.sort_and_pad(train_data, batch_size, num_steps)
        test_data = self.sort_and_pad(test_data, batch_size, num_steps)
        random.shuffle(train_data)
        random.shuffle(test_data)

        return train_data, test_data


if __name__ == "__main__":
    dp = DataProcessor()
    train_data, test_data = dp.load_dataset(20, 40)
    print(train_data[0][0][0])
    print(train_data[0][0][0])
    with open("./train_data.pkl", "wb") as fw:
        pickle.dump(train_data, fw)

    # test word_tag_map
    """
    test = [[["aa", "AA"], ["bb", "BB"], ["bb", "BB"], ["cc", "CC"]]]
    w_i, i_w, t_i, i_t = dp.word_tag_map(test)
    print(w_i)
    print("--------------------")
    print(i_w)
    print("--------------------")
    print(t_i)
    print("--------------------")
    print(i_t)
    """
    #train_data, test_data = dp.load_dataset(20, 40)
    #print(len(train_data))

