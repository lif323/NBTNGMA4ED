import tensorflow as tf
import pickle
import json

class DataProcessor():
    def __init__(self, path="./data/"):
        self.path = path
        self.train_file = self.path + "example.train"
        self.test_file = self.path + "example.test"
        self.map_file = self.path + "map_file.pkl"
        self.doc_file = self.path + "doc_dict.utf8"

    def load_sentences(self, path):
        sents = list()
        sent = list()
        with open(path, "r") as fr:
            for line in fr:
                line = line.rstrip()
                if len(line) == 0:
                    if len(sent) > 0:
                        sents.append(sent)
                        sent = list()
                else:
                    # 每行开头的 一个或多个空格 替换为 $
                    if line[0] == " ":
                        line = "$" + line[1:]
                        print("True")
                        word = line.split()
                    else:
                        word = line.split()
                    sent.append(word)
            if len(sent) > 0:
                sents.append(sent)
        return sents

    def word_tag_map(self, sents):
        word_list = [[x[0] for x in s] for s in sents]
        word_list = list()
        tag_list = list()
        for s in sents:
            for w in s:
                word_list.append(w[0])
                tag_list.append(w[-1])
        word_list = word_list + ["<PAD>", "<UNK>"]

        word_list = set(word_list)
        word_index = {word: index for word, index in enumerate(word_list)}
        index_word = {index: word for word, index in enumerate(word_list)}

        tag_list = set(tag_list)
        tag_index = {tag: index for index, tag in enumerate(tag_list)}
        index_tag = {index: tag for index, tag in enumerate(tag_list)}
        return word_index, index_word, tag_index, index_tag

    def process_data(self, sents, word_index, tag_index):
        non_index = tag_index["O"]
        data = []
        with open(self.doc_file, "r") as fr:
            data_doc = fr.readlines()
            doc_dict = json.load(data_doc)




    def load_dataset(self):
        train_sents = self.load_sentences(self.train_file)
        test_sents = self.load_sentences(self.test_file)
        word_index, index_word, tag_index, index_tag = self.word_tag_map(train_sents)

        with open(self.map_file, "wb") as fw:
            pickle.dump([word_index, index_word, tag_index, index_tag], fw)



if __name__ == "__main__":

    dp = DataProcessor()
    dp.load_dataset()

