import tensorflow as tf
import numpy as np
from data_processor import DataProcessor
import os
import time
import pickle
from sklearn.metrics import classification_report


class SentLstmLayer(tf.keras.layers.Layer):
    def __init__(self, lstm_dim):
        super(SentLstmLayer, self).__init__()
        self.sent_bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_dim, return_sequences=True, return_state=True))

    def call(self, inputs):
        outputs, h0, c0, h1, c1 = self.sent_bilstm(inputs)
        h = tf.concat([h0, h1], -1)
        return outputs, h

class Attention(tf.keras.layers.Layer):
    def __init__(self, is_doc=False):
        super(Attention, self).__init__()

    def get_s(self, source, targets):
        source_w = tf.matmul(source, self.W)
        source_w = tf.expand_dims(source_w, 1)
        prob = tf.matmul(source_w, targets, adjoint_b=True)
        prob = tf.squeeze(prob)
        prob = tf.tanh(prob)
        prob = tf.keras.activations.softmax(prob)
        prob = tf.expand_dims(prob, 1)

        attention_seq = tf.matmul(prob, targets)
        attention_seq = tf.squeeze(attention_seq)
        return attention_seq

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]))

    def call(self, inputs):
        num_step = inputs.shape[1]
        output = list()
        for i in range(num_step):
            atten_seq = self.get_s(inputs[:, i], inputs)
            output.append(atten_seq)
        outputs = tf.transpose(output, [1, 0, 2])
        return outputs


class DocAttention(tf.keras.layers.Layer):
    def __init__(self, is_doc=False):
        super(DocAttention, self).__init__()


    def get_s(self, source, targets):
        source_w = tf.matmul(source, self.W)
        source_w = tf.expand_dims(source_w, 1)
        prob = tf.matmul(source_w, targets, adjoint_b=True)
        prob = tf.add(prob, self.b)
        prob = tf.squeeze(prob)
        prob = tf.tanh(prob)
        prob = tf.keras.activations.softmax(prob)
        prob = tf.expand_dims(prob, 1)

        attention_seq = tf.matmul(prob, targets)
        attention_seq = tf.squeeze(attention_seq)
        return attention_seq

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]))
        self.b = self.add_weight(shape=(1,))

    def call(self, inputs, states, num_step):
        atten = self.get_s(states, inputs)
        atten = tf.expand_dims(atten, 1)
        atten = tf.tile(atten, [1, num_step, 1])
        return atten 

class Gate(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(Gate, self).__init__()
        self.dense = tf.keras.layers.Dense(output_dim, use_bias=True)

    def call(self, sent_atten, doc_atten):
        inputs = tf.concat([sent_atten, doc_atten], -1)
        gate = self.dense(inputs)
        gate = tf.keras.activations.sigmoid(gate)
        gate_ = tf.ones_like(gate) - gate
        outputs = gate * sent_atten + gate_ * doc_atten
        return outputs
        
    
class LstmDecoder(tf.keras.layers.Layer):
    def __init__(self, lstm_dim, output_dim):
        super(LstmDecoder, self).__init__()
        self.lstm_dim = lstm_dim
        self.output_dim = output_dim
        self.lstm_cell = tf.keras.layers.LSTMCell(lstm_dim)
        self.dense = tf.keras.layers.Dense(output_dim, use_bias=True)

    def get_pred_tags(self, h):
        y_pre = self.dense(h)
        tag_pre = tf.cast(tf.argmax(tf.keras.activations.softmax(y_pre), axis=-1), tf.float32)
        return y_pre, tag_pre
    
    def call(self, inputs):
        batch_size = inputs.shape[0]
        num_step = inputs.shape[1]
        outputs = list()
        tag_pre = tf.zeros([batch_size, self.output_dim])
        cell = tf.zeros([batch_size, self.lstm_dim])
        hidden = tf.zeros([batch_size, self.lstm_dim])
        for ts in range(num_step):
            output, (cell, hidden) = self.lstm_cell(tf.concat([inputs[:, ts], tag_pre], -1), (cell, hidden))
            tag_pre, tag_result = self.get_pred_tags(output)
            outputs.append(tag_pre)
        outputs = tf.transpose(outputs, [1, 0, 2])
        return outputs


class Model(tf.keras.Model):
    def __init__(self, vocab_size, num_tags, initial_embed, lstm_dim=100, word_embed_dim=100, types_embed_dim=20, types_num=59, subtypes_dim=20, subtypes_num=51):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.word_embed_dim = word_embed_dim
        self.word_embed = tf.keras.layers.Embedding(self.vocab_size, self.word_embed_dim, weights=[initial_embed])
        # doc embedding
        self.doc_embed = tf.keras.layers.Embedding(self.vocab_size, self.word_embed_dim, weights=[initial_embed])

        self.types_num = types_num
        self.types_embed_dim = types_embed_dim
        self.types_embed = tf.keras.layers.Embedding(self.types_num, self.types_embed_dim)

        self.subtypes_num = subtypes_num
        self.subtypes_dim = subtypes_dim
        self.subtypes_embed = tf.keras.layers.Embedding(self.subtypes_num, self.subtypes_dim)
        self.sent_layer = SentLstmLayer(lstm_dim)
        self.attention = Attention()
        self.doc_bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_dim))
        self.lstm_decoder = LstmDecoder(lstm_dim, lstm_dim)
        self.lstm_decoder2 = LstmDecoder(lstm_dim, lstm_dim)
        self.tag_attention = Attention()
        self.doc_attention = DocAttention()
        self.gate = Gate(lstm_dim * 2)
        self.dense = tf.keras.layers.Dense(num_tags)


    def call(self, inputs):
        # doc_around_sents (20, 8, 40)
        # word_as_num (20, 40)
        # types (20, 40)
        # subtypes (20, 40)
        # tags (20, 40)
        _, doc_around_sents, word_as_num, types, subtypes, tags = inputs
        # 非padding
        used = tf.sign(tf.abs(word_as_num))
        signal = tf.math.equal(used, tf.ones_like(used))
        lengths = tf.reduce_sum(used, axis=-1)
        # 句子嵌入
        # [20, 40, 140]
        sent_embed = self.sent_embedding(word_as_num, types, subtypes)
        # doc_around_sents [20, 8, 40]
        # doc_embed [20, 8, 200]
        doc_embed = self.doc_embedding(doc_around_sents)
        # outputs [20, 40, 200]
        # h [20, 200]
        outputs, h = self.sent_layer(sent_embed)
        doc_atten = self.doc_attention(doc_embed, h, word_as_num.shape[1])
        # sent_att_outputs [20, 40, 200]
        sent_att = self.attention(outputs)
        # gate_output [20 40 200]
        # cr_t = g_t * sh_t + (1 - g_t)
        gate_output = self.gate(sent_att, doc_atten)
        # xr_t = [e_t, cr_t]
        net_inputs = tf.concat([sent_embed, gate_output], -1)
        outputs = self.lstm_decoder(net_inputs)
        tag_attention = self.tag_attention(outputs)
        net_inputs = tf.concat([tag_attention, sent_embed], -1)
        outputs = self.lstm_decoder2(net_inputs)
        pred = self.dense(outputs)
        loss_sum, loss_mean  = self.get_loss(pred, tags, signal)
        return loss_sum, loss_mean, pred

    def doc_embedding(self, doc_around_sents):
        num_step = doc_around_sents.shape[1]
        states_list = list()
        for i in range(num_step):
            embed_doc = self.doc_embed(doc_around_sents[:, i])
            states = self.doc_bilstm(embed_doc)
            states_list.append(states)
        states_list = tf.transpose(states_list, [1, 0, 2])
        return states_list

    def sent_embedding(self, word_as_num, types, subtypes):
        word_embed = self.word_embed(word_as_num)
        type_embed = self.types_embed(types)
        subtype_embed = self.subtypes_embed(subtypes)
        embed = tf.concat([word_embed, type_embed, subtype_embed], -1)
        return embed

    def get_loss(self, logits, tags, signal):
        losses = tf.keras.losses.sparse_categorical_crossentropy(tags, logits, from_logits=True)
        
        # used losses
        losses = tf.boolean_mask(losses, signal)
        # used tag
        tags = tf.boolean_mask(tags, signal)
        # 非O处标记1, others 0
        tag_not_o = tf.sign(tf.abs(tags))
        ones = tf.ones_like(tag_not_o)
        # O处标记1, others 0
        tag_is_o = ones - tag_not_o

        tag_is_o = tf.cast(tag_is_o, dtype=tf.float32)
        tag_not_o = tf.cast(tag_not_o, dtype=tf.float32)
        alpha = 5
        losses = alpha * (losses * tag_not_o) + losses * tag_is_o
        loss_sum = tf.reduce_sum(losses)
        loss_mean = tf.reduce_mean(losses)
        return loss_sum, loss_mean

@tf.function
def train_step(opti, batch, model):
    with tf.GradientTape() as tape:
        loss_sum, loss_mean, _  = model(batch)
    grad = tape.gradient(loss_sum, model.trainable_variables)
    opti.apply_gradients(zip(grad, model.trainable_variables))
    return loss_sum, loss_mean


def test_when_train(data_processor, model, test_data):
    index_tag = data_processor.index_tag
    # vocab size
    n_words = data_processor.n_words
    # tags num
    num_tag = data_processor.num_tags
    @tf.function
    def test_step(batch, model):
        _, _, pred = model(batch)
        return pred

    def decode(pred, lengths):
        ret = list()
        for pred, length in zip(pred, lengths):
            pred = pred[:length]
            pred = tf.argmax(pred, axis = -1).numpy()
            ret.append(pred)
        return ret

    def get_lengths(word_as_num):
        lengths = tf.reduce_sum(tf.sign(tf.abs(word_as_num)), -1).numpy()
        return lengths

    golds = list()
    preds = list()
    appear_tag = set()
    for batch in test_data:
        pred = test_step(batch, model)
        words = batch[0]
        tags = batch[-1]
        words_as_num = batch[2]
        lengths = get_lengths(words_as_num)
        ret = decode(pred, lengths)
        for i in range(len(words)):
            golds.extend(tags[i][:lengths[i]])
            preds.extend(ret[i])
            appear_tag.update(tags[i][:lengths[i]])
            appear_tag.update(ret[i])

    print("==========================================")
    index_tag = data_processor.index_tag
    tags_name = [index_tag[i] for i in list(appear_tag)]
    metrics = classification_report(golds, preds, target_names = tags_name, zero_division=0)
    print(metrics)

def train():
    # 数据集参数
    batch_size = 20
    # 句子长度
    step_num = 40
    data_processor = DataProcessor()
    train_data, test_data = data_processor.load_dataset(batch_size, step_num)
    # 模型参数
    # word embedding
    word_embed_dim = 100
    # vocab size
    n_words = data_processor.n_words
    # tags num
    num_tag = data_processor.num_tags
    types_embed_dim = 20
    subtypes_embed_dim = 20
    embed_path = "./data/100.utf8"
    # 定义模型
    initial_embed = data_processor.load_word2vec(embed_path, 100)
    model = Model(n_words, num_tag, initial_embed=initial_embed)
    # 定义优化器
    opti = tf.keras.optimizers.Adam(learning_rate=0.0001)
    start = None
    for epoch in range(5):
        for idx, batch in enumerate(train_data):
            if start is None:
                start = time.time()
            loss_sum, loss_mean  = train_step(opti, batch, model)
            if (idx + 1) % 100 == 0:
                ends = time.time()
                cost = ends - start
                start = time.time()
                weights = model.get_weights()
                with open("./model/model.pkl", "wb") as fw:
                    pickle.dump(weights, fw)
                print(idx + 1, "---->", loss_mean.numpy(), "---> time cost: ", cost)
                test_when_train(data_processor, model, test_data)

if __name__ =="__main__":
    train()
