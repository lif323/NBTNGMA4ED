import tensorflow as tf
import numpy as np
from data_processor import DataProcessor
import os


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
        self.is_doc = is_doc

    def get_s(self, source, targets):
        source_w = tf.matmul(source, self.W)
        source_w = tf.expand_dims(source_w, 1)
        prob = tf.matmul(source_w, targets, adjoint_b=True)
        if self.is_doc:
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
        self.b = self.add_weight(shape=(1, input_shape[-1]), trainable=self.is_doc)

    def call(self, inputs):
        num_step = inputs.shape[1]
        output = list()
        for i in range(num_step):
            atten_seq = self.get_s(inputs[:, i], inputs)
            output.append(atten_seq)
        outputs = tf.transpose(output, [1, 0, 2])
        return outputs


class LSTM_decoder(tf.keras.layers.Layer):
    def __init__(self, lstm_dim, output_dim):
        super(LSTM_decoder, self).__init__()
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
    def __init__(self, vocab_size, num_tags, lstm_dim=100, word_embed_dim=100, types_embed_dim=20, types_num=59, subtypes_dim=20, subtypes_num=51):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.word_embed_dim = word_embed_dim
        self.word_embed = tf.keras.layers.Embedding(self.vocab_size, self.word_embed_dim)
        # doc embedding
        # self.doc_embed = tf.keras.layers.Embedding(self.vocab_size, self.word_embed_dim)

        self.types_num = types_num
        self.types_embed_dim = types_embed_dim
        self.types_embed = tf.keras.layers.Embedding(self.types_num, self.types_embed_dim)

        self.subtypes_num = subtypes_num
        self.subtypes_dim = subtypes_dim
        self.subtypes_embed = tf.keras.layers.Embedding(self.subtypes_num, self.subtypes_dim)


        self.sent_layer = SentLstmLayer(lstm_dim)

        self.attention = Attention()

        #self.doc_bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_dim))

        self.lstm_decoder = LSTM_decoder(lstm_dim, lstm_dim)
        self.lstm_decoder2 = LSTM_decoder(lstm_dim, lstm_dim)
        self.tag_attention = Attention()
        #self.doc_attention = Attention(is_doc=True)

        self.dense = tf.keras.layers.Dense(num_tags)


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
        losses = tf.keras.losses.sparse_categorical_crossentropy(tags, logits)
        """
        losses = tf.boolean_mask(losses, signal)

        tag_NOT_O = tf.sign(tf.abs(tags))
        tag_NOT_O = tf.boolean_mask(tag_NOT_O, signal)

        ones = tf.ones_like(tag_NOT_O)
        tag_is_O = ones - tag_NOT_O
        tag_is_O = tf.cast(tag_is_O, dtype=tf.float32)
        tag_NOT_O = tf.cast(tag_NOT_O, dtype=tf.float32)

        alpha = 5
        losses = alpha * (losses * tag_NOT_O) + losses * tag_is_O
        """
        loss = tf.reduce_sum(losses)
        return loss

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
        # 句子嵌入
        # [20, 40, 140]
        sent_embed = self.sent_embedding(word_as_num, types, subtypes)
        #doc_embed = self.doc_embedding(doc_around_sents)

        # outputs [20, 40, 200]
        # h [20, 200]
        outputs, h = self.sent_layer(sent_embed)
        # sent_att_outputs [20, 40, 200]
        sent_att_outputs = self.attention(outputs)
        net_inputs = tf.concat([sent_embed, sent_att_outputs], -1)
        outputs = self.lstm_decoder(net_inputs)

        tag_attention = self.tag_attention(outputs)
        net_inputs = tf.concat([tag_attention, sent_embed], -1)
        outputs = self.lstm_decoder2(net_inputs)
        pred = self.dense(outputs)
        loss = self.get_loss(pred, tags, signal)
        return loss


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
    # 定义模型
    model = Model(n_words, num_tag, word_embed_dim)
    # 定义优化器
    opti = tf.keras.optimizers.Adam(learning_rate=0.00001)
    for idx, batch in enumerate(train_data):
        with tf.GradientTape() as tape:
            loss = model(batch)
        grad = tape.gradient(loss, model.trainable_variables)
        opti.apply_gradients(zip(grad, model.trainable_variables))
        if (idx + 1) % 100 == 0:
            print(idx + 1, "---->", loss.numpy())

if __name__ =="__main__":
    train()
