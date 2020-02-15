# -*- coding:utf-8 -*-

import tensorflow as tf

"""wd_1_1_cnn_concat
title 部分使用 TextCNN；content 部分使用 TextCNN； 两部分输出直接 concat。
"""


class Settings(object):
    def __init__(self):
        self.model_name = 'wd_1_1_cnn_concat'
        self.title_len = 30
        self.content_len = 150
        self.topic_len= 5
        self.filter_sizes = [2, 3, 4, 5, 7]
        self.n_filter = 256
        self.fc_hidden_size = 1024
        self.n_class = 13258


class TextCNN(object):
    """
    title: inputs->textcnn->output_title
    content: inputs->textcnn->output_content
    concat[output_title, output_content] -> fc+bn+relu -> sigmoid_entropy.
    """

    def __init__(self, W_embedding, settings):
        self.model_name = settings.model_name
        self.title_len = settings.title_len
        self.content_len = settings.content_len
        self.topic_len = settings.topic_len
        self.filter_sizes = settings.filter_sizes
        self.n_filter = settings.n_filter
        self.n_filter_total = self.n_filter * len(self.filter_sizes)
        self.n_class = settings.n_class
        self.fc_hidden_size = settings.fc_hidden_size
        self._global_step = tf.Variable(0, trainable=False, name='Global_Step')
        self.update_emas = list()
        # placeholders
        self._tst = tf.placeholder(tf.bool)
        self._keep_prob = tf.placeholder(tf.float32, [])
        self._batch_size = tf.placeholder(tf.int32, [])

        with tf.name_scope('Inputs'):
            self._X1_inputs = tf.placeholder(tf.int64, [None, self.title_len], name='X1_inputs')
            self._X2_inputs = tf.placeholder(tf.int64, [None, self.content_len], name='X2_inputs')
            self._X3_inputs = tf.placeholder(tf.int64, [None, self.topic_len], name='X3_inputs')
            self._y_inputs = tf.placeholder(tf.float32, [None, self.n_class], name='y_input')
            self.dag = tf.placeholder(tf.int64, [None, 6], name='dag')
            self.mask = tf.placeholder(tf.float32, [None, 6], name='mask')

        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable(name='embedding', shape=W_embedding.shape,
                                             initializer=tf.constant_initializer(W_embedding), trainable=True)
        self.embedding_size = W_embedding.shape[1]

        with tf.variable_scope('cnn_text'):
            output_title = self.cnn_inference(self._X1_inputs, self.title_len)

        with tf.variable_scope('hcnn_content'):
            output_content = self.cnn_inference(self._X2_inputs, self.content_len)

        with tf.variable_scope('topic-layer-1'):
            self.topicEmbedding = tf.nn.embedding_lookup(self.embedding, self._X3_inputs)

            self.topicEmbedding=tf.transpose(self.topicEmbedding, perm=[0,2,1])
            self.topicEmbedding=tf.reshape(self.topicEmbedding,[-1,self.topic_len])


            W = tf.Variable(tf.truncated_normal([self.topic_len, 1], stddev=0.1), name="W_topic_1")
            b = tf.Variable(tf.constant(0.1, tf.float32, shape=[1], name="b_topic_2"))

            self.topic_fc= tf.nn.xw_plus_b(self.topicEmbedding, W, b)

            self.topic_fc = tf.reshape(self.topic_fc,[-1,self.embedding_size])
            self.topic_fc_bn_relu = tf.nn.relu(self.topic_fc,name="relu_topic_1")

            
        with tf.variable_scope('topic-layer-2'):
            W = self.weight_variable([self.embedding_size, self.fc_hidden_size], name='W_topic_2')

            self.topic_fc_bn_relu=tf.matmul(self.topic_fc_bn_relu,W)
            b = self.bias_variable([self.fc_hidden_size], name='b_topic_2')

            self.topic_fc_bn_relu,update_ema_fc=self.batchnorm(self.topic_fc_bn_relu, b, convolutional=False)
            self.update_emas.append(update_ema_fc)

            self.topic_fc_bn_relu_out =tf.nn.dropout(self.topic_fc_bn_relu, self.keep_prob)

        with tf.variable_scope('fatherSons'):
            self.dag1=tf.reshape(self.dag,[-1])
            self.dag2 = tf.nn.embedding_lookup(self.topic_fc_bn_relu_out, self.dag1) 
            father=tf.split(self.dag2,[1,6-1],0)[0] #1*256
            fatherSon=tf.concat([father,father,father,father,father,father],0)
            fatherSon=tf.concat([fatherSon,self.dag2],1) #6*512

            W3 = tf.Variable( tf.random_uniform([200, 2*self.fc_hidden_size], -0.1, 0.1), name="W3",trainable=True)
            b3 = tf.Variable(tf.constant(0.1, shape=[200]), name="b3",trainable=True)
            u=tf.Variable( tf.random_uniform([200,1], -0.1, 0.1), name="u",trainable=True)

            alpha=tf.matmul(tf.nn.tanh(tf.nn.bias_add(tf.matmul(fatherSon,W3,transpose_b=True) ,b3)),u)
            alpha = tf.reshape(alpha, [-1])
            alpha = alpha+tf.reshape(self.mask,[-1])
            alpha = tf.nn.softmax(alpha)
            self.alpha = tf.reshape(alpha, [-1,1])  #6*1

            newFather=tf.reduce_sum(self.alpha*self.dag2,0)
            newFather=tf.reshape(newFather,[-1,self.fc_hidden_size])

            temp1=tf.split(self.topic_fc_bn_relu_out,[self.dag1[0],1,self.n_class-1-self.dag1[0]],0)[0]
            temp2=tf.split(self.topic_fc_bn_relu_out,[self.dag1[0],1,self.n_class-1-self.dag1[0]],0)[2]
 
            self.topic_fc_bn_relu_out=tf.concat([temp1,newFather,temp2],0)
       

        with tf.variable_scope('fc-bn-layer-1'):
            output = tf.concat([output_title, output_content], axis=1)
            W_fc = self.weight_variable([self.n_filter_total * 2, self.fc_hidden_size], name='Weight_fc')

            h_fc = tf.matmul(output, W_fc, name='h_fc')
            beta_fc = tf.Variable(tf.constant(0.1, tf.float32, shape=[self.fc_hidden_size], name="beta_fc"))

            fc_bn, update_ema_fc = self.batchnorm(h_fc, beta_fc, convolutional=False)
            self.update_emas.append(update_ema_fc)
            self.fc_bn_relu = tf.nn.relu(fc_bn, name="relu")
            fc_bn_drop = tf.nn.dropout(self.fc_bn_relu, self.keep_prob)


        with tf.variable_scope('out_layer'):

            b_out = self.bias_variable([self.n_class], name='bias_out')

            self._y_pred = tf.nn.xw_plus_b(fc_bn_drop, tf.transpose(self.topic_fc_bn_relu_out), b_out, name='y_pred')  # 每个类别的分数 scores

        with tf.name_scope('loss'):
            self._loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self._y_pred, labels=self._y_inputs))



    @property
    def tst(self):
        return self._tst

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def global_step(self):
        return self._global_step

    @property
    def X1_inputs(self):
        return self._X1_inputs

    @property
    def X2_inputs(self):
        return self._X2_inputs
    @property
    def X3_inputs(self):
        return self._X3_inputs

    @property
    def y_inputs(self):
        return self._y_inputs

    @property
    def y_pred(self):
        return self._y_pred

    @property
    def loss(self):
        return self._loss

    def weight_variable(self, shape, name):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def batchnorm(self, Ylogits, offset, convolutional=False):
        """batchnormalization.
        Args:
            Ylogits: 1D向量或者是3D的卷积结果。
            num_updates: 迭代的global_step
            offset：表示beta，全局均值；在 RELU 激活中一般初始化为 0.1。
            scale：表示lambda，全局方差；在 sigmoid 激活中需要，这 RELU 激活中作用不大。
            m: 表示batch均值；v:表示batch方差。
            bnepsilon：一个很小的浮点数，防止除以 0.
        Returns:
            Ybn: 和 Ylogits 的维度一样，就是经过 Batch Normalization 处理的结果。
            update_moving_everages：更新mean和variance，主要是给最后的 test 使用。
        """
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999,
                                                           self._global_step)  # adding the iteration prevents from averaging across non-existing iterations
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_everages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(self.tst, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(self.tst, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_everages

    def cnn_inference(self, X_inputs, n_step):
        """TextCNN 模型。
        Args:
            X_inputs: tensor.shape=(batch_size, n_step)
        Returns:
            title_outputs: tensor.shape=(batch_size, self.n_filter_total)
        """
        inputs = tf.nn.embedding_lookup(self.embedding, X_inputs)
        inputs = tf.expand_dims(inputs, -1)
        pooled_outputs = list()
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.n_filter]
                W_filter = self.weight_variable(shape=filter_shape, name='W_filter')
                beta = self.bias_variable(shape=[self.n_filter], name='beta_filter')

                conv = tf.nn.conv2d(inputs, W_filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                conv_bn, update_ema = self.batchnorm(conv, beta, convolutional=True)  # 在激活层前面加 BN
                # Apply nonlinearity, batch norm scaling is not useful with relus
                # batch norm offsets are used instead of biases,使用 BN 层的 offset，不要 biases
                h = tf.nn.relu(conv_bn, name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, n_step - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)
                self.update_emas.append(update_ema)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, self.n_filter_total])
        return h_pool_flat  # shape = [batch_size, self.n_filter_total]


