# coding=UTF-8
import heapq
import random
import math
import tensorflow as tf
import numpy as np
import os
import io
import time
import datetime
import network
from tensorflow.contrib import learn
import tensorflow.contrib.slim as slim  

# Parameters
# ==================================================
import sys







batchSize=128
testBatchSize=3000
topic_length=5
topic_num=13258
question_length=30
questionDescription_length=150
#data file
#下面是embedding file
embeddingFile="/home/share/liyongqi/data2/zhihuFenzhi2/zhihuFenzhi2Embedding.npy"
#下面是question——topic data
trainFile="/home/share/liyongqi/data2/zhihuFenzhi2/zhihuFenzhi2QuestionAndTopicIdAndQuestionDescriptionVocabularyTrain.txt"

testFile="/home/share/liyongqi/data2/zhihuFenzhi2/zhihuFenzhi2QuestionAndTopicIdAndQuestionDescriptionVocabularyTest.txt"

#下面是topic的文本
dataTopic="/home/share/liyongqi/data2/zhihuFenzhi2/zhihuFenzhi2TopicIdAndTopicNameVocabulary.txt"

dataDAG = "/home/share/liyongqi/data2/zhihuFenzhi2/zhihuFenzhi2Dag.txt"
dataMask="/home/share/liyongqi/data2/zhihuFenzhi2/zhihuFenzhi2DagMask.txt"

settings = network.Settings()
# Data Preparation
# ==================================================

# Load data
print("Loading data...liyongqi")

w=np.load(embeddingFile)
print(w.shape)

#下面的方法是将一个语句补齐到定长
def vocabulary(question,question_length):
    q=[]
    for s in range(len(question)):
        arry=question[s].rstrip().strip('\n').split(' ')
        if(len(arry)>=question_length):
            for num in range(question_length):
                q.append(arry[num])
        if(len(arry)<question_length):
            for num in range(len(arry)):
                q.append(arry[num])
            for num in range(question_length-len(arry)):
                q.append(0)
    return q



data3 = list(io.open(trainFile, "r",encoding='utf-8').readlines())




Alltest=list(io.open(testFile, "r",encoding='utf-8').readlines())

dataTopic=list(io.open(dataTopic, "r",encoding='utf-8').readlines())

dataTopic=vocabulary(dataTopic,topic_length)

dataTopic=np.array(dataTopic)
dataTopic=dataTopic.reshape((topic_num,topic_length))


DAG=[]
with io.open(dataDAG, 'r',encoding="utf8") as f:
    lines=f.readlines()
    for s in lines:
        s=s.strip().split('\t')
        for i in s:
            DAG.append(int(i))
DAG=np.array(DAG)
DAG=DAG.reshape(topic_num,6)

Mask=[]
with io.open(dataMask, 'r',encoding="utf8") as f:
    lines=f.readlines()
    for s in lines:
    	s=s.strip()
        Mask.append(0)
        for i in range(int(s)):
            Mask.append(0)
        for i in range((5-int(s))):
            Mask.append(-100000)

Mask=np.array(Mask)
Mask=Mask.reshape(topic_num,6)

def getAllTest(test,testBatchSize,num):
        out=[]
        out1=[]
    

        qtl=test[num*testBatchSize*3:(num+1)*testBatchSize*3]
        qtl=np.array(qtl)
        qtl=qtl.reshape([-1,3])
        question=qtl[:,0]
        
        labels=qtl[:,1]
        
        questionDescription=qtl[:,2]

        for i in range(len(labels)):
            array=labels[i].strip('\n').split(" ")
            a=np.zeros((topic_num))
            out2=[]
            for j in range(len(array)):
                if(array[j]==''):
                    continue
               
                a[int(array[j])]=1
                out2.append(int(array[j]))
            out.append(a.tolist())
            out1.append(out2)
        labels=out
        question=vocabulary(question.tolist(),question_length)

        questionDescription=vocabulary(questionDescription.tolist(),questionDescription_length)
        

        return labels,question,questionDescription,out1

def score_eval(predict_label_and_marked_label_list):
    """
    :param predict_label_and_marked_label_list: 一个元组列表。例如
    [ ([1, 2, 3, 4, 5], [4, 5, 6, 7]),
      ([3, 2, 1, 4, 7], [5, 7, 3])
     ]
    需要注意这里 predict_label 是去重复的，例如 [1,2,3,2,4,1,6]，去重后变成[1,2,3,4,6]
    
    marked_label_list 本身没有顺序性，但提交结果有，例如上例的命中情况分别为
    [0，0，0，1，1]   (4，5命中)
    [1，0，0，0，1]   (3，7命中)
    """
    right_label_num = 0  #总命中标签数量
    right_label_at_pos_num = [0, 0, 0, 0, 0]  #在各个位置上总命中数量
    sample_num = 0    #总问题数量
    all_marked_label_num = 0    #总标签数量
    for predict_labels, marked_labels in predict_label_and_marked_label_list:
        sample_num += 1
        marked_label_set = set(marked_labels)
        all_marked_label_num += len(marked_label_set)
        for pos, label in zip(range(0, min(len(predict_labels), 5)), predict_labels):
            if label in marked_label_set:     #命中
                right_label_num += 1
                right_label_at_pos_num[pos] += 1

    precision = 0.0
    for pos, right_num in zip(range(0, 5), right_label_at_pos_num):
        precision += ((right_num / float(sample_num))) / math.log(2.0 + pos)  # 下标0-4 映射到 pos1-5 + 1，所以最终+2
    recall = float(right_label_num) / all_marked_label_num

    return precision, recall, (precision * recall) / (precision + recall )



def getBatch(data3,batchSize,num):
        qtl=data3[num*batchSize*3:(num+1)*batchSize*3]
        qtl=np.array(qtl)
        qtl=qtl.reshape([-1,3])
        question=qtl[:,0]
        labels=qtl[:,1]
        questionDescription=qtl[:,2]
        out=[]
        for i in range(len(labels)):
            array=labels[i].rstrip().strip('\n').split(" ")
            a=np.zeros((topic_num))
            for j in range(len(array)):
                a[int(array[j])]=1
            out.append(a.tolist())
        labels=out
        question=vocabulary(question.tolist(),question_length)
        questionDescription=vocabulary(questionDescription.tolist(),questionDescription_length)
        return question,labels,questionDescription


# Placeholders for input, output and dropout
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
sess = tf.InteractiveSession(config=config)




model = network.TextCNN(w, settings)



#learning_rate = tf.train.exponential_decay(lr, model.global_step, decay_step, decay_rate, staircase=True)






sess.run(tf.global_variables_initializer())


saver = tf.train.Saver()

saver.restore(sess,  "./train3/model.ckpt")

with open("train3loss1", "a") as f:

                    
                
                    predict_labels_list = list()  # 所有的预测结果
                    marked_labels_list = list()

                    li=len(Alltest)/testBatchSize/3
                    li=int(li)
                    print(li)
                    for li in range(li):
                    	print(li)
                        dagdag=np.array(DAG[0%topic_num])
                        dagdag=dagdag.reshape([-1,6])
                        maskmask=np.array(Mask[0%topic_num])
                        maskmask=maskmask.reshape([-1,6])

                        labelTest,questionTest,questionDescriptionTest,label=getAllTest(Alltest,testBatchSize,li)

                        marked_labels_list.extend(label)
                        labelTest=np.array(labelTest)
                        questionTest=np.array(questionTest)
                        questionDescriptionTest=np.array(questionDescriptionTest)

                        labelTest=labelTest.reshape([testBatchSize, topic_num])

                        questionTest=questionTest.reshape([testBatchSize,  question_length])
                        questionDescriptionTest=questionDescriptionTest.reshape([testBatchSize,questionDescription_length])

                        question_outputs,fc_bn_relu= sess.run([model._y_pred,model.fc_bn_relu],feed_dict = {model.X1_inputs: questionTest, model.X2_inputs: questionDescriptionTest,model.X3_inputs: dataTopic, model.y_inputs: labelTest,model.dag:dagdag,model.mask:maskmask,model.batch_size: testBatchSize, model.tst: True, model.keep_prob: 1.0})

                        predict_labels = map(lambda label: label.argsort()[-1:-6:-1], question_outputs)  # 取最大的5个下标
                        predict_labels_list.extend(predict_labels)
                    
                    predict_label_and_marked_label_list = zip(predict_labels_list, marked_labels_list)
                    precision, recall, f1 = score_eval(predict_label_and_marked_label_list)
                    print('recall',recall)
                    print('precision',precision)
                    print('f1',f1)


#       
sess.close()



