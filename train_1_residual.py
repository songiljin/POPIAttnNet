
import numpy as np 
import os
import time
import config
import tensorflow as tf
from make_dataset import Dataset
from model_residual import model_1
import config
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, confusion_matrix
import os
from predict import test
from aaindex_1 import peptide_into_property

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
class_dataset=Dataset()
train_dataset,val_dataset,test_dataset=class_dataset.train_val_test_dataset()
os.environ['TF_CPP_MIN_LOG'] = '2'

def accuary(prediction,label):
    i=0
    j=0
    prediction=tf.argmax(prediction,axis=-1)
    for p in prediction:
        if p.numpy()==label[i].numpy():
            j+=1
        i+=1
    
    return j
def main():
    model = model_1()
    model.build(input_shape=(None,config.peptide_length,553))
    optimizer = tf.keras.optimizers.Adam(lr=1e-4)
    model.summary()

    for epoch in range(config.EPOCHS):
        acc_score =0
        for step,(x,y) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x)
                y_onehot = tf.one_hot(y,depth=2)
                acc = accuracy_score(y,tf.argmax(logits,axis=-1))
                acc_score+=acc
                #loss=tf.keras.losses.SparseCategoricalCrossentropy(logits,y)
                loss = tf.losses.categorical_crossentropy(y_onehot,logits,from_logits=True)
                loss = tf.reduce_mean(loss)

            gradient = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(gradient,model.trainable_variables))

            if step % 20 == 0:
                print(epoch,step,'loss:',float(loss))
             

        total_num = 0
        total_correct = 0
        for x,y in val_dataset:
            logits = model(x)
            accuary_score = accuary(logits,y) 
            prob = tf.nn.softmax(logits,axis=1)
            pred = tf.argmax(prob,axis=1)
            pred = tf.cast(pred,dtype=tf.int32)

            correct = tf.cast(tf.equal(pred,y),dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += correct
            acc = total_correct/total_num
            
        print("acc:",accuary_score/64)
        print("acc:",acc )
        
        total_num = 0
        total_correct = 0
        for x,y in test_dataset:
            logits = model(x)
            accuary_score = accuary(logits,y) 
            prob = tf.nn.softmax(logits,axis=1)
            pred = tf.argmax(prob,axis=1)
            pred = tf.cast(pred,dtype=tf.int32)

            correct = tf.cast(tf.equal(pred,y),dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += correct
            acc = total_correct/total_num

        print("acc:",acc,"acc:",accuary_score/64 )
        #print(pred)
        #print(y)
        
        #from aaindex_1 import peptide_into_property
    
        #pep,labels=test()
        #test_datas = peptide_into_property(pep,0,max_min=0)
        #predictions = model.predict(test_datas,batch_size=64)
        #predictions = tf.nn.softmax(predictions,axis=1)
        #prediction = np.argmax(predictions,axis=1)
        ##print(predictions)
        #print(prediction)
        #print(labels)
        #i=0
        #j=0
        #for p in prediction:
        #    if p==labels[i]:
          #      j+=1
         #   i+=1
        #print("acc:",j)
            


if __name__ == '__main__':
    main()