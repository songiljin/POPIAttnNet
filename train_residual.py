
import numpy as np 
import os
import time
import config
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from make_dataset import Dataset
from model import model_1
import config
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, confusion_matrix
import os
from aaindex_1 import peptide_into_property

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#test_pep_seqs,test_labels=class_dataset.get_test_pep()
#test_datasets = peptide_into_property(test_pep_seqs,0,max_min=True)
#test_labels = np.array(test_labels)
os.environ['TF_CPP_MIN_LOG'] = '2'

def test():
    with open(r'123.txt') as f:
        total = f.readlines()
        peptides = []
        labels = []
        for line in total:
            peptide = line[:-2]
            label = line[-2]
            if len(peptide)==9:
                peptides.append(peptide)
                labels.append(label)
            print(line)
    return peptides,labels

def make_dataset(dataset,label):
    
    BUFFER_SIZE = len(dataset)
    BATCH_SIZE = config.BATCH_SIZE
    datasets = tf.data.Dataset.from_tensor_slices((dataset,label)).shuffle(BUFFER_SIZE)
    datasets = datasets.cache().batch(BATCH_SIZE, drop_remainder=True)
    
    return datasets

#test_dataset = make_dataset(test_datasets,test_labels)

def main():
    
    class_dataset=Dataset()
    train_dataset,val_dataset,test_dataset=class_dataset.train_val_test_dataset()
    model = model_1()
    model.build(input_shape=(None,config.peptide_length,553))
    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    model.summary()

    model.compile(loss='SparseCategoricalCrossentropy', optimizer=Adam(), metrics=['accuracy'])
    model.fit(train_dataset, batch_size=64, epochs=10, verbose=1, validation_data=val_dataset)
    
    test_loss,test_acc= model.evaluate(test_dataset)
    print("test_loss:",test_loss,"test_acc",test_acc)
    
    #from aaindex_1 import peptide_into_property
    
    #pep,labels=test()
    #test_datas = peptide_into_property(pep,0,max_min=True)
    #predictions = model.predict(test_datas)
    #prediction = np.argmax(predictions,axis=1)
    #print(predictions)
    #print(prediction)
    #print(labels)

if __name__ == '__main__':
    main()