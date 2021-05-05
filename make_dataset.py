from aaindex_1 import peptide_into_property
from read_csv import MHC_Read_data,Processing_peptides,mhc_data
import config
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np

    
class Dataset():
    def __init__(self):
        
        self.mhc = config.mhc_type
        self.train_peptides,self.train_labels,self.positive_length,self.val_peptides,self.val_labels,self.test_peptides,self.test_labels=    mhc_data(self.mhc)
           
    
    def get_dataset(self,peptides,labels,positive_length):
        
        datasets = peptide_into_property(peptides)  #property_length=553
        c_datasets = datasets.astype('float32')
        c_datasets = c_datasets[:2*positive_length]
        labels = labels[:2*positive_length]
        labels = np.array(labels)
        shuffle_index = np.random.permutation(len(c_datasets))
        c_datasets = c_datasets[shuffle_index]
        labels = labels[shuffle_index]
    
        return c_datasets[:2*positive_length],labels[:2*positive_length],positive_length
    

    
        
#dataset,label,positive_length=get_dataset(all_peptide_seqs,label,positive_length)


    def make_dataset(self,dataset,label):
    
        BUFFER_SIZE = len(dataset)
        BATCH_SIZE = config.BATCH_SIZE
        datasets = tf.data.Dataset.from_tensor_slices((dataset,label)).shuffle(BUFFER_SIZE)
        datasets = datasets.cache().batch(BATCH_SIZE, drop_remainder=True)
        print(1)
        
        return datasets

        #test_dataset=make_dataset(test_data,test_label)
    def train_val_test_dataset(self):

        train_data,train_label,positive_length = self.get_dataset(self.train_peptides,self.train_labels,self.positive_length)
        val_data =  peptide_into_property(self.val_peptides)
        val_data = val_data.astype('float32')
        val_label = self.val_labels
        test_data = peptide_into_property(self.test_peptides)
        test_data = test_data.astype('float32')
        test_label = self.test_labels
        
        #标准化
        train_data = train_data.reshape((len(train_data),config.peptide_length*553))
        val_data = val_data.reshape((len(val_data),config.peptide_length*553))
        test_data = test_data.reshape((len(test_data),config.peptide_length*553))
        
        standardscaler = StandardScaler()
        train_data = standardscaler.fit_transform(train_data)
        val_data = standardscaler.transform(val_data)
        test_data = standardscaler.transform(test_data)
        
        train_data = train_data.reshape((len(train_data),config.peptide_length,553))
        val_data = val_data.reshape((len(val_data),config.peptide_length,553))
        test_data = test_data.reshape((len(test_data),config.peptide_length,553))
        #数据集
        train_dataset = self.make_dataset(train_data,train_label)
        val_dataset = self.make_dataset(val_data,val_label)
        test_dataset = self.make_dataset(test_data,test_label)
        
        return train_dataset,val_dataset,test_dataset
    
class_dataset=Dataset()
train_dataset,val_dataset,test_dataset=class_dataset.train_val_test_dataset()
train_peptides,train_labels,positive_length,val_peptides,val_labels,test_peptides,test_labels=    mhc_data(config.mhc_type)