import pandas as pd
import config
import numpy as np

file = 'mhc.csv'
total_data=pd.read_csv(file)
mhci_data=total_data.loc[lambda df:
                              df.MHCType=='MHC-I']
human_mhci_data=mhci_data.loc[lambda df:
                                    df.DatasetType=='Human']


human_mhci_data_nan = human_mhci_data.loc[lambda df:
                                 df.MHC_Restriction!='']
    

NONE_VIN = (human_mhci_data["MHC_Restriction"].isnull()) | (human_mhci_data["MHC_Restriction"].apply(lambda x: str(x).isspace()))
df_not_null = human_mhci_data[~NONE_VIN]
hlai_names = df_not_null["MHC_Restriction"]
#hlai_names = hlai_names.dropna()
hlai_names_set = set(hlai_names)
hla_seqs = []
hla_names_seqs_dict = {}
with open("hla.txt","r") as f:
    for sentence in f.read().splitlines():
        if "HLA" in sentence:
            sentence = sentence.strip(" \t")
            hla_seqs.append(sentence)
            hla_name = sentence.split(" ")[0]
            hla_seq = sentence.split(" ")[-1]
            hla_names_seqs_dict[hla_name[1:-1]] = hla_seq[1:-2]
 
hla_names = []
j=0
for name in hlai_names:
    j+=1
    for i in name.split("|"):
                hla_names.append(i)
hla_names = set(hla_names)

same_number = 0
for i in hla_names_seqs_dict.keys():
    if i in hla_names:
        same_number +=1
        
class MHC_Read_data():
    
    def __init__(self,file):
        
        self.total_data=pd.read_csv(file)
        self.mhci_data=self.total_data.loc[lambda df:
                              df.MHCType=='MHC-I']
        self.human_mhci_data=self.mhci_data.loc[lambda df:
                                    df.DatasetType=='Human']
        self.mhcii_data=self.total_data.loc[lambda df:
                              df.MHCType=='MHC-II']
        self.human_mhcii_data=self.mhcii_data.loc[lambda df:
                                    df.DatasetType=='Human']
    
    def human_mhci_subclass(self,HLA_Subtype):
        
        human_mhci_sub_positive=self.human_mhci_data.loc[lambda df:
                                        df.Immunogenicity=='Positive']
        human_mhci_sub_positive=human_mhci_sub_positive.fillna('nan', axis = 0)
        human_mhci_sub_positive=human_mhci_sub_positive[human_mhci_sub_positive
                            ['MHC_Restriction'].str.contains(HLA_Subtype)]
        human_mhci_sub_positive=human_mhci_sub_positive.loc[
            lambda df:df.Immunogenicity_Evidence!='nan']
        human_mhci_sub_negative=self.human_mhci_data.loc[
            lambda df:df.Immunogenicity=='Negative']
    
        return human_mhci_sub_positive,human_mhci_sub_negative
    
    def human_mhci(self):
        
        human_mhci_positive=self.human_mhci_data.loc[lambda df:
                                        df.Immunogenicity=='Positive']

        human_mhci_negative=self.human_mhci_data.loc[lambda df:
                                        df.Immunogenicity=='Negative']
        
        return human_mhci_positive,human_mhci_negative
        

    
    def get_sequences(self,positive,negative):
         
         positive_peptide_seqs = list(positive["Peptide"])
         positive_hla_seqs_name = list(positive["MHC_Restriction"])
         negative_peptide_seqs = list(negative["Peptide"])
         negative_hla_seqs_name = list(negative["MHC_Restriction"])
             
        
         return positive_peptide_seqs,positive_hla_seqs_name,negative_peptide_seqs,negative_hla_seqs_name
        
read_data=MHC_Read_data('mhc.csv')
human_mhci_positive,human_mhci_negative=  read_data.human_mhci()     
human_mhci_positive_seqs,pos_hla_seqs_names,human_mhci_negative_seqs,neg_hla_seqs_names=  read_data.get_sequences(
    human_mhci_positive,human_mhci_negative)   

pos_hla_seqs = []
for hla_name in pos_hla_seqs_names:
    if hla_name in hla_names_seqs_dict.keys():
        pos_hla_seqs.append(hla_names_seqs_dict[hla_name])
    else:
        pos_hla_seqs.append(hla_name)
hla_names_seqs_dict