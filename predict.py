def test():
    with open(r'123.txt') as f:
        total = f.readlines()
        peptides = []
        labels = []
        j=0
        for line in total:
            peptide = line[:-3]
            label = line[-3]
            #label = int(label)
            #j+=1
            #print(j)
            if len(peptide)==9:
                #if label=='1':
                    j+=1
                    peptides.append(peptide)
                    labels.append(label)

            #print(line)
            i_labels=[]
            for label in labels:
                label=int(label)
                i_labels.append(label)
    return peptides,i_labels

pep,labels=test()

from aaindex_1 import peptide_into_property

test_datas = peptide_into_property(pep,0,max_min=True)