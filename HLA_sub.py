def get_subtype_name_seq(file):
    subtype_name_seq = {}
    with open(file,"r") as f:
        for line in f.readlines():
            if line[0]==">":
                subtype_name = "HLA-" + line.split(" ")[1]
                subtype_name = subtype_name[:11]
                subtype_seq = ""
            else:
                subtype_seq += line.split("\n")[0]
        
            subtype_name_seq[subtype_name] = subtype_seq
    return subtype_name_seq            
            
file = "hla_prot.fasta"
subtype_name_seq = get_subtype_name_seq(file)