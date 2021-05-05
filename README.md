# POPIAttnNet is a Biological prediction tool
Prediction of immunogenicity of the MHC-binding peptide
POPI is a prediction model for predicting the immunogenicity of HLA- binding peptides, which uses Bi-LSTM and attention mechanism to perform two-class prediction, i.e. immunogenicity and non-immunogenicity.
Original data can be obtained from read_csv.py.You can get the data set of tensorflow from make_dataset.py.The model_residual.py is the final model.the model.py is the part of the model_residual.py.The model can be trained from train_1_residual.py, and the trained model can be obtained.

See doc file for details.
The script of the  POPIAttnNet is Python3.7, and Tensorflow2.3.0 is used to build the model.
