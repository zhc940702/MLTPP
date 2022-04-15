# MLTPP
# MLTPP
A capsule network-based peptide therapeutic property predictor named MLTPP is able to predict the therapeutic properties of peptides and gives a 5-dimensional 0-1 vector as the prediction result.



# Requirements
* python == 3.6
* pytorch == 1.1
* Numpy == 1.18.3
* scikit-learn == 0.21.3


# Files:

1.data

seqfeature.pckl: The peptide encoding sequence feature vector.

profeature.pckl: The peptide physicochemical property encoding feature vector.

IUfeature.pckl: The peptide disorder score vector.

ss3feature.pckl: The peptide secondary structure feature encoding feature vector.

glbfeature.pckl: The global feature encoding vector for peptides.

kmerfeature.pckl: The peptide sequence feature vector extracted based on kmer.

tngfeature.pckl: The peptide tng feature encoding vector.

label.pckl: Each peptide has a 5-dimensional 0-1 vector representing the 5 therapeutic property labels of the peptide, namely antimicrobial peptides (AMPs), anticancer peptides (ACPs), antidiabetic peptides (ADPs), antihypertensives Peptides (AHPs), anti-inflammatory peptides (AIPs), where 1 means the peptide has this therapeutic property and 0 means it does not.
If you want to view the value stored in the file, you can run the following command:

```bash
import pickle
import numpy as np
gii = open(‘data1’ + '/' + ' seqfeature.pckl', 'rb')
drug_feature_one = pickle.load(gii)
```


2.Code

capsnet.py: The network framework of MLTPP written by pytorch takes the feature vectors and labels of peptides in the data as input, and outputs a 5-dimensional 0-1 vector as the prediction result.

single_label.py: computing evaluation metric. This function can evaluate the prediction performance of a multi-label classifier. Evaluation indicators are: Precision, Coverage, Absolutely true rate, Absolutely false rate and Accuracy.

cross_validation.py: This function can test the predictive performance of our model under ten-fold cross-validation.


# Train and test folds
python cross_validation.py --rawdata_dir /Your path --model_dir /Your path --num_epochs Your number --batch_size Your number

rawdata_dir: All input data should be placed in the folder of this path. (The data folder we uploaded contains all the required data.)

model_dir: Define the path to save the model.

num_epochs: Define the maximum number of epochs.

batch_size: Define the number of batch size for training and testing.

All files of Data and Code should be stored in the same folder to run the model.

Example:

```bash
python cross_validation.py --rawdata_dir /data --model_dir /save_model --num_epochs 50 --batch_size 128
```

# Contact 
If you have any questions or suggestions with the code, please let us know. Contact Haochen Zhao at zhaohaochen@csu.edu.cn
