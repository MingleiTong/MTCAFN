# MTCAFN
The scripts are uesd for Keras and Tensorflow(backend). 

Dataset:  
We used ShanghaiTech Dataset which can be downloaded at  https://github.com/svishwa/crowdcount-mcnn  
Crowd Counting Dataset can be downloaded at http://crcv.ucf.edu/data/ucf-cc-50/  

Hardware:  
GTX TITAN XPascal  
16G memory  

Software:  
Tensorflow  
Keras  
numpy  
opencv  

Content:  
MTnet.py  
The script is to build a multi-task network.  
MTtrain.py  
The script is to train the network and save the model.  
MTtest.py  
The script is to test the saved model.  

Following are the results on Shanghai Tech A and B dataset:
    
     |     |  MAE    |   MSE    |
     ----------------------------
     | A   |  88.1   |   137.2  |
     ----------------------------
     | B   |  18.8   |   31.3   |
