# MTCAFN
The scripts are uesd for Keras and Tensorflow(backend). 

Data Setup:
1. Download ShanghaiTech Dataset from  
Dropbox: https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0  
or Baidu Disk: http://pan.baidu.com/s/1nuAYslz  

2. Download UCF_CC_50 Dataset from  
http://crcv.ucf.edu/projects/crowdCounting/index.php  

3. WorldExpo'10 obtained from  
http://www.ee.cuhk.edu.hk/~xgwang/expo.html  
The dataset is available. Shanghai Jiao Tong University has the copyright of the dataset.  
So we contacted Prof. Xie (xierong@sjtu.edu.cn) to get the download link.  

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
