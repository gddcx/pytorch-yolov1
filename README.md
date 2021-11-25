# yolov1
Implemented YOLOv1 based on pytorch.  
There are two version with different backbone.   
Folder "resnet" means using the resnet50 as the backbone and folder "darknet" means using the backbone presented in the paper.
All checkpoint files can be downloaded on https://drive.google.com/drive/folders/1Aru9-JREqFBYoK3R52BB-K-tFrM25Woi?usp=sharing

## darknet
To compare the performance with paper, I have implemented the model presented in paper.  
### pretrain
I have pretrained the first 20 convolutional layers on ILSVRC2012-1k training set and evaluate on validation set.  
The top-5 accuracy is **87.78%**  
Change the data path and use the follow command to run the codes.
```bash
./decompress.sh # decompress the dataset
python pretrain.py 20211125  # the string 20211125 is the index of this train.
python eval.py # evaluate the model
```
### detection
The detection model obtain **43.8 mAP** on VOC2007 val set. 
```bash
python train.py # train the detection model with sacred
python eval.py # evaluate the model
```
## resnet
The detection model obtain **51.2 mAP** on VOC2007 val set. 
```bash
python train.py # train the detection model with sacred
python eval.py # evaluate the model
```

## To be optimized
The performance is worse than that presented in paper and some other implement. I am still trying to find the reason...
