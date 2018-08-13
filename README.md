# semantic-segmentation   

### 1简介  
构建FCN-8s网络模型，使用PASCAL VOC2012中语义分割数据集中的数据，完成网络训练和结果预测    

### 2数据集    
VOC数据集下载网址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/    

VOC数据集目录结构如下：  

├── local  
│   ├── VOC2006  
│   └── VOC2007  
├── results  
│   ├── VOC2006  
│   │   └── Main  
│   └── VOC2007  
│       ├── Layout  
│       ├── Main  
│       └── Segmentation  
├── VOC2007  
│   ├── Annotations  
│   ├── ImageSets  
│   │   ├── Layout  
│   │   ├── Main  
│   │   └── Segmentation  
│   ├── JPEGImages  
│   ├── SegmentationClass  
│   └── SegmentationObject  
├── VOC2012  
│   ├── Annotations  
│   ├── ImageSets  
│   │   ├── Action  
│   │   ├── Layout  
│   │   ├── Main  
│   │   └── Segmentation  
│   ├── JPEGImages  
│   ├── SegmentationClass  
│   └── SegmentationObject  
└── VOCcode  
  
数据集位于VOC2012/ImageSets/Segmentation中，分为train.txt 1464张图片和val.txt1449张图片。其中语义分割标签位于VOC2012/SegmentationClass,不是数据集中所有的图片都有语义分类的标签。   
语义分割标签用颜色来标志不同的物体，该数据集中共有20种不同的物体分类，以1～20的数字编号，加上编号为0的背景分类，该数据集中共有21种分类。编号与颜色的对应关系如下：   
```python
# class
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']

# RGB color for each class
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]
```

### 3代码实现   
1）使用convert_fcn_dataset.py读取原始图像（jpg）和标签（png），按照格式填充feature_dict，将原始数据转化为TFRecord格式  
2）下载并使用vgg16预训练模型   
3）将vgg网络的全连接层改为全卷积（same padding），并根据论文进行对feature map进行两次2倍上采样和element相加过程，最终得到的feature map再进行一次8倍上采样处理，得到与原图大小相同的feature map，对比groundtruth进行网络训练    
4）增加CRF对验证集图片进行inference     

### 4结果验证   
1）test image  
![](val_1400_img.jpg 'test image')    
2）groundtruth image  
![](val_1400_annotation.jpg 'groundtruth image ')    
3）prediction image   
![](val_1400_prediction.jpg 'prediction image ')    
4）CRFed prediction image    
![](val_1400_prediction_crfed.jpg 'CRFed prediction image')    

**参考论文**   
[《Fully Convolutional Networks for Semantic Segmentation》](https://arxiv.org/abs/1411.4038)
