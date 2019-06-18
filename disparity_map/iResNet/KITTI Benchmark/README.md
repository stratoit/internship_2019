# Model for KITTI Dataset

iRestNet Model for KITTI Benchmark dataset

## About

This folder contains implemnation of [iRestNet Model](https://arxiv.org/abs/1712.01039) for [KITTI benchmark data set](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php). The model is modified and different from my inital model because the same model won't work for all input dimensions as it has strided convolutions of various size.

## Package Requirements 

**apt requirements**

1. python3
2. pip3
4. gir1.2-clutter-1.0 
5. gir1.2-clutter-gst-3.0 
6. gir1.2-gtkclutter-1.0 
```
sudo apt install python3 python3-pip gir1.2-clutter-1.0 gir1.2-clutter-gst-3.0 gir1.2-gtkclutter-1.0
```

**pip requirements**

1. matplotlib
2. keras
3. tensorflow
4. PIL
5. numpy
6. os
7. pydot

Install tensorflow from [official website](https://www.tensorflow.org/install)

Then proceed with this command
```
pip3 install matplotlib numpy pydot keras PIL 
```

## Dataset

Visit [KITTI benchmark data set](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php) website and download the first file and place its contents in the same folder as the code.

So the folder has three items 
1. *File* : iResNet_v2_KITTI.ipynb
2. *Folder* : testing 
3. *Folder* : training

## Usage

Execute the iResNet_v2_KITTI.ipynb file sequentially in any any ipython console and everything would work. 
> **Note** : It might not work for other input sizes so keep that in mind before editing the code.

## Specifications

1. **Input Size** : *375 x 1242 x 3*
2. **Output Size** : *375 x 1242 x 3*
3. **Trainable parameters** : 19,720,780
4. **Non trainable params** : 0
5. **Varied input resizing method** : ANTIALIASING
6. **Trainable layers** : 79(*6 layers trained with shared weights for left and right images*)
7. **Dropout** : No 
8. **Batch Normalization** : No

> **Note**:Size mentioned in format *HxWxC*

