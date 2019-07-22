# **ncnn_106landmarks**
# **1. add landmark**
## **(1). the original model comes from repositories:**
### https://github.com/zuoqing1988/ZQCNN  by zuoqing
## **(2). the process of converting mxnet model to ncnn formate is:**
### https://github.com/zuoqing1988/ZQCNN/issues/50

## **(3). new landmark model: landmark_big.param && landmark_big.bin has also uploaded**
### you can change the model name in the test code to test the model:
### change "landmark.param", "landmark.bin" to "landmark_big.param" && "landmark_big.bin"

## **(4). add input size: 96x96 model:**
### you need to change the input size 48 x 48 to 96 x 96: 
#### ncnn::Mat ncnn_in = ncnn::Mat::from_pixels_resize(img_src.data,
####        ncnn::Mat::PIXEL_BGR, img_src.cols, img_src.rows, 48, 48);
#### **To:**
#### ncnn::Mat ncnn_in = ncnn::Mat::from_pixels_resize(img_src.data,
####        ncnn::Mat::PIXEL_BGR, img_src.cols, img_src.rows, 96, 96);

## **(5).add input size: 112x112 model:**
#### the most effect model.

# **2. add retinaface detection**
## the code refer to the repositories:
### https://github.com/Charrin/RetinaFace-Cpp by Charrin
## model comes from:
### https://github.com/Charrin/RetinaFace-Cpp/tree/master/Demo/ncnn/models
### **result:**
### ![图片](https://github.com/MirrorYuChen/ncnn_106landmarks/blob/master/images/result.jpg)

# **3.add jetson nano project based on vulkan**
## **(1). build vulkan ncnn:**
### https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-vulkan
## **(2). build the project:**
### >mkdir build && cd build && cmake .. && make -j3
### >./main
# **TODO:**
## (1). Optimaize the code
## (2). add face detection: MTCNN
## (3). add face recognize: Mobilefacenet
