# **ncnn_106landmarks:**
# **I promise i'll update this project forever!**
# **1. add landmark**
## **(1). the original model comes from repositories:**
### https://github.com/zuoqing1988/ZQCNN  by zuoqing
## **(2). the process of converting mxnet model to ncnn formate is:**
### https://github.com/zuoqing1988/ZQCNN/issues/50

## **(3). new landmark model: landmark_big.param && landmark_big.bin has also uploaded, input size: 48x48**
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

# **4.add mobilefacenet**
# **5.use openmp to optimize for loops**
## **test result:**
## **do not use vulkan:**
'''
start face detect.
4 faces detected.
start keypoints extract.
keypoints extract end.
start keypoints extract.
keypoints extract end.
start keypoints extract.
keypoints extract end.
start keypoints extract.
keypoints extract end.
time cost: 137.495ms
'''
## **use vulkan:**
'''
[0 NVIDIA Tegra X1 (nvgpu)]  queueC=0[16]  queueT=0[16]  memU=2  memDL=2  memHV=2
[0 NVIDIA Tegra X1 (nvgpu)]  fp16p=1  fp16s=1  fp16a=0  int8s=1  int8a=0
start face detect.
4 faces detected.
start keypoints extract.
keypoints extract end.
start keypoints extract.
keypoints extract end.
start keypoints extract.
keypoints extract end.
start keypoints extract.
keypoints extract end.
time cost: 553.328ms
'''
## **why so strange?**

# **6.add face alignment interface**
## **the interface refer to the project insightface:**
## https://github.com/deepinsight/insightface/tree/master/cpp-align
## **and the issue:**
## https://github.com/deepinsight/insightface/issues/333

# **7.refactor the project**
## **reduce the coupling of modules**

# **8.add mtcnn**
## **You only need to change line 14 of face_engine.cpp to change face detector.**
## **For Example:**
## **detector_(new Mtcnn()) To detector_(new RetinaFace())**

# **9.optimize the structure of the project**
# **10.add centerface:**
## https://github.com/Star-Clouds/centerface
## **please update ncnn to newest version, speed will impove a lot.**

# **10.add centerface detection**

# **11.add facedatabase and face tracking:**
## **refer to the project seetaface2:**
## https://github.com/seetafaceengine/SeetaFace2

# **TODO:**
- [x] refactor the project
- [x] optimize the speed