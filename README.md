# **ncnn_example:**
# **I promise i'll update this project forever!**
# 20201024:
## 1.add retinaface keypoints
## 2.add visual studio scripts
# 20200806:
## 1.add insightface 106 [landmarks](!https://github.com/deepinsight/insightface/tree/master/alignment/coordinateReg)
## 2.optimize the CMakeLists.txt
# 2020.03.10:
## 1.add mask detect code
## 2.optimize the name of retinaface 
## 3.update ncnn lib version to newest:  [20200226](!https://github.com/Tencent/ncnn/tree/20200226)
# 2020.02.23: 
## 1. add vulkan option
## 2. fix the network name of mobilenetssd 
# 2020.02.22: 
## 1. split the model files from the project
## 2. add classifier && add object detecter 
# **How to use?**
## 1. download the models from baiduyun: [baidu](https://pan.baidu.com/s/15wg10Ry6-5a2wa5MIJbNww)(code: w48b) and [google](https://drive.google.com/drive/folders/1kZ96ehstrlDMIH5HF8NHYYkD40yttxmd?usp=sharing)
## 2. put models to directory: ncnn_example/data/models 
## 3. compile the project:
```
>> cd ncnn_example && mkdir build && cd build && make -j3 
```
## 4. run the project:
```
>> cd src && ./face && ./object && ./classifier
```
## 5. result
### face result:
### ![图片](https://github.com/MirrorYuChen/ncnn_example/blob/master/data/images/result.jpg)
### mask result:
### ![图片](https://github.com/MirrorYuChen/ncnn_example/blob/master/data/images/mask_result.jpg)
### object result:
### ![图片](https://github.com/MirrorYuChen/ncnn_example/blob/master/data/images/object_result.jpg)
### classifier result:
### ![图片](https://github.com/MirrorYuChen/ncnn_example/blob/master/data/images/classify_result.jpg)

# 6. TODO:
- [x] optimize the speed by using block strategy
- [x] add yolo
- [x] add pose
- [x] refactor the project

# 7. references
## https://github.com/Tencent/ncnn
## https://github.com/zuoqing1988/ZQCNN
## https://github.com/Charrin/RetinaFace-Cpp
## https://github.com/deepinsight/insightface
## https://github.com/Star-Clouds/centerface
## https://github.com/seetafaceengine/SeetaFace2
