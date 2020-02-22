# **ncnn_example:**
# **I promise i'll update this project forever!**
# 2020.02.22: 
## 1.split the model files from the project
## 2. add classifier && add object detecter 
# **How to use?**
## 1.download the models from baiduyun: [baidu](https://pan.baidu.com/s/15wg10Ry6-5a2wa5MIJbNww)(code: w48b)
## 2.put models to directory: ncnn_example/data/models 
## 3.compile the project:
```
>> cd ncnn_example && mkdir build && cd build && make -j3 
```
## 4.run the project:
```
>> cd src && ./face && ./object && ./classifier
```
## 5.result
### face result:
### ![图片](https://github.com/MirrorYuChen/ncnn_example/blob/master/data/images/result.jpg)
### object result:
### ![图片](https://github.com/MirrorYuChen/ncnn_example/blob/master/data/images/object_result.jpg)
### classifier result:
### ![图片](https://github.com/MirrorYuChen/ncnn_example/blob/master/data/images/classify_result.jpg)

# **TODO:**
- [x] refactor the project
- [x] optimize the speed

# project refer to:
## https://github.com/Tencent/ncnn
## https://github.com/zuoqing1988/ZQCNN
## https://github.com/Charrin/RetinaFace-Cpp
## https://github.com/deepinsight/insightface
## https://github.com/Star-Clouds/centerface
## https://github.com/seetafaceengine/SeetaFace2