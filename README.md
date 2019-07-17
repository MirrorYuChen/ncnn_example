# **ncnn_106landmarks**
## **1. the original model comes from repositories:**
### https://github.com/zuoqing1988/ZQCNN  by zuoqing
## **2. the process of converting mxnet model to ncnn formate is:**
### https://github.com/zuoqing1988/ZQCNN/issues/50

## **3. new landmark model: landmark_big.param && landmark_big.bin has also uploaded**
### you can change the model name in the test code to test the model:
### change "landmark.param", "landmark.bin" to "landmark_big.param" && "landmark_big.bin"

## **4. add input size: 96x96 model:**
### you need to change the input size 48 x 48 to 96 x 96: 
#### ncnn::Mat ncnn_in = ncnn::Mat::from_pixels_resize(img_src.data,
####        ncnn::Mat::PIXEL_BGR, img_src.cols, img_src.rows, 48, 48);
#### **To:**
#### ncnn::Mat ncnn_in = ncnn::Mat::from_pixels_resize(img_src.data,
####        ncnn::Mat::PIXEL_BGR, img_src.cols, img_src.rows, 96, 96);
