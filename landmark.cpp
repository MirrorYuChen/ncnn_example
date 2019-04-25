#include <iostream>
#include "opencv2/opencv.hpp"
#include "ncnn/net.h"

int main(int argc, char* argv[]) {
    cv::Mat img_src = cv::imread("./images/test.jpg");
    if (img_src.empty()) {
        std::cout << "the input image is empty." << std::endl;
        return -1;
    }
    cv::resize(img_src, img_src, cv::Size(480, 480));
    ncnn::Net net;
    if (net.load_param("./model/landmark.param") == -1 ||
        net.load_model("./model/landmark.bin") == -1) {
        std::cout << "load model failed." << std::endl;
        return -1;
    }

    ncnn::Extractor ex = net.create_extractor();
    ncnn::Mat ncnn_in = ncnn::Mat::from_pixels_resize(img_src.data,
        ncnn::Mat::PIXEL_BGR, img_src.cols, img_src.rows, 48, 48);
    const float meanVals[3] = { 127.5f, 127.5f, 127.5f };
    const float normVals[3] = { 0.0078125f, 0.0078125f, 0.0078125f };
    ncnn_in.substract_mean_normalize(meanVals, normVals);
    ex.input("data", ncnn_in);
    ncnn::Mat ncnn_out;
    ex.extract("bn6_3", ncnn_out);
    std::cout << "channels: " << ncnn_out.w << std::endl;
    for (int i = 0; i < 106; ++i) {
        std::cout << "ncnn out: " << ncnn_out[2 * i] << " " << ncnn_out[2 * i + 1] << std::endl;
        float x = abs(ncnn_out[2 * i] * img_src.cols);
        float y = abs(ncnn_out[2 * i + 1]  * img_src.rows);
        cv::Point curr_pt = cv::Point(x, y);
        cv::circle(img_src, curr_pt, 4, cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("result", img_src);
    cv::waitKey(0);
    return 0;
}