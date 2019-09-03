#ifndef _PREPROCESS_H_
#define _PREPROCESS_H_

#include "common.h"

cv::Mat MeanAxis0(const cv::Mat &src);

cv::Mat ElementwiseMinus(const cv::Mat &A, const cv::Mat &B);

cv::Mat VarAxis0(const cv::Mat &src);

int MatrixRank(cv::Mat M);

cv::Mat SimilarTransform(const cv::Mat& src, const cv::Mat& dst);

cv::Mat Align(const cv::Mat& img_src, const std::vector<cv::Point2f>& keypoints);

#endif
