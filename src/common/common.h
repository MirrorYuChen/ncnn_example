#ifndef _COMMON_H_
#define _COMMON_H_

#include <vector>
#include <string>
#include "opencv2/core.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

#define kFaceFeatureDim 128
const int threads_num = 2;

struct FaceInfo {
	FaceInfo() {}
	FaceInfo(const cv::Rect& face, const int& score) {
		face_ = face;
		score_ = score;
	}
	cv::Rect face_;
	float score_;
	float keypoints_[10];
};

struct TrackedFaceInfo {
	FaceInfo face_info_;
	float iou_score_;
};

struct QueryResult {
    std::string name_;
    float sim_;
};

int RatioAnchors(const cv::Rect & anchor,
	const std::vector<float>& ratios, std::vector<cv::Rect>* anchors);

int ScaleAnchors(const std::vector<cv::Rect>& ratio_anchors,
	const std::vector<float>& scales, std::vector<cv::Rect>* anchors);

int GenerateAnchors(const int & base_size,
	const std::vector<float>& ratios, const std::vector<float> scales,
	std::vector<cv::Rect>* anchors);

float InterRectArea(const cv::Rect & a,
	const cv::Rect & b);

int ComputeIOU(const cv::Rect & rect1,
	const cv::Rect & rect2, float * iou,
	const std::string& type = "UNION");

int NMS(const std::vector<FaceInfo>& faces, std::vector<FaceInfo> * result,
	const float& threshold, const std::string& type = "UNION");

float CalculSimilarity(const std::vector<float>&feature1, const std::vector<float>& feature2);

#endif // !_COMMON_H_

