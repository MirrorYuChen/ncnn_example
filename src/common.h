#ifndef _COMMON_H_
#define _COMMON_H_

#include <vector>
#include "opencv2/core.hpp"

struct FaceInfo {
	FaceInfo() {}
	FaceInfo(const cv::Rect& face, const int& score) {
		face_ = face;
		score_ = score;
	}
	cv::Rect face_;
	float score_;
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
	const cv::Rect & rect2, float * iou);

int NMS(const std::vector<FaceInfo>& faces, std::vector<FaceInfo> * result);

#endif // !_COMMON_H_

