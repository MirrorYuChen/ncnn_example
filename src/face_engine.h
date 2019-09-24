#include <vector>
#include "common/common.h"
#include "opencv2/core.hpp"

class FaceEngine {
public:
	FaceEngine();
	~FaceEngine();
	int LoadModel(const char* root_path);
	int Detect(const cv::Mat& img_src, std::vector<FaceInfo>* faces);
	int ExtractKeypoints(const cv::Mat& img_src,
		const cv::Rect& face, std::vector<cv::Point2f>* keypoints);
	int ExtractFeature(const cv::Mat& img_face, std::vector<float>* feature);
	int AlignFace(const cv::Mat& img_src, const std::vector<cv::Point2f>& keypoints, cv::Mat* face_aligned);

private:
	class Impl;
	Impl* impl_;

};