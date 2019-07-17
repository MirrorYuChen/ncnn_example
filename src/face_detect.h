#include <vector>
#include "common.h"
#include "opencv2/core.hpp"

class FaceDetector {
 public:
	FaceDetector();
	~FaceDetector();
    int LoadModel(const char* root_path);
    int Detect(const cv::Mat& img_src, std::vector<FaceInfo>* faces);
	int ExtractKeypoints(const cv::Mat& img_src,
		const cv::Rect& face, std::vector<cv::Point>* keypoints);

 private:
    class Impl;
    Impl* impl;

};