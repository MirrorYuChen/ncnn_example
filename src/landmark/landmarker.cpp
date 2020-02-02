#include "landmarker.h"
#include "zqlandmark/zq_landmarker.h"

namespace mirror {
Landmarker::Landmarker() {

}

Landmarker::~Landmarker() {

}

int Landmarker::LoadModel(const char * root_path) {
	return 0;
}

int Landmarker::ExtractKeypoints(const cv::Mat & img_src,
	const cv::Rect & face, std::vector<cv::Point2f>* keypoints) {
	return 0;
}

Landmarker* ZQLandmarkerFactory::CreateLandmarker() {
	return new ZQLandmarker();
}

}
