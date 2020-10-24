#ifndef _FACE_LANDMARKER_H_
#define _FACE_LANDMARKER_H_

#include "opencv2/core.hpp"

namespace mirror {
// 抽象类
class Landmarker {
public:
	virtual ~Landmarker() {};
	virtual int LoadModel(const char* root_path) = 0;
	virtual int ExtractKeypoints(const cv::Mat& img_src,
		const cv::Rect& face, std::vector<cv::Point2f>* keypoints) = 0;
};

// 工厂基类
class LandmarkerFactory {
public:
	virtual Landmarker* CreateLandmarker() = 0;
	virtual ~LandmarkerFactory() {}
};

// 不同landmark检测器工厂
class ZQLandmarkerFactory : public LandmarkerFactory {
public:
	ZQLandmarkerFactory(){}
	Landmarker* CreateLandmarker();
	~ZQLandmarkerFactory() {}
};

class InsightfaceLandmarkerFactory : public LandmarkerFactory {
public:
	InsightfaceLandmarkerFactory(){}
	Landmarker* CreateLandmarker();
	~InsightfaceLandmarkerFactory() {}
};

}

#endif // !_FACE_LANDMARKER_H_

