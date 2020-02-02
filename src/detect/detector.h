#ifndef _FACE_DETECTOR_H_
#define _FACE_DETECTOR_H_

#include "opencv2/core.hpp"
#include "../common/common.h"

namespace mirror {
// 抽象类
class Detector {
public:
	Detector();
	virtual ~Detector();
	virtual int LoadModel(const char* root_path);
	virtual int Detect(const cv::Mat& img_src, std::vector<FaceInfo>* faces);

};

// 工厂基类
class DetectorFactory {
public:
	DetectorFactory() {}
	virtual Detector* CreateDetector() = 0;
	virtual ~DetectorFactory() {};
};

// 不同人脸检测器
class CenterfaceFactory : public DetectorFactory {
public:
	CenterfaceFactory() {}
	~CenterfaceFactory() {}
	Detector* CreateDetector();
};

class MtcnnFactory : public DetectorFactory {
public:
	MtcnnFactory() {}
	~MtcnnFactory() {}
	Detector* CreateDetector();

};

class RetinafaceFactory : public DetectorFactory {
public:
	RetinafaceFactory() {}
	~RetinafaceFactory() {}
	Detector* CreateDetector();
};

}

#endif // !_FACE_DETECTOR_H_

