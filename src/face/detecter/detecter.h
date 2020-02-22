#ifndef _FACE_DETECTER_H_
#define _FACE_DETECTER_H_

#include "opencv2/core.hpp"
#include "../common/common.h"

namespace mirror {
// 抽象类
class Detecter {
public:
	virtual ~Detecter() {};
	virtual int LoadModel(const char* root_path) = 0;
	virtual int DetectFace(const cv::Mat& img_src, std::vector<FaceInfo>* faces) = 0;

};

// 工厂基类
class DetecterFactory {
public:
	virtual Detecter* CreateDetecter() = 0;
	virtual ~DetecterFactory() {};
};

// 不同人脸检测器
class CenterfaceFactory : public DetecterFactory {
public:
	CenterfaceFactory() {}
	~CenterfaceFactory() {}
	Detecter* CreateDetecter();
};

class MtcnnFactory : public DetecterFactory {
public:
	MtcnnFactory() {}
	~MtcnnFactory() {}
	Detecter* CreateDetecter();

};

class RetinafaceFactory : public DetecterFactory {
public:
	RetinafaceFactory() {}
	~RetinafaceFactory() {}
	Detecter* CreateDetecter();
};

}

#endif // !_FACE_DETECTER_H_

