#ifndef _FACE_DETECTOR_H_
#define _FACE_DETECTOR_H_

#include "opencv2/core.hpp"
#include "../common/common.h"

class Detector {
public:
	Detector();
	virtual ~Detector();
	virtual int LoadModel(const char* root_path);
	virtual int Detect(const cv::Mat& img_src, std::vector<FaceInfo>* faces);

};


#endif // !_FACE_DETECTOR_H_

