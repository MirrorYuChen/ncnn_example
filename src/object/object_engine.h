#ifndef _OBJECT_DETECTOR_H_
#define _OBJECT_DETECTOR_H_

#include <vector>
#include "opencv2/core.hpp"
#include "../common/common.h"

namespace mirror {

class ObjectEngine {
public:
	ObjectEngine();
	~ObjectEngine();

	int LoadModel(const char* root_path);
	int DetectObject(const cv::Mat& img_src, std::vector<ObjectInfo>* objects);

private:
	class Impl;
	Impl* impl_;
};

}

#endif // !_OBJECT_DETECTOR_H_

