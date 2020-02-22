#ifndef _OBJECT_DETECTER_H_
#define _OBJECT_DETECTER_H_

#include <vector>
#include "opencv2/core.hpp"
#include "../object_engine.h"

namespace mirror {
class ObjectDetecter {
public:
	virtual ~ObjectDetecter() {}
	virtual int LoadModel(const char* root_path) = 0;
	virtual int DetectObject(const cv::Mat& img_src, std::vector<ObjectInfo>* objects) = 0;
};

}



#endif // !_OBJECT_DETECT_H_

