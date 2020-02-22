#ifndef _CLASSIFIER_ENGINE_H_
#define _CLASSIFIER_ENGINE_H_

#include <iostream>
#include <string>
#include <vector>
#include "opencv2/core.hpp"
#include "../common/common.h"

namespace mirror {
class ClassifierEngine {
public:
	ClassifierEngine();
	~ClassifierEngine();
	int LoadModel(const char* root_path);
	int Classify(const cv::Mat& img_src, std::vector<ImageInfo>* images);

private:
	class Impl;
	Impl* impl_;

};

}

#endif // !_CLASSIFIER_ENGINE_H_

