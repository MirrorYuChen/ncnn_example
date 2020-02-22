#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include "../classifier_engine.h"

namespace mirror {

class Classifier {
public:
	virtual ~Classifier() {}
	virtual int LoadModel(const char* root_path) = 0;
	virtual int Classify(const cv::Mat& img_src, std::vector<ImageInfo>* images) = 0;
};



}




#endif // !_CLASSIFIER_H_

