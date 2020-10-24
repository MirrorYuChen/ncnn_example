#ifndef _CLASSIFIER_ENGINE_H_
#define _CLASSIFIER_ENGINE_H_

#include <iostream>
#include <string>
#include <vector>
#include "opencv2/core.hpp"
#include "../common/common.h"

#if defined(_MSC_VER) || defined(_WIN32) || defined(_WIN64)
    #ifdef CLASSIFIER_EXPORTS
        #define CLASSIFIER_API __declspec(dllexport)
    #else
        #define CLASSIFIER_API __declspec(dllimport)
    #endif
#else
    #define CLASSIFIER_API __attribute__ ((visibility("default")))
#endif

namespace mirror {
class ClassifierEngine {
public:
	CLASSIFIER_API ClassifierEngine();
	CLASSIFIER_API ~ClassifierEngine();
	CLASSIFIER_API int LoadModel(const char* root_path);
	CLASSIFIER_API int Classify(const cv::Mat& img_src, std::vector<ImageInfo>* images);

private:
	class Impl;
	Impl* impl_;

};

}

#endif // !_CLASSIFIER_ENGINE_H_

