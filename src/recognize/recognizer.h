#ifndef _FACE_RECOGNIZER_H_
#define _FACE_RECOGNIZER_H_

#include <vector>
#include "opencv2/core.hpp"
#include "../common/common.h"

class Recognizer {
public:
	Recognizer();
	virtual ~Recognizer();
	virtual int LoadModel(const char* root_path);
	virtual int ExtractFeature(const cv::Mat& img_face, std::vector<float>* feature);

};


#endif // !_FACE_RECOGNIZER_H_

