#ifndef _FACE_RECOGNIZER_H_
#define _FACE_RECOGNIZER_H_

#include <vector>
#include "opencv2/core.hpp"
#include "../common/common.h"

namespace mirror {
class Recognizer {
public:
	virtual ~Recognizer() {};
	virtual int LoadModel(const char* root_path) = 0;
	virtual int ExtractFeature(const cv::Mat& img_face, std::vector<float>* feature) = 0;

};

class RecognizerFactory {
public:
	virtual Recognizer* CreateRecognizer() = 0;
	virtual ~RecognizerFactory() {}

};

class MobilefacenetRecognizerFactory : public RecognizerFactory {
public:
	MobilefacenetRecognizerFactory() {};
	Recognizer* CreateRecognizer();
	~MobilefacenetRecognizerFactory() {}
};	

}

#endif // !_FACE_RECOGNIZER_H_

