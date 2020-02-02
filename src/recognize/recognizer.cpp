#include "recognizer.h"
#include "./mobilefacenet/mobilefacenet.h"

namespace mirror {
Recognizer::Recognizer() {
}

Recognizer::~Recognizer() {

}

int Recognizer::LoadModel(const char * root_path) {
	return 0;
}

int Recognizer::ExtractFeature(const cv::Mat & img_face,
	std::vector<float>* feature) {
	return 0;
}

Recognizer* MobilefacenetRecognizerFactory::CreateRecognizer() {
	return new Mobilefacenet();
}

}
