#include "recognizer.h"

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
