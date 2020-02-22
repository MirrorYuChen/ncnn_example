#ifndef _CLASSIFIER_MOBILENET_H_
#define _CLASSIFIER_MOBILENET_H_

#include "../classifier.h"
#include "ncnn/net.h"

namespace mirror {

class Mobilenet : public Classifier {
public:
	Mobilenet();
	~Mobilenet();
	int LoadModel(const char* root_path);
	int Classify(const cv::Mat& img_src, std::vector<ImageInfo>* images);

private:
	bool initialized_;
	ncnn::Net* mobilenet_;
	std::vector<std::string> labels_;
	const float meanVals[3] = { 103.94f, 116.78f, 123.68f };
	const float normVals[3] = { 0.017f,  0.017f,  0.017f };
	const cv::Size inputSize = cv::Size(224, 224);

	int LoadLabels(const char* root_path);

};

}



#endif // !_CLASSIFIER_MOBILENET_H_

