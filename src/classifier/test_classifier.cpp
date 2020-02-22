#include "classifier_engine.h"
#include "opencv2/opencv.hpp"

int main(int argc, char* argv[]) {
	const char* img_path = "../../data/images/dog.jpg";
	cv::Mat img_src = cv::imread(img_path);

	const char* root_path = "../../data/models";
	mirror::ClassifierEngine* classifier_engine = new mirror::ClassifierEngine();

	classifier_engine->LoadModel(root_path);
	std::vector<mirror::ImageInfo> images;
	classifier_engine->Classify(img_src, &images);

	int topk = images.size();
	for (int i = 0; i < topk; ++i) {
		cv::putText(img_src, images[i].label_, cv::Point(10, 10 + 30 * i),
			0, 0.5, cv::Scalar(255, 100, 0), 2, 2);
	}

	cv::imshow("result", img_src);
	cv::waitKey(0);
	delete classifier_engine;
	classifier_engine = nullptr;

	cv::imwrite("../../data/images/classify_result.jpg", img_src);

	return 0;

}

