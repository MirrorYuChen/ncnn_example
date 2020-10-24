#define OBJECT_EXPORTS
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "object_engine.h"
#include <iostream>

int main(int argc, char* argv[]) {
	const char* img_path = "../../data/images/cat.jpg";
	cv::Mat img_src = cv::imread(img_path);

	const char* model_root_path = "../../data/models";
	mirror::ObjectEngine* object_engine = new mirror::ObjectEngine();

	object_engine->LoadModel(model_root_path);

	double start = static_cast<double>(cv::getTickCount());
	std::vector<mirror::ObjectInfo> objects;
	object_engine->DetectObject(img_src, &objects);

	int num_objects = static_cast<int>(objects.size());
	for (int i = 0; i < num_objects; ++i) {
		cv::rectangle(img_src, objects[i].location_, cv::Scalar(255, 0, 255), 2);

		char text[256];
		sprintf(text, "%s %.1f%%", objects[i].name_.c_str(), objects[i].score_ * 100);
		int baseLine = 0;
		cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		cv::putText(img_src, text, cv::Point(objects[i].location_.x,
			objects[i].location_.y + label_size.height),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
	}
	double end = static_cast<double>(cv::getTickCount());
	double time_cost = (end - start) / cv::getTickFrequency() * 1000;
	std::cout << "time cost: " << time_cost << " ms." << std::endl;
	cv::imwrite("../../data/images/object_result.jpg", img_src);
	cv::imshow("result", img_src);
	cv::waitKey(0);

	delete object_engine;
	object_engine = nullptr;

	return 0;
}