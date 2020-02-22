#define _CRT_SECURE_NO_WARNINGS
#include "mobilenet.h"
#include <algorithm>
#include <string>


namespace mirror {
	Mobilenet::Mobilenet() {
		mobilenet_ = new ncnn::Net();
		initialized_ = false;
	}

	Mobilenet::~Mobilenet() {
		if (mobilenet_) {
			mobilenet_->clear();
		}
	}

	int Mobilenet::LoadModel(const char * root_path) {
		std::cout << "start load model." << std::endl;
		std::string param_file = std::string(root_path) + "/mobilenet.param";
		std::string model_file = std::string(root_path) + "/mobilenet.bin";
		if (mobilenet_->load_param(param_file.c_str()) == -1 ||
			mobilenet_->load_model(model_file.c_str()) == -1 ||
			LoadLabels(root_path) != 0) {
			std::cout << "load model or label file failed." << std::endl;
			return 10000;
		}
		initialized_ = true;
		std::cout << "end load model." << std::endl;

		return 0;
	}
	int Mobilenet::Classify(const cv::Mat & img_src, std::vector<ImageInfo>* images) {
		std::cout << "start classify." << std::endl;
		images->clear();
		if (!initialized_) {
			std::cout << "model uninitialized." << std::endl;
			return 10000;
		}
		if (img_src.empty()) {
			std::cout << "input empty." << std::endl;
			return 10001;
		}
		ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_src.data, ncnn::Mat::PIXEL_BGR2RGB,
			img_src.cols, img_src.rows, inputSize.width, inputSize.height);
		in.substract_mean_normalize(meanVals, normVals);

		ncnn::Extractor ex = mobilenet_->create_extractor();
		ex.input("data", in);
		ncnn::Mat out;
		ex.extract("prob", out);
		std::cout << "out w: " << out.w << std::endl;
		
		std::vector<std::pair<float, int>> scores;
		for (int i = 0; i < out.w; ++i) {
			scores.push_back(std::make_pair(out[i], i));
		}

		int topk = 5;
		std::partial_sort(scores.begin(), scores.begin() + topk, scores.end(),
			std::greater< std::pair<float, int> >());

		for (int i = 0; i < topk; ++i) {
			ImageInfo image_info;
			image_info.label_ = labels_[scores[i].second];
			image_info.score_ = scores[i].first;
			images->push_back(image_info);
		}

		std::cout << "end classify." << std::endl;
		return 0;
	}

	int Mobilenet::LoadLabels(const char * root_path) {
		std::string label_file = std::string(root_path) + "/label.txt";
		FILE* fp = fopen(label_file.c_str(), "r");

		while (!feof(fp)) {
			char str[1024];
			if (nullptr == fgets(str, 1024, fp)) continue;
			std::string str_s(str);

			if (str_s.length() > 0) {
				for (int i = 0; i < str_s.length(); i++) {
					if (str_s[i] == ' ') {
						std::string strr = str_s.substr(i, str_s.length() - i - 1);
						labels_.push_back(strr);
						i = str_s.length();
					}
				}
			}
		}
		return 0;
	}
}


