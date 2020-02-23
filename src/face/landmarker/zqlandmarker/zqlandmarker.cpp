#include "zqlandmarker.h"
#include <iostream>
#include <string>

#if MIRROR_VULKAN
#include "gpu.h"
#endif // MIRROR_VULKAN

namespace mirror {
ZQLandmarker::ZQLandmarker() {
	zq_landmarker_net_ = new ncnn::Net();
	initialized = false;
#if MIRROR_VULKAN
	ncnn::create_gpu_instance();	
    zq_landmarker_net_->opt.use_vulkan_compute = true;
#endif // MIRROR_VULKAN
}

ZQLandmarker::~ZQLandmarker() {
	zq_landmarker_net_->clear();
#if MIRROR_VULKAN
	ncnn::destroy_gpu_instance();
#endif // MIRROR_VULKAN	
}

int ZQLandmarker::LoadModel(const char * root_path) {
	std::string fl_param = std::string(root_path) + "/fl.param";
	std::string fl_bin = std::string(root_path) + "/fl.bin";
	if (zq_landmarker_net_->load_param(fl_param.c_str()) == -1 ||
		zq_landmarker_net_->load_model(fl_bin.c_str()) == -1) {
		std::cout << "load face landmark model failed." << std::endl;
		return 10000;
	}
	initialized = true;
	return 0;
}

int ZQLandmarker::ExtractKeypoints(const cv::Mat & img_src,
	const cv::Rect & face, std::vector<cv::Point2f>* keypoints) {
	std::cout << "start extract keypoints." << std::endl;
	keypoints->clear();
	if (!initialized) {
		std::cout << "zq landmarker unitialized." << std::endl;
		return 10000;
	}

	if (img_src.empty()) {
		std::cout << "input empty." << std::endl;
		return 10001;
	}

	cv::Mat img_face = img_src(face).clone();
	ncnn::Extractor ex = zq_landmarker_net_->create_extractor();
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_face.data,
		ncnn::Mat::PIXEL_BGR, img_face.cols, img_face.rows, 112, 112);
	in.substract_mean_normalize(meanVals, normVals);
	ex.input("data", in);
	ncnn::Mat out;
	ex.extract("bn6_3", out);

	for (int i = 0; i < 106; ++i) {
		float x = abs(out[2 * i] * img_face.cols) + face.x;
		float y = abs(out[2 * i + 1] * img_face.rows) + face.y;
		keypoints->push_back(cv::Point2f(x, y));
	}


	std::cout << "end extract keypoints." << std::endl;
	return 0;
}

}
