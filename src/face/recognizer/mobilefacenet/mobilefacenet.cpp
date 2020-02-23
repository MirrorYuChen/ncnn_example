#include "mobilefacenet.h"
#include <iostream>
#include <string>
#if MIRROR_VULKAN
#include "gpu.h"
#endif // MIRROR_VULKAN

namespace mirror {
Mobilefacenet::Mobilefacenet() {
	mobileface_net_ = new ncnn::Net();
	initialized_ = false;
#if MIRROR_VULKAN
	ncnn::create_gpu_instance();	
    mobileface_net_->opt.use_vulkan_compute = true;
#endif // MIRROR_VULKAN
}

Mobilefacenet::~Mobilefacenet() {
	mobileface_net_->clear();
#if MIRROR_VULKAN
	ncnn::destroy_gpu_instance();
#endif // MIRROR_VULKAN	
}

int Mobilefacenet::LoadModel(const char * root_path) {
	std::string fr_param = std::string(root_path) + "/fr.param";
	std::string fr_bin = std::string(root_path) + "/fr.bin";
	if (mobileface_net_->load_param(fr_param.c_str()) == -1 ||
		mobileface_net_->load_model(fr_bin.c_str()) == -1) {
		std::cout << "load face recognize model failed." << std::endl;
		return 10000;
	}

	initialized_ = true;
	return 0;
}

int Mobilefacenet::ExtractFeature(const cv::Mat & img_face,
	std::vector<float>* feature) {
	std::cout << "start extract feature." << std::endl;
	feature->clear();
	if (!initialized_) {
		std::cout << "mobilefacenet model uninitialized." << std::endl;
		return 10000;
	}
	if (img_face.empty()) {
		std::cout << "input empty." << std::endl;
		return 10001;
	}

	cv::Mat face_cpy = img_face.clone();
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(face_cpy.data,
		ncnn::Mat::PIXEL_BGR2RGB, face_cpy.cols, face_cpy.rows, 112, 112);
	feature->resize(kFaceFeatureDim);
	ncnn::Extractor ex = mobileface_net_->create_extractor();
	ex.input("data", in);
	ncnn::Mat out;
	ex.extract("fc1", out);
	for (int i = 0; i < kFaceFeatureDim; ++i) {
		feature->at(i) = out[i];
	}

	std::cout << "end extract feature." << std::endl;

	return 0;
}

}


