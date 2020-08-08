#include "insightface.h"
#include <iostream>
#include <string>
#include "../../common/common.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#if MIRROR_VULKAN
#include "gpu.h"
#endif // MIRROR_VULKAN

namespace mirror {
InsightfaceLandmarker::InsightfaceLandmarker() {
	insightface_landmarker_net_ = new ncnn::Net();
	initialized = false;
#if MIRROR_VULKAN
	ncnn::create_gpu_instance();	
    insightface_landmarker_net_->opt.use_vulkan_compute = true;
#endif // MIRROR_VULKAN
}

InsightfaceLandmarker::~InsightfaceLandmarker() {
	insightface_landmarker_net_->clear();
#if MIRROR_VULKAN
	ncnn::destroy_gpu_instance();
#endif // MIRROR_VULKAN	
}

int InsightfaceLandmarker::LoadModel(const char * root_path) {
	std::string fl_param = std::string(root_path) + "/2d106.param";
	std::string fl_bin = std::string(root_path) + "/2d106.bin";
	if (insightface_landmarker_net_->load_param(fl_param.c_str()) == -1 ||
		insightface_landmarker_net_->load_model(fl_bin.c_str()) == -1) {
		std::cout << "load face landmark model failed." << std::endl;
		return 10000;
	}
	initialized = true;
	return 0;
}

int InsightfaceLandmarker::ExtractKeypoints(const cv::Mat & img_src,
	const cv::Rect & face, std::vector<cv::Point2f>* keypoints) {
	std::cout << "start extract keypoints." << std::endl;
	keypoints->clear();
	if (!initialized) {
		std::cout << "insightface landmarker unitialized." << std::endl;
		return 10000;
	}

	if (img_src.empty()) {
		std::cout << "input empty." << std::endl;
		return 10001;
	}

	// 1 enlarge the face rect
	cv::Rect face_enlarged = face;
	const float enlarge_scale = 1.2f;
	EnlargeRect(enlarge_scale, &face_enlarged);
	face_enlarged = face_enlarged & cv::Rect(0, 0, img_src.cols, img_src.rows);
	cv::Mat img_face = img_src(face_enlarged).clone();

	// 2 resize make border
	int max_side = MAX(img_face.cols, img_face.rows);
	float resize_scale = static_cast<float> (max_side) / 192;
	int face_width_resized = img_face.cols / resize_scale;
	int face_height_resized = img_face.rows / resize_scale;
	cv::Rect roi = cv::Rect((192 - face_width_resized) / 2, (192 - face_height_resized) / 2,
						face_width_resized, face_height_resized);
	cv::Mat face_in = cv::Mat(cv::Size(192, 192), CV_8UC3, cv::Scalar(0,0,0));
	cv::resize(img_face, face_in(roi), cv::Size(face_width_resized, face_height_resized));
	// cv::imwrite("resize_face.jpg", face_in);
	// 3 compute the offset
	float offset_x = roi.x * resize_scale;
	float offset_y = roi.y * resize_scale;

	// 4 start infer
	ncnn::Extractor ex = insightface_landmarker_net_->create_extractor();
	ncnn::Mat in = ncnn::Mat::from_pixels(face_in.data,
		ncnn::Mat::PIXEL_BGR, face_in.cols, face_in.rows);
	ex.input("data", in);
	ncnn::Mat out;
	ex.extract("fc1", out);

	for (int i = 0; i < 106; ++i) {
		float x = (out[2 * i] + 1.0f) * img_face.cols / 2 + face_enlarged.x - offset_x;
		float y = (out[2 * i + 1] + 1.0f) * img_face.rows / 2 + face_enlarged.y - offset_y;
		keypoints->push_back(cv::Point2f(x, y));
	}


	std::cout << "end extract keypoints." << std::endl;
	return 0;
}

}
