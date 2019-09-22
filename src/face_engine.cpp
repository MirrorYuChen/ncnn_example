#include "face_engine.h"
#include <iostream>
#include <string>
#include "ncnn/net.h"
#include "common.h"
#include "aligner.h"
#include "retinaface.h"

class FaceEngine::Impl {
public:
	Impl() : detector_(new RetinaFace()),
		flnet_(new ncnn::Net()),
		frnet_(new ncnn::Net()),
		aligner_(new Aligner()),
		initialized(false) {
		// ncnn::create_gpu_instance();	
		// fdnet_->opt.use_vulkan_compute = 1;
		// flnet_->opt.use_vulkan_compute = 1;
		// frnet_->opt.use_vulkan_compute = 1;
	}
	~Impl() {
		flnet_->clear();
		frnet_->clear();
		// ncnn::destroy_gpu_instance();
	}

	Detector* detector_;
	ncnn::Net* flnet_;
	ncnn::Net* frnet_;
	Aligner * aligner_;

	bool initialized;

	const float flMeanVals[3] = { 127.5f, 127.5f, 127.5f };
	const float flNormVals[3] = { 0.0078125f, 0.0078125f, 0.0078125f };
	const int RPNs[3] = { 32, 16, 8 };
	int LoadModel(const char* root_path);
	int ExtractKeypoints(const cv::Mat& img_src,
		const cv::Rect& face, std::vector<cv::Point2f>* keypoints);
	int ExtractFeature(const cv::Mat& img_face, std::vector<float>* feature);

};

int FaceEngine::Impl::LoadModel(const char * root_path) {
	if (detector_->LoadModel(root_path) != 0) {
		return 10000;
	}

	std::string fl_param = std::string(root_path) + "/fl.param";
	std::string fl_bin = std::string(root_path) + "/fl.bin";
	if (flnet_->load_param(fl_param.c_str()) == -1 ||
		flnet_->load_model(fl_bin.c_str()) == -1) {
		std::cout << "load face landmark model failed." << std::endl;
		return 10000;
	}

	std::string fr_param = std::string(root_path) + "/fr.param";
	std::string fr_bin = std::string(root_path) + "/fr.bin";
	if (frnet_->load_param(fr_param.c_str()) == -1 ||
		frnet_->load_model(fr_bin.c_str()) == -1) {
		std::cout << "load face recognize model failed." << std::endl;
		return 10000;
	}

	initialized = true;
	return 0;
}

int FaceEngine::Impl::ExtractKeypoints(const cv::Mat & img_src,
	const cv::Rect & face, std::vector<cv::Point2f>* keypoints) {
	std::cout << "start keypoints extract." << std::endl;
	if (!initialized) {
		std::cout << "model uninitialized." << std::endl;
		return 10000;
	}
	if (img_src.empty()) {
		std::cout << "input empty." << std::endl;
		return 10001;
	}
	keypoints->clear();
	cv::Mat img_face = img_src(face).clone();
	ncnn::Extractor ex = flnet_->create_extractor();
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_face.data,
		ncnn::Mat::PIXEL_BGR, img_face.cols, img_face.rows, 112, 112);
	in.substract_mean_normalize(flMeanVals, flNormVals);
	ex.input("data", in);
	ncnn::Mat out;
	ex.extract("bn6_3", out);

	for (int i = 0; i < 106; ++i) {
		float x = abs(out[2 * i] * img_face.cols) + face.x;
		float y = abs(out[2 * i + 1] * img_face.rows) + face.y;
		keypoints->push_back(cv::Point2f(x, y));
	}
	std::cout << "keypoints extract end." << std::endl;
	return 0;
}

int FaceEngine::Impl::ExtractFeature(const cv::Mat& img_face,
	std::vector<float>* feature) {
	std::cout << "start extract feature." << std::endl;
	feature->clear();
	if (!initialized) {
		std::cout << "model uninitialized." << std::endl;
		return 10000;
	}
	if (img_face.empty()) {
		std::cout << "input empty" << std::endl;
		return 10001;
	}
	cv::Mat face_cpy = img_face.clone();
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(face_cpy.data,
		ncnn::Mat::PIXEL_BGR2RGB, face_cpy.cols, face_cpy.rows, 112, 112);
	feature->resize(kFaceFeatureDim);
	ncnn::Extractor ex = frnet_->create_extractor();
	ex.input("data", in);
	ncnn::Mat out;
	ex.extract("fc1", out);
	for (int i = 0; i < kFaceFeatureDim; ++i) {
		feature->at(i) = out[i];
	}
	std::cout << "extract feature end." << std::endl;
	return 0;
}

FaceEngine::FaceEngine() {
	impl_ = new FaceEngine::Impl();
}

FaceEngine::~FaceEngine() {
	if (impl_) {
		delete impl_;
	}
}

int FaceEngine::LoadModel(const char * root_path) {
	return impl_->LoadModel(root_path);
}

int FaceEngine::Detect(const cv::Mat & img_src, std::vector<FaceInfo>* faces) {
	return impl_->detector_->Detect(img_src, faces);
}

int FaceEngine::ExtractKeypoints(const cv::Mat & img_src,
	const cv::Rect & face, std::vector<cv::Point2f>* keypoints) {
	return impl_->ExtractKeypoints(img_src, face, keypoints);
}

int FaceEngine::ExtractFeature(const cv::Mat& img_face, std::vector<float>* feature) {
	return impl_->ExtractFeature(img_face, feature);
}

int FaceEngine::AlignFace(const cv::Mat& img_src,
	const std::vector<cv::Point2f>& keypoints, cv::Mat* face_aligned) {
	return impl_->aligner_->Align(img_src, keypoints, face_aligned);
}
