#include "face_engine.h"
#include <iostream>
#include <string>
#include "align/aligner.h"
#include "detect/mtcnn/mtcnn.h"
#include "detect/retinaface/retinaface.h"
#include "detect/centerface/centerface.h"
#include "landmark/zqlandmark/zq_landmarker.h"
#include "recognize/mobilefacenet/mobilefacenet.h"

class FaceEngine::Impl {
public:
	Impl() : detector_(new CenterFace()),
		landmarker_(new ZQLandmarker()),
		recognizer_(new Mobilefacenet()),
		aligner_(new Aligner()),
		initialized_(false) {
		// ncnn::create_gpu_instance();	
		// fdnet_->opt.use_vulkan_compute = 1;
		// flnet_->opt.use_vulkan_compute = 1;
		// frnet_->opt.use_vulkan_compute = 1;
	}
	~Impl() {
		// ncnn::destroy_gpu_instance();
	}

	Detector* detector_;
	Landmarker* landmarker_;
	Recognizer* recognizer_;
	Aligner * aligner_;

	bool initialized_;
	int LoadModel(const char* root_path);

};

int FaceEngine::Impl::LoadModel(const char * root_path) {
	if (detector_->LoadModel(root_path) != 0) {
		return 10000;
	}

	if (landmarker_->LoadModel(root_path) != 0)	{
		return 10000;
	}

	if (recognizer_->LoadModel(root_path) != 0) {
		return 10000;
	}

	initialized_ = true;
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
	return impl_->landmarker_->ExtractKeypoints(img_src, face, keypoints);
}

int FaceEngine::ExtractFeature(const cv::Mat& img_face, std::vector<float>* feature) {
	return impl_->recognizer_->ExtractFeature(img_face, feature);
}

int FaceEngine::AlignFace(const cv::Mat& img_src,
	const std::vector<cv::Point2f>& keypoints, cv::Mat* face_aligned) {
	return impl_->aligner_->Align(img_src, keypoints, face_aligned);
}
