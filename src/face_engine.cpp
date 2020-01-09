#include "face_engine.h"
#include <iostream>
#include <string>
#include "align/aligner.h"
#include "database/face_database.h"
#include "detect/mtcnn/mtcnn.h"
#include "detect/retinaface/retinaface.h"
#include "detect/centerface/centerface.h"
#include "track/tracker.h"
#include "landmark/zqlandmark/zq_landmarker.h"
#include "recognize/mobilefacenet/mobilefacenet.h"

class FaceEngine::Impl {
public:
	Impl() : database_(new FaceDatabase()), 
		detector_(new CenterFace()),
		tracker_(new Tracker()),
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
		if (database_) {
			delete database_;
			database_ = nullptr;
		}
		if (detector_) {
			delete detector_;
			detector_ = nullptr;
		}

		if (tracker_) {
			delete tracker_;
			tracker_ = nullptr;
		}

		if (landmarker_) {
			delete landmarker_;
			landmarker_ = nullptr;
		}

		if (recognizer_) {
			delete recognizer_;
			recognizer_ = nullptr;
		}

		if (aligner_) {
			delete aligner_;
			aligner_ = nullptr;
		}

		// ncnn::destroy_gpu_instance();
	}

	FaceDatabase* database_;
	Detector* detector_;
	Tracker* tracker_;
	Landmarker* landmarker_;
	Recognizer* recognizer_;
	Aligner * aligner_;

    std::string db_name_;
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

	db_name_ = std::string(root_path);
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

int FaceEngine::Track(const std::vector<FaceInfo>& curr_faces, std::vector<TrackedFaceInfo>* faces) {
	return impl_->tracker_->Track(curr_faces, faces);
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

int FaceEngine::Insert(const std::vector<float>& feat, const std::string& name) {
	return impl_->database_->Insert(feat, name);
}

int FaceEngine::Delete(int64_t index) {
	return impl_->database_->Delete(index);
}

int64_t FaceEngine::QueryTop(const std::vector<float>& feat, QueryResult *query_result) {
	return impl_->database_->QueryTop(feat, query_result);
}

int FaceEngine::Save() {
	return impl_->database_->Save(impl_->db_name_.c_str());
}

int FaceEngine::Load() {
	return impl_->database_->Load(impl_->db_name_.c_str());
}
