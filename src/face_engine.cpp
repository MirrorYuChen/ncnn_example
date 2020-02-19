#include "face_engine.h"
#include <iostream>
#include <string>
#include "aligner/aligner.h"
#include "database/face_database.h"
#include "tracker/tracker.h"
#include "landmarker/landmarker.h"
#include "recognizer/recognizer.h"
#include "detecter/detecter.h"

namespace mirror {
class FaceEngine::Impl {
public:
	Impl() {
		detecter_factory_ = new RetinafaceFactory();
		landmarker_factory_ = new ZQLandmarkerFactory();
		recognizer_factory_ = new MobilefacenetRecognizerFactory();

		database_ = new FaceDatabase();
		tracker_ = new Tracker();
		detecter_ = detecter_factory_->CreateDetecter();
		landmarker_ = landmarker_factory_->CreateLandmarker();
		recognizer_ = recognizer_factory_->CreateRecognizer();
		aligner_ = new Aligner();
		initialized_ = false;

		// ncnn::create_gpu_instance();	
		// fdnet_->opt.use_vulkan_compute = 1;
		// flnet_->opt.use_vulkan_compute = 1;
		// frnet_->opt.use_vulkan_compute = 1;
	}
	~Impl() {
		if (detecter_factory_) {
			delete detecter_factory_;
			detecter_factory_ = nullptr;
		}
		if (landmarker_factory_) {
			delete landmarker_factory_;
			landmarker_factory_ = nullptr;
		}
		if (recognizer_factory_) {
			delete recognizer_factory_;
			recognizer_factory_ = nullptr;
		}
		if (database_) {
			delete database_;
			database_ = nullptr;
		}
		if (detecter_) {
			delete detecter_;
			detecter_ = nullptr;
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

public:
	DetecterFactory* detecter_factory_;
	LandmarkerFactory* landmarker_factory_;
	RecognizerFactory* recognizer_factory_;

	FaceDatabase* database_;
	Tracker* tracker_;
	Detecter* detecter_;
	Landmarker* landmarker_;
	Recognizer* recognizer_;
	Aligner * aligner_;

    std::string db_name_;
	bool initialized_;
	int LoadModel(const char* root_path);

};

int FaceEngine::Impl::LoadModel(const char * root_path) {
	if (detecter_->LoadModel(root_path) != 0) {
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
	return impl_->detecter_->Detect(img_src, faces);
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

int FaceEngine::Delete(const std::string& name) {
	return impl_->database_->Delete(name);
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

}
