#include "face_engine.h"
#include <iostream>

#include "detecter/detecter.h"
#include "tracker/tracker.h"
#include "landmarker/landmarker.h"
#include "aligner/aligner.h"
#include "recognizer/recognizer.h"
#include "database/face_database.h"


namespace mirror {
class FaceEngine::Impl {
public:
    Impl() {
        detecter_factory_ = new AnticonvFactory();
        landmarker_factory_ = new ZQLandmarkerFactory();
        recognizer_factory_ = new MobilefacenetRecognizerFactory();
        
        detecter_ = detecter_factory_->CreateDetecter();
        landmarker_ = landmarker_factory_->CreateLandmarker();
        recognizer_ = recognizer_factory_->CreateRecognizer();

		tracker_ = new Tracker();
        aligner_ = new Aligner();
        database_ = new FaceDatabase();
        initialized_ = false;
    }

    ~Impl() {
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

        if (database_) {
            delete database_;
            database_ = nullptr;
        }

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
    }

    int LoadModel(const char* root_path) {
        if (detecter_->LoadModel(root_path) != 0) {
            std::cout << "load face detecter failed." << std::endl;
            return 10000;
        }

        if (landmarker_->LoadModel(root_path) != 0) {
            std::cout << "load face landmarker failed." << std::endl;
            return 10000;
        }

        if (recognizer_->LoadModel(root_path) != 0) {
            std::cout << "load face recognizer failed." << std::endl;
            return 10000;
        }

        db_name_ = std::string(root_path);
        initialized_ = true;

        return 0;
    }
	inline int Track(const std::vector<FaceInfo>& curr_faces, std::vector<TrackedFaceInfo>* faces) {
		return tracker_->Track(curr_faces, faces);
	}
    inline int DetectFace(const cv::Mat& img_src, std::vector<FaceInfo>* faces) {
        return detecter_->DetectFace(img_src, faces);
    }
    inline int ExtractKeypoints(const cv::Mat& img_src,
		const cv::Rect& face, std::vector<cv::Point2f>* keypoints) {
        return landmarker_->ExtractKeypoints(img_src, face, keypoints);
    }
    inline int AlignFace(const cv::Mat& img_src, const std::vector<cv::Point2f>& keypoints, cv::Mat * face_aligned) {
        return aligner_->AlignFace(img_src, keypoints, face_aligned);
    }
    inline int ExtractFeature(const cv::Mat& img_face, std::vector<float>* feat) {
        return recognizer_->ExtractFeature(img_face, feat);
    }

    inline int Insert(const std::vector<float>& feat, const std::string& name) {
        return database_->Insert(feat, name);
    }
    inline int Delete(const std::string& name) {
        return database_->Delete(name);
    }
	inline int64_t QueryTop(const std::vector<float>& feat, QueryResult *query_result = nullptr) {
        return database_->QueryTop(feat, query_result);
    }
    inline int Save() {
        return  database_->Save(db_name_.c_str());
    }
    inline int Load() {
        return database_->Load(db_name_.c_str());
    }

private:
    DetecterFactory* detecter_factory_ = nullptr;
    LandmarkerFactory* landmarker_factory_ = nullptr;
    RecognizerFactory* recognizer_factory_ = nullptr;

private:
    bool initialized_;
    std::string db_name_;
    Aligner* aligner_ = nullptr;
    Detecter* detecter_ = nullptr;
	Tracker* tracker_ = nullptr;
    Landmarker* landmarker_ = nullptr;
    Recognizer* recognizer_ = nullptr;
    FaceDatabase* database_ = nullptr;
};

FaceEngine::FaceEngine() {
    impl_ = new FaceEngine::Impl();
}

FaceEngine::~FaceEngine() {
    if (impl_) {
        delete impl_;
        impl_ = nullptr;
    }
}

int FaceEngine::LoadModel(const char* root_path) {
    return impl_->LoadModel(root_path);
}

int FaceEngine::Track(const std::vector<FaceInfo>& curr_faces,
	std::vector<TrackedFaceInfo>* faces) {
	
}

int FaceEngine::DetectFace(const cv::Mat& img_src, std::vector<FaceInfo>* faces) {
    return impl_->DetectFace(img_src, faces);
}

int FaceEngine::ExtractKeypoints(const cv::Mat& img_src,
	const cv::Rect& face, std::vector<cv::Point2f>* keypoints) {
    return impl_->ExtractKeypoints(img_src, face, keypoints);
}

int FaceEngine::AlignFace(const cv::Mat& img_src, const std::vector<cv::Point2f>& keypoints, cv::Mat* face_aligned) {
    return impl_->AlignFace(img_src, keypoints, face_aligned);
}

int FaceEngine::ExtractFeature(const cv::Mat& img_face, std::vector<float>* feat) {
    return impl_->ExtractFeature(img_face, feat);
}

int FaceEngine::Insert(const std::vector<float>& feat, const std::string& name) {
    return impl_->Insert(feat, name);
}

int FaceEngine::Delete(const std::string& name) {
    return impl_->Delete(name);
}

int64_t FaceEngine::QueryTop(const std::vector<float>& feat,
    QueryResult* query_result) {
    return impl_->QueryTop(feat, query_result);
}

int FaceEngine::Save() {
    return impl_->Save();
}

int FaceEngine::Load() {
    return impl_->Load();
}

}