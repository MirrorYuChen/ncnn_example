#include "classifier_engine.h"
#include "classifier/classifier.h"
#include "classifier/mobilenet/mobilenet.h"

namespace mirror {

class ClassifierEngine::Impl {
public:
	Impl() {
		classifier_ = new Mobilenet();
	}
	~Impl() {
		if (classifier_) {
			delete classifier_;
			classifier_ = nullptr;
		}
	}
	int LoadModel(const char* root_path);
	int Classify(const cv::Mat& img_src, std::vector<ImageInfo>* images);

private:
	Classifier* classifier_;

};

ClassifierEngine::ClassifierEngine() {
	impl_ = new ClassifierEngine::Impl();
}

ClassifierEngine::~ClassifierEngine() {
	if (impl_) {
		delete impl_;
		impl_ = nullptr;
	}
}

int ClassifierEngine::LoadModel(const char * root_path) {
	return impl_->LoadModel(root_path);
}

int ClassifierEngine::Classify(const cv::Mat & img_src,
	std::vector<ImageInfo>* images) {
	return impl_->Classify(img_src, images);
}



int ClassifierEngine::Impl::LoadModel(const char * root_path) {
	return classifier_->LoadModel(root_path);
}

int ClassifierEngine::Impl::Classify(const cv::Mat & img_src, std::vector<ImageInfo>* images) {
	return classifier_->Classify(img_src, images);
}

}


