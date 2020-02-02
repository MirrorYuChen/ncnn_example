#include "detector.h"
#include "centerface/centerface.h"
#include "mtcnn/mtcnn.h"
#include "retinaface/retinaface.h"

namespace mirror {
Detector::Detector() {

}

Detector::~Detector() {

}

int Detector::LoadModel(const char * root_path) {
	return 0;
}

int Detector::Detect(const cv::Mat & img_src, std::vector<FaceInfo>* faces) {
	return 0;
}

Detector* CenterfaceFactory::CreateDetector() {
	return new CenterFace();
}

Detector* MtcnnFactory::CreateDetector() {
	return new Mtcnn();
}

Detector* RetinafaceFactory::CreateDetector() {
	return new RetinaFace();
}


}
