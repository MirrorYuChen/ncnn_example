#include "detector.h"
#include "centerface/centerface.h"
#include "mtcnn/mtcnn.h"
#include "retinaface/retinaface.h"

namespace mirror {
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
