#ifndef _RETINAFACE_H_
#define _RETINAFACE_H_

#include "detector.h"
#include "ncnn/net.h"

using ANCHORS = std::vector<cv::Rect>;
class RetinaFace : public Detector {
public:
	RetinaFace();
	~RetinaFace();
	int LoadModel(const char* root_path);
	int Detect(const cv::Mat& img_src, std::vector<FaceInfo>* faces);

private:
	ncnn::Net* retina_net_;
	std::vector<ANCHORS> anchors_generated_;
	bool initialized;
	const int RPNs[3] = { 32, 16, 8 };

};



#endif // !_RETINAFACE_H_

