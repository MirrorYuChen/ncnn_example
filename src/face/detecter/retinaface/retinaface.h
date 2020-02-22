#ifndef _RETINAFACE_H_
#define _RETINAFACE_H_

#include "../detecter.h"
#include "ncnn/net.h"

namespace mirror {
using ANCHORS = std::vector<cv::Rect>;
class RetinaFace : public Detecter {
public:
	RetinaFace();
	~RetinaFace();
	int LoadModel(const char* root_path);
	int DetectFace(const cv::Mat& img_src, std::vector<FaceInfo>* faces);

private:
	ncnn::Net* retina_net_;
	std::vector<ANCHORS> anchors_generated_;
	bool initialized_;
	const int RPNs[3] = { 32, 16, 8 };
	const float iouThreshold_ = 0.4f;

};

}

#endif // !_RETINAFACE_H_

