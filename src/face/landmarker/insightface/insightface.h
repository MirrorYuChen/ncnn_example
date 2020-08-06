#ifndef _FACE_INSIGHTFACE_LANDMARKER_H_
#define _FACE_INSIGHTFACE_LANDMARKER_H_

#include "../landmarker.h"
#include "ncnn/net.h"

namespace mirror {
class InsightfaceLandmarker : public Landmarker {
public:
	InsightfaceLandmarker();
	~InsightfaceLandmarker();

	int LoadModel(const char* root_path);
	int ExtractKeypoints(const cv::Mat& img_src,
		const cv::Rect& face, std::vector<cv::Point2f>* keypoints);

private:
	ncnn::Net* insightface_landmarker_net_;
	bool initialized;
};

}

#endif // !_FACE_INSIGHTFACE_LANDMARKER_H_

