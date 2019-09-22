#ifndef _FACE_LANDMARKER_H_
#define _FACE_LANDMARKER_H_

#include "opencv2/core.hpp"

class Landmarker {
public:
	Landmarker();
	virtual ~Landmarker();
	virtual int LoadModel(const char* root_path);
	virtual int ExtractKeypoints(const cv::Mat& img_src,
		const cv::Rect& face, std::vector<cv::Point2f>* keypoints);
};


#endif // !_FACE_LANDMARKER_H_

