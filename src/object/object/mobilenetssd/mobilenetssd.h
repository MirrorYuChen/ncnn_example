#ifndef _OBJECT_MOBILENETSSD_H_
#define _OBJECT_MOBILENETSSD_H_

#include "../object_detecter.h"
#include "ncnn/net.h"

namespace mirror {
class MobilenetSSD : public ObjectDetecter {
public:
	MobilenetSSD();
	~MobilenetSSD();

	int LoadModel(const char* model_path);
	int DetectObject(const cv::Mat& img_src, std::vector<ObjectInfo>* objects);

private:
	ncnn::Net* mobilenetssd_ = nullptr;
	bool initialized_;
	const float meanVals[3] = { 0.5f, 0.5f, 0.5f };
	const float normVals[3] = { 0.007843f, 0.007843f, 0.007843f };
	const float scoreThreshold_ = 0.7f;
	const float nmsThreshold_ = 0.5f;
	std::vector<std::string> class_names = { 
		"background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair",
		"cow", "diningtable", "dog", "horse",
		"motorbike", "person", "pottedplant",
		"sheep", "sofa", "train", "tvmonitor"
	};
};

}
#endif // !_OBJECT_MOBILENETSSD_H_

