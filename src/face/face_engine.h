#include <vector>
#include "../common/common.h"
#include "opencv2/core.hpp"

namespace mirror {
class FaceEngine {
public:
	FaceEngine();
	~FaceEngine();
	int LoadModel(const char* root_path);
	int DetectFace(const cv::Mat& img_src, std::vector<FaceInfo>* faces);
	int Track(const std::vector<FaceInfo>& curr_faces,
		std::vector<TrackedFaceInfo>* faces);
	int ExtractKeypoints(const cv::Mat& img_src,
		const cv::Rect& face, std::vector<cv::Point2f>* keypoints);
	int ExtractFeature(const cv::Mat& img_face, std::vector<float>* feature);
	int AlignFace(const cv::Mat& img_src, const std::vector<cv::Point2f>& keypoints, cv::Mat* face_aligned);

	// database operation
    int Insert(const std::vector<float>& feat, const std::string& name);
	int Delete(const std::string& name);
	int64_t QueryTop(const std::vector<float>& feat, QueryResult *query_result = nullptr);
    int Save();
    int Load();


private:
	class Impl;
	Impl* impl_;

};

}
