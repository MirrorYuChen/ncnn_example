#include "opencv2/opencv.hpp"
#include "face_detect.h"
#include "ncnn/net.h"


int main(int argc, char* argv[]) {
	cv::Mat img_src = cv::imread("../images/test_1.jpg");
	float factor_x = img_src.cols / 300.0f;
	float factor_y = img_src.rows / 300.0f;
	const char* root_path = "../models";

	ncnn::create_gpu_instance();	
	FaceDetector face_detector;
	face_detector.LoadModel(root_path);
	std::vector<FaceInfo> faces;
	face_detector.Detect(img_src, &faces);
	for (int i = 0; i < static_cast<int>(faces.size()); ++i) {
		cv::Rect face = faces.at(i).face_;
		face.x = face.x * factor_x;
		face.y = face.y * factor_y;
		face.width = face.width * factor_x;
		face.height = face.height * factor_y;
		cv::rectangle(img_src, face, cv::Scalar(0, 255, 0), 2);
		std::vector<cv::Point> keypoints;
		face_detector.ExtractKeypoints(img_src, face, &keypoints);
		for (int j = 0; j < static_cast<int>(keypoints.size()); ++j) {
			cv::circle(img_src, keypoints[j], 1, cv::Scalar(0, 0, 255), 1);
		}
	}
	cv::imwrite("../images/result.jpg", img_src);
	cv::imshow("result", img_src);
	cv::waitKey(0);

	ncnn::destroy_gpu_instance();
	return 0;

}
