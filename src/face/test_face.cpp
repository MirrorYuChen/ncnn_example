#include "opencv2/opencv.hpp"
#include "face_engine.h"

using namespace mirror;

int TestLandmark(int argc, char* argv[]) {
	const char* img_file = "../../data/images/4.jpg";
	cv::Mat img_src = cv::imread(img_file);
	const char* root_path = "../../data/models";

	double start = static_cast<double>(cv::getTickCount());
	
	FaceEngine* face_engine = new FaceEngine();
	face_engine->LoadModel(root_path);
	std::vector<FaceInfo> faces;
	face_engine->DetectFace(img_src, &faces);
	for (int i = 0; i < static_cast<int>(faces.size()); ++i) {
		cv::Rect face = faces.at(i).location_;
		cv::rectangle(img_src, face, cv::Scalar(0, 255, 0), 2);
		std::vector<cv::Point2f> keypoints;
		face_engine->ExtractKeypoints(img_src, face, &keypoints);
		for (int j = 0; j < static_cast<int>(keypoints.size()); ++j) {
			cv::circle(img_src, keypoints[j], 1, cv::Scalar(0, 0, 255), 1);
		}
	}

	double end = static_cast<double>(cv::getTickCount());
	double time_cost = (end - start) / cv::getTickFrequency() * 1000;
	std::cout << "time cost: " << time_cost << "ms" << std::endl;
	cv::imwrite("../../images/result.jpg", img_src);
	cv::imshow("result", img_src);
	cv::waitKey(0);

	delete face_engine;
	face_engine = nullptr;

	return 0;

}

int TestRecognize(int argc, char* argv[]) {
	const char* img_file = "../data/images/4.jpg";
	cv::Mat img_src = cv::imread(img_file);
	const char* root_path = "../data/models";

	double start = static_cast<double>(cv::getTickCount());
	FaceEngine* face_engine = new FaceEngine();
	face_engine->LoadModel(root_path);
	std::vector<FaceInfo> faces;
	face_engine->DetectFace(img_src, &faces);

	cv::Mat face1 = img_src(faces[0].location_).clone();
	cv::Mat face2 = img_src(faces[1].location_).clone();
	std::vector<float> feature1, feature2;
	face_engine->ExtractFeature(face1, &feature1);
	face_engine->ExtractFeature(face2, &feature2);
	float sim = CalculateSimilarity(feature1, feature2);

	double end = static_cast<double>(cv::getTickCount());
	double time_cost = (end - start) / cv::getTickFrequency() * 1000;
	std::cout << "time cost: " << time_cost << "ms" << std::endl;

	for (int i = 0; i < static_cast<int>(faces.size()); ++i) {
		cv::Rect face = faces.at(i).location_;
		cv::rectangle(img_src, face, cv::Scalar(0, 255, 0), 2);		
	}
	cv::imwrite("../data/images/face1.jpg", face1);
	cv::imwrite("../data/images/face2.jpg", face2);
	cv::imwrite("result.jpg", img_src);
	std::cout << "similarity is: " << sim << std::endl;

	delete face_engine;
	face_engine = nullptr;
	return 0;

}

int TestAlignFace(int argc, char* argv[]) {
	const char* img_file = "../../data/images/4.jpg";
	cv::Mat img_src = cv::imread(img_file);
	const char* root_path = "../../data/models";

	double start = static_cast<double>(cv::getTickCount());
	
	FaceEngine* face_engine = new FaceEngine();
	face_engine->LoadModel(root_path);
	std::vector<FaceInfo> faces;
	face_engine->DetectFace(img_src, &faces);
	for (int i = 0; i < static_cast<int>(faces.size()); ++i) {
		cv::Rect face = faces.at(i).location_;
		std::vector<cv::Point2f> keypoints;
		face_engine->ExtractKeypoints(img_src, face, &keypoints);
		cv::Mat face_aligned;
		face_engine->AlignFace(img_src, keypoints, &face_aligned);
		std::string name = std::to_string(i) + ".jpg";
		cv::imwrite(name.c_str(), face_aligned);
		for (int j = 0; j < static_cast<int>(keypoints.size()); ++j) {
			cv::circle(img_src, keypoints[j], 1, cv::Scalar(0, 0, 255), 1);
		}
		cv::rectangle(img_src, face, cv::Scalar(0, 255, 0), 2);
	}
	cv::imshow("result", img_src);
	cv::waitKey(0);

	delete face_engine;
	face_engine = nullptr;
	
	return 0;
}

int TestDetecter(int argc, char* argv[]) {
	const char* img_file = "../../data/images/4.jpg";
	cv::Mat img_src = cv::imread(img_file);
	const char* root_path = "../../data/models";

	FaceEngine* face_engine = new FaceEngine();
	face_engine->LoadModel(root_path);
	std::vector<FaceInfo> faces;
	double start = static_cast<double>(cv::getTickCount());
	face_engine->DetectFace(img_src, &faces);
	double end = static_cast<double>(cv::getTickCount());
	double time_cost = (end - start) / cv::getTickFrequency() * 1000;
	std::cout << "time cost: " << time_cost << "ms" << std::endl;

	for (int i = 0; i < static_cast<int>(faces.size()); ++i) {
		FaceInfo face_info = faces.at(i);
		cv::rectangle(img_src, face_info.location_, cv::Scalar(0, 255, 0), 2);
#if 0
		for (int num = 0; num < 5; ++num) {
			cv::Point curr_pt = cv::Point(face_info.keypoints_[num],
										  face_info.keypoints_[num + 5]);
			cv::circle(img_src, curr_pt, 2, cv::Scalar(255, 0, 255), 2);
		}	
#endif 
	}
	cv::imwrite("../../data/images/mtcnn_result.jpg", img_src);
	cv::imshow("result", img_src);
	cv::waitKey(0);

	delete face_engine;
	face_engine = nullptr;

	return 0;
}

int TestTrack(int argc, char* argv[]) {
	const char* img_file = "../../data/images/4.jpg";
	cv::Mat img_src = cv::imread(img_file);
	cv::VideoCapture cam(0);
	if (!cam.isOpened()) {
		std::cout << "open camera failed." << std::endl;
		return -1;
	}

	const char* root_path = "../../data/models";
	FaceEngine* face_engine = new FaceEngine();
	face_engine->LoadModel(root_path);

	cv::Mat frame;
	while (true) {
		cam >> frame;
		if (frame.empty()) {
			continue;
		}
		std::vector<FaceInfo> curr_faces;
		face_engine->DetectFace(frame, &curr_faces);
		std::vector<TrackedFaceInfo> faces;
		face_engine->Track(curr_faces, &faces);

		for (int i = 0; i < static_cast<int>(faces.size()); ++i) {
			TrackedFaceInfo tracked_face_info = faces.at(i);
			cv::rectangle(frame, tracked_face_info.face_info_.location_, cv::Scalar(0, 255, 0), 2);
		}

		cv::imshow("result", frame);
		if (cv::waitKey(60) == 'q') {
			break;
		}
	}

	delete face_engine;
	face_engine = nullptr;

	return 0;
}

int TestDatabase(int argc, char* argv[]) {
    const char* img_path = "../../data/images/4.jpg";
    cv::Mat img_src = cv::imread(img_path);
    if (img_src.empty()) {
        std::cout << "load image failed." << std::endl;
        return 10001;
    }

    const char* root_path = "../../data/models";
    FaceEngine* face_engine = new FaceEngine();
    face_engine->LoadModel(root_path);
    face_engine->Load();
    std::vector<FaceInfo> faces;
    face_engine->DetectFace(img_src, &faces);

    int faces_num = static_cast<int>(faces.size());
    std::cout << "faces number: " << faces_num << std::endl;
    for (int i = 0; i < faces_num; ++i) {
        cv::Rect face = faces.at(i).location_;
		cv::rectangle(img_src, face, cv::Scalar(0, 255, 0), 2);
        std::vector<float> feat;
        face_engine->ExtractFeature(img_src(face).clone(), &feat);

#if 1
        face_engine->Insert(feat, "face" + std::to_string(i));
#endif

#if 0
        QueryResult query_result;
        face_engine->QueryTop(feat, &query_result);
        std::cout << i << "-th face is: " << query_result.name_ <<
            " similarity is: " << query_result.sim_ << std::endl;   
#endif

    }
    face_engine->Save();
    cv::imwrite("../../data/images/result.jpg", img_src);

	delete face_engine;
	face_engine = nullptr;

    return 0;
}


int main(int argc, char* argv[]) {
	return TestLandmark(argc, argv);
	// return TestRecognize(argc, argv);
	// return TestAlignFace(argc, argv);
	// return TestDetecter(argc, argv);
	// return TestTrack(argc, argv);
	// return TestDatabase(argc, argv);
}
