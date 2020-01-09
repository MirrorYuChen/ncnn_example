#include "mtcnn.h"
#include <iostream>
#include "opencv2/imgproc.hpp"

Mtcnn::Mtcnn() :
	pnet_(new ncnn::Net()),
	rnet_(new ncnn::Net()),
	onet_(new ncnn::Net()),
	pnet_size_(12),
	min_face_size_(40),
	scale_factor_(0.709f),
	initialized_(false) {}

Mtcnn::~Mtcnn() {
	pnet_->clear();
	rnet_->clear();
	onet_->clear();
}

int Mtcnn::LoadModel(const char * root_path) {
	std::string pnet_param = std::string(root_path) + "/pnet.param";
	std::string pnet_bin = std::string(root_path) + "/pnet.bin";
	if (pnet_->load_param(pnet_param.c_str()) == -1 ||
		pnet_->load_model(pnet_bin.c_str()) == -1) {
		std::cout << "Load pnet model failed." << std::endl;
		std::cout << "pnet param: " << pnet_param << std::endl;
		std::cout << "pnet bin: " << pnet_bin << std::endl;
		return 10000;
	}
	std::string rnet_param = std::string(root_path) + "/rnet.param";
	std::string rnet_bin = std::string(root_path) + "/rnet.bin";
	if (rnet_->load_param(rnet_param.c_str()) == -1 ||
		rnet_->load_model(rnet_bin.c_str()) == -1) {
		std::cout << "Load rnet model failed." << std::endl;
		std::cout << "rnet param: " << rnet_param << std::endl;
		std::cout << "rnet bin: " << rnet_bin << std::endl;
		return 10000;
	}
	std::string onet_param = std::string(root_path) + "/onet.param";
	std::string onet_bin = std::string(root_path) + "/onet.bin";
	if (onet_->load_param(onet_param.c_str()) == -1 ||
		onet_->load_model(onet_bin.c_str()) == -1) {
		std::cout << "Load onet model failed." << std::endl;
		std::cout << "onet param: " << onet_param << std::endl;
		std::cout << "onet bin: " << onet_bin << std::endl;
		return 10000;
	}
	initialized_ = true;
	return 0;
}

int Mtcnn::Detect(const cv::Mat & img_src,
	std::vector<FaceInfo>* faces) {
	if (img_src.empty()) {
		std::cout << "input empty." << std::endl;
		return 10001;
	}
	if (!initialized_) {
		std::cout << "model unintialized." << std::endl;
		return 10000;
	}
	cv::Size max_size = cv::Size(img_src.cols, img_src.rows);
	cv::Mat img_cpy = img_src.clone();
	std::vector<FaceInfo> first_bboxes, second_bboxes;
	std::vector<FaceInfo> first_bboxes_result;
	PDetect(img_cpy, &first_bboxes);
	NMS(first_bboxes, &first_bboxes_result, nms_threshold_[0]);
	Refine(&first_bboxes_result, max_size);

	RDetect(img_cpy, first_bboxes_result, &second_bboxes);
	std::vector<FaceInfo> second_bboxes_result;
	NMS(second_bboxes, &second_bboxes_result, nms_threshold_[1]);
	Refine(&second_bboxes_result, max_size);

	std::vector<FaceInfo> third_bboxes;
	ODetect(img_cpy, second_bboxes_result, &third_bboxes);
	NMS(third_bboxes, faces, nms_threshold_[2], "MIN");
	Refine(faces, max_size);
	return 0;
}

int Mtcnn::PDetect(const cv::Mat & img_src,
	std::vector<FaceInfo>* first_bboxes) {
	first_bboxes->clear();
	int width = img_src.cols;
	int height = img_src.rows;
	float min_side = MIN(width, height);
	float curr_scale = float(pnet_size_) / min_face_size_;
	min_side *= curr_scale;
	std::vector<float> scales;
	while (min_side > pnet_size_) {
		scales.push_back(curr_scale);
		min_side *= scale_factor_;
		curr_scale *= scale_factor_;
	}

	// mutiscale resize the image
	for (int i = 0; i < static_cast<size_t>(scales.size()); ++i) {
		int w = static_cast<int>(width * scales[i]);
		int h = static_cast<int>(height * scales[i]);
		cv::Mat img_resized;
		cv::resize(img_src, img_resized, cv::Size(w, h));
		std::cout << "width: " << w << " height: " << h << std::endl;
		ncnn::Mat in = ncnn::Mat::from_pixels(img_resized.data,
			ncnn::Mat::PIXEL_BGR2RGB, img_resized.cols, img_resized.rows);
		in.substract_mean_normalize(meanVals, normVals);
		ncnn::Extractor ex = pnet_->create_extractor();
		//ex.set_num_threads(2);
		ex.set_light_mode(true);
		ex.input("data", in);
		ncnn::Mat score_mat, location_mat;
		ex.extract("prob1", score_mat);
		ex.extract("conv4-2", location_mat);
		const int stride = 2;
		const int cell_size = 12;
		for (int row = 0; row < score_mat.h; ++row) {
			for (int col = 0; col < score_mat.w; ++col) {
				int index = row * score_mat.w + col;
				// pnet output: 1x1x2  no-face && face
				// face score: channel(1)
				float score = score_mat.channel(1)[index];
				if (score < threshold_[0]) continue;

				// 1. generated bounding box
				int x1 = round((stride * col + 1) / scales[i]);
				int y1 = round((stride * row + 1) / scales[i]);
				int x2 = round((stride * col + 1 + cell_size) / scales[i]);
				int y2 = round((stride * row + 1 + cell_size) / scales[i]);

				// 2. regression bounding box
				float x1_reg = location_mat.channel(0)[index];
				float y1_reg = location_mat.channel(1)[index];
				float x2_reg = location_mat.channel(2)[index];
				float y2_reg = location_mat.channel(3)[index];

				int bbox_width = x2 - x1 + 1;
				int bbox_height = y2 - y1 + 1;

				FaceInfo face_info;
				face_info.score_ = score;
				face_info.face_.x = x1 + x1_reg * bbox_width;
				face_info.face_.y = y1 + y1_reg * bbox_height;
				face_info.face_.width = x2 + x2_reg * bbox_width - face_info.face_.x;
				face_info.face_.height = y2 + y2_reg * bbox_height - face_info.face_.y;

				if (face_info.face_.x + face_info.face_.width >= width - 1 ||
					face_info.face_.y + face_info.face_.height >= height - 1) {
					continue;
				}
				first_bboxes->push_back(face_info);
			}
		}
	}
	return 0;
}

int Mtcnn::RDetect(const cv::Mat & img_src,
	const std::vector<FaceInfo>& first_bboxes,
	std::vector<FaceInfo>* second_bboxes) {
	second_bboxes->clear();
	for (int i = 0; i < static_cast<int>(first_bboxes.size()); ++i) {
		cv::Rect face = first_bboxes.at(i).face_ & cv::Rect(0,0, img_src.cols, img_src.rows);
		std::cout << "rnet input: " << face << std::endl;
		cv::Mat img_face = img_src(face).clone();
		ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_face.data,
			ncnn::Mat::PIXEL_BGR2RGB, img_face.cols, img_face.rows, 24, 24);
		in.substract_mean_normalize(meanVals, normVals);
		ncnn::Extractor ex = rnet_->create_extractor();
		ex.set_light_mode(true);
		ex.set_num_threads(2);
		ex.input("data", in);
		ncnn::Mat score_mat, location_mat;
		ex.extract("prob1", score_mat);
		ex.extract("conv5-2", location_mat);
		float score = score_mat[1];
		if (score < threshold_[1]) continue;
		float x_reg = location_mat[0];
		float y_reg = location_mat[1];
		float w_reg = location_mat[2];
		float h_reg = location_mat[3];

		FaceInfo face_info;
		face_info.score_ = score;
		face_info.face_.x = face.x + x_reg * face.width;
		face_info.face_.y = face.y + y_reg * face.height;
		face_info.face_.width = face.x + face.width +
			w_reg * face.width - face_info.face_.x;
		face_info.face_.height = face.y + face.height +
			h_reg * face.height - face_info.face_.y;
		second_bboxes->push_back(face_info);
	}
	return 0;
}

int Mtcnn::ODetect(const cv::Mat & img_src,
	const std::vector<FaceInfo>& second_bboxes,
	std::vector<FaceInfo>* third_bboxes) {
	third_bboxes->clear();
	for (int i = 0; i < static_cast<int>(second_bboxes.size()); ++i) {
		cv::Rect face = second_bboxes.at(i).face_ & cv::Rect(0,0, img_src.cols, img_src.rows);
		cv::Mat img_face = img_src(face).clone();
		cv::Mat img_resized;
		cv::resize(img_face, img_resized, cv::Size(48, 48));
		ncnn::Mat in = ncnn::Mat::from_pixels(img_resized.data,
			ncnn::Mat::PIXEL_BGR, img_resized.cols, img_resized.rows);
		in.substract_mean_normalize(meanVals, normVals);
		ncnn::Extractor ex = onet_->create_extractor();
		ex.set_light_mode(true);
		ex.set_num_threads(2);
		ex.input("data", in);
		ncnn::Mat score_mat, location_mat, keypoints_mat;
		ex.extract("prob1", score_mat);
		ex.extract("conv6-2", location_mat);
		ex.extract("conv6-3", keypoints_mat);
		float score = score_mat[1];
		if (score < threshold_[1]) continue;
		float x_reg = location_mat[0];
		float y_reg = location_mat[1];
		float w_reg = location_mat[2];
		float h_reg = location_mat[3];

		FaceInfo face_info;
		face_info.score_ = score;
		face_info.face_.x = face.x + x_reg * face.width;
		face_info.face_.y = face.y + y_reg * face.height;
		face_info.face_.width = face.x + face.width +
			w_reg * face.width - face_info.face_.x;
		face_info.face_.height = face.y + face.height +
			h_reg * face.height - face_info.face_.y;

		for (int num = 0; num < 5; num++) {
			face_info.keypoints_[num] = face.x + face.width * keypoints_mat[num];
			face_info.keypoints_[num + 5] = face.y + face.height * keypoints_mat[num + 5];
		}

		third_bboxes->push_back(face_info);
	}
	return 0;
}

int Mtcnn::Refine(std::vector<FaceInfo>* bboxes, const cv::Size max_size) {
	int num_boxes = static_cast<int>(bboxes->size());
	for (int i = 0; i < num_boxes; ++i) {
		FaceInfo face_info = bboxes->at(i);
		int width = face_info.face_.width;
		int height = face_info.face_.height;
		float max_side = MAX(width, height);

		face_info.face_.x = face_info.face_.x + 0.5 * width - 0.5 * max_side;
		face_info.face_.y = face_info.face_.y + 0.5 * height - 0.5 * max_side;
		face_info.face_.width = max_side;
		face_info.face_.height = max_side;

		face_info.face_.x = MAX(face_info.face_.x, 0);
		face_info.face_.y = MAX(face_info.face_.y, 0);
		if (face_info.face_.x + max_side > max_size.width) {
			face_info.face_.width = max_size.width - 1;
		}
		if (face_info.face_.y + max_side > max_size.height) {
			face_info.face_.height = max_size.height - 1;
		}
		bboxes->at(i) = face_info;
	}
	
	return 0;
}
