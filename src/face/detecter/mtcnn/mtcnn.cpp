#include "mtcnn.h"
#include <iostream>
#include "opencv2/imgproc.hpp"

#if MIRROR_VULKAN
#include "gpu.h"
#endif // MIRROR_VULKAN

namespace mirror {
Mtcnn::Mtcnn() :
	pnet_(new ncnn::Net()),
	rnet_(new ncnn::Net()),
	onet_(new ncnn::Net()),
	pnet_size_(12),
	min_face_size_(40),
	scale_factor_(0.709f),
	initialized_(false) {
#if MIRROR_VULKAN
	ncnn::create_gpu_instance();	
    pnet_->opt.use_vulkan_compute = true;
	rnet_->opt.use_vulkan_compute = true;
	onet_->opt.use_vulkan_compute = true;
#endif // MIRROR_VULKAN
}

Mtcnn::~Mtcnn() {
	if (pnet_) {
		pnet_->clear();
	}
	if (rnet_) {
		rnet_->clear();
	}
	if (onet_) {
		onet_->clear();
	}
#if MIRROR_VULKAN
	ncnn::destroy_gpu_instance();
#endif // MIRROR_VULKAN	
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

int Mtcnn::DetectFace(const cv::Mat & img_src,
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
	ncnn::Mat img_in = ncnn::Mat::from_pixels(img_src.data,
		ncnn::Mat::PIXEL_BGR2RGB, img_src.cols, img_src.rows);
	img_in.substract_mean_normalize(meanVals, normVals);
	
	std::vector<FaceInfo> first_bboxes, second_bboxes;
	std::vector<FaceInfo> first_bboxes_result;
	PDetect(img_in, &first_bboxes);
	NMS(first_bboxes, &first_bboxes_result, nms_threshold_[0]);
	Refine(&first_bboxes_result, max_size);

	RDetect(img_in, first_bboxes_result, &second_bboxes);
	std::vector<FaceInfo> second_bboxes_result;
	NMS(second_bboxes, &second_bboxes_result, nms_threshold_[1]);
	Refine(&second_bboxes_result, max_size);

	std::vector<FaceInfo> third_bboxes;
	ODetect(img_in, second_bboxes_result, &third_bboxes);
	NMS(third_bboxes, faces, nms_threshold_[2], "MIN");
	Refine(faces, max_size);
	return 0;
}

int Mtcnn::PDetect(const ncnn::Mat & img_in,
	std::vector<FaceInfo>* first_bboxes) {
	first_bboxes->clear();
	int width = img_in.w;
	int height = img_in.h;
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
		ncnn::Mat img_resized;
		ncnn::resize_bilinear(img_in, img_resized, w, h);
		ncnn::Extractor ex = pnet_->create_extractor();
		//ex.set_num_threads(2);
		ex.set_light_mode(true);
		ex.input("data", img_resized);
		ncnn::Mat score_mat, location_mat;
		ex.extract("prob1", score_mat);
		ex.extract("conv4-2", location_mat);
		const int stride = 2;
		const int cell_size = 12;
		for (int h = 0; h < score_mat.h; ++h) {
			for (int w = 0; w < score_mat.w; ++w) {
				int index = h * score_mat.w + w;
				// pnet output: 1x1x2  no-face && face
				// face score: channel(1)
				float score = score_mat.channel(1)[index];
				if (score < threshold_[0]) continue;

				// 1. generated bounding box
				int x1 = round((stride * w + 1) / scales[i]);
				int y1 = round((stride * h + 1) / scales[i]);
				int x2 = round((stride * w + 1 + cell_size) / scales[i]);
				int y2 = round((stride * h + 1 + cell_size) / scales[i]);

				// 2. regression bounding box
				float x1_reg = location_mat.channel(0)[index];
				float y1_reg = location_mat.channel(1)[index];
				float x2_reg = location_mat.channel(2)[index];
				float y2_reg = location_mat.channel(3)[index];

				int bbox_width = x2 - x1 + 1;
				int bbox_height = y2 - y1 + 1;

				FaceInfo face_info;
				face_info.score_ = score;
				face_info.location_.x = x1 + x1_reg * bbox_width;
				face_info.location_.y = y1 + y1_reg * bbox_height;
				face_info.location_.width = x2 + x2_reg * bbox_width - face_info.location_.x;
				face_info.location_.height = y2 + y2_reg * bbox_height - face_info.location_.y;
				face_info.location_ = face_info.location_ & cv::Rect(0, 0, width, height);
				first_bboxes->push_back(face_info);
			}
		}
	}
	return 0;
}

int Mtcnn::RDetect(const ncnn::Mat & img_in,
	const std::vector<FaceInfo>& first_bboxes,
	std::vector<FaceInfo>* second_bboxes) {
	second_bboxes->clear();
	for (int i = 0; i < static_cast<int>(first_bboxes.size()); ++i) {
		cv::Rect face = first_bboxes.at(i).location_ & cv::Rect(0, 0, img_in.w, img_in.h);
		ncnn::Mat img_face, img_resized;
		ncnn::copy_cut_border(img_in, img_face, face.y, img_in.h - face.br().y, face.x, img_in.w - face.br().x);
		ncnn::resize_bilinear(img_face, img_resized, 24, 24);
		ncnn::Extractor ex = rnet_->create_extractor();
		ex.set_light_mode(true);
		ex.set_num_threads(2);
		ex.input("data", img_resized);
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
		face_info.location_.x = face.x + x_reg * face.width;
		face_info.location_.y = face.y + y_reg * face.height;
		face_info.location_.width = face.x + face.width +
			w_reg * face.width - face_info.location_.x;
		face_info.location_.height = face.y + face.height +
			h_reg * face.height - face_info.location_.y;
		second_bboxes->push_back(face_info);
	}
	return 0;
}

int Mtcnn::ODetect(const ncnn::Mat & img_in,
	const std::vector<FaceInfo>& second_bboxes,
	std::vector<FaceInfo>* third_bboxes) {
	third_bboxes->clear();
	for (int i = 0; i < static_cast<int>(second_bboxes.size()); ++i) {
		cv::Rect face = second_bboxes.at(i).location_ & cv::Rect(0, 0, img_in.w, img_in.h);
		ncnn::Mat img_face, img_resized;
		ncnn::copy_cut_border(img_in, img_face, face.y, img_in.h - face.br().y, face.x, img_in.w - face.br().x);
		ncnn::resize_bilinear(img_face, img_resized, 48, 48);

		ncnn::Extractor ex = onet_->create_extractor();
		ex.set_light_mode(true);
		ex.set_num_threads(2);
		ex.input("data", img_resized);
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
		face_info.location_.x = face.x + x_reg * face.width;
		face_info.location_.y = face.y + y_reg * face.height;
		face_info.location_.width = face.x + face.width +
			w_reg * face.width - face_info.location_.x;
		face_info.location_.height = face.y + face.height +
			h_reg * face.height - face_info.location_.y;

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
		int width = face_info.location_.width;
		int height = face_info.location_.height;
		float max_side = MAX(width, height);

		face_info.location_.x = face_info.location_.x + 0.5 * width - 0.5 * max_side;
		face_info.location_.y = face_info.location_.y + 0.5 * height - 0.5 * max_side;
		face_info.location_.width = max_side;
		face_info.location_.height = max_side;
		face_info.location_ = face_info.location_ & cv::Rect(0, 0, max_size.width, max_size.height);
		bboxes->at(i) = face_info;
	}
	
	return 0;
}

}
