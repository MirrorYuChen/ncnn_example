#include "face_detect.h"
#include <iostream>
#include <string>
#include "ncnn/net.h"
#include "common.h"

using ANCHORS = std::vector<cv::Rect>;
class FaceDetector::Impl {
 public:
    Impl() : fdnet_(new ncnn::Net()),
             flnet_(new ncnn::Net()),
             initialized(false) {
		ncnn::create_gpu_instance();	
	}
    ~Impl() {
		fdnet_->clear();
		flnet_->clear();
		ncnn::destroy_gpu_instance();
	}

    ncnn::Net* fdnet_;
    ncnn::Net* flnet_;
    bool initialized;

	const float flMeanVals[3] = { 127.5f, 127.5f, 127.5f };
	const float flNormVals[3] = { 0.0078125f, 0.0078125f, 0.0078125f };
	const int RPNs[3] = {32, 16, 8};
    int LoadModel(const char* root_path);
	int Detect(const cv::Mat& img_src, std::vector<FaceInfo>* faces);
	int ExtractKeypoints(const cv::Mat& img_src,
		const cv::Rect& face, std::vector<cv::Point>* keypoints);

 private:
	std::vector<ANCHORS> anchors_generated_;

};

int FaceDetector::Impl::LoadModel(const char * root_path) {
	std::string fd_param = std::string(root_path) + "/fd.param";
	std::string fd_bin = std::string(root_path) + "/fd.bin";
	fdnet_->opt.use_vulkan_compute = 1;
	if (fdnet_->load_param(fd_param.c_str()) == -1 ||
		fdnet_->load_model(fd_bin.c_str()) == -1) {
		std::cout << "load face detect model failed." << std::endl;
		return 10000;
	}
	std::string fl_param = std::string(root_path) + "/lnet112.param";
	std::string fl_bin = std::string(root_path) + "/lnet112.bin";
	flnet_->opt.use_vulkan_compute = 1;
	if (flnet_->load_param(fl_param.c_str()) == -1 ||
		flnet_->load_model(fl_bin.c_str()) == -1) {
		std::cout << "load face landmark model failed." << std::endl;
		return 10000;
	}

	// generate anchors
	for (int i = 0; i < 3; ++i) {
		ANCHORS anchors;
		if (0 == i) {
			GenerateAnchors(16, { 1.0f }, { 32, 16 }, &anchors);
		}
		else if (1 == i) {
			GenerateAnchors(16, { 1.0f }, { 8, 4 }, &anchors);
		}
		else {
			GenerateAnchors(16, { 1.0f }, { 2, 1 }, &anchors);
		}
		anchors_generated_.push_back(anchors);
	}

	initialized = true;
	return 0;
}

int FaceDetector::Impl::Detect(const cv::Mat & img_src,
	std::vector<FaceInfo>* faces) {
	faces->clear();
	std::cout << "start face detect." << std::endl;
	if (!initialized) {
		std::cout << "model uninitialized." << std::endl;
		return 10000;
	}
	if (img_src.empty()) {
		std::cout << "input empty." << std::endl;
		return 10001;
	}
	cv::Mat img_cpy = img_src.clone();
	ncnn::Extractor ex = fdnet_->create_extractor();
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_cpy.data,
		ncnn::Mat::PIXEL_BGR2RGB, img_cpy.cols, img_cpy.rows, 300, 300);
	ex.input("data", in);

	std::vector<float> scores;
	std::vector<FaceInfo> faces_tmp;
	for (int i = 0; i < 3; ++i) {
		std::string class_layer_name = "face_rpn_cls_prob_reshape_stride" + std::to_string(RPNs[i]);
		std::string bbox_layer_name = "face_rpn_bbox_pred_stride" + std::to_string(RPNs[i]);
		std::string landmark_layer_name = "face_rpn_landmark_pred_stride" + std::to_string(RPNs[i]);

		ncnn::Mat class_mat, bbox_mat, landmark_mat;
		ex.extract(class_layer_name.c_str(), class_mat);
		ex.extract(bbox_layer_name.c_str(), bbox_mat);
		ex.extract(landmark_layer_name.c_str(), landmark_mat);

		ANCHORS anchors = anchors_generated_.at(i);
		int width = class_mat.w;
		int height = class_mat.h;
		int anchor_num = static_cast<int>(anchors.size());
		for (int h = 0; h < height; ++h)	{
			for (int w = 0; w < width; ++w) {
				int index = h * width + w;
				for (int a = 0; a < anchor_num; ++a) {
					float score = class_mat.channel(anchor_num + a)[index];
					if (score < 0.8) {
						continue;
					}
					scores.push_back(score);
					cv::Rect box = cv::Rect(w * RPNs[i]+ anchors[a].x,
						h * RPNs[i] + anchors[a].y,
						anchors[a].width,
						anchors[a].height);

					float delta_x = bbox_mat.channel(a * 4 + 0)[index];
					float delta_y = bbox_mat.channel(a * 4 + 1)[index];
					float delta_w = bbox_mat.channel(a * 4 + 2)[index];
					float delta_h = bbox_mat.channel(a * 4 + 3)[index];
					cv::Point2f center = cv::Point2f(box.x + box.width * 0.5f,
						box.y + box.height * 0.5f);
					center.x = center.x + delta_x * box.width;
					center.y = center.y + delta_y * box.height;
					float curr_width = std::exp(delta_w) * (box.width + 1);
					float curr_height = std::exp(delta_h) * (box.height + 1);
					cv::Rect curr_box = cv::Rect(
						center.x - curr_width * 0.5f,
						center.y - curr_height* 0.5f,
						curr_width,
						curr_height);
					curr_box.x = MAX(curr_box.x, 0);
					curr_box.y = MAX(curr_box.y, 0);
					curr_box.width = MIN(300 - curr_box.x, curr_box.width);
					curr_box.height = MIN(300 - curr_box.y, curr_box.height);

					FaceInfo face(curr_box, score);
					faces_tmp.push_back(face);
				}
			}
		}
	}

	std::sort(faces_tmp.begin(), faces_tmp.end(), [](FaceInfo face1, FaceInfo face2) { return face1.score_ < face2.score_; });
	NMS(faces_tmp, faces);
	std::cout << faces->size() << " faces detected." << std::endl;

	return 0;
}

int FaceDetector::Impl::ExtractKeypoints(const cv::Mat & img_src,
	const cv::Rect & face, std::vector<cv::Point>* keypoints) {
	std::cout << "start keypoints extract." << std::endl;
	if (!initialized) {
		std::cout << "model uninitialized." << std::endl;
		return 10000;
	}
	if (img_src.empty()) {
		std::cout << "input empty." << std::endl;
		return 10001;
	}
	keypoints->clear();
	cv::Mat img_face = img_src(face).clone();
	ncnn::Extractor ex = flnet_->create_extractor();
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_face.data,
		ncnn::Mat::PIXEL_BGR, img_face.cols, img_face.rows, 112, 112);
	in.substract_mean_normalize(flMeanVals, flNormVals);
	ex.input("data", in);
	ncnn::Mat out;
	ex.extract("bn6_3", out);
	for (int i = 0; i < 106; ++i) {
		float x = abs(out[2 * i] * img_face.cols) + face.x;
		float y = abs(out[2 * i + 1] * img_face.rows) + face.y;
		keypoints->push_back(cv::Point(x, y));
	}
	std::cout << "keypoints extract end." << std::endl;
	return 0;
}

FaceDetector::FaceDetector() {
	impl = new FaceDetector::Impl();
}

FaceDetector::~FaceDetector() {
	if (impl) {
		delete impl;
	}
}

int FaceDetector::LoadModel(const char * root_path) {
	impl->LoadModel(root_path);
	return 0;
}

int FaceDetector::Detect(const cv::Mat & img_src, std::vector<FaceInfo>* faces) {
	impl->Detect(img_src, faces);
	return 0;
}

int FaceDetector::ExtractKeypoints(const cv::Mat & img_src,
	const cv::Rect & face, std::vector<cv::Point>* keypoints) {
	impl->ExtractKeypoints(img_src, face, keypoints);
	return 0;
}
