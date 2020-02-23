#include "centerface.h"
#include <iostream>
#include "opencv2/imgproc.hpp"

#if MIRROR_VULKAN
#include "gpu.h"
#endif // MIRROR_VULKAN

namespace mirror {
CenterFace::CenterFace() {
    centernet_ = new ncnn::Net();
    initialized_ = false;
#if MIRROR_VULKAN
	ncnn::create_gpu_instance();	
    centernet_->opt.use_vulkan_compute = true;
#endif // MIRROR_VULKAN
}

CenterFace::~CenterFace(){
    if (centernet_) {
        centernet_->clear();
    }
#if MIRROR_VULKAN
	ncnn::destroy_gpu_instance();
#endif // MIRROR_VULKAN	
}

int CenterFace::LoadModel(const char* root_path) {
    std::cout << "start load model." << std::endl;
    std::string param_file = std::string(root_path) + "/centerface.param";
	std::string model_file = std::string(root_path) + "/centerface.bin";
    if (centernet_->load_param(param_file.c_str()) == -1 ||
        centernet_->load_model(model_file.c_str()) == -1) {
        std::cout << "load model failed." << std::endl;
        return 10000;
	}

    initialized_ = true;
    std::cout << "end load model." << std::endl;
    return 0;
}

int CenterFace::DetectFace(const cv::Mat& img_src,
    std::vector<FaceInfo>* faces) {
    std::cout << "start detect." << std::endl;
    faces->clear();
    if (!initialized_) {
        std::cout << "model uninitialized." << std::endl;
        return 10000;
    }
    if (img_src.empty()) {
        std::cout << "input empty." << std::endl;
        return 10001;
    }
    int img_width = img_src.cols;
	int img_height = img_src.rows;

	int img_width_new  = img_width / 32 * 32;
	int img_height_new = img_height / 32 * 32;
    float scale_x = static_cast<float>(img_width)  / img_width_new;
    float scale_y = static_cast<float>(img_height) / img_height_new;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_src.data, ncnn::Mat::PIXEL_BGR2RGB,
		img_width, img_height, img_width_new, img_height_new);
	ncnn::Extractor ex = centernet_->create_extractor();
	ex.input("input.1", in);
	ncnn::Mat mat_heatmap, mat_scale, mat_offset, mat_landmark;
	ex.extract("537", mat_heatmap);
	ex.extract("538", mat_scale);
	ex.extract("539", mat_offset);
	ex.extract("540", mat_landmark);

    int height = mat_heatmap.h;
	int width = mat_heatmap.w;
    std::vector<FaceInfo> faces_tmp;
	for (int h = 0; h < height; ++h) {
		for (int w = 0; w < width; ++w) {
			int index = h * width + w;
			float score = mat_heatmap[index];
			if (score < scoreThreshold_) {
				continue;
			}
			float s0 = 4 * exp(mat_scale.channel(0)[index]);
			float s1 = 4 * exp(mat_scale.channel(1)[index]);
			float o0 = mat_offset.channel(0)[index];
			float o1 = mat_offset.channel(1)[index];

			float ymin = MAX(0, 4 * (h + o0 + 0.5) - 0.5 * s0);
			float xmin = MAX(0, 4 * (w + o1 + 0.5) - 0.5 * s1);
			float ymax = MIN(ymin + s0, img_height_new);
			float xmax = MIN(xmin + s1, img_width_new);

            FaceInfo face_info;
            face_info.score_ = score;
            face_info.location_.x = scale_x * xmin;
            face_info.location_.y = scale_y * ymin;
            face_info.location_.width = scale_x * (xmax - xmin);
            face_info.location_.height = scale_y * (ymax - ymin);

            for (int num = 0; num < 5; ++num) {
                face_info.keypoints_[num    ] = scale_x * (s1 * mat_landmark.channel(2 * num + 1)[index] + xmin);
                face_info.keypoints_[num + 5] = scale_y * (s0 * mat_landmark.channel(2 * num + 0)[index] + ymin);
            }
            faces_tmp.push_back(face_info);
		}
	}
    NMS(faces_tmp, faces, nmsThreshold_);
    std::cout << "end detect." << std::endl;
    return 0;
}

}
