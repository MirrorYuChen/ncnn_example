#include "aligner.h"
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace mirror {
class Aligner::Impl {
public:
	int AlignFace(const cv::Mat& img_src, const std::vector<cv::Point2f>& keypoints, cv::Mat* face_aligned);

private:
	cv::Mat MeanAxis0(const cv::Mat &src);
	cv::Mat ElementwiseMinus(const cv::Mat &A, const cv::Mat &B);
	cv::Mat VarAxis0(const cv::Mat &src);
	int MatrixRank(cv::Mat M);
	cv::Mat SimilarTransform(const cv::Mat& src, const cv::Mat& dst);

	float points_dst[5][2] = {
		{ 30.2946f + 8.0f, 51.6963f },
		{ 65.5318f + 8.0f, 51.5014f },
		{ 48.0252f + 8.0f, 71.7366f },
		{ 33.5493f + 8.0f, 92.3655f },
		{ 62.7299f + 8.0f, 92.2041f }
	};
};


Aligner::Aligner() {
	impl_ = new Impl();
}

Aligner::~Aligner() {
	if (impl_) {
		delete impl_;
	}
}

int Aligner::AlignFace(const cv::Mat & img_src,
	const std::vector<cv::Point2f>& keypoints, cv::Mat * face_aligned) {
	return impl_->AlignFace(img_src, keypoints, face_aligned);
}

int Aligner::Impl::AlignFace(const cv::Mat & img_src,
	const std::vector<cv::Point2f>& keypoints, cv::Mat * face_aligned) {
	std::cout << "start align face." << std::endl;
	if (img_src.empty()) {
		std::cout << "input empty." << std::endl;
		return 10001;
	}
	if (keypoints.size() == 0) {
		std::cout << "keypoints empty." << std::endl;
		return 10001;
	}

	float points_src[5][2] = {
		{keypoints[104].x, keypoints[104].y},
		{keypoints[105].x, keypoints[105].y},
		{keypoints[46].x,  keypoints[46].y },
		{keypoints[84].x,  keypoints[84].y },
		{keypoints[90].x, keypoints[90].y}
	};

	cv::Mat src_mat(5, 2, CV_32FC1, points_src);
	cv::Mat dst_mat(5, 2, CV_32FC1, points_dst);

	cv::Mat transform = SimilarTransform(src_mat, dst_mat);

	face_aligned->create(112, 112, CV_32FC3);

	cv::Mat transfer_mat = transform(cv::Rect(0, 0, 3, 2));
	cv::warpAffine(img_src.clone(), *face_aligned, transfer_mat, cv::Size(112, 112), 1, 0, 0);

	std::cout << "end align face." << std::endl;
	return 0;
}

cv::Mat Aligner::Impl::MeanAxis0(const cv::Mat & src) {
	int num = src.rows;
	int dim = src.cols;

	// x1 y1
	// x2 y2

	cv::Mat output(1, dim, CV_32FC1);
	for (int i = 0; i < dim; i++) {
		float sum = 0;
		for (int j = 0; j < num; j++) {
			sum += src.at<float>(j, i);
		}
		output.at<float>(0, i) = sum / num;
	}

	return output;
}

cv::Mat Aligner::Impl::ElementwiseMinus(const cv::Mat & A, const cv::Mat & B) {
	cv::Mat output(A.rows, A.cols, A.type());
	assert(B.cols == A.cols);
	if (B.cols == A.cols) {
		for (int i = 0; i < A.rows; i++) {
			for (int j = 0; j < B.cols; j++) {
				output.at<float>(i, j) = A.at<float>(i, j) - B.at<float>(0, j);
			}
		}
	}

	return output;	
}

cv::Mat Aligner::Impl::VarAxis0(const cv::Mat & src) {
	cv::Mat temp_ = ElementwiseMinus(src, MeanAxis0(src));
	cv::multiply(temp_, temp_, temp_);
	return MeanAxis0(temp_);
}

int Aligner::Impl::MatrixRank(cv::Mat M) {
	cv::Mat w, u, vt;
	cv::SVD::compute(M, w, u, vt);
	cv::Mat1b nonZeroSingularValues = w > 0.0001;
	int rank = countNonZero(nonZeroSingularValues);
	return rank;
}

/*
References: "Least-squares estimation of transformation parameters between two point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
Anthor: Jack Yu
*/
cv::Mat Aligner::Impl::SimilarTransform(const cv::Mat & src, const cv::Mat & dst) {
	int num = src.rows;
	int dim = src.cols;
	cv::Mat src_mean = MeanAxis0(src);
	cv::Mat dst_mean = MeanAxis0(dst);
	cv::Mat src_demean = ElementwiseMinus(src, src_mean);
	cv::Mat dst_demean = ElementwiseMinus(dst, dst_mean);
	cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
	cv::Mat d(dim, 1, CV_32F);
	d.setTo(1.0f);
	if (cv::determinant(A) < 0) {
		d.at<float>(dim - 1, 0) = -1;

	}
	cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
	cv::Mat U, S, V;
	cv::SVD::compute(A, S, U, V);

	// the SVD function in opencv differ from scipy .

	int rank = MatrixRank(A);
	if (rank == 0) {
		assert(rank == 0);

	}
	else if (rank == dim - 1) {
		if (cv::determinant(U) * cv::determinant(V) > 0) {
			T.rowRange(0, dim).colRange(0, dim) = U * V;
		}
		else {
			int s = d.at<float>(dim - 1, 0) = -1;
			d.at<float>(dim - 1, 0) = -1;

			T.rowRange(0, dim).colRange(0, dim) = U * V;
			cv::Mat diag_ = cv::Mat::diag(d);
			cv::Mat twp = diag_ * V; //np.dot(np.diag(d), V.T)
			cv::Mat B = cv::Mat::zeros(3, 3, CV_8UC1);
			cv::Mat C = B.diag(0);
			T.rowRange(0, dim).colRange(0, dim) = U * twp;
			d.at<float>(dim - 1, 0) = s;
		}
	}
	else {
		cv::Mat diag_ = cv::Mat::diag(d);
		cv::Mat twp = diag_ * V.t(); //np.dot(np.diag(d), V.T)
		cv::Mat res = U * twp; // U
		T.rowRange(0, dim).colRange(0, dim) = -U.t()* twp;
	}
	cv::Mat var_ = VarAxis0(src_demean);
	float val = cv::sum(var_).val[0];
	cv::Mat res;
	cv::multiply(d, S, res);
	float scale = 1.0 / val * cv::sum(res).val[0];
	T.rowRange(0, dim).colRange(0, dim) = -T.rowRange(0, dim).colRange(0, dim).t();
	cv::Mat  temp1 = T.rowRange(0, dim).colRange(0, dim); // T[:dim, :dim]
	cv::Mat  temp2 = src_mean.t();
	cv::Mat  temp3 = temp1 * temp2;
	cv::Mat temp4 = scale * temp3;
	T.rowRange(0, dim).colRange(dim, dim + 1) = -(temp4 - dst_mean.t());
	T.rowRange(0, dim).colRange(0, dim) *= scale;
	return T;
}

}
