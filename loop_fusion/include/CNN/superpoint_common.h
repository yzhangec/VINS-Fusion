#pragma once

#include <ATen/ATen.h>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <torch/csrc/api/include/torch/types.h>
#include <torch/csrc/autograd/variable.h>
#include "../src/utility/tic_toc.h"

#define SP_DESC_RAW_LEN 256
// #define USE_PCA

namespace loop_closure {
void getKeyPoints(const cv::Mat &prob, float threshold, std::vector<cv::Point2f> &keypoints,
                  int width, int height, int max_num);
void computeDescriptors(const torch::Tensor &mProb, const torch::Tensor &desc,
                        const std::vector<cv::Point2f> &keypoints,
                        std::vector<float> &local_descriptors, int width, int height,
                        const Eigen::MatrixXf &pca_comp_T, const Eigen::RowVectorXf &pca_mean);
} // namespace loop_closure