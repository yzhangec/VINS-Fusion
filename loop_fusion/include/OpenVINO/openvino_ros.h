#include <iostream>
#include <thread>
#include <vector>

#include <Eigen/Eigen>
#include <ros/ros.h>

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "OpenVINO/superpoint_help_functions.h"

class OpenVINOInference {
public:
  struct Config {
    std::string model_path;
    std::string model_name;
    std::string device = "CPU"; // Default device

    int input_width = 640;  // Default input width
    int input_height = 480; // Default input height
    bool verbose = false;   // Verbose output
  };

  OpenVINOInference() = default;

  OpenVINOInference(const Config &config) : config_(config) {
    // Step 1: Initialize OpenVINO Runtime Core
    core_ = std::make_shared<ov::Core>();

    // Step 2: Compile the model
    ov::AnyMap model_config = {{ov::hint::inference_precision(ov::element::f32)}};
    model_ = core_->read_model(config_.model_path);
    compiled_model_ = core_->compile_model(model_, config_.device, model_config);

    // Step 3: Create inference request
    infer_request_ = compiled_model_.create_infer_request();
  }

  std::vector<float> inference(const cv::Mat &input, bool verbose = false) {
    cv::Mat _input;
    if (input.channels() == 3) {
      if (verbose || config_.verbose)
        std::cout << "Covert color to gray" << std::endl;
      cv::cvtColor(input, _input, cv::COLOR_BGR2GRAY);
    } else {
      _input = input;
    }
    if (_input.cols != config_.input_width || _input.rows != config_.input_height) {
      if (verbose || config_.verbose)
        std::cout << "Resize image from " << _input.size() << " to "
                  << cv::Size(config_.input_width, config_.input_height) << std::endl;
      cv::resize(_input, _input, cv::Size(config_.input_width, config_.input_height));
    }

    _input.convertTo(_input, CV_32F);

    // Step 4: Prepare input data
    ov::Tensor input_tensor = infer_request_.get_input_tensor();

    // Copy data to the input tensor
    std::memcpy(input_tensor.data(), _input.data, _input.total() * _input.elemSize());

    // Step 5: Perform inference
    infer_request_.infer();

    // Handle MobileNetVLAD specific output
    ov::Tensor output_tensor = infer_request_.get_output_tensor(0);
    return std::vector<float>(output_tensor.data<float>(),
                              output_tensor.data<float>() + output_tensor.get_size());
  }

  void inference(const cv::Mat &input, std::vector<cv::Point2f> &keypoints,
                 std::vector<float> &local_descriptors, bool verbose = false) {
    cv::Mat _input;
    if (input.channels() == 3) {
      if (verbose || config_.verbose)
        std::cout << "Covert color to gray" << std::endl;
      cv::cvtColor(input, _input, cv::COLOR_BGR2GRAY);
    } else {
      _input = input;
    }
    if (_input.cols != config_.input_width || _input.rows != config_.input_height) {
      if (verbose || config_.verbose)
        std::cout << "Resize image from " << _input.size() << " to "
                  << cv::Size(config_.input_width, config_.input_height) << std::endl;
      cv::resize(_input, _input, cv::Size(config_.input_width, config_.input_height));
    }

    _input.convertTo(_input, CV_32F, 1 / 255.0);

    // Step 4: Prepare input data
    ov::Tensor input_tensor = infer_request_.get_input_tensor();

    // Copy data to the input tensor
    std::memcpy(input_tensor.data(), _input.data, _input.total() * _input.elemSize());

    // Step 5: Perform inference
    infer_request_.infer();

    // Handle SuperPoint specific output
    ov::Tensor semi_output = infer_request_.get_output_tensor(0);
    ov::Tensor desc_output = infer_request_.get_output_tensor(1);

    // Get the data from output tensors
    float *semi_data = semi_output.data<float>();
    float *desc_data = desc_output.data<float>();

    // Get output shapes
    ov::Shape semi_shape = semi_output.get_shape();
    ov::Shape desc_shape = desc_output.get_shape();

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto mProb =
        at::from_blob(semi_data, {1, 1, config_.input_height, config_.input_width}, options);
    auto mDesc = at::from_blob(
        desc_data, {1, 256, config_.input_height / 8, config_.input_width / 8}, options);
    cv::Mat Prob(config_.input_height, config_.input_width, CV_32F, semi_data);

    double thres = 0.015;
    int max_num = 200;
    getKeyPoints(Prob, thres, keypoints, config_.input_width, config_.input_height, max_num);

    Eigen::MatrixXf pca_comp_T;
    Eigen::RowVectorXf pca_mean;
    computeDescriptors(mProb, mDesc, keypoints, local_descriptors, config_.input_width,
                       config_.input_height, pca_comp_T, pca_mean);
  }

private:
  Config config_;
  std::shared_ptr<ov::Core> core_;
  std::shared_ptr<ov::Model> model_;
  ov::CompiledModel compiled_model_;
  ov::InferRequest infer_request_;
};
