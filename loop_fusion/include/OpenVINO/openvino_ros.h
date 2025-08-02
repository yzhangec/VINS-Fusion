#include <iostream>
#include <thread>
#include <vector>

#include <Eigen/Eigen>
#include <ros/ros.h>

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

class OpenVINOInference {
public:
  struct Config {
    std::string model_path;
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
    model_ = core_->read_model(config_.model_path);
    compiled_model_ = core_->compile_model(model_, config_.device);

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

    _input.convertTo(_input, CV_32F, 1 / 255.0);

    // Step 4: Prepare input data
    ov::Tensor input_tensor = infer_request_.get_input_tensor();

    // Copy data to the input tensor
    std::memcpy(input_tensor.data(), _input.data, _input.total() * _input.elemSize());

    // Step 5: Perform inference
    infer_request_.infer();

    // Step 6: Get output data
    ov::Tensor output_tensor = infer_request_.get_output_tensor();
    return std::vector<float>(output_tensor.data<float>(),
                              output_tensor.data<float>() + output_tensor.get_size());
  }

private:
  Config config_;
  std::shared_ptr<ov::Core> core_;
  std::shared_ptr<ov::Model> model_;
  ov::CompiledModel compiled_model_;
  ov::InferRequest infer_request_;
};
