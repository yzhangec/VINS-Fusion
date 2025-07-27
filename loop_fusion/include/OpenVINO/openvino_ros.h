#include <iostream>
#include <thread>
#include <vector>

#include <Eigen/Eigen>
#include <ros/ros.h>

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

class OpenVINOInference {
public:
  OpenVINOInference() = default;

  OpenVINOInference(const std::string &model_path, const std::string &device = "CPU") {
    // Step 1: Initialize OpenVINO Runtime Core
    core = std::make_shared<ov::Core>();

    // Step 2: Compile the model
    model = core->read_model(model_path);
    compiled_model = core->compile_model(model, device);

    // Step 3: Create inference request
    infer_request = compiled_model.create_infer_request();
  }

  std::vector<float> inference(const cv::Mat &input, bool verbose = false) {
    cv::Mat _input;
    if (input.channels() == 3) {
      if (verbose)
        std::cout << "Covert color to gray" << std::endl;
      cv::cvtColor(input, _input, cv::COLOR_BGR2GRAY);
    } else {
      _input = input;
    }
    if (_input.cols != 640 || _input.rows != 480) {
      if (verbose)
        std::cout << "Resize image from " << _input.size() << " to " << cv::Size(640, 480)
                  << std::endl;
      cv::resize(_input, _input, cv::Size(640, 480));
    }

    // Step 4: Prepare input data
    ov::Tensor input_tensor = infer_request.get_input_tensor();

    // Copy data to the input tensor
    std::memcpy(input_tensor.data(), _input.data,
                _input.total() * _input.elemSize());

    // Step 5: Perform inference
    infer_request.infer();

    // Step 6: Get output data
    ov::Tensor output_tensor = infer_request.get_output_tensor();
    return std::vector<float>(output_tensor.data<float>(),
                              output_tensor.data<float>() + output_tensor.get_size());
  }

private:
  std::shared_ptr<ov::Core> core;
  std::shared_ptr<ov::Model> model;
  ov::CompiledModel compiled_model;
  ov::InferRequest infer_request;
};
