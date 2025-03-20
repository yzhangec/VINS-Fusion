#include "onnx_generic.h"
#include <Eigen/Dense>

namespace loop_closure {
class SuperPointONNX : public ONNXInferenceGeneric {
  Eigen::MatrixXf pca_comp_T;
  Eigen::RowVectorXf pca_mean;
  float *results_desc_ = nullptr;
  float *results_semi_ = nullptr;
  std::array<int64_t, 4> output_shape_desc_;
  std::array<int64_t, 3> output_shape_semi_;
  std::array<int64_t, 4> input_shape_;
  std::vector<Ort::Value> output_tensors_;
  int max_num = 200;

public:
  double thres = 0.015;
  SuperPointONNX(std::string engine_path, std::string _pca_comp, std::string _pca_mean, int _width,
                 int _height, float _thres = 0.015, int _max_num = 200, bool _enable_perf = false);

  void inference(const cv::Mat &input, std::vector<cv::Point2f> &keypoints,
                 std::vector<float> &local_descriptors);
  void doInference(const unsigned char *input, const uint32_t batchSize) override;
};
} // namespace loop_closure