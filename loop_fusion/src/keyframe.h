/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "parameters.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#define MIN_LOOP_NUM 25

using namespace Eigen;
using namespace std;

class Keyframe {
public:
  Keyframe(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i,
           cv::Mat &_image, vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv,
           vector<cv::Point2f> &_point_2d_normal, vector<double> &_point_id, int _sequence,
           std::vector<float> &_global_desc, std::vector<float> &_local_desc, int _img_seq);
  bool findConnection(Keyframe *old_kf, bool use_gt = false);
  void getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
  void getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
  void updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
  void updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
  void updateLoop(Eigen::Matrix<double, 8, 1> &_loop_info);

  void setGrountTruthPose(Eigen::Vector3d &_T_w_i_gt, Eigen::Matrix3d &_R_w_i_gt) {
    T_w_i_gt = _T_w_i_gt;
    R_w_i_gt = _R_w_i_gt;
  }
  void getGroundTruthPose(Eigen::Vector3d &_T_w_i_gt, Eigen::Matrix3d &_R_w_i_gt) {
    _T_w_i_gt = T_w_i_gt;
    _R_w_i_gt = R_w_i_gt;
  }

  Eigen::Vector3d getLoopRelativeT();
  double getLoopRelativeYaw();
  Eigen::Quaterniond getLoopRelativeQ();

  // get T_loop_cur
  bool estimateTBetweenFrames(vector<cv::Point3f> &cur_pts_3d, vector<cv::Point2f> &loop_pts_2d,
                              Matrix3d &R, Vector3d &t);
  void drawFeatureOnImage(cv::Mat &image, const std::vector<cv::Point2f> &pts,
                          const cv::Scalar &color = cv::Scalar(0, 255, 0, 0));
  void drawLineOnImage(cv::Mat &image, const std::vector<cv::Point2f> &pts0,
                       const std::vector<cv::Point2f> &pts1,
                       const cv::Scalar &color = cv::Scalar(0, 0, 0, 0));
  double time_stamp;
  int index;
  int local_index;
  Eigen::Vector3d vio_T_w_i;
  Eigen::Matrix3d vio_R_w_i;
  Eigen::Vector3d T_w_i;
  Eigen::Matrix3d R_w_i;
  Eigen::Vector3d T_w_i_gt;
  Eigen::Matrix3d R_w_i_gt;
  Eigen::Vector3d origin_vio_T;
  Eigen::Matrix3d origin_vio_R;
  cv::Mat image;
  cv::Mat thumbnail;
  vector<cv::Point3f> point_3d;
  vector<cv::Point2f> point_2d_uv;
  vector<cv::Point2f> point_2d_norm;
  vector<double> point_id;
  vector<cv::KeyPoint> keypoints;
  vector<cv::KeyPoint> keypoints_norm;
  vector<cv::KeyPoint> window_keypoints;
  int sequence;
  int img_seq;

  bool has_loop;
  int loop_index;
  Eigen::Matrix<double, 8, 1> loop_info;

  std::vector<float> global_desc; // NetVLAD
  std::vector<float> local_desc;  // SuperPoint
};
