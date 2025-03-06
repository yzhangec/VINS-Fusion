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

#include <fstream>
#include <map>
#include <mutex>
#include <queue>
#include <stdio.h>
#include <thread>

#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float32MultiArray.h>

#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "utility/visualization.h"

Estimator estimator;

std::string POSE_TOPIC = "/vins_estimator/odometry";

queue<nav_msgs::Odometry> pose_buf;
queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
std::mutex m_buf;

Eigen::Matrix3d R_cam_l_r, R_imu_cam0, R_imu_cam1;
Eigen::Vector3d t_cam_l_r, t_imu_cam0, t_imu_cam1;

std::vector<camodocal::CameraPtr> stereo_camera_;

ros::Publisher pub_landmarks, pub_last_landmarks, pub_cov, pub_feature_img;

gtsam::NonlinearFactorGraph graph;
gtsam::Values initialEstimate;

// log file
std::ofstream log_file;

void pose_callback(const nav_msgs::OdometryConstPtr &pose_msg) {
  m_buf.lock();
  pose_buf.push(*pose_msg);
  m_buf.unlock();
}

void img0_callback(const sensor_msgs::ImageConstPtr &img_msg) {
  m_buf.lock();
  img0_buf.push(img_msg);
  m_buf.unlock();
}

void img1_callback(const sensor_msgs::ImageConstPtr &img_msg) {
  m_buf.lock();
  img1_buf.push(img_msg);
  m_buf.unlock();
}

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg) {
  cv_bridge::CvImageConstPtr ptr;
  if (img_msg->encoding == "8UC1") {
    sensor_msgs::Image img;
    img.header = img_msg->header;
    img.height = img_msg->height;
    img.width = img_msg->width;
    img.is_bigendian = img_msg->is_bigendian;
    img.step = img_msg->step;
    img.data = img_msg->data;
    img.encoding = "mono8";
    ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
  } else
    ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

  cv::Mat img = ptr->image.clone();
  return img;
}

void drawFeature(const cv::Mat &image, cv::Mat &image_with_feature,
                 const vector<cv::Point2f> &n_pts) {
  cv::cvtColor(image, image_with_feature, CV_GRAY2BGR);
  for (size_t i = 0; i < n_pts.size(); i++) {
    cv::circle(image_with_feature, n_pts[i], 2, cv::Scalar(0, 0, 255), 2);
  }
}

template <class T> void reduceVector(vector<T> &v, vector<uchar> status) {
  int j = 0;
  for (int i = 0; i < int(v.size()); i++)
    if (status[i])
      v[j++] = v[i];
  v.resize(j);
}

void triangulatePoint(const Eigen::Matrix<double, 3, 4> &Pose0,
                      const Eigen::Matrix<double, 3, 4> &Pose1, const Eigen::Vector2d &point0,
                      const Eigen::Vector2d &point1, Eigen::Vector3d &point_3d) {
  Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
  Eigen::Vector4d triangulated_point;

  design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
  design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
  design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
  design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);

  triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();

  point_3d(0) = triangulated_point(0) / triangulated_point(3);
  point_3d(1) = triangulated_point(1) / triangulated_point(3);
  point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

void generate3dPoints(const std::vector<cv::Point2f> &left_pts,
                      const std::vector<cv::Point2f> &right_pts,
                      std::vector<cv::Point3f> &cur_pts_3d, std::vector<uchar> &status) {

  Eigen::Matrix<double, 3, 4> P1, P2;

  P1 << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0;
  P2.block(0, 0, 3, 3) = (R_cam_l_r.transpose());
  P2.block(0, 3, 3, 1) = -P2.block(0, 0, 3, 3) * t_cam_l_r;

  for (unsigned int i = 0; i < left_pts.size(); ++i) {
    Vector2d pl(left_pts[i].x, left_pts[i].y);
    Vector2d pr(right_pts[i].x, right_pts[i].y);
    Vector3d pt3;
    triangulatePoint(P1, P2, pl, pr, pt3);

    double depth = pt3[2];
    if (depth > 0.0 && depth < 10.0) {
      cur_pts_3d.push_back(cv::Point3f(pt3[0], pt3[1], pt3[2]));
      status.push_back(1);
    } else {
      status.push_back(0);
    }
  }
}

void undistortedPts(const std::vector<cv::Point2f> &pts, std::vector<cv::Point2f> &un_pts,
                    const camodocal::CameraPtr &cam) {
  un_pts.clear();
  for (unsigned int i = 0; i < pts.size(); i++) {
    Eigen::Vector2d a(pts[i].x, pts[i].y);
    Eigen::Vector3d b;
    // Convert pixel coordinates to undistorted normalized coordinates.
    cam->liftProjective(a, b);
    un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
  }
}

void publishLandmarks(const ros::Time &ts, const std::vector<cv::Point3f> &landmarks) {
  pcl::PointCloud<pcl::PointXYZRGB> pointcloud;
  for (size_t i = 0; i < landmarks.size(); i++) {
    pcl::PointXYZRGB point;
    point.x = landmarks[i].x;
    point.y = landmarks[i].y;
    point.z = landmarks[i].z;
    point.r = 255;
    point.g = 0;
    point.b = 0;
    pointcloud.push_back(point);
  }

  pointcloud.width = pointcloud.points.size();
  pointcloud.height = 1;

  sensor_msgs::PointCloud2 pointcloud_msg;
  pcl::toROSMsg(pointcloud, pointcloud_msg);
  pointcloud_msg.header.frame_id = "world";
  pointcloud_msg.header.stamp = ts;
  pub_landmarks.publish(pointcloud_msg);
}

void publishLastLandmarks(const ros::Time &ts, const std::vector<cv::Point3f> &landmarks) {
  pcl::PointCloud<pcl::PointXYZRGB> pointcloud;
  for (size_t i = 0; i < landmarks.size(); i++) {
    pcl::PointXYZRGB point;
    point.x = landmarks[i].x;
    point.y = landmarks[i].y;
    point.z = landmarks[i].z;
    point.r = 0;
    point.g = 0;
    point.b = 255;
    pointcloud.push_back(point);
  }

  pointcloud.width = pointcloud.points.size();
  pointcloud.height = 1;

  sensor_msgs::PointCloud2 pointcloud_msg;
  pcl::toROSMsg(pointcloud, pointcloud_msg);
  pointcloud_msg.header.frame_id = "world";
  pointcloud_msg.header.stamp = ts;
  pub_last_landmarks.publish(pointcloud_msg);
}

void computePointsCovariance(const vector<cv::Point3f> &points_A,
                             const vector<cv::Point3f> &points_B, Matrix3d &covariance) {
  if (points_A.size() != points_B.size()) {
    ROS_ERROR("Error: Number of points in A and B must be equal!");
  }

  if (points_A.size() < 3) {
    ROS_ERROR("Error: At least 3 points are required!");
  }

  int N = points_A.size();
  Vector3d mean_A = Vector3d::Zero();
  Vector3d mean_B = Vector3d::Zero();

  // Compute mean of points A and B
  for (int i = 0; i < N; ++i) {
    mean_A += Vector3d(points_A[i].x, points_A[i].y, points_A[i].z);
    mean_B += Vector3d(points_B[i].x, points_B[i].y, points_B[i].z);
  }
  mean_A /= N;
  mean_B /= N;

  // Compute covariance matrix
  covariance.setZero();

  for (int i = 0; i < N; ++i) {
    Vector3d diff_A = Vector3d(points_A[i].x, points_A[i].y, points_A[i].z) - mean_A;
    Vector3d diff_B = Vector3d(points_B[i].x, points_B[i].y, points_B[i].z) - mean_B;
    covariance += diff_B * diff_A.transpose();
  }
  covariance /= (N - 1);

  // cout << "covariance:\n" << covariance << endl;
}

Matrix3d skewSymmetric(const Vector3d &v) {
  Matrix3d S;
  S << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return S;
}

// Function to compute pose covariance considering an arbitrary covariance matrix for
// correspondences
Matrix<double, 6, 6>
computePoseCovariance(const vector<cv::Point3f> &points_A, // Source point cloud
                      const vector<cv::Point3f> &points_B, // Target point cloud
                      const Matrix3d &R,                   // Rotation from A to B
                      const Vector3d &t,                   // Translation from A to B
                      const Matrix3d &covariances_AB       // Per-point covariance matrices
) {
  int N = points_A.size();
  if (N < 3) {
    cerr << "Error: At least 3 points are required!" << endl;
    // exit(EXIT_FAILURE);

    // Return identity matrix if less than 3 points
    return Matrix<double, 6, 6>::Identity();
  }

  // Construct Jacobian J (3N x 6)
  MatrixXd J(3 * N, 6);
  MatrixXd Sigma_Y(3 * N, 3 * N); // Covariance of Y (block diagonal)

  Sigma_Y.setZero();

  for (int i = 0; i < N; ++i) {
    Vector3d p_A = Vector3d(points_A[i].x, points_A[i].y, points_A[i].z);
    // Vector3d p_B = Vector3d(points_B[i].x, points_B[i].y, points_B[i].z);

    // Compute Jacobian block
    Matrix3d skew = skewSymmetric(R * p_A);
    J.block<3, 3>(3 * i, 0) = -skew;                // Rotation part
    J.block<3, 3>(3 * i, 3) = Matrix3d::Identity(); // Translation part

    // Insert per-point covariance into block diagonal of Sigma_Y
    Sigma_Y.block<3, 3>(3 * i, 3 * i) = covariances_AB;
  }

  // Compute (Jᵀ Σ_Y⁻¹ J)⁻¹
  Matrix<double, 6, 6> Sigma_T = (J.transpose() * Sigma_Y.inverse() * J).inverse();

  return Sigma_T;
}

cv::Mat last_image0;
vector<cv::Point2f> last_features;
vector<cv::Point3f> last_landmarks;
nav_msgs::Odometry last_pose;

// extract images with same timestamp from two topics
void sync_process() {
  while (1) {
    cv::Mat image0, image1;
    vector<cv::Point2f> n_pts, n_pts_right;
    vector<cv::Point3f> landmarks;

    nav_msgs::Odometry pose;
    std_msgs::Header header;
    double time = 0;
    m_buf.lock();
    if (!img0_buf.empty() && !img1_buf.empty()) {
      double time0 = img0_buf.front()->header.stamp.toSec();
      double time1 = img1_buf.front()->header.stamp.toSec();
      // 0.003s sync tolerance
      if (time0 < time1 - 0.08) {
        img0_buf.pop();
        printf("throw img0\n");
      } else if (time0 > time1 + 0.08) {
        img1_buf.pop();
        printf("throw img1\n");
      } else {
        time = img0_buf.front()->header.stamp.toSec();
        header = img0_buf.front()->header;
        image0 = getImageFromMsg(img0_buf.front());
        img0_buf.pop();
        image1 = getImageFromMsg(img1_buf.front());
        img1_buf.pop();

        // find pose with same timestamp
        while (!pose_buf.empty()) {
          if (pose_buf.front().header.stamp.toSec() < time - 0.08) {
            pose_buf.pop();
          } else if (pose_buf.front().header.stamp.toSec() > time + 0.08) {
            break;
          } else {
            pose = pose_buf.front();
            pose_buf.pop();
            break;
          }
        }
        // printf("find img0 and img1\n");

        if (pose.header.stamp.toSec() != 0) {
          // synced images and pose
          // continue;

          // std::cout << std::setprecision(12);
          // std::cout << "img0 ts: " << time0 << " img1 ts: " << time1 << std::endl;
          // std::cout << "pose ts: " << pose.header.stamp.toSec() << std::endl;

          cv::goodFeaturesToTrack(image0, n_pts, 100, 0.01, 30);

          vector<cv::Point2f> reverseLeftPts;
          vector<uchar> status, statusRightLeft;
          vector<float> err;
          cv::calcOpticalFlowPyrLK(image0, image1, n_pts, n_pts_right, status, err,
                                   cv::Size(21, 21), 3);

          cv::calcOpticalFlowPyrLK(image1, image0, n_pts_right, reverseLeftPts, statusRightLeft,
                                   err, cv::Size(21, 21), 3);

          for (size_t i = 0; i < status.size(); i++) {
            if (status[i] && statusRightLeft[i])
              status[i] = 1;
            else
              status[i] = 0;
          }
          reduceVector(n_pts, status);
          reduceVector(n_pts_right, status);

          std::vector<cv::Point2f> un_pts0, un_pts1;
          undistortedPts(n_pts, un_pts0, stereo_camera_[0]);
          undistortedPts(n_pts_right, un_pts1, stereo_camera_[1]);

          std::vector<uchar> status_3d;
          generate3dPoints(un_pts0, un_pts1, landmarks, status_3d);
          reduceVector(n_pts, status_3d);
          reduceVector(n_pts_right, status_3d);

          // std::cout << "n_pts size: " << n_pts.size() << std::endl;
          // std::cout << "landmarks size: " << landmarks.size() << std::endl;

          // pose is T_w_c, landmarks is p_c, convert p_c to  p_w
          Eigen::Matrix3d R_w_b =
              Eigen::Quaterniond(pose.pose.pose.orientation.w, pose.pose.pose.orientation.x,
                                 pose.pose.pose.orientation.y, pose.pose.pose.orientation.z)
                  .toRotationMatrix();
          Eigen::Vector3d t_w_b(pose.pose.pose.position.x, pose.pose.pose.position.y,
                                pose.pose.pose.position.z);
          Eigen::Matrix3d R_w_c = R_w_b * R_imu_cam0;
          Eigen::Vector3d t_w_c = R_w_b * t_imu_cam0 + t_w_b;

          // std::cout << "R_w_c: " << R_w_c << std::endl;
          // std::cout << "t_w_c: " << t_w_c << std::endl;

          for (size_t i = 0; i < landmarks.size(); i++) {
            Eigen::Vector3d pt_c, pt_w;
            pt_c << landmarks[i].x, landmarks[i].y, landmarks[i].z;
            pt_w = R_w_c * pt_c + t_w_c;

            landmarks[i].x = pt_w(0);
            landmarks[i].y = pt_w(1);
            landmarks[i].z = pt_w(2);
          }

          // draw feature on image
          if (0) {
            cv::Mat image0_with_feature, image1_with_feature;
            drawFeature(image0, image0_with_feature, n_pts);
            drawFeature(image1, image1_with_feature, n_pts_right);

            cv::Mat image_hconcat;
            cv::hconcat(image0_with_feature, image1_with_feature, image_hconcat);
            // cv::imshow("feature", image_hconcat);
            // cv::waitKey(1);

            // publishLandmarks(header.stamp, landmarks);
            // publishLastLandmarks(header.stamp, last_landmarks);
          }

          if (!last_image0.empty()) {
            std::vector<cv::Point3f> last_landmarks_cur;
            if (last_features.size() >= 10) {
              vector<cv::Point2f> last_features_on_cur;
              status.clear();
              err.clear();
              cv::calcOpticalFlowPyrLK(last_image0, image0, last_features, last_features_on_cur,
                                       status, err, cv::Size(21, 21), 3);
              // std::cout << "last_features_on_cur size: " << last_features_on_cur.size() <<
              // std::endl;

              vector<cv::Point2f> last_features_on_cur_right;
              vector<uchar> last_features_status, last_features_status_right;
              err.clear();
              cv::calcOpticalFlowPyrLK(image0, image1, last_features_on_cur,
                                       last_features_on_cur_right, last_features_status, err,
                                       cv::Size(21, 21), 3);
              reverseLeftPts.clear();
              err.clear();
              cv::calcOpticalFlowPyrLK(image1, image0, last_features_on_cur_right, reverseLeftPts,
                                       last_features_status_right, err, cv::Size(21, 21), 3);

              for (size_t i = 0; i < last_features_status.size(); i++) {
                if (last_features_status[i] && last_features_status_right[i])
                  last_features_status[i] = 1;
                else
                  last_features_status[i] = 0;
              }
              // std::cout << "last_features_on_cur_right size: " <<
              // last_features_on_cur_right.size()
              // << std::endl;
              reduceVector(last_features_on_cur, last_features_status);
              reduceVector(last_features_on_cur_right, last_features_status);
              reduceVector(last_landmarks, last_features_status);
              // std::cout << "last_features_on_cur size: " << last_features_on_cur.size() <<
              // std::endl; std::cout << "last_features_on_cur_right size: " <<
              // last_features_on_cur_right.size()
              //           << std::endl;
              // std::cout << "last_landmarks size: " << last_landmarks.size() << std::endl;

              std::vector<cv::Point2f> last_un_pts0, last_un_pts1;
              undistortedPts(last_features_on_cur, last_un_pts0, stereo_camera_[0]);
              undistortedPts(last_features_on_cur_right, last_un_pts1, stereo_camera_[1]);

              std::vector<uchar> last_status_3d;
              generate3dPoints(last_un_pts0, last_un_pts1, last_landmarks_cur, last_status_3d);
              // std::cout << "last_landmarks_cur size: " << last_landmarks_cur.size() << std::endl;

              // reduceVector(last_landmarks_cur, last_status_3d);
              reduceVector(last_landmarks, last_status_3d);

              // std::cout << "last_landmarks_cur size: " << last_landmarks_cur.size() << std::endl;
              std::cout << "last_landmarks size: " << last_landmarks.size() << std::endl;

              // pose is T_w_c, landmarks is p_c, convert p_c to  p_w
              for (size_t i = 0; i < last_landmarks_cur.size(); i++) {
                Eigen::Vector3d pt_c, pt_w;
                pt_c << last_landmarks_cur[i].x, last_landmarks_cur[i].y, last_landmarks_cur[i].z;
                pt_w = R_w_c * pt_c + t_w_c;

                last_landmarks_cur[i].x = pt_w(0);
                last_landmarks_cur[i].y = pt_w(1);
                last_landmarks_cur[i].z = pt_w(2);
              }

              if (1) {
                publishLandmarks(header.stamp, last_landmarks_cur);
                publishLastLandmarks(header.stamp, last_landmarks);
              }
            }

            Matrix<double, 6, 6> poseCovariance = Matrix<double, 6, 6>::Identity() * 1e-10;
            double poseCovDet = 1e-10;
            double poseCovTrace = poseCovariance.trace();
            if (last_landmarks_cur.size() >= 3 && last_landmarks.size() >= 3) {
              // calculate the covariance of last_landmarks_cur and landmarks
              Eigen::Matrix<double, 3, 3> cov;
              computePointsCovariance(last_landmarks_cur, last_landmarks, cov);

              // Compute pose covariance considering heterogeneous noise
              poseCovariance =
                  computePoseCovariance(last_landmarks_cur, last_landmarks, R_w_b, t_w_b, cov);

              poseCovDet = poseCovariance.determinant();
              poseCovTrace = poseCovariance.trace();

              // SVD decomposition
              // Eigen::JacobiSVD<Eigen::MatrixXd> svd(poseCovariance,
              //                                       Eigen::ComputeThinU | Eigen::ComputeThinV);
              // Eigen::VectorXd singularValues = svd.singularValues();
              // std::cout << "Singular values: " << singularValues.transpose() << std::endl;

              // only keep diagnal entry
              for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                  if (i != j) {
                    poseCovariance(i, j) = 0;
                  }
                }
              }

              // diagnal entry > 1 then set to 1
              for (int i = 0; i < 6; i++) {
                if (poseCovariance(i, i) > 1) {
                  poseCovariance(i, i) = 1;
                }
              }

            } else {
              poseCovariance << 0.1, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0,
                  1e-4, 0, 0, 0, 0, 0, 0, 1e-4, 0, 0, 0, 0, 0, 0, 1e-4;
            }

            // if nan
            if (poseCovariance.hasNaN()) {
              ROS_ERROR("Pose covariance has NaN!");
              poseCovariance << 0.1, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0,
                  1e-4, 0, 0, 0, 0, 0, 0, 1e-4, 0, 0, 0, 0, 0, 0, 1e-4;
              poseCovDet = 1e-10;
              poseCovTrace = poseCovariance.trace();
            }

            // Display result
            cout << "Pose Covariance Matrix:\n" << poseCovariance << endl;
            cout << "det(poseCovariance): " << poseCovDet << endl;
            cout << "trace(poseCovariance): " << poseCovTrace << endl;

            Eigen::Matrix3d R_w_b_last = Eigen::Quaterniond(last_pose.pose.pose.orientation.w,
                                                            last_pose.pose.pose.orientation.x,
                                                            last_pose.pose.pose.orientation.y,
                                                            last_pose.pose.pose.orientation.z)
                                             .toRotationMatrix();
            Eigen::Vector3d t_w_b_last(last_pose.pose.pose.position.x,
                                       last_pose.pose.pose.position.y,
                                       last_pose.pose.pose.position.z);

            // print R_w_c_last and t_w_c_last
            // std::cout << "R_w_c_last: " << R_w_c_last << std::endl;
            // std::cout << "t_w_c_last: " << t_w_c_last << std::endl;

            Eigen::Matrix3d R_last_cur = R_w_b_last.transpose() * R_w_b;
            Eigen::Vector3d t_last_cur = R_w_b_last.transpose() * (t_w_b - t_w_b_last);

            // print R_last_cur and t_last_cur
            std::cout << "R_last_cur: " << endl << R_last_cur << std::endl;
            std::cout << "t_last_cur: " << endl << t_last_cur << std::endl;

            gtsam::Pose3 odometry =
                gtsam::Pose3(gtsam::Rot3(R_last_cur(0, 0), R_last_cur(0, 1), R_last_cur(0, 2),
                                         R_last_cur(1, 0), R_last_cur(1, 1), R_last_cur(1, 2),
                                         R_last_cur(2, 0), R_last_cur(2, 1), R_last_cur(2, 2)),
                             gtsam::Point3(t_last_cur(0), t_last_cur(1), t_last_cur(2)));

            // auto odometryNoise =
            //     gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(1e-4));
            auto odometryNoise = gtsam::noiseModel::Gaussian::Covariance(poseCovariance);
            // auto odometryNoise =
            //     gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(poseCovDet * 1e5));
            int i = graph.size();
            graph.add(gtsam::BetweenFactor<gtsam::Pose3>(i, i + 1, odometry, odometryNoise));

            gtsam::Pose3 init_pose = gtsam::Pose3(
                gtsam::Rot3(pose.pose.pose.orientation.w, pose.pose.pose.orientation.x,
                            pose.pose.pose.orientation.y, pose.pose.pose.orientation.z),
                gtsam::Point3(pose.pose.pose.position.x, pose.pose.pose.position.y,
                              pose.pose.pose.position.z));
            initialEstimate.insert(graph.size(), init_pose);

            // initialEstimate.print("Initial estimate:\n");

            // gtsam::LevenbergMarquardtOptimizer optimizer(graph, initialEstimate);
            // gtsam::Values result = optimizer.optimize();

            // graph.print("Final factor graph:\n");

            gtsam::Marginals marginals(graph, initialEstimate);
            double d_opt = marginals.marginalCovariance(graph.size() - 1).determinant();
            double t_opt = marginals.marginalCovariance(graph.size() - 1).trace();
            cout << "graph size: " << graph.size() << ", D-opt: " << d_opt << ", T-opt: " << t_opt
                 << endl;

            std_msgs::Float32MultiArray cov_msg;
            cov_msg.data.push_back(d_opt);
            cov_msg.data.push_back(poseCovDet);
            pub_cov.publish(cov_msg);

            log_file << d_opt << " " << poseCovDet << " " << poseCovariance(0, 0) << " "
                     << poseCovariance(1, 1) << " " << poseCovariance(2, 2) << " "
                     << poseCovariance(3, 3) << " " << poseCovariance(4, 4) << " "
                     << poseCovariance(5, 5) << std::endl;

          } else {
            if (graph.size() == 0) {
              gtsam::Pose3 priorMean =
                  gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(0, 0, 0));
              auto priorNoise =
                  gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(1e-15));
              graph.add(gtsam::PriorFactor<gtsam::Pose3>(1, priorMean, priorNoise));
              ROS_INFO("Added prior factor");

              gtsam::Pose3 init_pose = gtsam::Pose3(
                  gtsam::Rot3(pose.pose.pose.orientation.w, pose.pose.pose.orientation.x,
                              pose.pose.pose.orientation.y, pose.pose.pose.orientation.z),
                  gtsam::Point3(pose.pose.pose.position.x, pose.pose.pose.position.y,
                                pose.pose.pose.position.z));
              initialEstimate.insert(1, init_pose);

              // gtsam::LevenbergMarquardtOptimizer optimizer(graph, initialEstimate);
              // gtsam::Values result = optimizer.optimize();
            } else {
              ROS_ERROR("No last features");
            }
          }

          last_image0 = image0;
          last_features = n_pts;
          last_landmarks = landmarks;
          last_pose = pose;
        }
      }
    }
    m_buf.unlock();

    std::chrono::milliseconds dura(2); // 2ms, 500Hz
    std::this_thread::sleep_for(dura);
  }
}

// sync callback
void syncCallback(const sensor_msgs::ImageConstPtr &img0_msg,
                  const sensor_msgs::ImageConstPtr &img1_msg,
                  const nav_msgs::OdometryConstPtr &pose_msg) {
  // print three timestamps
  std::cout << std::setprecision(16);
  std::cout << "receive img0 ts: " << img0_msg->header.stamp.toSec()
            << ", img1 ts: " << img1_msg->header.stamp.toSec()
            << ", pose ts: " << pose_msg->header.stamp.toSec() << std::endl;

  cv::Mat image0, image1;
  vector<cv::Point2f> n_pts, n_pts_right;
  vector<cv::Point3f> landmarks;

  nav_msgs::Odometry pose;
  std_msgs::Header header;

  image0 = getImageFromMsg(img0_msg);
  image1 = getImageFromMsg(img1_msg);
  pose = *pose_msg;
  
  cv::goodFeaturesToTrack(image0, n_pts, 100, 0.01, 30);

  vector<cv::Point2f> reverseLeftPts;
  vector<uchar> status, statusRightLeft;
  vector<float> err;
  cv::calcOpticalFlowPyrLK(image0, image1, n_pts, n_pts_right, status, err,
                           cv::Size(21, 21), 3);

  cv::calcOpticalFlowPyrLK(image1, image0, n_pts_right, reverseLeftPts, statusRightLeft,
                           err, cv::Size(21, 21), 3);

  for (size_t i = 0; i < status.size(); i++) {
    if (status[i] && statusRightLeft[i])
      status[i] = 1;
    else
      status[i] = 0;
  }
  reduceVector(n_pts, status);
  reduceVector(n_pts_right, status);

  std::vector<cv::Point2f> un_pts0, un_pts1;
  undistortedPts(n_pts, un_pts0, stereo_camera_[0]);
  undistortedPts(n_pts_right, un_pts1, stereo_camera_[1]);

  std::vector<uchar> status_3d;
  generate3dPoints(un_pts0, un_pts1, landmarks, status_3d);
  reduceVector(n_pts, status_3d);
  reduceVector(n_pts_right, status_3d);

  // std::cout << "n_pts size: " << n_pts.size() << std::endl;
  // std::cout << "landmarks size: " << landmarks.size() << std::endl;

  // pose is T_w_c, landmarks is p_c, convert p_c to  p_w
  Eigen::Matrix3d R_w_b =
      Eigen::Quaterniond(pose.pose.pose.orientation.w, pose.pose.pose.orientation.x,
                         pose.pose.pose.orientation.y, pose.pose.pose.orientation.z)
          .toRotationMatrix();
  Eigen::Vector3d t_w_b(pose.pose.pose.position.x, pose.pose.pose.position.y,
                        pose.pose.pose.position.z);
  Eigen::Matrix3d R_w_c = R_w_b * R_imu_cam0;
  Eigen::Vector3d t_w_c = R_w_b * t_imu_cam0 + t_w_b;

  // std::cout << "R_w_c: " << R_w_c << std::endl;
  // std::cout << "t_w_c: " << t_w_c << std::endl;

  for (size_t i = 0; i < landmarks.size(); i++) {
    Eigen::Vector3d pt_c, pt_w;
    pt_c << landmarks[i].x, landmarks[i].y, landmarks[i].z;
    pt_w = R_w_c * pt_c + t_w_c;

    landmarks[i].x = pt_w(0);
    landmarks[i].y = pt_w(1);
    landmarks[i].z = pt_w(2);
  }

  // draw feature on image
  if (0) {
    cv::Mat image0_with_feature, image1_with_feature;
    drawFeature(image0, image0_with_feature, n_pts);
    drawFeature(image1, image1_with_feature, n_pts_right);

    cv::Mat image_hconcat;
    cv::hconcat(image0_with_feature, image1_with_feature, image_hconcat);
    // cv::imshow("feature", image_hconcat);
    // cv::waitKey(1);

    // publishLandmarks(header.stamp, landmarks);
    // publishLastLandmarks(header.stamp, last_landmarks);
  }

  if (!last_image0.empty()) {
    std::vector<cv::Point3f> last_landmarks_cur;
    if (last_features.size() >= 10) {
      vector<cv::Point2f> last_features_on_cur;
      status.clear();
      err.clear();
      cv::calcOpticalFlowPyrLK(last_image0, image0, last_features, last_features_on_cur,
                               status, err, cv::Size(21, 21), 3);
      // std::cout << "last_features_on_cur size: " << last_features_on_cur.size() <<
      // std::endl;

      vector<cv::Point2f> last_features_on_cur_right;
      vector<uchar> last_features_status, last_features_status_right;
      err.clear();
      cv::calcOpticalFlowPyrLK(image0, image1, last_features_on_cur,
                               last_features_on_cur_right, last_features_status, err,
                               cv::Size(21, 21), 3);
      reverseLeftPts.clear();
      err.clear();
      cv::calcOpticalFlowPyrLK(image1, image0, last_features_on_cur_right, reverseLeftPts,
                               last_features_status_right, err, cv::Size(21, 21), 3);

      for (size_t i = 0; i < last_features_status.size(); i++) {
        if (last_features_status[i] && last_features_status_right[i])
          last_features_status[i] = 1;
        else
          last_features_status[i] = 0;
      }
      // std::cout << "last_features_on_cur_right size: " <<
      // last_features_on_cur_right.size()
      // << std::endl;
      reduceVector(last_features_on_cur, last_features_status);
      reduceVector(last_features_on_cur_right, last_features_status);
      reduceVector(last_landmarks, last_features_status);
      // std::cout << "last_features_on_cur size: " << last_features_on_cur.size() <<
      // std::endl; std::cout << "last_features_on_cur_right size: " <<
      // last_features_on_cur_right.size()
      //           << std::endl;
      // std::cout << "last_landmarks size: " << last_landmarks.size() << std::endl;

      std::vector<cv::Point2f> last_un_pts0, last_un_pts1;
      undistortedPts(last_features_on_cur, last_un_pts0, stereo_camera_[0]);
      undistortedPts(last_features_on_cur_right, last_un_pts1, stereo_camera_[1]);

      std::vector<uchar> last_status_3d;
      generate3dPoints(last_un_pts0, last_un_pts1, last_landmarks_cur, last_status_3d);
      // std::cout << "last_landmarks_cur size: " << last_landmarks_cur.size() << std::endl;

      // reduceVector(last_landmarks_cur, last_status_3d);
      reduceVector(last_landmarks, last_status_3d);

      // std::cout << "last_landmarks_cur size: " << last_landmarks_cur.size() << std::endl;
      std::cout << "last_landmarks size: " << last_landmarks.size() << std::endl;

      // pose is T_w_c, landmarks is p_c, convert p_c to  p_w
      for (size_t i = 0; i < last_landmarks_cur.size(); i++) {
        Eigen::Vector3d pt_c, pt_w;
        pt_c << last_landmarks_cur[i].x, last_landmarks_cur[i].y, last_landmarks_cur[i].z;
        pt_w = R_w_c * pt_c + t_w_c;

        last_landmarks_cur[i].x = pt_w(0);
        last_landmarks_cur[i].y = pt_w(1);
        last_landmarks_cur[i].z = pt_w(2);
      }

      if (1) {
        publishLandmarks(header.stamp, last_landmarks_cur);
        publishLastLandmarks(header.stamp, last_landmarks);
      }
    }

    Matrix<double, 6, 6> poseCovariance = Matrix<double, 6, 6>::Identity() * 1e-10;
    double poseCovDet = 1e-10;
    double poseCovTrace = poseCovariance.trace();
    if (last_landmarks_cur.size() >= 3 && last_landmarks.size() >= 3) {
      // calculate the covariance of last_landmarks_cur and landmarks
      Eigen::Matrix<double, 3, 3> cov;
      computePointsCovariance(last_landmarks_cur, last_landmarks, cov);

      // Compute pose covariance considering heterogeneous noise
      poseCovariance =
          computePoseCovariance(last_landmarks_cur, last_landmarks, R_w_b, t_w_b, cov);

      poseCovDet = poseCovariance.determinant();
      poseCovTrace = poseCovariance.trace();

      // SVD decomposition
      // Eigen::JacobiSVD<Eigen::MatrixXd> svd(poseCovariance,
      //                                       Eigen::ComputeThinU | Eigen::ComputeThinV);
      // Eigen::VectorXd singularValues = svd.singularValues();
      // std::cout << "Singular values: " << singularValues.transpose() << std::endl;

      // only keep diagnal entry
      for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
          if (i != j) {
            poseCovariance(i, j) = 0;
          }
        }
      }

      // diagnal entry > 1 then set to 1
      for (int i = 0; i < 6; i++) {
        if (poseCovariance(i, i) > 1) {
          poseCovariance(i, i) = 1;
        }
      }

    } else {
      poseCovariance << 0.1, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0,
          1e-4, 0, 0, 0, 0, 0, 0, 1e-4, 0, 0, 0, 0, 0, 0, 1e-4;
    }

    // if nan
    if (poseCovariance.hasNaN()) {
      ROS_ERROR("Pose covariance has NaN!");
      poseCovariance << 0.1, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0,
          1e-4, 0, 0, 0, 0, 0, 0, 1e-4, 0, 0, 0, 0, 0, 0, 1e-4;
      poseCovDet = 1e-10;
      poseCovTrace = poseCovariance.trace();
    }

    // Display result
    cout << "Pose Covariance Matrix:\n" << poseCovariance << endl;
    cout << "det(poseCovariance): " << poseCovDet << endl;
    cout << "trace(poseCovariance): " << poseCovTrace << endl;

    Eigen::Matrix3d R_w_b_last = Eigen::Quaterniond(last_pose.pose.pose.orientation.w,
                                                    last_pose.pose.pose.orientation.x,
                                                    last_pose.pose.pose.orientation.y,
                                                    last_pose.pose.pose.orientation.z)
                                     .toRotationMatrix();
    Eigen::Vector3d t_w_b_last(last_pose.pose.pose.position.x,
                               last_pose.pose.pose.position.y,
                               last_pose.pose.pose.position.z);

    // print R_w_c_last and t_w_c_last
    // std::cout << "R_w_c_last: " << R_w_c_last << std::endl;
    // std::cout << "t_w_c_last: " << t_w_c_last << std::endl;

    Eigen::Matrix3d R_last_cur = R_w_b_last.transpose() * R_w_b;
    Eigen::Vector3d t_last_cur = R_w_b_last.transpose() * (t_w_b - t_w_b_last);

    // print R_last_cur and t_last_cur
    std::cout << "R_last_cur: " << endl << R_last_cur << std::endl;
    std::cout << "t_last_cur: " << endl << t_last_cur << std::endl;

    gtsam::Pose3 odometry =
        gtsam::Pose3(gtsam::Rot3(R_last_cur(0, 0), R_last_cur(0, 1), R_last_cur(0, 2),
                                 R_last_cur(1, 0), R_last_cur(1, 1), R_last_cur(1, 2),
                                 R_last_cur(2, 0), R_last_cur(2, 1), R_last_cur(2, 2)),
                     gtsam::Point3(t_last_cur(0), t_last_cur(1), t_last_cur(2)));

    // auto odometryNoise =
    //     gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(1e-4));
    auto odometryNoise = gtsam::noiseModel::Gaussian::Covariance(poseCovariance);
    // auto odometryNoise =
    //     gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(poseCovDet * 1e5));
    int i = graph.size();
    graph.add(gtsam::BetweenFactor<gtsam::Pose3>(i, i + 1, odometry, odometryNoise));

    gtsam::Pose3 init_pose = gtsam::Pose3(
        gtsam::Rot3(pose.pose.pose.orientation.w, pose.pose.pose.orientation.x,
                    pose.pose.pose.orientation.y, pose.pose.pose.orientation.z),
        gtsam::Point3(pose.pose.pose.position.x, pose.pose.pose.position.y,
                      pose.pose.pose.position.z));
    initialEstimate.insert(graph.size(), init_pose);

    // initialEstimate.print("Initial estimate:\n");

    // gtsam::LevenbergMarquardtOptimizer optimizer(graph, initialEstimate);
    // gtsam::Values result = optimizer.optimize();

    // graph.print("Final factor graph:\n");

    gtsam::Marginals marginals(graph, initialEstimate);
    double d_opt = marginals.marginalCovariance(graph.size() - 1).determinant();
    double t_opt = marginals.marginalCovariance(graph.size() - 1).trace();
    cout << "graph size: " << graph.size() << ", D-opt: " << d_opt << ", T-opt: " << t_opt
         << endl;

    std_msgs::Float32MultiArray cov_msg;
    cov_msg.data.push_back(d_opt);
    cov_msg.data.push_back(poseCovDet);
    pub_cov.publish(cov_msg);

    log_file << d_opt << " " << poseCovDet << " " << poseCovariance(0, 0) << " "
             << poseCovariance(1, 1) << " " << poseCovariance(2, 2) << " "
             << poseCovariance(3, 3) << " " << poseCovariance(4, 4) << " "
             << poseCovariance(5, 5) << std::endl;

  } else {
    if (graph.size() == 0) {
      gtsam::Pose3 priorMean =
          gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(0, 0, 0));
      auto priorNoise =
          gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(1e-15));
      graph.add(gtsam::PriorFactor<gtsam::Pose3>(1, priorMean, priorNoise));
      ROS_INFO("Added prior factor");

      gtsam::Pose3 init_pose = gtsam::Pose3(
          gtsam::Rot3(pose.pose.pose.orientation.w, pose.pose.pose.orientation.x,
                      pose.pose.pose.orientation.y, pose.pose.pose.orientation.z),
          gtsam::Point3(pose.pose.pose.position.x, pose.pose.pose.position.y,
                        pose.pose.pose.position.z));
      initialEstimate.insert(1, init_pose);

      // gtsam::LevenbergMarquardtOptimizer optimizer(graph, initialEstimate);
      // gtsam::Values result = optimizer.optimize();
    } else {
      ROS_ERROR("No last features");
    }
  }

  last_image0 = image0;
  last_features = n_pts;
  last_landmarks = landmarks;
  last_pose = pose;

}

int main(int argc, char **argv) {
  ros::init(argc, argv, "vins_estimator");
  ros::NodeHandle n("~");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

  ros::Duration(1).sleep();

  if (argc != 2) {
    printf("please intput: rosrun vins vins_node [config file] \n"
           "for example: rosrun vins vins_node "
           "~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml \n");
    return 1;
  }

  string config_file = argv[1];
  printf("config_file: %s\n", argv[1]);

  readParameters(config_file);
  estimator.setParameter();

  R_imu_cam0 = RIC[0];
  R_imu_cam1 = RIC[1];
  t_imu_cam0 = TIC[0];
  t_imu_cam1 = TIC[1];

  R_cam_l_r = R_imu_cam0.transpose() * R_imu_cam1;
  t_cam_l_r = R_imu_cam0.transpose() * (t_imu_cam1 - t_imu_cam0);

  std::string cam0_file, cam1_file;
  camodocal::CameraPtr cam0, cam1;
  cam0_file =
      "/home/eason/workspace/exploration_ws/src/VINS-Fusion/config/realsense_d435i/left.yaml";
  cam1_file =
      "/home/eason/workspace/exploration_ws/src/VINS-Fusion/config/realsense_d435i/right.yaml";
  cam0 = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(cam0_file);
  cam1 = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(cam1_file);
  stereo_camera_.push_back(cam0);
  stereo_camera_.push_back(cam1);

#ifdef EIGEN_DONT_PARALLELIZE
  ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

  // ROS_WARN("waiting for image and imu...");

  registerPub(n);

  pub_landmarks = n.advertise<sensor_msgs::PointCloud2>("/vins_estimator/landmarks", 1);
  pub_last_landmarks = n.advertise<sensor_msgs::PointCloud2>("/vins_estimator/last_landmarks", 1);
  pub_cov = n.advertise<std_msgs::Float32MultiArray>("/vins_estimator/cov", 1);
  pub_feature_img = n.advertise<sensor_msgs::Image>("/vins_estimator/cov_feature_img", 1);

  ros::Subscriber sub_pose = n.subscribe(POSE_TOPIC, 2000, pose_callback);
  ros::Subscriber sub_img0 = n.subscribe(IMAGE0_TOPIC, 100, img0_callback);
  ros::Subscriber sub_img1 = n.subscribe(IMAGE1_TOPIC, 100, img1_callback);

  message_filters::Subscriber<sensor_msgs::Image> image0_sub(n, IMAGE0_TOPIC, 1);
  message_filters::Subscriber<sensor_msgs::Image> image1_sub(n, IMAGE1_TOPIC, 1);
  message_filters::Subscriber<nav_msgs::Odometry> odom_sub(n, POSE_TOPIC, 1);

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image,
                                                          nav_msgs::Odometry>
      MySyncPolicy;
  message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image0_sub, image1_sub,
                                                   odom_sub);
  sync.registerCallback(boost::bind(&syncCallback, _1, _2, _3));

  log_file.open("/home/eason/workspace/exploration_ws/src/VINS-Fusion/log/covariance.log");

  // std::thread sync_thread{sync_process};
  ros::spin();

  return 0;
}
