#include <fstream>
#include <map>
#include <mutex>
#include <queue>
#include <stdio.h>
#include <thread>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/String.h>

#include <gtsam/base/serialization.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/export.hpp> // BOOST_CLASS_EXPORT_GUID
#include <boost/serialization/serialization.hpp>

#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "exploration_utils/SerializedGraphMsg.h"
#include "utility/visualization.h"

using namespace std;

// namespace boost {
// namespace serialization {
// template <class Archive, typename Derived>
// void serialize(Archive &ar, Eigen::EigenBase<Derived> &g, const unsigned int version) {
//   ar &boost::serialization::make_array(g.derived().data(), g.size());
// }
// } // namespace serialization
// } // namespace boost

// /* Create GUIDs for Noisemodels */
// /* ************************************************************************* */
// // BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Constrained, "gtsam_noiseModel_Constrained");
// BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Diagonal, "gtsam_noiseModel_Diagonal");
// BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Gaussian, "gtsam_noiseModel_Gaussian");
// BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Unit, "gtsam_noiseModel_Unit");
// BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Isotropic, "gtsam_noiseModel_Isotropic");
// // BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Robust, "gtsam_noiseModel_Robust");

// // BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::mEstimator::Base,
// "gtsam_noiseModel_mEstimator_Base");
// // BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::mEstimator::Null,
// "gtsam_noiseModel_mEstimator_Null");
// // BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::mEstimator::Fair,
// "gtsam_noiseModel_mEstimator_Fair");
// // BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::mEstimator::Huber,
// // "gtsam_noiseModel_mEstimator_Huber");
// // BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::mEstimator::Tukey,
// // "gtsam_noiseModel_mEstimator_Tukey");

// // BOOST_CLASS_EXPORT_GUID(gtsam::SharedNoiseModel, "gtsam_SharedNoiseModel");
// // BOOST_CLASS_EXPORT_GUID(gtsam::SharedDiagonal, "gtsam_SharedDiagonal");

// /* Create GUIDs for geometry */
// /* ************************************************************************* */
// // GTSAM_VALUE_EXPORT(gtsam::Point2);
// // GTSAM_VALUE_EXPORT(gtsam::Point3);
// // GTSAM_VALUE_EXPORT(gtsam::Rot2);
// // GTSAM_VALUE_EXPORT(gtsam::Rot3);
// // GTSAM_VALUE_EXPORT(gtsam::Pose2);
// // GTSAM_VALUE_EXPORT(gtsam::Pose3);

// BOOST_CLASS_EXPORT_GUID(gtsam::BetweenFactor<gtsam::Point3>,
// "gtsam::BetweenFactor<gtsam::Point3>");
// BOOST_CLASS_EXPORT_GUID(gtsam::BetweenFactor<gtsam::Rot3>, "gtsam::BetweenFactor<gtsam::Rot3>");
// BOOST_CLASS_EXPORT_GUID(gtsam::BetweenFactor<gtsam::Pose3>,
// "gtsam::BetweenFactor<gtsam::Pose3>"); BOOST_CLASS_EXPORT_GUID(gtsam::PriorFactor<gtsam::Point3>,
// "gtsam::PriorFactor<gtsam::Point3>"); BOOST_CLASS_EXPORT_GUID(gtsam::PriorFactor<gtsam::Rot3>,
// "gtsam::PriorFactor<gtsam::Rot3>"); BOOST_CLASS_EXPORT_GUID(gtsam::PriorFactor<gtsam::Pose3>,
// "gtsam::PriorFactor<gtsam::Pose3>");

// BOOST_CLASS_EXPORT_GUID(gtsam::Values, "gtsam_Values");
// BOOST_CLASS_EXPORT_GUID(gtsam::NonlinearFactorGraph, "gtsam_NonlinearFactorGraph");

BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Constrained, "gtsam_noiseModel_Constrained");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Diagonal, "gtsam_noiseModel_Diagonal");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Gaussian, "gtsam_noiseModel_Gaussian");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Unit, "gtsam_noiseModel_Unit");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Isotropic, "gtsam_noiseModel_Isotropic");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Robust, "gtsam_noiseModel_Robust");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::mEstimator::Base, "gtsam_noiseModel_mEstimator_Base");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::mEstimator::Null, "gtsam_noiseModel_mEstimator_Null");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::mEstimator::Fair, "gtsam_noiseModel_mEstimator_Fair");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::mEstimator::Huber, "gtsam_noiseModel_mEstimator_Huber");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::mEstimator::Tukey, "gtsam_noiseModel_mEstimator_Tukey");
BOOST_CLASS_EXPORT_GUID(gtsam::SharedNoiseModel, "gtsam_SharedNoiseModel");
BOOST_CLASS_EXPORT_GUID(gtsam::SharedDiagonal, "gtsam_SharedDiagonal");
BOOST_CLASS_EXPORT_GUID(gtsam::JacobianFactor, "gtsam::JacobianFactor");
BOOST_CLASS_EXPORT_GUID(gtsam::BetweenFactor<gtsam::Pose3>, "gtsam::BetweenFactor<gtsam::Pose3>");
BOOST_CLASS_EXPORT_GUID(gtsam::BetweenFactor<gtsam::Point3>, "gtsam::BetweenFactor<gtsam::Point3>");
BOOST_CLASS_EXPORT_GUID(gtsam::BetweenFactor<gtsam::Rot3>, "gtsam::BetweenFactor<gtsam::Rot3>");
BOOST_CLASS_EXPORT_GUID(gtsam::PriorFactor<gtsam::Point3>, "gtsam::PriorFactor<gtsam::Point3>");
BOOST_CLASS_EXPORT_GUID(gtsam::PriorFactor<gtsam::Rot3>, "gtsam::PriorFactor<gtsam::Rot3>");
BOOST_CLASS_EXPORT_GUID(gtsam::PriorFactor<gtsam::Pose3>, "gtsam::PriorFactor<gtsam::Pose3>");
BOOST_CLASS_EXPORT_GUID(gtsam::Values, "gtsam::Values");
BOOST_CLASS_EXPORT_GUID(gtsam::NonlinearFactorGraph, "gtsam::NonlinearFactorGraph");
BOOST_CLASS_EXPORT_GUID(gtsam::FactorGraph<gtsam::NonlinearFactor>,
                        "gtsam::FactorGraph<gtsam::NonlinearFactor>");
GTSAM_VALUE_EXPORT(gtsam::Point3);
GTSAM_VALUE_EXPORT(gtsam::Rot3);
GTSAM_VALUE_EXPORT(gtsam::Pose3);

// Estimator estimator;

std::string POSE_TOPIC = "/vins_estimator/odometry";

queue<nav_msgs::Odometry> pose_buf;
queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
std::mutex m_buf;

Eigen::Matrix3d R_cam_l_r, R_imu_cam0, R_imu_cam1;
Eigen::Vector3d t_cam_l_r, t_imu_cam0, t_imu_cam1;

std::vector<camodocal::CameraPtr> stereo_camera_;

ros::Publisher pub_landmarks, pub_last_landmarks, pub_cov, pub_feature_img, pub_pose_array,
    pub_serialized_graph;

gtsam::NonlinearFactorGraph graph;
gtsam::Values initialEstimate;

double graph_node_distance = 0.5;

int num_poses = 0;

// log file
std::ofstream log_file;

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg) {
  ROS_INFO_ONCE("Received image %d x %d, encoding: %s", img_msg->width, img_msg->height,
                img_msg->encoding.c_str());

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

Eigen::MatrixXd onlyDiagnal(const Eigen::MatrixXd &mat) {
  if (mat.rows() != mat.cols()) {
    ROS_ERROR("Error: Matrix must be square!");
  }

  Eigen::MatrixXd diag = Eigen::MatrixXd::Zero(mat.rows(), mat.cols());
  for (int i = 0; i < mat.rows(); ++i) {
    diag(i, i) = mat(i, i);
  }

  return diag;
}

cv::Mat last_image0;
vector<cv::Point2f> last_features;
vector<cv::Point3f> last_landmarks;
nav_msgs::Odometry last_pose;

std::vector<Matrix<double, 6, 6>> cov_buffer;
geometry_msgs::PoseArray pose_array;

// sync callback
void syncCallback(const sensor_msgs::ImageConstPtr &img0_msg,
                  const sensor_msgs::ImageConstPtr &img1_msg,
                  const nav_msgs::OdometryConstPtr &pose_msg) {
  ros::Time t_sync = ros::Time::now();

  double t_optical_flow = 0.0;

  ros::Time t_optical_flow_start, t_optical_flow_end;
  // print three timestamps
  // std::cout << std::setprecision(16);
  // std::cout << "receive img0 ts: " << img0_msg->header.stamp.toSec()
  //           << ", img1 ts: " << img1_msg->header.stamp.toSec()
  //           << ", pose ts: " << pose_msg->header.stamp.toSec() << std::endl;

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
  t_optical_flow_start = ros::Time::now();
  cv::calcOpticalFlowPyrLK(image0, image1, n_pts, n_pts_right, status, err, cv::Size(10, 10), 3);

  cv::calcOpticalFlowPyrLK(image1, image0, n_pts_right, reverseLeftPts, statusRightLeft, err,
                           cv::Size(10, 10), 3);
  t_optical_flow_end = ros::Time::now();
  t_optical_flow += (t_optical_flow_end - t_optical_flow_start).toSec();

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

      t_optical_flow_start = ros::Time::now();

      cv::calcOpticalFlowPyrLK(last_image0, image0, last_features, last_features_on_cur, status,
                               err, cv::Size(10, 10), 3);
      // std::cout << "last_features_on_cur size: " << last_features_on_cur.size() <<
      // std::endl;

      vector<cv::Point2f> last_features_on_cur_right;
      vector<uchar> last_features_status, last_features_status_right;
      err.clear();
      cv::calcOpticalFlowPyrLK(image0, image1, last_features_on_cur, last_features_on_cur_right,
                               last_features_status, err, cv::Size(10, 10), 3);
      reverseLeftPts.clear();
      err.clear();
      cv::calcOpticalFlowPyrLK(image1, image0, last_features_on_cur_right, reverseLeftPts,
                               last_features_status_right, err, cv::Size(10, 10), 3);
      t_optical_flow_end = ros::Time::now();
      t_optical_flow += (t_optical_flow_end - t_optical_flow_start).toSec();

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
      // std::cout << "last_landmarks size: " << last_landmarks.size() << std::endl;

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

    ros::Time t_pose_cov = ros::Time::now();

    Matrix<double, 6, 6> poseCovariance = Matrix<double, 6, 6>::Identity() * 1e-10;
    double poseCovDet = 1e-10;
    double poseCovTrace = poseCovariance.trace();
    if (last_landmarks_cur.size() >= 3 && last_landmarks.size() >= 3) {
      // calculate the covariance of last_landmarks_cur and landmarks
      Eigen::Matrix<double, 3, 3> cov;
      computePointsCovariance(last_landmarks_cur, last_landmarks, cov);

      // Compute pose covariance considering heterogeneous noise
      poseCovariance = computePoseCovariance(last_landmarks_cur, last_landmarks, R_w_b, t_w_b, cov);

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
        if (poseCovariance(i, i) < 1e-10) {
          poseCovariance(i, i) = 1e-10;
        }
      }

    } else {
      poseCovariance << 0.1, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 1e-4,
          0, 0, 0, 0, 0, 0, 1e-4, 0, 0, 0, 0, 0, 0, 1e-4;
    }

    // if nan
    if (poseCovariance.hasNaN()) {
      ROS_ERROR("Pose covariance has NaN!");
      poseCovariance << 0.1, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 1e-4,
          0, 0, 0, 0, 0, 0, 1e-4, 0, 0, 0, 0, 0, 0, 1e-4;
      poseCovDet = 1e-10;
      poseCovTrace = poseCovariance.trace();
    }

    // cout << "Pose Covariance time: " << (ros::Time::now() - t_pose_cov).toSec() << endl;

    // Display result
    // cout << "Pose Covariance Matrix:\n" << poseCovariance << endl;
    // cout << "det(poseCovariance): " << poseCovDet << endl;
    // cout << "trace(poseCovariance): " << poseCovTrace << endl;

    // get last pose on graph
    gtsam::Pose3 last_pose_on_graph = initialEstimate.at<gtsam::Pose3>(num_poses - 1);

    Eigen::Matrix3d R_w_b_last = last_pose_on_graph.rotation().matrix();
    Eigen::Vector3d t_w_b_last(last_pose_on_graph.translation().x(),
                               last_pose_on_graph.translation().y(),
                               last_pose_on_graph.translation().z());

    // print R_w_c_last and t_w_c_last
    // std::cout << "R_w_c_last: " << R_w_c_last << std::endl;
    // std::cout << "t_w_c_last: " << t_w_c_last << std::endl;

    Eigen::Matrix3d R_last_cur = R_w_b_last.transpose() * R_w_b;
    Eigen::Vector3d t_last_cur = R_w_b_last.transpose() * (t_w_b - t_w_b_last);

    // print R_last_cur and t_last_cur
    // std::cout << "R_last_cur: " << endl << R_last_cur << std::endl;
    // std::cout << "t_last_cur: " << endl << t_last_cur << std::endl;

    if (t_last_cur.norm() > graph_node_distance) {
      std::cout << "add factor" << std::endl;

      gtsam::Pose3 odometry =
          gtsam::Pose3(gtsam::Rot3(R_last_cur(0, 0), R_last_cur(0, 1), R_last_cur(0, 2),
                                   R_last_cur(1, 0), R_last_cur(1, 1), R_last_cur(1, 2),
                                   R_last_cur(2, 0), R_last_cur(2, 1), R_last_cur(2, 2)),
                       gtsam::Point3(t_last_cur(0), t_last_cur(1), t_last_cur(2)));

      // auto odometryNoise =
      //     gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(1e-4));
      // auto odometryNoise = gtsam::noiseModel::Gaussian::Covariance(poseCovariance);
      std::cout << "poseCovariance: " << poseCovariance << std::endl;
      auto odometryNoise = gtsam::noiseModel::Gaussian::Covariance(onlyDiagnal(poseCovariance));
      // auto odometryNoise =
      //     gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(poseCovDet * 1e5));
      // int i = graph.size();
      graph.add(
          gtsam::BetweenFactor<gtsam::Pose3>(num_poses - 1, num_poses, odometry, odometryNoise));

      gtsam::Pose3 init_pose =
          gtsam::Pose3(gtsam::Rot3(pose.pose.pose.orientation.w, pose.pose.pose.orientation.x,
                                   pose.pose.pose.orientation.y, pose.pose.pose.orientation.z),
                       gtsam::Point3(pose.pose.pose.position.x, pose.pose.pose.position.y,
                                     pose.pose.pose.position.z));
      initialEstimate.insert(num_poses, init_pose);
      num_poses++;

      pose_array.header = pose.header;
      pose_array.poses.push_back(pose.pose.pose);
      pub_pose_array.publish(pose_array);
      // initialEstimate.print("Initial estimate:\n");

      // gtsam::LevenbergMarquardtOptimizer optimizer(graph, initialEstimate);
      // gtsam::Values result = optimizer.optimize();

      // graph.print("Final factor graph:\n");

      auto getDOptimal = [&](const Eigen::MatrixXd &mat) {
        // // eigen values
        // Eigen::EigenSolver<Eigen::MatrixXd> es(mat);
        // Eigen::VectorXcd eigenvalues = es.eigenvalues();

        // // print eigenvalues
        // std::cout << "Eigenvalues:\n" << eigenvalues.transpose() << std::endl;

        // // d-optimal: exp(1/n * sum(log(eigenvalues)))
        // double d_optimal = 0;
        // for (int i = 0; i < eigenvalues.size(); ++i) {
        //   d_optimal += log(eigenvalues(i).real());
        // }
        // d_optimal /= 6;
        // d_optimal = exp(d_optimal);

        // print mat.determinant()
        // std::cout << "Determinant of the Matrix: " << mat.determinant() << std::endl;
        // print pow(mat.determinant(), 1.0 / mat.rows())
        // std::cout << "Pow of the Matrix: " << pow(mat.determinant(), 1.0 / mat.rows()) <<
        // std::endl;

        double d_optimal = pow(mat.determinant(), 1.0 / mat.rows());

        // int n = mat.rows() + 1;
        // double d_optimal = exp(log(mat.determinant()) / (double)n);
        // d_optimal *= pow(n, 1.0 / n);

        return d_optimal;
      };

      gtsam::Marginals marginals(graph, initialEstimate);
      Matrix<double, 6, 6> nodeCovariance = marginals.marginalCovariance(num_poses - 1);
      Matrix<double, 6, 6> nodeInformation = marginals.marginalInformation(num_poses - 1);
      double d_opt = getDOptimal(nodeCovariance);
      double d_opt_info = getDOptimal(nodeInformation);

      cout << "Node Covariance Matrix:\n" << nodeCovariance << endl;
      cout << "Node Information Matrix:\n" << nodeInformation << endl;

      cout << "graph size: " << graph.size() << ", last pose D-opt: " << d_opt << endl;
      cout << "graph size: " << graph.size() << ", last pose D-opt info: " << d_opt_info << endl;
      cout << "num_poses: " << num_poses << endl;

      // terminate program if d_opt is nan
      if (std::isnan(d_opt)) {
        ROS_ERROR("D-optimal is NaN!");
        exit(EXIT_FAILURE);
      }

      // log_file << d_opt << " " << t_opt << " " << nodeCovariance(0, 0) << " "
      //          << nodeCovariance(1, 1) << " " << nodeCovariance(2, 2) << " " << nodeCovariance(3,
      //          3)
      //          << " " << nodeCovariance(4, 4) << " " << nodeCovariance(5, 5) << std::endl;

      // log_file << d_opt << " " << t_opt << " " << poseCovDet << " " << poseCovTrace << std::endl;

      ros::Time t_cov = ros::Time::now();
      Eigen::MatrixXd A = Eigen::MatrixXd::Zero(num_poses, num_poses);

      for (const auto &factor : graph) {
        if (auto betweenFactor =
                dynamic_cast<const gtsam::BetweenFactor<gtsam::Pose3> *>(factor.get())) {
          gtsam::Key key1 = betweenFactor->key1();
          gtsam::Key key2 = betweenFactor->key2();
          // std::cout << "Key1: " << key1 << ", Key2: " << key2 << std::endl;
          A(key1, key2) = 1;
          A(key2, key1) = 1;
        }
      }
      // cout << "A:\n" << A << endl;

      gtsam::Matrix D = gtsam::Matrix::Zero(A.rows(), A.cols());
      for (int i = 0; i < A.rows(); ++i) {
        D(i, i) = A.row(i).sum();
      }

      // cout << "D:\n" << D << endl;

      gtsam::Matrix laplacian = D - A;

      // Laplacian matrix of the pose graph is weighted by the D-optimal of the inverse of the
      // covariance matrices for each edge in the pose graph.
      gtsam::Matrix weighted_laplacian = laplacian;

      // get edge weight
      auto getEdgeWeight = [&](const gtsam::BetweenFactor<gtsam::Pose3> &factor) {
        gtsam::Matrix edge_fim = factor.noiseModel()->sigmas().asDiagonal().inverse();
        // lp-norm of fim
        // double lp_norm = edge_fim.lpNorm<Eigen::Infinity>();
        // cout << "Edge FIM:\n" << edge_fim << endl;
        // cout << "Edge LP-Norm: " << lp_norm << endl;
        // return lp_norm;

        return getDOptimal(edge_fim);
      };

      // set diagonal to 0
      for (int i = 0; i < weighted_laplacian.rows(); ++i) {
        weighted_laplacian(i, i) = 0;
      }

      for (const auto &factor : graph) {
        if (auto betweenFactor =
                dynamic_cast<const gtsam::BetweenFactor<gtsam::Pose3> *>(factor.get())) {
          gtsam::Key key1 = betweenFactor->key1();
          // cout << "Key1: " << key1 << endl;
          gtsam::Key key2 = betweenFactor->key2();
          // cout << "Key2: " << key2 << endl;
          double edge_weight = getEdgeWeight(*betweenFactor);
          // cout << "Edge weight:\n" << edge_weight << endl;
          // double d_optimal = edge_weight.determinant();
          // cout << "D-Optimal: " << d_optimal << endl;

          weighted_laplacian(key1, key2) *= edge_weight;
          weighted_laplacian(key2, key1) *= edge_weight;

          // diagnal
          weighted_laplacian(key1, key1) += edge_weight;
          weighted_laplacian(key2, key2) += edge_weight;
        }
      }

      // calculate eigenvalues of weighted_laplacian
      Eigen::EigenSolver<Eigen::MatrixXd> es(weighted_laplacian);
      Eigen::VectorXcd eigenvalues = es.eigenvalues();

      // print eigenvalues
      std::cout << "Eigenvalues:\n" << eigenvalues.transpose() << std::endl;

      double product_eigenvalues_nonzero = 1;
      for (int i = 0; i < eigenvalues.size(); ++i) {
        if (eigenvalues(i).real() > 1e-10) {
          product_eigenvalues_nonzero *= eigenvalues(i).real();
        }
      }

      // print eigenvalues product
      std::cout << "Eigenvalues product: " << product_eigenvalues_nonzero << std::endl;
      std::cout << "Eigenvalues product / n: "
                << product_eigenvalues_nonzero / weighted_laplacian.rows() << std::endl;

      // remove the last row and column (remove an arbitrary node)
      weighted_laplacian.conservativeResize(weighted_laplacian.rows() - 1,
                                            weighted_laplacian.cols() - 1);

      // calculate eigenvalues of reduced weighted_laplacian
      Eigen::EigenSolver<Eigen::MatrixXd> es_reduced(weighted_laplacian);
      Eigen::VectorXcd eigenvalues_reduced = es_reduced.eigenvalues();

      // print eigenvalues
      std::cout << "Eigenvalues Reduced:\n" << eigenvalues_reduced.transpose() << std::endl;
      std::cout << "Eigenvalues Reduced product: " << eigenvalues_reduced.prod() << std::endl;

      // print the determinant of the reduced weighted_laplacian
      std::cout << "Determinant of the Reduced Weighted Laplacian Matrix: "
                << weighted_laplacian.determinant() << std::endl;

      // std::cout << "Laplacian Matrix:\n" << laplacian << std::endl;
      // std::cout << "Weighted Laplacian Matrix:\n" << weighted_laplacian << std::endl;

      // determinant of the Laplacian matrix
      // std::cout << "Determinant of the Laplacian Matrix: " << laplacian.determinant() <<
      // std::endl;
      double d_opt_weighted_laplacian = getDOptimal(weighted_laplacian);
      std::cout << "D-opt of the Weighted Laplacian Matrix: " << d_opt_weighted_laplacian
                << std::endl;
      // trace
      // std::cout << "Trace of the Weighted Laplacian Matrix: " << weighted_laplacian.trace()
      //           << std::endl;

      std_msgs::Float32MultiArray cov_msg;
      cov_msg.data.push_back(d_opt);
      cov_msg.data.push_back(d_opt_weighted_laplacian);
      pub_cov.publish(cov_msg);

      cout << "cov time: " << (ros::Time::now() - t_cov).toSec() << endl;

      exploration_utils::SerializedGraphMsg serialized_msg;
      serialized_msg.header.stamp = ros::Time::now();
      serialized_msg.serialized_graph = gtsam::serialize(graph);
      serialized_msg.serialized_values = gtsam::serialize(initialEstimate);

      pub_serialized_graph.publish(serialized_msg);

      // deserialize
      // gtsam::NonlinearFactorGraph graph_deserialized;
      // gtsam::deserialize(serialized_graph, graph_deserialized);
      // graph_deserialized.print("Deserialized Factor Graph:\n");
    }
  } else {
    if (graph.size() == 0) {
      gtsam::Pose3 priorMean = gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(0, 0, 0));
      auto priorNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(1e-1));
      graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, priorMean, priorNoise));
      ROS_INFO("Added prior factor");

      gtsam::Pose3 init_pose =
          gtsam::Pose3(gtsam::Rot3(pose.pose.pose.orientation.w, pose.pose.pose.orientation.x,
                                   pose.pose.pose.orientation.y, pose.pose.pose.orientation.z),
                       gtsam::Point3(pose.pose.pose.position.x, pose.pose.pose.position.y, 1.0));
      initialEstimate.insert(0, init_pose);

      num_poses++;

      pose_array.header = pose.header;
      geometry_msgs::Pose pose_msg;
      pose_msg = pose.pose.pose;
      pose_msg.position.z = 1.0;
      pose_array.poses.push_back(pose_msg);
      pub_pose_array.publish(pose_array);

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

  // std::cout << "sync time: " << (ros::Time::now() - t_sync).toSec() << std::endl;
  // std::cout << "optical flow time: " << t_optical_flow << std::endl;
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
  // estimator.setParameter();

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

  // registerPub(n);

  // read ros param
  n.param("graph_node_distance", graph_node_distance, 1.0);

  pub_landmarks = n.advertise<sensor_msgs::PointCloud2>("/vins_estimator/landmarks", 1);
  pub_last_landmarks = n.advertise<sensor_msgs::PointCloud2>("/vins_estimator/last_landmarks", 1);
  pub_cov = n.advertise<std_msgs::Float32MultiArray>("/vins_estimator/cov", 1);
  pub_feature_img = n.advertise<sensor_msgs::Image>("/vins_estimator/cov_feature_img", 1);
  pub_pose_array = n.advertise<geometry_msgs::PoseArray>("/vins_estimator/graph_pose_array", 1);
  pub_serialized_graph =
      n.advertise<exploration_utils::SerializedGraphMsg>("/vins_estimator/serialized_graph", 1);

  message_filters::Subscriber<sensor_msgs::Image> image0_sub(n, IMAGE0_TOPIC, 1);
  message_filters::Subscriber<sensor_msgs::Image> image1_sub(n, IMAGE1_TOPIC, 1);
  message_filters::Subscriber<nav_msgs::Odometry> odom_sub(n, POSE_TOPIC, 1);

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image,
                                                          nav_msgs::Odometry>
      MySyncPolicy;
  message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image0_sub, image1_sub,
                                                   odom_sub);
  sync.registerCallback(boost::bind(&syncCallback, _1, _2, _3));

  // log_file.open("/home/eason/workspace/exploration_ws/src/VINS-Fusion/log/covariance.log");

  ros::Rate rate(100);
  while (ros::ok()) {
    ros::spinOnce();
    rate.sleep();
  }

  // log_file.close();

  std::cout << "Covariance estimator finished!" << std::endl;

  return 0;
}
