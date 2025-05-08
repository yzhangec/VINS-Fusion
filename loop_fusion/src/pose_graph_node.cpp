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

#include "keyframe.h"
#include "parameters.h"
#include "pose_graph.h"
#include "utility/CameraPoseVisualization.h"
#include "utility/tic_toc.h"
#include <cv_bridge/cv_bridge.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <mutex>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <queue>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Bool.h>
#include <thread>
#include <vector>
#include <visualization_msgs/Marker.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>

#include "CNN/mobilenetvlad_onnx.h"
#include "CNN/onnx_generic.h"
#include "CNN/superpoint_onnx.h"
#include "backward/backward.hpp"
#include "loop_fusion/StereoImage.h"

namespace backward {
backward::SignalHandling sh;
}

#define SKIP_FIRST_CNT 10
using namespace std;
using namespace loop_closure;

// queue<loop_fusion::StereoImagePtr> image_buf;
// queue<sensor_msgs::PointCloudConstPtr> point_buf;
queue<nav_msgs::Odometry::ConstPtr> pose_buf, pose_gt_buf;
queue<Eigen::Vector3d> odometry_buf;

queue<std::tuple<sensor_msgs::ImageConstPtr, sensor_msgs::ImageConstPtr,
                 nav_msgs::Odometry::ConstPtr>>
    image_pose_buf;
queue<std::tuple<sensor_msgs::ImageConstPtr, sensor_msgs::ImageConstPtr,
                 nav_msgs::Odometry::ConstPtr, nav_msgs::Odometry::ConstPtr>>
    image_pose_gt_buf;

std::mutex m_buf;
std::mutex m_process;
int frame_index = 0;
int sequence = 1;
PoseGraph posegraph;
int skip_first_cnt = 0;
int SKIP_CNT;
int skip_cnt = 0;
bool load_flag = 0;
bool start_flag = 0;
double SKIP_DIS = 0;
bool has_gt_pose = true;

int VISUALIZATION_SHIFT_X;
int VISUALIZATION_SHIFT_Y;
int ROW;
int COL;
int DEBUG_IMAGE;

Eigen::Vector3d tic;
Eigen::Matrix3d qic;
ros::Publisher pub_match_img;
ros::Publisher pub_camera_pose_visual;
ros::Publisher pub_odometry_rect;
ros::Publisher debug_marker_array_pub_;

std::string BRIEF_PATTERN_FILE;
std::string POSE_GRAPH_SAVE_PATH;
std::string VINS_RESULT_PATH;
CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
Eigen::Vector3d last_t(-100, -100, -100);
double last_image_time = -1;

ros::Publisher pub_point_cloud, pub_margin_cloud;

MobileNetVLADONNX *netvlad_onnx = nullptr;
SuperPointONNX *superpoint_onnx = nullptr;
Eigen::Matrix4d T_cam_l_r, T_i_c;
std::vector<camodocal::CameraPtr> stereo_camera_;

// typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image>
//     ImageSyncPolicy;
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image,
                                                        nav_msgs::Odometry>
    ImageOdometrySyncPolicy;
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image,
                                                        nav_msgs::Odometry, nav_msgs::Odometry>
    ImageOdometryGTSyncPolicy;

void drawFeatureOnImage(cv::Mat &image, const std::vector<cv::Point2f> &pts,
                        const cv::Scalar &color) {
  for (const cv::Point2f &pt : pts) {
    cv::circle(image, pt, 3, color, 1);
  }
}

bool inBorder(const cv::Point2f &pt, const int &row, const int &col) {
  const int BORDER_SIZE = 1;
  int img_x = cvRound(pt.x);
  int img_y = cvRound(pt.y);
  return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y &&
         img_y < row - BORDER_SIZE;
}

double distance(cv::Point2f pt1, cv::Point2f pt2) {
  double dx = pt1.x - pt2.x;
  double dy = pt1.y - pt2.y;
  return sqrt(dx * dx + dy * dy);
}

template <typename Derived> void reduceVector(std::vector<Derived> &v, std::vector<uchar> status) {
  int j = 0;
  for (int i = 0; i < int(v.size()); i++)
    if (status[i])
      v[j++] = v[i];
  v.resize(j);
}

template <typename Derived>
void reduceDescriptorVector(std::vector<Derived> &v, std::vector<uchar> status) {
  // v.size FEATURE_DESC_SIZE * status.size()
  int j = 0;
  int FEATURE_DESC_SIZE = 256;
  for (int i = 0; i < int(status.size()); i++)
    if (status[i]) {
      for (int k = 0; k < FEATURE_DESC_SIZE; k++) {
        v[j * FEATURE_DESC_SIZE + k] = v[i * FEATURE_DESC_SIZE + k];
      }
      j++;
    }
  v.resize(j * FEATURE_DESC_SIZE);
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
  P2.block(0, 0, 3, 3) = T_cam_l_r.block(0, 0, 3, 3).transpose();
  P2.block(0, 3, 3, 1) = -P2.block(0, 0, 3, 3) * T_cam_l_r.block(0, 3, 3, 1);

  status.clear();

  for (unsigned int i = 0; i < left_pts.size(); ++i) {
    Vector2d pl(left_pts[i].x, left_pts[i].y);
    Vector2d pr(right_pts[i].x, right_pts[i].y);
    Vector3d pt3;
    triangulatePoint(P1, P2, pl, pr, pt3);

    if (pt3[2] > 0 && pt3[2] < 5) {
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
    // 将像素坐标转化为无畸变的归一化坐标
    cam->liftProjective(a, b);
    un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
  }
}

// void new_sequence() {
//   printf("new sequence\n");
//   sequence++;
//   printf("sequence cnt %d \n", sequence);
//   if (sequence > 5) {
//     ROS_WARN("only support 5 sequences since it's boring to copy code for more sequences.");
//     ROS_BREAK();
//   }
//   posegraph.posegraph_visualization->reset();
//   posegraph.publish();
//   m_buf.lock();
//   while (!image_buf.empty())
//     image_buf.pop();
//   while (!point_buf.empty())
//     point_buf.pop();
//   while (!pose_buf.empty())
//     pose_buf.pop();
//   while (!odometry_buf.empty())
//     odometry_buf.pop();
//   m_buf.unlock();
// }

void publishLandmarks(const std::vector<cv::Point3f> &landmarks) {
  visualization_msgs::MarkerArray mk_array;
  visualization_msgs::Marker mk;
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.id = 0;
  mk.ns = "Loop landmarks";
  mk.type = visualization_msgs::Marker::POINTS;
  mk.color.r = 1.0;
  mk.color.g = 1.0;
  mk.color.b = 1.0;
  mk.color.a = 1.0;
  mk.scale.x = 0.05;
  mk.scale.y = 0.05;
  mk.scale.z = 0.05;

  for (const cv::Point3f &point : landmarks) {
    geometry_msgs::Point pt;
    pt.x = point.x;
    pt.y = point.y;
    pt.z = point.z;
    mk.points.push_back(pt);
  }
  mk_array.markers.push_back(mk);

  debug_marker_array_pub_.publish(mk_array);
}

// void image_callback(const sensor_msgs::ImageConstPtr &image0_msg,
//                     const sensor_msgs::ImageConstPtr &image1_msg) {
//   // ROS_INFO("image_callback!");

//   loop_fusion::StereoImagePtr stereo_image_msg(new loop_fusion::StereoImage);
//   stereo_image_msg->header = image0_msg->header;
//   stereo_image_msg->image0 = *image0_msg;
//   stereo_image_msg->image1 = *image1_msg;

//   m_buf.lock();
//   image_buf.push(stereo_image_msg);
//   m_buf.unlock();
//   // printf(" image time %f \n", image_msg->header.stamp.toSec());

//   // detect unstable camera stream
//   // if (last_image_time == -1)
//   //   last_image_time = image_msg->header.stamp.toSec();
//   // else if (image_msg->header.stamp.toSec() - last_image_time > 1.0 ||
//   //          image_msg->header.stamp.toSec() < last_image_time) {
//   //   ROS_WARN("image discontinue! detect a new sequence!");
//   //   // new_sequence();
//   // }
//   // last_image_time = image_msg->header.stamp.toSec();
// }

void image_pose_callback(const sensor_msgs::ImageConstPtr &image0_msg,
                         const sensor_msgs::ImageConstPtr &image1_msg,
                         const nav_msgs::Odometry::ConstPtr &pose_msg) {
  m_buf.lock();
  // std::cout << "image_pose_callback" << std::endl;
  // std::cout << "image0_msg ts:" << image0_msg->header.stamp << std::endl;
  // std::cout << "image1_msg ts:" << image1_msg->header.stamp << std::endl;
  // std::cout << "pose_msg ts:" << pose_msg->header.stamp << std::endl;
  image_pose_buf.push(std::make_tuple(image0_msg, image1_msg, pose_msg));
  m_buf.unlock();
}

void image_pose_gt_callback(const sensor_msgs::ImageConstPtr &image0_msg,
                            const sensor_msgs::ImageConstPtr &image1_msg,
                            const nav_msgs::Odometry::ConstPtr &pose_msg,
                            const nav_msgs::Odometry::ConstPtr &pose_gt_msg) {
  m_buf.lock();
  // std::cout << "image_pose_gt_callback" << std::endl;
  // std::cout << "image0_msg ts:" << image0_msg->header.stamp << std::endl;
  // std::cout << "image1_msg ts:" << image1_msg->header.stamp << std::endl;
  // std::cout << "pose_msg ts:" << pose_msg->header.stamp << std::endl;
  // std::cout << "pose_gt_msg ts:" << pose_gt_msg->header.stamp << std::endl;
  image_pose_gt_buf.push(std::make_tuple(image0_msg, image1_msg, pose_msg, pose_gt_msg));
  m_buf.unlock();
}

void point_callback(const sensor_msgs::PointCloudConstPtr &point_msg) {
  // ROS_INFO("point_callback!");
  // m_buf.lock();
  // point_buf.push(point_msg);
  // m_buf.unlock();

  sensor_msgs::PointCloud point_cloud;
  point_cloud.header = point_msg->header;
  for (unsigned int i = 0; i < point_msg->points.size(); i++) {
    cv::Point3f p_3d;
    p_3d.x = point_msg->points[i].x;
    p_3d.y = point_msg->points[i].y;
    p_3d.z = point_msg->points[i].z;
    Eigen::Vector3d tmp =
        posegraph.r_drift * Eigen::Vector3d(p_3d.x, p_3d.y, p_3d.z) + posegraph.t_drift;
    geometry_msgs::Point32 p;
    p.x = tmp(0);
    p.y = tmp(1);
    p.z = tmp(2);
    point_cloud.points.push_back(p);
  }
  pub_point_cloud.publish(point_cloud);
}

void margin_point_callback(const sensor_msgs::PointCloud2ConstPtr &point_msg) {

  // ROS_INFO("margin_point_callback!");
  pcl::PointCloud<pcl::PointXYZ> cloud;
  pcl::fromROSMsg(*point_msg, cloud);

  for (size_t i = 0; i < cloud.points.size(); i++) {
    Eigen::Vector3d p_3d(cloud.points[i].x, cloud.points[i].y, cloud.points[i].z);
    Eigen::Vector3d tmp = posegraph.r_drift * p_3d + posegraph.t_drift;
    cloud.points[i].x = tmp(0);
    cloud.points[i].y = tmp(1);
    cloud.points[i].z = tmp(2);
  }

  sensor_msgs::PointCloud2 point_cloud;
  pcl::toROSMsg(cloud, point_cloud);
  point_cloud.header = point_msg->header;
  pub_margin_cloud.publish(point_cloud);

  // sensor_msgs::PointCloud point_cloud;
  // point_cloud.header = point_msg->header;
  // for (unsigned int i = 0; i < point_msg->points.size(); i++) {
  //   cv::Point3f p_3d;
  //   p_3d.x = point_msg->points[i].x;
  //   p_3d.y = point_msg->points[i].y;
  //   p_3d.z = point_msg->points[i].z;
  //   Eigen::Vector3d tmp =
  //       posegraph.r_drift * Eigen::Vector3d(p_3d.x, p_3d.y, p_3d.z) + posegraph.t_drift;
  //   geometry_msgs::Point32 p;
  //   p.x = tmp(0);
  //   p.y = tmp(1);
  //   p.z = tmp(2);
  //   point_cloud.points.push_back(p);
  // }
  // pub_margin_cloud.publish(point_cloud);
}

void pose_callback(const nav_msgs::Odometry::ConstPtr &pose_msg) {
  m_buf.lock();
  pose_buf.push(pose_msg);
  m_buf.unlock();
}

void pose_gt_callback(const nav_msgs::Odometry::ConstPtr &pose_msg) {
  m_buf.lock();
  pose_gt_buf.push(pose_msg);
  m_buf.unlock();
}

void vio_callback(const nav_msgs::Odometry::ConstPtr &pose_msg) {
  // ROS_INFO("vio_callback!");
  Vector3d vio_t(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y,
                 pose_msg->pose.pose.position.z);
  Quaterniond vio_q;
  vio_q.w() = pose_msg->pose.pose.orientation.w;
  vio_q.x() = pose_msg->pose.pose.orientation.x;
  vio_q.y() = pose_msg->pose.pose.orientation.y;
  vio_q.z() = pose_msg->pose.pose.orientation.z;

  vio_t = posegraph.w_r_vio * vio_t + posegraph.w_t_vio;
  vio_q = posegraph.w_r_vio * vio_q;

  vio_t = posegraph.r_drift * vio_t + posegraph.t_drift;
  vio_q = posegraph.r_drift * vio_q;

  nav_msgs::Odometry odometry;
  odometry.header = pose_msg->header;
  odometry.header.frame_id = "world";
  odometry.pose.pose.position.x = vio_t.x();
  odometry.pose.pose.position.y = vio_t.y();
  odometry.pose.pose.position.z = vio_t.z();
  odometry.pose.pose.orientation.x = vio_q.x();
  odometry.pose.pose.orientation.y = vio_q.y();
  odometry.pose.pose.orientation.z = vio_q.z();
  odometry.pose.pose.orientation.w = vio_q.w();
  pub_odometry_rect.publish(odometry);

  // Vector3d vio_t_cam;
  // Quaterniond vio_q_cam;
  // vio_t_cam = vio_t + vio_q * tic;
  // vio_q_cam = vio_q * qic;

  // cameraposevisual.reset();
  // cameraposevisual.add_pose(vio_t_cam, vio_q_cam);
  // cameraposevisual.publish_by(pub_camera_pose_visual, pose_msg->header);
}

void extrinsic_callback(const nav_msgs::Odometry::ConstPtr &pose_msg) {
  m_process.lock();
  tic = Vector3d(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y,
                 pose_msg->pose.pose.position.z);
  qic = Quaterniond(pose_msg->pose.pose.orientation.w, pose_msg->pose.pose.orientation.x,
                    pose_msg->pose.pose.orientation.y, pose_msg->pose.pose.orientation.z)
            .toRotationMatrix();
  m_process.unlock();
}

void process() {
  while (true) {
    sensor_msgs::Image::ConstPtr image0_msg = NULL, image1_msg = NULL;
    nav_msgs::Odometry::ConstPtr pose_msg = NULL;
    nav_msgs::Odometry::ConstPtr pose_gt_msg = NULL;

    // if (!has_gt_pose) {
    //   m_buf.lock();
    //   if (!image_buf.empty() && !pose_buf.empty()) {
    //     if (image_buf.front()->header.stamp.toSec() > pose_buf.front()->header.stamp.toSec()) {
    //       pose_buf.pop();
    //       printf("throw pose at beginning\n");
    //     } else if (image_buf.back()->header.stamp.toSec() >=
    //                pose_buf.front()->header.stamp.toSec()) {
    //       pose_msg = pose_buf.front();
    //       pose_buf.pop();
    //       while (!pose_buf.empty())
    //         pose_buf.pop();
    //       while (image_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
    //         image_buf.pop();
    //       image_msg = image_buf.front();
    //       image_buf.pop();
    //     }
    //   }
    //   m_buf.unlock();
    // } else {
    //   m_buf.lock();
    //   if (!image_buf.empty() && !pose_buf.empty() && !pose_gt_buf.empty()) {
    //     if (image_buf.front()->header.stamp.toSec() > pose_buf.front()->header.stamp.toSec()) {
    //       pose_buf.pop();
    //       printf("throw pose at beginning\n");
    //     } else if (image_buf.front()->header.stamp.toSec() >
    //                pose_gt_buf.front()->header.stamp.toSec()) {
    //       pose_gt_buf.pop();
    //       printf("throw pose_gt at beginning\n");
    //     } else if (pose_buf.front()->header.stamp.toSec() >
    //                pose_gt_buf.front()->header.stamp.toSec()) {
    //       pose_gt_buf.pop();
    //       printf("throw pose_gt at beginning\n");
    //     } else if (image_buf.back()->header.stamp.toSec() >=
    //                    pose_buf.front()->header.stamp.toSec() &&
    //                image_buf.back()->header.stamp.toSec() >=
    //                    pose_gt_buf.front()->header.stamp.toSec()) {
    //       pose_msg = pose_buf.front();
    //       pose_buf.pop();
    //       pose_gt_msg = pose_gt_buf.front();
    //       pose_gt_buf.pop();
    //       while (!pose_buf.empty())
    //         pose_buf.pop();
    //       while (!pose_gt_buf.empty())
    //         pose_gt_buf.pop();
    //       while (image_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
    //         image_buf.pop();
    //       image_msg = image_buf.front();
    //       image_buf.pop();
    //     }
    //   }

    //   m_buf.unlock();
    // }

    m_buf.lock();
    // remove old tuples if tuple size is too large
    while (image_pose_buf.size() > 5) {
      image_pose_buf.pop();
      printf("throw image_pose at beginning\n");
    }

    if (!has_gt_pose) {
      // get msg from tuple
      if (!image_pose_buf.empty()) {
        std::tuple<sensor_msgs::ImageConstPtr, sensor_msgs::ImageConstPtr,
                   nav_msgs::Odometry::ConstPtr>
            tuple = image_pose_buf.front();
        image_pose_buf.pop();
        image0_msg = std::get<0>(tuple);
        image1_msg = std::get<1>(tuple);
        pose_msg = std::get<2>(tuple);
      }
    } else {
      if (!image_pose_gt_buf.empty()) {
        std::tuple<sensor_msgs::ImageConstPtr, sensor_msgs::ImageConstPtr,
                   nav_msgs::Odometry::ConstPtr, nav_msgs::Odometry::ConstPtr>
            tuple = image_pose_gt_buf.front();
        image_pose_gt_buf.pop();
        image0_msg = std::get<0>(tuple);
        image1_msg = std::get<1>(tuple);
        pose_msg = std::get<2>(tuple);
        pose_gt_msg = std::get<3>(tuple);
      }
    }
    m_buf.unlock();

    if (pose_msg != NULL) {
      // printf("pose time %f \n", pose_msg->header.stamp.toSec());
      // if (has_gt_pose)
      //   printf("pose_gt time %f \n", pose_gt_msg->header.stamp.toSec());
      // printf("image0 time %f \n", image0_msg->header.stamp.toSec());
      // printf("image1 time %f \n", image1_msg->header.stamp.toSec());

      // skip fisrt few
      if (skip_first_cnt < SKIP_FIRST_CNT) {
        skip_first_cnt++;
        continue;
      }

      if (skip_cnt < SKIP_CNT) {
        skip_cnt++;
        continue;
      } else {
        skip_cnt = 0;
      }

      // cv_bridge::CvImageConstPtr ptr0, ptr1;
      // image0_msg = image_msg->image0;
      // image1_msg = image_msg->image1;
      // if (image0_msg.encoding == "8UC1") {
      //   sensor_msgs::Image img0, img1;
      //   img0.header = image0_msg.header;
      //   img0.height = image0_msg.height;
      //   img0.width = image0_msg.width;
      //   img0.is_bigendian = image0_msg.is_bigendian;
      //   img0.step = image0_msg.step;
      //   img0.data = image0_msg.data;
      //   img0.encoding = "mono8";

      //   img1.header = image1_msg.header;
      //   img1.height = image1_msg.height;
      //   img1.width = image1_msg.width;
      //   img1.is_bigendian = image1_msg.is_bigendian;
      //   img1.step = image1_msg.step;
      //   img1.data = image1_msg.data;
      //   img1.encoding = "mono8";

      //   ptr0 = cv_bridge::toCvCopy(img0, sensor_msgs::image_encodings::MONO8);
      //   ptr1 = cv_bridge::toCvCopy(img1, sensor_msgs::image_encodings::MONO8);
      // } else {
      //   ptr0 = cv_bridge::toCvCopy(image0_msg, sensor_msgs::image_encodings::MONO8);
      //   ptr1 = cv_bridge::toCvCopy(image1_msg, sensor_msgs::image_encodings::MONO8);
      // }

      cv_bridge::CvImageConstPtr ptr0, ptr1;
      ptr0 = cv_bridge::toCvCopy(*image0_msg, sensor_msgs::image_encodings::MONO8);
      ptr1 = cv_bridge::toCvCopy(*image1_msg, sensor_msgs::image_encodings::MONO8);

      cv::Mat image0 = ptr0->image;
      cv::Mat image1 = ptr1->image;

      // build keyframe
      Vector3d T = Vector3d(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y,
                            pose_msg->pose.pose.position.z);
      Matrix3d R = Quaterniond(pose_msg->pose.pose.orientation.w, pose_msg->pose.pose.orientation.x,
                               pose_msg->pose.pose.orientation.y, pose_msg->pose.pose.orientation.z)
                       .toRotationMatrix();
      if ((T - last_t).norm() > SKIP_DIS) {
        vector<cv::Point3f> point_3d;
        vector<cv::Point2f> point_2d_uv;
        vector<cv::Point2f> point_2d_normal;
        vector<double> point_id;

        std::vector<float> global_desc;
        std::vector<float> local_desc;

        if (COL == 848) {
          global_desc = netvlad_onnx->inference(image0(cv::Range(0, 480), cv::Range(124, 764)));
          superpoint_onnx->inference(image0(cv::Range(0, 480), cv::Range(124, 764)), point_2d_uv,
                                     local_desc);
          for (int i = 0; i < (int)point_2d_uv.size(); i++) {
            point_2d_uv[i].x += 124;
            point_2d_uv[i].y += 0;
          }
        } else if (COL == 640) {
          global_desc = netvlad_onnx->inference(image0);
          superpoint_onnx->inference(image0, point_2d_uv, local_desc);
        } else {
          ROS_ERROR("image size not supported");
          ROS_BREAK();
        }

        posegraph.faiss_index.add(1, global_desc.data());

        if (point_2d_uv.size() < 10) {
          // ROS_WARN("feature points less than 10, skip");
          continue;
        }

        std::vector<cv::Point2f> landmarks_2d_cam1, un_pts0, un_pts1;
        std::vector<uchar> status;
        std::vector<float> err;

        cv::calcOpticalFlowPyrLK(image0, image1, point_2d_uv, landmarks_2d_cam1, status, err,
                                 cv::Size(21, 21), 3);

        std::vector<uchar> status_rl;
        std::vector<cv::Point2f> reverseLeftPts;
        cv::calcOpticalFlowPyrLK(image1, image0, landmarks_2d_cam1, reverseLeftPts, status_rl, err,
                                 cv::Size(21, 21), 3);
        for (size_t i = 0; i < status.size(); i++) {
          if (status[i] && status_rl[i] && inBorder(landmarks_2d_cam1[i], ROW, COL) &&
              distance(point_2d_uv[i], reverseLeftPts[i]) <= 0.9)
            status[i] = 1;
          else
            status[i] = 0;
        }

        reduceVector(point_2d_uv, status);
        reduceVector(landmarks_2d_cam1, status);
        reduceDescriptorVector(local_desc, status);
        undistortedPts(point_2d_uv, point_2d_normal, stereo_camera_[0]);
        undistortedPts(landmarks_2d_cam1, un_pts1, stereo_camera_[1]);
        generate3dPoints(point_2d_normal, un_pts1, point_3d, status);
        reduceVector(point_2d_uv, status);
        reduceVector(point_2d_normal, status);
        reduceVector(landmarks_2d_cam1, status);
        reduceDescriptorVector(local_desc, status);

        for (size_t i = 0; i < point_3d.size(); i++) {
          Eigen::Vector3d pt_c, pt_i, pt_w;
          pt_c << point_3d[i].x, point_3d[i].y, point_3d[i].z;
          pt_i = T_i_c.block(0, 0, 3, 3) * pt_c + T_i_c.block(0, 3, 3, 1);
          pt_w = R * pt_i + T;
          point_3d[i].x = pt_w(0);
          point_3d[i].y = pt_w(1);
          point_3d[i].z = pt_w(2);
        }

        // for (unsigned int i = 0; i < point_msg->points.size(); i++) {
        //   cv::Point3f p_3d;
        //   p_3d.x = point_msg->points[i].x;
        //   p_3d.y = point_msg->points[i].y;
        //   p_3d.z = point_msg->points[i].z;
        //   point_3d.push_back(p_3d);

        //   cv::Point2f p_2d_uv, p_2d_normal;
        //   double p_id;
        //   p_2d_normal.x = point_msg->channels[i].values[0];
        //   p_2d_normal.y = point_msg->channels[i].values[1];
        //   p_2d_uv.x = point_msg->channels[i].values[2];
        //   p_2d_uv.y = point_msg->channels[i].values[3];
        //   p_id = point_msg->channels[i].values[4];
        //   point_2d_normal.push_back(p_2d_normal);
        //   point_2d_uv.push_back(p_2d_uv);
        //   point_id.push_back(p_id);

        //   // printf("u %f, v %f \n", p_2d_uv.x, p_2d_uv.y);
        // }

        if (0) {
          cv::Mat show_img0, show_img1, show_img;
          show_img0 = image0.clone();
          show_img1 = image1.clone();
          drawFeatureOnImage(show_img0, point_2d_uv, cv::Scalar(255, 255, 255));
          drawFeatureOnImage(show_img1, landmarks_2d_cam1, cv::Scalar(255, 255, 255));
          cv::hconcat(show_img0, show_img1, show_img);
          cv::imshow("loop feature", show_img);
          cv::waitKey(1);
          publishLandmarks(point_3d);
        }

        Keyframe *keyframe = new Keyframe(
            pose_msg->header.stamp.toSec(), frame_index, T, R, image0, point_3d, point_2d_uv,
            point_2d_normal, point_id, sequence, global_desc, local_desc, image0_msg->header.seq);
        if (has_gt_pose) {
          Eigen::Vector3d T_w_i_gt =
              Eigen::Vector3d(pose_gt_msg->pose.pose.position.x, pose_gt_msg->pose.pose.position.y,
                              pose_gt_msg->pose.pose.position.z);
          Eigen::Matrix3d R_w_i_gt = Eigen::Quaterniond(pose_gt_msg->pose.pose.orientation.w,
                                                        pose_gt_msg->pose.pose.orientation.x,
                                                        pose_gt_msg->pose.pose.orientation.y,
                                                        pose_gt_msg->pose.pose.orientation.z)
                                         .toRotationMatrix();

          keyframe->setGrountTruthPose(T_w_i_gt, R_w_i_gt);
        }
        m_process.lock();
        start_flag = 1;
        posegraph.addKeyFrame(keyframe, has_gt_pose);
        m_process.unlock();
        frame_index++;
        last_t = T;
      }
    }
    std::chrono::milliseconds dura(5);
    std::this_thread::sleep_for(dura);
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "loop_fusion");
  ros::NodeHandle n("~");
  posegraph.registerPub(n);

  VISUALIZATION_SHIFT_X = 0;
  VISUALIZATION_SHIFT_Y = 0;
  SKIP_CNT = 0;
  SKIP_DIS = 0;

  if (argc != 2) {
    printf("please intput: rosrun loop_fusion loop_fusion_node [config file] \n"
           "for example: rosrun loop_fusion loop_fusion_node "
           "/home/tony-ws1/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml \n");
    return 0;
  }

  string config_file = argv[1];
  printf("config_file: %s\n", argv[1]);

  // if not exists
  if (access(config_file.c_str(), 0) == -1) {
    printf("config file not exists, please check the path\n");
    return 0;
  }

  cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    std::cerr << "ERROR: Wrong path to settings" << std::endl;
  }

  cameraposevisual.setScale(0.1);
  cameraposevisual.setLineWidth(0.01);

  std::string IMAGE0_TOPIC, IMAGE1_TOPIC;

  ROW = fsSettings["image_height"];
  COL = fsSettings["image_width"];

  int pn = config_file.find_last_of('/');
  std::string configPath = config_file.substr(0, pn);
  std::string cam0Calib, cam1Calib;
  fsSettings["cam0_calib"] >> cam0Calib;
  fsSettings["cam1_calib"] >> cam1Calib;
  std::string cam0_file, cam1_file;
  camodocal::CameraPtr cam0, cam1;
  cam0_file = configPath + "/" + cam0Calib;
  cam1_file = configPath + "/" + cam1Calib;
  cam0 = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(cam0_file);
  cam1 = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(cam1_file);
  stereo_camera_.push_back(cam0);
  stereo_camera_.push_back(cam1);

  fsSettings["image0_topic"] >> IMAGE0_TOPIC;
  fsSettings["image1_topic"] >> IMAGE1_TOPIC;
  fsSettings["pose_graph_save_path"] >> POSE_GRAPH_SAVE_PATH;
  fsSettings["output_path"] >> VINS_RESULT_PATH;
  fsSettings["save_image"] >> DEBUG_IMAGE;
  fsSettings["has_gt_pose"] >> has_gt_pose;

  VINS_RESULT_PATH = VINS_RESULT_PATH + "/vio_loop.csv";
  std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
  fout.close();

  int USE_IMU = fsSettings["imu"];
  posegraph.setIMUFlag(USE_IMU);

  netvlad_onnx = new MobileNetVLADONNX(
      string(getenv("HOME")) + "/source/cnn_models/mobilenetvlad_480x640.onnx", 640, 480);
  superpoint_onnx =
      new SuperPointONNX(string(getenv("HOME")) + "/source/cnn_models/superpoint_v1_480x640.onnx",
                         std::string(), std::string(), 640, 480, 0.2, 200);

  cv::Mat cv_Tbl, cv_Tbr;
  Eigen::Matrix4d Tbl, Tbr;
  fsSettings["body_T_cam0"] >> cv_Tbl;
  fsSettings["body_T_cam1"] >> cv_Tbr;
  cv::cv2eigen(cv_Tbl, Tbl);
  cv::cv2eigen(cv_Tbr, Tbr);
  T_cam_l_r = Tbl.inverse() * Tbr;
  T_i_c = Tbl;
  fsSettings.release();

  ros::Subscriber sub_vio = n.subscribe("/vins_estimator/odometry", 1, vio_callback);
  // ros::Subscriber sub_pose = n.subscribe("/vins_estimator/keyframe_pose", 1, pose_callback);
  // ros::Subscriber sub_pose_gt = n.subscribe("/uav_simulator/odometry", 1, pose_gt_callback);
  ros::Subscriber sub_extrinsic = n.subscribe("/vins_estimator/extrinsic", 1, extrinsic_callback);
  ros::Subscriber sub_point = n.subscribe("/vins_estimator/keyframe_point", 1, point_callback);
  ros::Subscriber sub_margin_point =
      n.subscribe("/vins_estimator/margin_cloud", 1, margin_point_callback);

  // ros::Subscriber sub_image = n.subscribe(IMAGE_TOPIC, 2000, image_callback);

  message_filters::Subscriber<sensor_msgs::Image> image0_sub_;
  message_filters::Subscriber<sensor_msgs::Image> image1_sub_;
  message_filters::Subscriber<nav_msgs::Odometry> pose_sub_;
  message_filters::Subscriber<nav_msgs::Odometry> pose_gt_sub_;

  // message_filters::Synchronizer<ImageSyncPolicy> stereo_sync_(ImageSyncPolicy(100), image0_sub_,
  //                                                             image1_sub_);
  message_filters::Synchronizer<ImageOdometrySyncPolicy> stereo_pose_sync_(
      ImageOdometrySyncPolicy(100), image0_sub_, image1_sub_, pose_sub_);
  message_filters::Synchronizer<ImageOdometryGTSyncPolicy> stereo_pose_gt_sync_(
      ImageOdometryGTSyncPolicy(100), image0_sub_, image1_sub_, pose_sub_, pose_gt_sub_);

  image0_sub_.subscribe(n, IMAGE0_TOPIC, 100);
  image1_sub_.subscribe(n, IMAGE1_TOPIC, 100);
  pose_sub_.subscribe(n, "/vins_estimator/keyframe_pose", 100);
  pose_gt_sub_.subscribe(n, "/uav_simulator/odometry", 100);

  // stereo_sync_.registerCallback(boost::bind(&image_callback, _1, _2));
  if (!has_gt_pose)
    stereo_pose_sync_.registerCallback(boost::bind(&image_pose_callback, _1, _2, _3));
  else
    stereo_pose_gt_sync_.registerCallback(boost::bind(&image_pose_gt_callback, _1, _2, _3, _4));

  // pub_match_img = n.advertise<sensor_msgs::Image>("match_image", 1000);
  // pub_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("camera_pose_visual",
  // 1000);
  pub_point_cloud = n.advertise<sensor_msgs::PointCloud>("point_cloud_rect", 1000);
  pub_margin_cloud = n.advertise<sensor_msgs::PointCloud2>("margin_cloud_rect", 1000);
  pub_odometry_rect = n.advertise<nav_msgs::Odometry>("odometry_rect", 1000);
  debug_marker_array_pub_ = n.advertise<visualization_msgs::MarkerArray>("debug", 100);

  std::thread measurement_process;
  measurement_process = std::thread(process);

  ros::spin();

  return 0;
}
