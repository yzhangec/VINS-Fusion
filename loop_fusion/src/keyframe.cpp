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

#define NAME(x) #x

template <typename Derived> static void reduceVector(vector<Derived> &v, vector<uchar> status) {
  int j = 0;
  for (int i = 0; i < int(v.size()); i++)
    if (status[i])
      v[j++] = v[i];
  v.resize(j);
}

template <typename Derived>
void checkSize(std::vector<Derived> &v, const std::string &name = std::string()) {
  std::cout << name << ".size() = " << v.size() << std::endl;
}

void Keyframe::drawFeatureOnImage(cv::Mat &image, const std::vector<cv::Point2f> &pts,
                                  const cv::Scalar &color) {
  for (const cv::Point2f &pt : pts) {
    cv::circle(image, pt, 3, color, 1);
  }
}

void Keyframe::drawLineOnImage(cv::Mat &image, const std::vector<cv::Point2f> &pts0,
                               const std::vector<cv::Point2f> &pts1, const cv::Scalar &color) {
  for (int i = 0; i < pts0.size(); i++) {
    cv::line(image, pts0[i], pts1[i] + cv::Point2f(image.size().width * 0.5, 0), color, 1);
  }
}

// create keyframe online
Keyframe::Keyframe(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i,
                   cv::Mat &_image, vector<cv::Point3f> &_point_3d,
                   vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_norm,
                   vector<double> &_point_id, int _sequence, std::vector<float> &_global_desc,
                   std::vector<float> &_local_desc) {
  time_stamp = _time_stamp;
  index = _index;
  vio_T_w_i = _vio_T_w_i;
  vio_R_w_i = _vio_R_w_i;
  T_w_i = vio_T_w_i;
  R_w_i = vio_R_w_i;
  origin_vio_T = vio_T_w_i;
  origin_vio_R = vio_R_w_i;
  image = _image.clone();
  cv::resize(image, thumbnail, cv::Size(80, 60));
  point_3d = _point_3d;
  point_2d_uv = _point_2d_uv;
  point_2d_norm = _point_2d_norm;
  point_id = _point_id;
  has_loop = false;
  loop_index = -1;
  has_fast_point = false;
  loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
  sequence = _sequence;
  global_desc = _global_desc;
  local_desc = _local_desc;
  // computeWindowBRIEFPoint();
  // computeBRIEFPoint();
  // if (!DEBUG_IMAGE)
  //   image.release();
}

// load previous keyframe
Keyframe::Keyframe(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i,
                   Vector3d &_T_w_i, Matrix3d &_R_w_i, cv::Mat &_image, int _loop_index,
                   Eigen::Matrix<double, 8, 1> &_loop_info, vector<cv::KeyPoint> &_keypoints,
                   vector<cv::KeyPoint> &_keypoints_norm,
                   vector<BRIEF::bitset> &_brief_descriptors) {
  time_stamp = _time_stamp;
  index = _index;
  // vio_T_w_i = _vio_T_w_i;
  // vio_R_w_i = _vio_R_w_i;
  vio_T_w_i = _T_w_i;
  vio_R_w_i = _R_w_i;
  T_w_i = _T_w_i;
  R_w_i = _R_w_i;
  if (DEBUG_IMAGE) {
    image = _image.clone();
    cv::resize(image, thumbnail, cv::Size(80, 60));
  }
  if (_loop_index != -1)
    has_loop = true;
  else
    has_loop = false;
  loop_index = _loop_index;
  loop_info = _loop_info;
  has_fast_point = false;
  sequence = 0;
  keypoints = _keypoints;
  keypoints_norm = _keypoints_norm;
  brief_descriptors = _brief_descriptors;
}

void Keyframe::computeWindowBRIEFPoint() {
  BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
  for (int i = 0; i < (int)point_2d_uv.size(); i++) {
    cv::KeyPoint key;
    key.pt = point_2d_uv[i];
    window_keypoints.push_back(key);
  }
  extractor(image, window_keypoints, window_brief_descriptors);
}

void Keyframe::computeBRIEFPoint() {
  BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
  const int fast_th = 20; // corner detector response threshold
  if (1)
    cv::FAST(image, keypoints, fast_th, true);
  else {
    vector<cv::Point2f> tmp_pts;
    cv::goodFeaturesToTrack(image, tmp_pts, 500, 0.01, 10);
    for (int i = 0; i < (int)tmp_pts.size(); i++) {
      cv::KeyPoint key;
      key.pt = tmp_pts[i];
      keypoints.push_back(key);
    }
  }
  extractor(image, keypoints, brief_descriptors);
  for (int i = 0; i < (int)keypoints.size(); i++) {
    Eigen::Vector3d tmp_p;
    m_camera->liftProjective(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
    cv::KeyPoint tmp_norm;
    tmp_norm.pt = cv::Point2f(tmp_p.x() / tmp_p.z(), tmp_p.y() / tmp_p.z());
    keypoints_norm.push_back(tmp_norm);
  }
}

void BriefExtractor::operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys,
                                vector<BRIEF::bitset> &descriptors) const {
  m_brief.compute(im, keys, descriptors);
}

bool Keyframe::searchInAera(const BRIEF::bitset window_descriptor,
                            const std::vector<BRIEF::bitset> &descriptors_old,
                            const std::vector<cv::KeyPoint> &keypoints_old,
                            const std::vector<cv::KeyPoint> &keypoints_old_norm,
                            cv::Point2f &best_match, cv::Point2f &best_match_norm) {
  cv::Point2f best_pt;
  int bestDist = 128;
  int bestIndex = -1;
  for (int i = 0; i < (int)descriptors_old.size(); i++) {

    int dis = HammingDis(window_descriptor, descriptors_old[i]);
    if (dis < bestDist) {
      bestDist = dis;
      bestIndex = i;
    }
  }
  // printf("best dist %d", bestDist);
  if (bestIndex != -1 && bestDist < 80) {
    best_match = keypoints_old[bestIndex].pt;
    best_match_norm = keypoints_old_norm[bestIndex].pt;
    return true;
  } else
    return false;
}

void Keyframe::searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
                                std::vector<cv::Point2f> &matched_2d_old_norm,
                                std::vector<uchar> &status,
                                const std::vector<BRIEF::bitset> &descriptors_old,
                                const std::vector<cv::KeyPoint> &keypoints_old,
                                const std::vector<cv::KeyPoint> &keypoints_old_norm) {
  for (int i = 0; i < (int)window_brief_descriptors.size(); i++) {
    cv::Point2f pt(0.f, 0.f);
    cv::Point2f pt_norm(0.f, 0.f);
    if (searchInAera(window_brief_descriptors[i], descriptors_old, keypoints_old,
                     keypoints_old_norm, pt, pt_norm))
      status.push_back(1);
    else
      status.push_back(0);
    matched_2d_old.push_back(pt);
    matched_2d_old_norm.push_back(pt_norm);
  }
}

void Keyframe::FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                      const std::vector<cv::Point2f> &matched_2d_old_norm,
                                      vector<uchar> &status) {
  int n = (int)matched_2d_cur_norm.size();
  for (int i = 0; i < n; i++)
    status.push_back(0);
  if (n >= 8) {
    vector<cv::Point2f> tmp_cur(n), tmp_old(n);
    for (int i = 0; i < (int)matched_2d_cur_norm.size(); i++) {
      double FOCAL_LENGTH = 460.0;
      double tmp_x, tmp_y;
      tmp_x = FOCAL_LENGTH * matched_2d_cur_norm[i].x + COL / 2.0;
      tmp_y = FOCAL_LENGTH * matched_2d_cur_norm[i].y + ROW / 2.0;
      tmp_cur[i] = cv::Point2f(tmp_x, tmp_y);

      tmp_x = FOCAL_LENGTH * matched_2d_old_norm[i].x + COL / 2.0;
      tmp_y = FOCAL_LENGTH * matched_2d_old_norm[i].y + ROW / 2.0;
      tmp_old[i] = cv::Point2f(tmp_x, tmp_y);
    }
    cv::findFundamentalMat(tmp_cur, tmp_old, cv::FM_RANSAC, 3.0, 0.9, status);
  }
}

void Keyframe::PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                         const std::vector<cv::Point3f> &matched_3d, std::vector<uchar> &status,
                         Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old) {
  // for (int i = 0; i < matched_3d.size(); i++)
  //	printf("3d x: %f, y: %f, z: %f\n",matched_3d[i].x, matched_3d[i].y, matched_3d[i].z );
  // printf("match size %d \n", matched_3d.size());
  cv::Mat r, rvec, t, D, tmp_r;
  cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
  Matrix3d R_inital;
  Vector3d P_inital;
  Matrix3d R_w_c = origin_vio_R * qic;
  Vector3d T_w_c = origin_vio_T + origin_vio_R * tic;

  R_inital = R_w_c.inverse();
  P_inital = -(R_inital * T_w_c);

  cv::eigen2cv(R_inital, tmp_r);
  cv::Rodrigues(tmp_r, rvec);
  cv::eigen2cv(P_inital, t);

  cv::Mat inliers;
  TicToc t_pnp_ransac;

  if (CV_MAJOR_VERSION < 3)
    solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 100,
                   inliers);
  else {
    if (CV_MINOR_VERSION < 2)
      solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, sqrt(10.0 / 460.0),
                     0.99, inliers);
    else
      solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 0.99,
                     inliers);
  }

  for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
    status.push_back(0);

  for (int i = 0; i < inliers.rows; i++) {
    int n = inliers.at<int>(i);
    status[n] = 1;
  }

  cv::Rodrigues(rvec, r);
  Matrix3d R_pnp, R_w_c_old;
  cv::cv2eigen(r, R_pnp);
  R_w_c_old = R_pnp.transpose();
  Vector3d T_pnp, T_w_c_old;
  cv::cv2eigen(t, T_pnp);
  T_w_c_old = R_w_c_old * (-T_pnp);

  PnP_R_old = R_w_c_old * qic.transpose();
  PnP_T_old = T_w_c_old - PnP_R_old * tic;
}

bool Keyframe::findConnection(Keyframe *old_kf) {
  TicToc tmp_t;
  // printf("find Connection\n");
  vector<cv::Point2f> matched_2d_cur, matched_2d_old;
  vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
  vector<cv::Point3f> matched_3d;
  vector<double> matched_id;
  vector<uchar> status;

  matched_3d = point_3d;
  matched_2d_cur = point_2d_uv;
  matched_2d_cur_norm = point_2d_norm;
  matched_id = point_id;

  TicToc t_match;
#if 0
		if (DEBUG_IMAGE)    
	    {
	        cv::Mat gray_img, loop_match_img;
	        cv::Mat old_img = old_kf->image;
	        cv::hconcat(image, old_img, gray_img);
	        cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)point_2d_uv.size(); i++)
	        {
	            cv::Point2f cur_pt = point_2d_uv[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)old_kf->keypoints.size(); i++)
	        {
	            cv::Point2f old_pt = old_kf->keypoints[i].pt;
	            old_pt.x += COL;
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        ostringstream path;
	        path << "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "0raw_point.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
#endif
  // printf("search by des\n");
  // searchByBRIEFDes(matched_2d_old, matched_2d_old_norm, status, old_kf->brief_descriptors,
  //                  old_kf->keypoints, old_kf->keypoints_norm);
  // reduceVector(matched_2d_cur, status);
  // reduceVector(matched_2d_old, status);
  // reduceVector(matched_2d_cur_norm, status);
  // reduceVector(matched_2d_old_norm, status);
  // reduceVector(matched_3d, status);
  // reduceVector(matched_id, status);
  // printf("search by des finish\n");

  cv::BFMatcher bfmatcher(cv::NORM_L2, true);
  std::vector<cv::DMatch> _matches;
  std::vector<unsigned char> mask;

  cv::Mat descriptors_a(point_2d_uv.size(), 256, CV_32F);
  memcpy(descriptors_a.data, local_desc.data(), local_desc.size() * sizeof(float));

  cv::Mat descriptors_b(old_kf->point_2d_uv.size(), 256, CV_32F);
  memcpy(descriptors_b.data, old_kf->local_desc.data(), old_kf->local_desc.size() * sizeof(float));

  bfmatcher.match(descriptors_a, descriptors_b, _matches);

  std::vector<cv::Point2f> pts0, pts1, pts1_norm;
  std::vector<cv::Point3f> pts0_3d;
  for (auto match : _matches) {
    int index_a = match.queryIdx;
    int index_b = match.trainIdx;

    pts0.push_back(point_2d_uv[index_a]);
    pts0_3d.push_back(point_3d[index_a]);
    pts1.push_back(old_kf->point_2d_uv[index_b]);
    pts1_norm.push_back(old_kf->point_2d_norm[index_b]);
  }

  cv::findHomography(pts1, pts0, cv::RANSAC, 3, mask);
  reduceVector(pts0, mask);
  reduceVector(pts0_3d, mask);
  reduceVector(pts1, mask);
  reduceVector(pts1_norm, mask);

  // checkSize(pts0, NAME(pts0));
  // checkSize(pts0_3d, NAME(pts0_3d));
  // checkSize(pts1, NAME(pts1));
  // checkSize(pts1_norm, NAME(pts1_norm));

#if 0
  if (DEBUG_IMAGE) {
    int gap = 10;
    cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
    cv::Mat gray_img, loop_match_img;
    cv::Mat old_img = old_kf->image;
    cv::hconcat(image, gap_image, gap_image);
    cv::hconcat(gap_image, old_img, gray_img);
    cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
    for (int i = 0; i < (int)matched_2d_cur.size(); i++) {
      cv::Point2f cur_pt = matched_2d_cur[i];
      cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
    }
    for (int i = 0; i < (int)matched_2d_old.size(); i++) {
      cv::Point2f old_pt = matched_2d_old[i];
      old_pt.x += (COL + gap);
      cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
    }
    for (int i = 0; i < (int)matched_2d_cur.size(); i++) {
      cv::Point2f old_pt = matched_2d_old[i];
      old_pt.x += (COL + gap);
      cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
    }

    ostringstream path, path1, path2;
    path << "/home/tony-ws1/raw_data/loop_image/" << index << "-" << old_kf->index << "-"
         << "1descriptor_match.jpg";
    cv::imwrite(path.str().c_str(), loop_match_img);
    /*
    path1 <<  "/home/tony-ws1/raw_data/loop_image/"
            << index << "-"
            << old_kf->index << "-" << "1descriptor_match_1.jpg";
    cv::imwrite( path1.str().c_str(), image);
    path2 <<  "/home/tony-ws1/raw_data/loop_image/"
            << index << "-"
            << old_kf->index << "-" << "1descriptor_match_2.jpg";
    cv::imwrite( path2.str().c_str(), old_img);
    */
  }
#endif
  status.clear();
/*
FundmantalMatrixRANSAC(matched_2d_cur_norm, matched_2d_old_norm, status);
reduceVector(matched_2d_cur, status);
reduceVector(matched_2d_old, status);
reduceVector(matched_2d_cur_norm, status);
reduceVector(matched_2d_old_norm, status);
reduceVector(matched_3d, status);
reduceVector(matched_id, status);
*/
#if 0
  if (DEBUG_IMAGE) {
    int gap = 10;
    cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
    cv::Mat gray_img, loop_match_img;
    cv::Mat old_img = old_kf->image;
    cv::hconcat(image, gap_image, gap_image);
    cv::hconcat(gap_image, old_img, gray_img);
    cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
    for (int i = 0; i < (int)matched_2d_cur.size(); i++) {
      cv::Point2f cur_pt = matched_2d_cur[i];
      cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
    }
    for (int i = 0; i < (int)matched_2d_old.size(); i++) {
      cv::Point2f old_pt = matched_2d_old[i];
      old_pt.x += (COL + gap);
      cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
    }
    for (int i = 0; i < (int)matched_2d_cur.size(); i++) {
      cv::Point2f old_pt = matched_2d_old[i];
      old_pt.x += (COL + gap);
      cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
    }

    ostringstream path;
    path << "/home/tony-ws1/raw_data/loop_image/" << index << "-" << old_kf->index << "-"
         << "2fundamental_match.jpg";
    cv::imwrite(path.str().c_str(), loop_match_img);
  }
#endif
  // PnP_T_old: T_loop_cur
  Eigen::Vector3d PnP_T_old;
  Eigen::Matrix3d PnP_R_old;
  Eigen::Vector3d relative_t;
  Quaterniond relative_q;
  double relative_yaw;
  // matched_2d_cur = point_2d_uv;
  // if ((int)matched_2d_cur.size() > MIN_LOOP_NUM) {
  //   status.clear();
  //   PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old);
  //   reduceVector(matched_2d_cur, status);
  //   reduceVector(matched_2d_old, status);
  //   reduceVector(matched_2d_cur_norm, status);
  //   reduceVector(matched_2d_old_norm, status);
  //   reduceVector(matched_3d, status);
  //   reduceVector(matched_id, status);

  // Calcuate T_loop_cur

  bool pnp_success = estimateTBetweenFrames(pts0_3d, pts1_norm, PnP_R_old, PnP_T_old);

#if 1
  if (DEBUG_IMAGE) {
    int gap = 10;
    cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
    cv::Mat gray_img, loop_match_img;
    cv::Mat old_img = old_kf->image;
    cv::hconcat(image, gap_image, gap_image);
    cv::hconcat(gap_image, old_img, gray_img);
    cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
    for (int i = 0; i < (int)matched_2d_cur.size(); i++) {
      cv::Point2f cur_pt = matched_2d_cur[i];
      cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
    }
    for (int i = 0; i < (int)matched_2d_old.size(); i++) {
      cv::Point2f old_pt = matched_2d_old[i];
      old_pt.x += (COL + gap);
      cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
    }
    for (int i = 0; i < (int)matched_2d_cur.size(); i++) {
      cv::Point2f old_pt = matched_2d_old[i];
      old_pt.x += (COL + gap);
      cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 2, 8, 0);
    }
    cv::Mat notation(50, COL + gap + COL, CV_8UC3, cv::Scalar(255, 255, 255));
    putText(notation, "current frame: " + to_string(index) + "  sequence: " + to_string(sequence),
            cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);

    putText(notation,
            "previous frame: " + to_string(old_kf->index) +
                "  sequence: " + to_string(old_kf->sequence),
            cv::Point2f(20 + COL + gap, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
    cv::vconcat(notation, loop_match_img, loop_match_img);

    /*
    ostringstream path;
    path <<  "/home/tony-ws1/raw_data/loop_image/"
            << index << "-"
            << old_kf->index << "-" << "3pnp_match.jpg";
    cv::imwrite( path.str().c_str(), loop_match_img);
    */
    if ((int)matched_2d_cur.size() > MIN_LOOP_NUM) {
      /*
      cv::imshow("loop connection",loop_match_img);
      cv::waitKey(10);
      */
      cv::Mat thumbimage;
      cv::resize(loop_match_img, thumbimage,
                 cv::Size(loop_match_img.cols / 2, loop_match_img.rows / 2));
      sensor_msgs::ImagePtr msg =
          cv_bridge::CvImage(std_msgs::Header(), "bgr8", thumbimage).toImageMsg();
      msg->header.stamp = ros::Time(time_stamp);
      pub_match_img.publish(msg);
    }
  }
#endif

  // if ((int)matched_2d_cur.size() > MIN_LOOP_NUM) {
  if (pnp_success) {
    cv::Mat show_img0, show_img1, show_loop_img;
    cv::cvtColor(image, show_img0, cv::COLOR_GRAY2RGB);
    cv::cvtColor(old_kf->image, show_img1, cv::COLOR_GRAY2RGB);
    drawFeatureOnImage(show_img0, point_2d_uv);
    drawFeatureOnImage(show_img1, old_kf->point_2d_uv);
    cv::hconcat(show_img0, show_img1, show_loop_img);
    drawLineOnImage(show_loop_img, pts0, pts1);
    static int loop_cnt = 0;
    cv::imshow("loop frame", show_loop_img);
    cv::imwrite("/home/eason/output/active_loop_debug/tmp_loop_pair/" + to_string(index) + "-" +
                    to_string(old_kf->index) + "-" + to_string(loop_cnt) + ".png",
                show_loop_img);
    cv::waitKey(1);

    relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
    relative_q = PnP_R_old.transpose() * origin_vio_R;
    relative_yaw =
        Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(PnP_R_old).x());
    // printf("PNP relative\n");
    cout << "pnp relative_t " << relative_t.transpose() << endl;
    cout << "pnp relative_yaw " << relative_yaw << endl;
    if (abs(relative_yaw) < 30.0 && relative_t.norm() < 20.0) {

      has_loop = true;
      loop_index = old_kf->index;
      loop_info << relative_t.x(), relative_t.y(), relative_t.z(), relative_q.w(), relative_q.x(),
          relative_q.y(), relative_q.z(), relative_yaw;
      // cout << "pnp relative_t " << relative_t.transpose() << endl;
      // cout << "pnp relative_q " << relative_q.w() << " " << relative_q.vec().transpose() << endl;
      return true;
    }
  }
  // printf("loop final use num %d %lf--------------- \n", (int)matched_2d_cur.size(),
  // t_match.toc());
  return false;
}

int Keyframe::HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b) {
  BRIEF::bitset xor_of_bitset = a ^ b;
  int dis = xor_of_bitset.count();
  return dis;
}

void Keyframe::getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i) {
  _T_w_i = vio_T_w_i;
  _R_w_i = vio_R_w_i;
}

void Keyframe::getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i) {
  _T_w_i = T_w_i;
  _R_w_i = R_w_i;
}

void Keyframe::updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i) {
  T_w_i = _T_w_i;
  R_w_i = _R_w_i;
}

void Keyframe::updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i) {
  vio_T_w_i = _T_w_i;
  vio_R_w_i = _R_w_i;
  T_w_i = vio_T_w_i;
  R_w_i = vio_R_w_i;
}

Eigen::Vector3d Keyframe::getLoopRelativeT() {
  return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2));
}

Eigen::Quaterniond Keyframe::getLoopRelativeQ() {
  return Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5), loop_info(6));
}

double Keyframe::getLoopRelativeYaw() { return loop_info(7); }

void Keyframe::updateLoop(Eigen::Matrix<double, 8, 1> &_loop_info) {
  if (abs(_loop_info(7)) < 30.0 &&
      Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0) {
    // printf("update loop info\n");
    loop_info = _loop_info;
  }
}

BriefExtractor::BriefExtractor(const std::string &pattern_file) {
  // The DVision::BRIEF extractor computes a random pattern by default when
  // the object is created.
  // We load the pattern that we used to build the vocabulary, to make
  // the descriptors compatible with the predefined vocabulary

  // loads the pattern
  cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
  if (!fs.isOpened())
    throw string("Could not open file ") + pattern_file;

  vector<int> x1, y1, x2, y2;
  fs["x1"] >> x1;
  fs["x2"] >> x2;
  fs["y1"] >> y1;
  fs["y2"] >> y2;

  m_brief.importPairs(x1, y1, x2, y2);
}

double reprojectionError(Eigen::Matrix3d &R, Eigen::Vector3d &t, cv::Point3f &cur_pts_3d,
                         cv::Point2f &loop_pts_2d) {
  Vector3d pt1(cur_pts_3d.x, cur_pts_3d.y, cur_pts_3d.z);
  Vector3d pt2 = R * pt1 + t;
  pt2 = pt2 / pt2[2];
  return sqrt(pow(pt2[0] - loop_pts_2d.x, 2) + pow(pt2[1] - loop_pts_2d.y, 2));
}

bool Keyframe::estimateTBetweenFrames(vector<cv::Point3f> &cur_pts_3d,
                                      vector<cv::Point2f> &loop_pts_2d, Matrix3d &R, Vector3d &t) {
  // To do: calculate relative pose between the key frame and the current frame using the matched
  // 2d-3d points

  if (int(loop_pts_2d.size()) < 20) {
    printf("feature tracking not enough \n");
    return false;
  }

  std::cout << "3d: " << cur_pts_3d.size() << ", 2d: " << loop_pts_2d.size() << std::endl;

  cv::Mat K, tmp_r, rvec, t1, inliers;
  K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
  tmp_r = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
  t1 = (cv::Mat_<double>(3, 1) << 0, 0, 0);

  cv::Rodrigues(tmp_r, rvec);

  if (!cv::solvePnPRansac(cur_pts_3d, loop_pts_2d, K, cv::Mat(), rvec, t1, true, 100, 3, 0.99,
                          inliers)) {
    printf("pnp failed ! \n");
    return false;
  }

  // convert to Eigen matrix
  cv::Rodrigues(rvec, tmp_r);
  for (int i = 0; i < 3; i++) {
    t(i) = t1.at<double>(i, 0);
    for (int j = 0; j < 3; j++) {
      R(i, j) = tmp_r.at<double>(i, j);
    }
  }

  // reject matched points with large reprojection error and redo PnP
  vector<uchar> status(cur_pts_3d.size(), 0);
  for (int i = 0, id = 0; i < inliers.rows; ++i) {
    id = inliers.at<int>(i);
    if (reprojectionError(R, t, cur_pts_3d[id], loop_pts_2d[id]) < 1.0) {
      status[id] = 1;
    }
  }
  reduceVector<cv::Point2f>(loop_pts_2d, status);
  reduceVector<cv::Point3f>(cur_pts_3d, status);

  if (int(loop_pts_2d.size()) < 4) {
    printf("feature tracking not enough 2 \n");
    return false;
  }

  cv::Rodrigues(tmp_r, rvec);
  t1.setTo(0);

  // std::cout << "key points: " << cur_pts_3d << std::endl;
  // std::cout << "cur 2d: " << loop_pts_2d << std::endl;

  if (!cv::solvePnPRansac(cur_pts_3d, loop_pts_2d, K, cv::Mat(), rvec, t1, true, 100, 10.0 / 460.0,
                          0.99, inliers)) {
    printf("pnp failed ! \n");
    return false;
  }

  // convert to Eigen matrix
  cv::Rodrigues(rvec, tmp_r);
  for (int i = 0; i < 3; i++) {
    t(i) = t1.at<double>(i, 0);
    for (int j = 0; j < 3; j++) {
      R(i, j) = tmp_r.at<double>(i, j);
    }
  }

  Matrix3d R_w_c_old;
  R_w_c_old = R.transpose();
  Vector3d T_w_c_old;
  T_w_c_old = R_w_c_old * (-t);

  R = R_w_c_old * qic.transpose();
  t = T_w_c_old - R * tic;

  printf("Successfully PnP! \n");

  return true;
}
