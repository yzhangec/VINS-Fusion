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

#include "pose_graph.h"

PoseGraph::PoseGraph() : faiss_index(4096) {
  posegraph_visualization = new CameraPoseVisualization(1.0, 0.0, 1.0, 1.0);
  posegraph_visualization->setScale(0.1);
  posegraph_visualization->setLineWidth(0.01);
  earliest_loop_index = -1;
  t_drift = Eigen::Vector3d(0, 0, 0);
  yaw_drift = 0;
  r_drift = Eigen::Matrix3d::Identity();
  w_t_vio = Eigen::Vector3d(0, 0, 0);
  w_r_vio = Eigen::Matrix3d::Identity();
  global_index = 0;
  sequence_cnt = 0;
  sequence_loop.push_back(0);
  base_sequence = 1;
  use_imu = 0;
}

PoseGraph::~PoseGraph() { t_optimization.detach(); }

void PoseGraph::registerPub(ros::NodeHandle &n) {
  // nav_msgs::Path, geometry_msgs/PoseStamped[] poses
  pub_pg_path = n.advertise<nav_msgs::Path>("pose_graph_path", 1000);
  pub_base_path = n.advertise<nav_msgs::Path>("base_path", 1000);
  pub_pose_graph = n.advertise<visualization_msgs::MarkerArray>("pose_graph", 1000);
  pub_opt = n.advertise<nav_msgs::Path>("path_optimization_trigger", 1000);
  for (int i = 1; i < 10; i++)
    pub_path[i] = n.advertise<nav_msgs::Path>("path_" + to_string(i), 1000);
  pub_loop_pairs = n.advertise<std_msgs::Int32MultiArray>("loop_pairs", 1000);
}

void PoseGraph::setIMUFlag(bool _use_imu) {
  use_imu = _use_imu;
  if (use_imu) {
    printf("VIO input, perfrom 4 DoF (x, y, z, yaw) pose graph optimization\n");
    t_optimization = std::thread(&PoseGraph::optimize4DoF, this);
  }
}

void PoseGraph::addKeyFrame(Keyframe *cur_kf, bool use_gt) {
  // shift to base frame
  Vector3d vio_P_cur;
  Matrix3d vio_R_cur;
  if (sequence_cnt != cur_kf->sequence) {
    sequence_cnt++;
    sequence_loop.push_back(0);
    w_t_vio = Eigen::Vector3d(0, 0, 0);
    w_r_vio = Eigen::Matrix3d::Identity();
    m_drift.lock();
    t_drift = Eigen::Vector3d(0, 0, 0);
    r_drift = Eigen::Matrix3d::Identity();
    m_drift.unlock();
  }

  cur_kf->getVioPose(vio_P_cur, vio_R_cur);
  vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
  vio_R_cur = w_r_vio * vio_R_cur;
  cur_kf->updateVioPose(vio_P_cur, vio_R_cur);
  cur_kf->index = global_index;
  global_index++;
  int loop_index = -1;
  TicToc tmp_t;
  loop_index = detectLoopML(cur_kf, cur_kf->index);
  // std::cout << "detect loop : " << loop_index << std::endl;

  if (loop_index != -1 && loop_index < cur_kf->index) {
    Keyframe *old_kf = getKeyFrame(loop_index);
    if (cur_kf->findConnection(old_kf, use_gt)) {
      printf(" %d detect loop with %d , old kf img_seq %d, new kf seq %d \n", cur_kf->index,
             loop_index, old_kf->img_seq, cur_kf->img_seq);
      printf("loop info %lf %lf %lf %lf %lf %lf %lf %lf \n", cur_kf->loop_info(0),
             cur_kf->loop_info(1), cur_kf->loop_info(2), cur_kf->loop_info(3), cur_kf->loop_info(4),
             cur_kf->loop_info(5), cur_kf->loop_info(6), cur_kf->loop_info(7));
      loop_pairs.push_back(std::make_pair(cur_kf->img_seq, old_kf->img_seq));

      if (earliest_loop_index > loop_index || earliest_loop_index == -1)
        earliest_loop_index = loop_index;

      Vector3d w_P_old, w_P_cur, vio_P_cur;
      Matrix3d w_R_old, w_R_cur, vio_R_cur;
      old_kf->getVioPose(w_P_old, w_R_old);
      cur_kf->getVioPose(vio_P_cur, vio_R_cur);

      Vector3d relative_t;
      Quaterniond relative_q;
      relative_t = cur_kf->getLoopRelativeT();
      relative_q = (cur_kf->getLoopRelativeQ()).toRotationMatrix();
      w_P_cur = w_R_old * relative_t + w_P_old;
      w_R_cur = w_R_old * relative_q; // relative_q = old_R_cur = w_R_old.transpose() * w_R_cur
      double shift_yaw;
      Matrix3d shift_r;
      Vector3d shift_t;
      if (use_imu) {
        // 4 DOF
        shift_yaw = Utility::R2ypr(w_R_cur).x() - Utility::R2ypr(vio_R_cur).x();
        shift_r = Utility::ypr2R(Vector3d(shift_yaw, 0, 0));
      } else
        shift_r = w_R_cur * vio_R_cur.transpose(); // w_R_vio, 6 DOF
      shift_t =
          w_P_cur - w_R_cur * vio_R_cur.transpose() * vio_P_cur; // w_P_vio = w_P_cur - vio_P_cur
      // shift vio pose of whole sequence to the world frame for different sequences
      if (old_kf->sequence != cur_kf->sequence && sequence_loop[cur_kf->sequence] == 0) {
        w_r_vio = shift_r;
        w_t_vio = shift_t;
        vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
        vio_R_cur = w_r_vio * vio_R_cur;
        cur_kf->updateVioPose(vio_P_cur, vio_R_cur);
        list<Keyframe *>::iterator it = keyframelist.begin();
        for (; it != keyframelist.end(); it++) {
          if ((*it)->sequence == cur_kf->sequence) {
            Vector3d vio_P_cur;
            Matrix3d vio_R_cur;
            (*it)->getVioPose(vio_P_cur, vio_R_cur);
            vio_P_cur = w_t_vio + w_r_vio * vio_P_cur;
            vio_R_cur = w_r_vio * vio_R_cur;
            (*it)->updateVioPose(vio_P_cur, vio_R_cur);
          }
        }
        sequence_loop[cur_kf->sequence] = 1;
      }

      // print relative pose
      // printf("relative_t %lf %lf %lf \n", relative_t.x(), relative_t.y(), relative_t.z());
      // printf("relative_q %lf %lf %lf %lf \n", relative_q.w(), relative_q.x(), relative_q.y(),
      //        relative_q.z());

      m_optimize_buf.lock();
      optimize_buf.push(cur_kf->index);
      m_optimize_buf.unlock();
    }
  }
  m_keyframelist.lock();
  Vector3d P;
  Matrix3d R;
  cur_kf->getVioPose(P, R);
  P = r_drift * P + t_drift;
  R = r_drift * R;
  cur_kf->updatePose(P, R);
  Quaterniond Q{R};
  geometry_msgs::PoseStamped pose_stamped;
  pose_stamped.header.stamp = ros::Time(cur_kf->time_stamp);
  pose_stamped.header.frame_id = "world";
  pose_stamped.pose.position.x = P.x() + VISUALIZATION_SHIFT_X;
  pose_stamped.pose.position.y = P.y() + VISUALIZATION_SHIFT_Y;
  pose_stamped.pose.position.z = P.z();
  pose_stamped.pose.orientation.x = Q.x();
  pose_stamped.pose.orientation.y = Q.y();
  pose_stamped.pose.orientation.z = Q.z();
  pose_stamped.pose.orientation.w = Q.w();
  path[sequence_cnt].poses.push_back(pose_stamped);
  path[sequence_cnt].header = pose_stamped.header;

  if (SAVE_LOOP_PATH) {
    ofstream loop_path_file(VINS_RESULT_PATH, ios::app);
    loop_path_file.setf(ios::fixed, ios::floatfield);
    loop_path_file.precision(0);
    loop_path_file << cur_kf->time_stamp * 1e9 << ",";
    loop_path_file.precision(5);
    loop_path_file << P.x() << "," << P.y() << "," << P.z() << "," << Q.w() << "," << Q.x() << ","
                   << Q.y() << "," << Q.z() << "," << endl;
    loop_path_file.close();
  }
  // draw local connection
  if (SHOW_S_EDGE) {
    list<Keyframe *>::reverse_iterator rit = keyframelist.rbegin();
    for (int i = 0; i < 4; i++) {
      if (rit == keyframelist.rend())
        break;
      Vector3d conncected_P;
      Matrix3d connected_R;
      if ((*rit)->sequence == cur_kf->sequence) {
        (*rit)->getPose(conncected_P, connected_R);
        posegraph_visualization->add_edge(P, conncected_P);
      }
      rit++;
    }
  }
  if (SHOW_L_EDGE) {
    if (cur_kf->has_loop) {
      // printf("has loop \n");
      Keyframe *connected_KF = getKeyFrame(cur_kf->loop_index);
      Vector3d connected_P, P0;
      Matrix3d connected_R, R0;
      connected_KF->getPose(connected_P, connected_R);
      // cur_kf->getVioPose(P0, R0);
      cur_kf->getPose(P0, R0);
      if (cur_kf->sequence > 0) {
        // printf("add loop into visual \n");
        posegraph_visualization->add_loopedge(
            P0, connected_P + Vector3d(VISUALIZATION_SHIFT_X, VISUALIZATION_SHIFT_Y, 0));
      }
    }
  }
  // posegraph_visualization->add_pose(P + Vector3d(VISUALIZATION_SHIFT_X, VISUALIZATION_SHIFT_Y,
  // 0), Q);

  keyframelist.push_back(cur_kf);
  publish();
  m_keyframelist.unlock();
}

Keyframe *PoseGraph::getKeyFrame(int index) {
  //    unique_lock<mutex> lock(m_keyframelist);
  list<Keyframe *>::iterator it = keyframelist.begin();
  for (; it != keyframelist.end(); it++) {
    if ((*it)->index == index)
      break;
  }
  if (it != keyframelist.end())
    return *it;
  else
    return NULL;
}

int PoseGraph::detectLoopML(Keyframe *keyframe, int frame_index) {
  float distances[1000] = {0};
  faiss::Index::idx_t labels[1000];
  faiss_index.search(1, keyframe->global_desc.data(), 100, distances, labels);
  int loop_index = -1;

  // std::cout << "distances";
  // for (int i = 0; i < 100; i++) {
  //   std::cout << distances[i] << " ";
  // }
  // std::cout << std::endl;

  // std::cout << "labels";
  // for (int i = 0; i < 100; i++) {
  //   std::cout << labels[i] << " ";
  // }
  // std::cout << std::endl;

  for (int i = 0; i < 100; i++) {
    if (labels[i] < 0) {
      continue;
    }

    // Set a distance threshold to avoid too many matches of nearby frames
    double thres = 0.8;
    if (labels[i] <= faiss_index.ntotal - 100 && distances[i] > thres) {
      loop_index = labels[i];
      thres = distances[i];
    }
  }

  return loop_index;
}

void PoseGraph::optimize4DoF() {
  while (true) {
    int cur_index = -1;
    int first_looped_index = -1;

    // Find nearest frames that has loop detection and  earliest looped frame index
    m_optimize_buf.lock();
    while (!optimize_buf.empty()) {
      cur_index = optimize_buf.front();
      first_looped_index = earliest_loop_index;
      optimize_buf.pop();
    }
    m_optimize_buf.unlock();

    if (cur_index != -1) {
      printf("optimize pose graph \n");
      TicToc tmp_t;
      m_keyframelist.lock();
      Keyframe *cur_kf = getKeyFrame(cur_index);

      int max_length = cur_index + 1;

      // w^t_i   w^q_i
      double t_array[max_length][3];
      Quaterniond q_array[max_length];
      double euler_array[max_length][3];
      double sequence_array[max_length];

      ceres::Problem problem;
      ceres::Solver::Options options;
      options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
      // options.minimizer_progress_to_stdout = true;
      // options.max_solver_time_in_seconds = SOLVER_TIME * 3;
      options.max_num_iterations = 5;
      ceres::Solver::Summary summary;
      ceres::LossFunction *loss_function;
      loss_function = new ceres::HuberLoss(0.1);
      // loss_function = new ceres::CauchyLoss(1.0);
      ceres::LocalParameterization *angle_local_parameterization =
          AngleLocalParameterization::Create();

      list<Keyframe *>::iterator it;

      int i = 0;
      for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
        // iterate between the earliest looped frame and the nearest frame with loop detection
        if ((*it)->index < first_looped_index)
          continue;

        (*it)->local_index = i;
        Quaterniond tmp_q;
        Matrix3d tmp_r;
        Vector3d tmp_t;
        (*it)->getVioPose(tmp_t, tmp_r);
        tmp_q = tmp_r;
        t_array[i][0] = tmp_t(0);
        t_array[i][1] = tmp_t(1);
        t_array[i][2] = tmp_t(2);
        q_array[i] = tmp_q;

        Vector3d euler_angle = Utility::R2ypr(tmp_q.toRotationMatrix());
        euler_array[i][0] = euler_angle.x();
        euler_array[i][1] = euler_angle.y();
        euler_array[i][2] = euler_angle.z();

        sequence_array[i] = (*it)->sequence;

        problem.AddParameterBlock(euler_array[i], 1, angle_local_parameterization);
        problem.AddParameterBlock(t_array[i], 3);

        if ((*it)->index == first_looped_index || (*it)->sequence == 0) {
          problem.SetParameterBlockConstant(euler_array[i]);
          problem.SetParameterBlockConstant(t_array[i]);
        }

        // add edge
        for (int j = 1; j < 5; j++) {
          if (i - j >= 0 && sequence_array[i] == sequence_array[i - j]) {
            Vector3d euler_conncected = Utility::R2ypr(q_array[i - j].toRotationMatrix());
            Vector3d relative_t(t_array[i][0] - t_array[i - j][0],
                                t_array[i][1] - t_array[i - j][1],
                                t_array[i][2] - t_array[i - j][2]);
            relative_t = q_array[i - j].inverse() * relative_t;
            double relative_yaw = euler_array[i][0] - euler_array[i - j][0];
            ceres::CostFunction *cost_function =
                FourDOFError::Create(relative_t.x(), relative_t.y(), relative_t.z(), relative_yaw,
                                     euler_conncected.y(), euler_conncected.z());
            problem.AddResidualBlock(cost_function, NULL, euler_array[i - j], t_array[i - j],
                                     euler_array[i], t_array[i]);
          }
        }

        // add loop edge
        if ((*it)->has_loop) {
          assert((*it)->loop_index >= first_looped_index);
          int connected_index = getKeyFrame((*it)->loop_index)->local_index;
          Vector3d euler_conncected = Utility::R2ypr(q_array[connected_index].toRotationMatrix());
          Vector3d relative_t;
          relative_t = (*it)->getLoopRelativeT();
          double relative_yaw = (*it)->getLoopRelativeYaw();
          ceres::CostFunction *cost_function =
              FourDOFWeightError::Create(relative_t.x(), relative_t.y(), relative_t.z(),
                                         relative_yaw, euler_conncected.y(), euler_conncected.z());
          problem.AddResidualBlock(cost_function, loss_function, euler_array[connected_index],
                                   t_array[connected_index], euler_array[i], t_array[i]);
        }

        if ((*it)->index == cur_index)
          break;
        i++;
      }
      m_keyframelist.unlock();

      ceres::Solve(options, &problem, &summary);
      // std::cout << summary.BriefReport() << "\n";

      // printf("pose optimization time: %f \n", tmp_t.toc());
      /*
      for (int j = 0 ; j < i; j++)
      {
          printf("optimize i: %d p: %f, %f, %f\n", j, t_array[j][0], t_array[j][1], t_array[j][2] );
      }
      */
      m_keyframelist.lock();
      i = 0;
      for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
        if ((*it)->index < first_looped_index)
          continue;

        Quaterniond tmp_q;
        tmp_q = Utility::ypr2R(Vector3d(euler_array[i][0], euler_array[i][1], euler_array[i][2]));
        Vector3d tmp_t = Vector3d(t_array[i][0], t_array[i][1], t_array[i][2]);
        Matrix3d tmp_r = tmp_q.toRotationMatrix();
        (*it)->updatePose(tmp_t, tmp_r);

        if ((*it)->index == cur_index)
          break;
        i++;
      }

      Vector3d cur_t, vio_t;
      Matrix3d cur_r, vio_r;
      cur_kf->getPose(cur_t, cur_r);
      cur_kf->getVioPose(vio_t, vio_r);
      m_drift.lock();
      yaw_drift = Utility::R2ypr(cur_r).x() - Utility::R2ypr(vio_r).x();
      r_drift = Utility::ypr2R(Vector3d(yaw_drift, 0, 0));
      t_drift = cur_t - r_drift * vio_t;
      m_drift.unlock();
      // cout << "t_drift " << t_drift.transpose() << endl;
      // cout << "r_drift " << Utility::R2ypr(r_drift).transpose() << endl;
      // cout << "yaw drift " << yaw_drift << endl;

      it++;
      for (; it != keyframelist.end(); it++) {
        Vector3d P;
        Matrix3d R;
        (*it)->getVioPose(P, R);
        P = r_drift * P + t_drift;
        R = r_drift * R;
        (*it)->updatePose(P, R);
      }
      m_keyframelist.unlock();
      updatePath();
    }

    std::chrono::milliseconds dura(2000);
    std::this_thread::sleep_for(dura);
  }
  return;
}

void PoseGraph::updatePath() {
  m_keyframelist.lock();
  list<Keyframe *>::iterator it;
  for (int i = 1; i <= sequence_cnt; i++) {
    path[i].poses.clear();
  }
  base_path.poses.clear();
  posegraph_visualization->reset();

  if (SAVE_LOOP_PATH) {
    ofstream loop_path_file_tmp(VINS_RESULT_PATH, ios::out);
    loop_path_file_tmp.close();
  }

  for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
    Vector3d P;
    Matrix3d R;
    (*it)->getPose(P, R);
    Quaterniond Q;
    Q = R;
    //        printf("path p: %f, %f, %f\n",  P.x(),  P.z(),  P.y() );

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time((*it)->time_stamp);
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose.position.x = P.x() + VISUALIZATION_SHIFT_X;
    pose_stamped.pose.position.y = P.y() + VISUALIZATION_SHIFT_Y;
    pose_stamped.pose.position.z = P.z();
    pose_stamped.pose.orientation.x = Q.x();
    pose_stamped.pose.orientation.y = Q.y();
    pose_stamped.pose.orientation.z = Q.z();
    pose_stamped.pose.orientation.w = Q.w();
    if ((*it)->sequence == 0) {
      base_path.poses.push_back(pose_stamped);
      base_path.header = pose_stamped.header;
    } else {
      path[(*it)->sequence].poses.push_back(pose_stamped);
      path[(*it)->sequence].header = pose_stamped.header;
    }

    if (SAVE_LOOP_PATH) {
      ofstream loop_path_file(VINS_RESULT_PATH, ios::app);
      loop_path_file.setf(ios::fixed, ios::floatfield);
      loop_path_file.precision(0);
      loop_path_file << (*it)->time_stamp * 1e9 << ",";
      loop_path_file.precision(5);
      loop_path_file << P.x() << "," << P.y() << "," << P.z() << "," << Q.w() << "," << Q.x() << ","
                     << Q.y() << "," << Q.z() << "," << endl;
      loop_path_file.close();
    }
    // draw local connection
    if (SHOW_S_EDGE) {
      list<Keyframe *>::reverse_iterator rit = keyframelist.rbegin();
      list<Keyframe *>::reverse_iterator lrit;
      for (; rit != keyframelist.rend(); rit++) {
        if ((*rit)->index == (*it)->index) {
          lrit = rit;
          lrit++;
          for (int i = 0; i < 4; i++) {
            if (lrit == keyframelist.rend())
              break;
            if ((*lrit)->sequence == (*it)->sequence) {
              Vector3d conncected_P;
              Matrix3d connected_R;
              (*lrit)->getPose(conncected_P, connected_R);
              posegraph_visualization->add_edge(P, conncected_P);
            }
            lrit++;
          }
          break;
        }
      }
    }
    if (SHOW_L_EDGE) {
      if ((*it)->has_loop && (*it)->sequence == sequence_cnt) {
        Keyframe *connected_KF = getKeyFrame((*it)->loop_index);
        Vector3d connected_P;
        Matrix3d connected_R;
        connected_KF->getPose(connected_P, connected_R);
        //(*it)->getVioPose(P, R);
        (*it)->getPose(P, R);
        if ((*it)->sequence > 0) {
          posegraph_visualization->add_loopedge(
              P, connected_P + Vector3d(VISUALIZATION_SHIFT_X, VISUALIZATION_SHIFT_Y, 0));
        }
      }
    }
  }
  publish();
  pub_opt.publish(path[1]);
  m_keyframelist.unlock();
}

void PoseGraph::publish() {
  for (int i = 1; i <= sequence_cnt; i++) {
    // if (sequence_loop[i] == true || i == base_sequence)
    if (1 || i == base_sequence) {
      pub_pg_path.publish(path[i]);
      pub_path[i].publish(path[i]);
      posegraph_visualization->publish_by(pub_pose_graph, path[sequence_cnt].header);
    }
  }
  pub_base_path.publish(base_path);
  // posegraph_visualization->publish_by(pub_pose_graph, path[sequence_cnt].header);

  if (!loop_pairs.empty()) {
    std_msgs::Int32MultiArray loop_pairs_msg;
    for (int i = 0; i < (int)loop_pairs.size(); i++) {
      loop_pairs_msg.data.push_back(loop_pairs[i].first);
      loop_pairs_msg.data.push_back(loop_pairs[i].second);
    }
    pub_loop_pairs.publish(loop_pairs_msg);
  }
}
