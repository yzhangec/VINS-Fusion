#include "loop_fusion/loop_closure.h"
#include "backward/backward.hpp"

namespace backward {
backward::SignalHandling sh;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "loop_closure_node");
  ros::NodeHandle nh("~");

  loop_closure::ActiveLoop active_loop(nh);

  ros::Duration(1.0).sleep();
  ros::spin();
  return 0;
}
