# plot covariance.log data
# column 1: cov of lastest graph node
# column 2: cov of current pose
# column 3,4,5,6,7,8: diag of cov of current pose


import matplotlib.pyplot as plt
import numpy as np

def plot_cov(file_path):
    data = np.loadtxt(file_path)

    # use different y-axis
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(data[:, 0], 'r-')
    ax2.plot(data[:, 1], 'b-')

    ax1.set_xlabel('time')

    ax1.set_ylabel('cov of lastest graph node', color='r')
    ax2.set_ylabel('cov of current pose', color='b')

    # subplot for diag of cov of current pose
    fig, ax = plt.subplots()
    ax.plot(data[:, 2], 'r-', label='cov_x')
    ax.plot(data[:, 3], 'g-', label='cov_y')
    ax.plot(data[:, 4], 'b-', label='cov_z')
    ax.plot(data[:, 5], 'c-', label='cov_roll')
    ax.plot(data[:, 6], 'm-', label='cov_pitch')
    ax.plot(data[:, 7], 'y-', label='cov_yaw')

    ax.set_xlabel('time')
    ax.set_ylabel('cov of current pose')
    ax.legend()

    plt.show()
    

if __name__ == '__main__':
    file_path = 'covariance.log'
    plot_cov(file_path)
