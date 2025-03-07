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

    ax1.plot(data[:, 0], 'r-', label='d_opt')
    ax2.plot(data[:, 1], 'b-', label='t_opt')

    # subplot for diag of cov of current pose
    fig, ax = plt.subplots()
    ax.plot(data[:, 2], 'r-', label='cov_0')
    ax.plot(data[:, 3], 'g-', label='cov_1')
    ax.plot(data[:, 4], 'b-', label='cov_2')
    ax.plot(data[:, 5], 'c-', label='cov_3')
    ax.plot(data[:, 6], 'm-', label='cov_4')
    ax.plot(data[:, 7], 'y-', label='cov_5')

    ax.set_xlabel('num')
    ax.set_ylabel('cov diag', color='k')
    ax.legend()

    plt.show()

if __name__ == '__main__':
    file_path = 'covariance.log'
    plot_cov(file_path)
