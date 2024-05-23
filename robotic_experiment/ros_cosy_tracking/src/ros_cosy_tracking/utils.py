import numpy as np
import pinocchio as pin
from example_robot_data import load
from geometry_msgs.msg import Pose
import matplotlib.pyplot as plt

def SE3_2_posemsg(T):
    T_pin = pin.SE3(T)
    xyz_qxyzw = pin.SE3ToXYZQUAT(T_pin)
    pose = Pose()
    pose.position.x = xyz_qxyzw[0]
    pose.position.y = xyz_qxyzw[1]
    pose.position.z = xyz_qxyzw[2]
    pose.orientation.x = xyz_qxyzw[3]
    pose.orientation.y = xyz_qxyzw[4]
    pose.orientation.z = xyz_qxyzw[5]
    pose.orientation.w = xyz_qxyzw[6]
    return pose


def posemsg_2_SE3(msg: Pose):
    pose_v = np.array([
        msg.position.x,
        msg.position.y,
        msg.position.z,
        msg.orientation.x,
        msg.orientation.y,
        msg.orientation.z,
        msg.orientation.w,
    ])
    return pin.XYZQUATToSE3(pose_v)


def create_panda7dof():
    r = load('panda')
    # freeze finger joints
    fixed_joints = ['panda_finger_joint1', 'panda_finger_joint2']
    fixed_ids = [r.model.getJointId(jname) for jname in fixed_joints]
    rmodel, [gmodel_col, gmodel_viz] = pin.buildReducedModel(
        r.model, [r.collision_model, r.visual_model],
        fixed_ids, r.q0
    )
    return pin.RobotWrapper(rmodel, gmodel_col, gmodel_viz)


def posiquat_2_SE3(x,q):
    pose = np.concatenate([x,q])
    return pin.XYZQUATToSE3(pose)


def SE3_2_posiquat(T: pin.SE3):
    pose = pin.SE3ToXYZQUAT(T)
    return pose[:3], pose[3:]


def plot_se3_errors(t_arr, T_b_EE_d_lst, T_b_EE_m_lst, img_path='errors_se3.png'):
    N = len(T_b_EE_m_lst)

    err_p_arr = np.zeros((N,3))
    err_o_arr = np.zeros((N,3))
    for i, (T_b_EE_m, T_b_EE_d) in enumerate(zip(T_b_EE_m_lst, T_b_EE_d_lst)):
        err_p_arr[i] = T_b_EE_m.translation - T_b_EE_d.translation
        err_o_arr[i] = np.rad2deg(pin.log3(T_b_EE_d.rotation.T @ T_b_EE_m.rotation))

    plt.figure()
    plt.subplot(2,1,1)
    for i in range(3):
        plt.plot(t_arr, err_p_arr[:,i], 'rgb'[i]+'.', markersize=2)
    plt.grid()
    plt.ylabel('error t (m)')
    plt.subplot(2,1,2)
    for i in range(3):
        plt.plot(t_arr, err_o_arr[:,i], 'rgb'[i]+'.', markersize=2)
    plt.xlabel('time (s)')
    plt.ylabel('error o (deg)')
    plt.grid()
    print('Saving', img_path)
    plt.savefig(img_path)



def plot_se3_abs(t_arr, T_b_EE_d_lst, T_b_EE_m_lst, img_path='abs_se3.png'):
    N = len(T_b_EE_m_lst)

    pd_arr = np.zeros((N,3))
    od_arr = np.zeros((N,3))
    pm_arr = np.zeros((N,3))
    om_arr = np.zeros((N,3))
    for i, (T_b_EE_m, T_b_EE_d) in enumerate(zip(T_b_EE_m_lst, T_b_EE_d_lst)):
        pd_arr[i] = T_b_EE_d.translation
        od_arr[i] = np.rad2deg(pin.log3(T_b_EE_d.rotation))
        pm_arr[i] = T_b_EE_m.translation
        om_arr[i] = np.rad2deg(pin.log3(T_b_EE_m.rotation))

    plt.figure()
    plt.subplot(2,1,1)
    for i in range(3):
        l = 'xyz'[i]
        plt.plot(t_arr, pd_arr[:,i], 'rgb'[i]+'--', markersize=2, label=f'pd_{l}')
        plt.plot(t_arr, pm_arr[:,i], 'rgb'[i]+'-', markersize=2, label=f'pm_{l}')
    plt.legend()
    plt.grid()
    plt.ylabel('translation (m)')
    plt.subplot(2,1,2)
    for i in range(3):
        l = 'xyz'[i]
        plt.plot(t_arr, od_arr[:,i], 'rgb'[i]+'--', markersize=2, label=f'od_{l}')
        plt.plot(t_arr, om_arr[:,i], 'rgb'[i]+'-', markersize=2, label=f'om_{l}')
    plt.legend()
    plt.xlabel('time (s)')
    plt.ylabel('angle axis (deg)')
    plt.grid()
    print('Saving', img_path)
    plt.savefig(img_path)



if __name__ == '__main__':
    N = 10
    t_arr = np.arange(10)
    Td = pin.SE3.Identity()
    Tm = pin.SE3.Identity()
    Td.translation[0] = 1

    plot_se3_abs(t_arr, N*[Td], N*[Tm])


