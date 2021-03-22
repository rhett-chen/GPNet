import _init_paths
import torch
import OpenEXR
import cv2
import Imath
from tools.nms import nms, nms2
from test import get7dofPoses
from tools.proposal import *
from dataset.grasp_dataset import generate_grid
from tools.depth2points import Depth2PointCloud, Depth2PointCloudAiRobot
from lib.gpnet import GraspPoseNet
import numpy as np
import os


def get_camera_info(path):
    camera_info = np.load(path)
    camera_info_dict = {}
    for item in camera_info:
        camera_info_dict[item['id'].decode()] = (item['position'], item['orientation'], item['calibration_matrix'])
    camera_pos = camera_info_dict['view0'][0]
    camera_ori = camera_info_dict['view0'][1]  # quaternion
    camera_intrinsic_calibration = camera_info_dict['view0'][2].reshape(3, 3)
    # print(camera_pos, camera_ori, camera_intrinsic_calibration)
    with open('/home/zibo/GPNet/output/camera_info.txt', 'w') as ci:
        ci.write(str(camera_pos) + '\n')
        ci.write(str(camera_ori) + '\n')
        ci.write(str(camera_intrinsic_calibration))
        ci.flush()
    return camera_pos, camera_ori, camera_intrinsic_calibration


def load_net(path):
    print('=> loading ', path)
    checkpoint = torch.load(path)
    net = GraspPoseNet(tanh=True, grid=True, training=False, bn=False, use_angle=True).cuda()
    pretrained_dict = checkpoint['state_dict']
    net.load_state_dict(pretrained_dict)
    epoch = checkpoint['epoch']
    print("\n=> loaded checkpoint (epoch {})".format(epoch))
    del checkpoint
    net.eval()
    return net


def preprocess_pc(pc, grid_len, grid_num):
    inf_idx = (pc != pc) | (np.abs(pc) > 30)
    pc[inf_idx] = 0.0
    pc_index = np.nonzero(pc[:, 2] > 0.002)[0]
    pc = pc[pc_index]
    pc_x = pc[:, 0]
    pc_y = pc[:, 1]
    pc_z = pc[:, 2]
    # del_idx = (pc_x < -0.22 / 2) | (pc_x > 0.22 / 2) | (pc_y < -0.22 / 2) | (pc_y > 0.22 / 2) | (pc_z > 0.22)
    del_idx = (pc_x < -0.15) | (pc_x > 0.18) | (pc_y < -0.15) | (pc_y > 0.15) | (pc_z > 0.2)
    pc = pc[np.logical_not(del_idx)]
    print('pc shape after filter: ', pc.shape)

    xyz_max = pc.max(0)
    xyz_min = pc.min(0)
    xyz_diff = xyz_max - xyz_min
    xyz_idx = np.where(xyz_diff < grid_len / grid_num)[0]
    print('xyz index shape: ', xyz_idx.shape)
    if xyz_idx.shape[0] > 0:
        xyz_max[xyz_idx] += (grid_len / grid_num)
        xyz_min[xyz_idx] -= (grid_len / grid_num)
    grids = generate_grid(grid_len, grid_num)
    print('grids shape: ', grids.shape)
    print('xyz min max diff: ', xyz_min, xyz_max, xyz_diff)
    # xyz min max diff:[0.2429963 0.0646993 0.1438633][0.4477120 0.1999853 0.1822021][0.2047157 0.13528598 0.03833897]
    grid_choose = (xyz_min.reshape(1, -1) <= grids) * (grids <= xyz_max.reshape(1, -1))
    print(grids)
    grid_choose = (grid_choose.sum(1) == 3)
    grids = grids[grid_choose]
    print('grids shape after choosing: ', grids.shape)
    if grids.shape[0] > 400:
        idx = np.random.choice(np.arange(grids.shape[0]), 400, replace=False)
        grids = grids[idx]
    contact_index = np.arange(pc.shape[0])
    pc = pc / np.array([0.22 / 2, 0.22 / 2, 0.22])
    return pc, grids, contact_index


def get_point_cloud_from_exr(path, pos, ori, intrinsic):
    exr_file = OpenEXR.InputFile(path)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = exr_file.header()['dataWindow']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    rgb = [np.frombuffer(exr_file.channel(c, PixType), dtype=np.float32) for c in 'RGB']
    img = np.reshape(rgb[0], (Size[1], Size[0]))
    depth = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)  # all float number indicate distance
    pc = Depth2PointCloud(depth, intrinsic, ori, pos, org_size=img.shape).transpose()
    inf_idx = (pc != pc) | (np.abs(pc) > 100)
    pc[inf_idx] = 0.0
    pc_index = np.nonzero(pc[:, 2] > 0.002)[0]
    pc = pc[pc_index]

    pc_x = pc[:, 0]
    pc_y = pc[:, 1]
    pc_z = pc[:, 2]
    del_idx = (pc_x < -0.22 / 2) | (pc_x > 0.22 / 2) | (pc_y < -0.22 / 2) | (pc_y > 0.22 / 2) | (pc_z > 0.22)
    pc = pc[np.logical_not(del_idx)]

    xyz_max = pc.max(0)
    xyz_min = pc.min(0)
    xyz_diff = xyz_max - xyz_min
    xyz_idx = np.where(xyz_diff < 0.22 / 10)[0]
    if xyz_idx.shape[0] > 0:
        xyz_max[xyz_idx] += (0.22 / 10)
        xyz_min[xyz_idx] -= (0.22 / 10)
    grids = generate_grid(0.22, 10)
    grid_choose = (xyz_min.reshape(1, -1) <= grids) * (grids <= xyz_max.reshape(1, -1))
    grid_choose = (grid_choose.sum(1) == 3)
    grids = grids[grid_choose]
    if grids.shape[0] > 400:
        idx = np.random.choice(np.arange(grids.shape[0]), 400, replace=False)
        grids = grids[idx]
    contact_index = np.arange(pc.shape[0])
    pc = pc / np.array([0.22 / 2, 0.22 / 2, 0.22])
    return pc, grids, contact_index


def get_grasp_pose(pc, grids, contact_index, grid_len, grid_num):
    pc_, grids_, contact_index_ = torch.tensor(pc).unsqueeze(0), torch.tensor(grids), torch.tensor(contact_index)
    pc1, grids1, contact_index1 = pc_.float().cuda(1), grids_.float().cuda(1), contact_index_.long().cuda(1)
    pc, grids, contact_index = pc_.float().cuda(0), grids_.float().cuda(0), contact_index_.long().cuda(0)
    del (pc_, grids_, contact_index_)
    radius = grid_len / grid_num * np.sqrt(3)
    print('proposal input shape ', pc1.shape, grids1.shape, contact_index1.shape)
    pairs_all_, local_points_ = getTestProposalsV3(pc1, grids1, contact_index1, grid_th=0.075)
    del (pc1, grids1, contact_index1)
    pairs_all, local_points = pairs_all_.cuda(0), local_points_.cuda(0)
    del (local_points_, pairs_all_)

    st = time.time()
    print('net input: pc & local point & pairs all shape: ', pc.shape, local_points.shape, pairs_all.shape)
    with torch.no_grad():
        prop_score, pred_score, pred_offset, pred_angle, prop_posi_idx = net(pc, local_points, pairs_all, scale=radius)
    print('\nforward time: ', time.time() - st)
    print('net output: pred score & offset & angle shape: ', pred_score.shape, pred_offset.shape, pred_angle.shape)
    posi_idx = torch.nonzero(pred_score.view(-1) >= 0.5).view(-1)
    print('pred score and positive index: ', pred_score, posi_idx)
    posi_scores = pred_score.view(-1)[posi_idx]
    prop_score = prop_score.view(-1)[posi_idx]
    prop_posi_idx = prop_posi_idx[posi_idx]
    pred_offset = pred_offset[0, posi_idx]
    pred_angle = pred_angle[0, posi_idx]
    pred_pairs = pairs_all[0, prop_posi_idx]

    print('pred pairs, offset, angles shape fed into get7dof: ', pred_pairs.shape)
    centers, widths, quaternions = get7dofPoses(pred_pairs, pred_offset, pred_angle, scale=radius)

    z = centers[:, 2]
    select = torch.nonzero((widths < 0.085) * (z > 0)).view(-1)
    centers, widths, quaternions = centers[select], widths[select], quaternions[select]
    select_pred_pairs = pred_pairs[select]
    posi_scores = posi_scores[select]
    # prop_score = prop_score[select]
    # pred_angle = pred_angle[select].view(-1)
    # posi_contacts = select_pred_pairs[:, 0]
    # posi_contacts_cpu = posi_contacts.cpu().numpy()
    posi_scores_cpu = posi_scores.cpu().numpy()
    centers = centers.cpu().numpy()
    widths = widths.cpu().numpy()
    quaternions = quaternions.cpu().numpy()
    # prop_score_cpu = prop_score.cpu().numpy()
    # pred_angle_cpu = pred_angle.cpu().numpy()

    print('posi grasp num:', posi_scores.size(0))
    keep = nms2(centers, quaternions, posi_scores_cpu, cent_th=0.04, ang_th=np.pi / 3)
    keep = np.array(keep, dtype=np.int32)
    centers = centers[keep]
    widths = widths[keep]
    quaternions = quaternions[keep]
    print(centers.shape, widths.shape, quaternions.shape)
    return centers, widths, quaternions


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    weight_path = "/home/zibo/GPNet/checkpoint_500.pth"
    net = load_net(weight_path)

    # camera_info_path = '/home/zibo/data_root/GPNet_release_data/images/cylinder014/CameraInfo.npy'
    # cam_pos, cam_ori, cam_intrinsic = get_camera_info(camera_info_path)
    # exr_img_path = '/home/zibo/data_root/GPNet_release_data/images/cylinder014/render0Depth0001.exr'
    # pc, grids, contact_index = get_point_cloud_from_exr(exr_img_path, cam_pos, cam_ori, cam_intrinsic)

    grid_length = 0.22
    grid_num = 10
    point_cloud = np.load("/home/zibo/GPNet/pc.npy")  # input point cloud data
    pc_mean = np.mean(point_cloud, 0)
    point_cloud -= np.expand_dims(pc_mean, 0)

    pc, grids, contact_index = preprocess_pc(point_cloud, grid_length, grid_num)
    centers, widths, quaternions = get_grasp_pose(pc, grids, contact_index, grid_length, grid_num)
    with open('/home/zibo/GPNet/output/grasp_pose_6.txt', 'w') as gp:
        for center, width, quat in zip(centers, widths, quaternions):
            center = center + pc_mean
            gp.write(str(center) + '\t' + str(width) + '\t' + str(quat) + '\n')
        gp.flush()
        print('=> Successfully saved grasp pose!')
    data = []
    for center, quat in zip(centers, quaternions):
        center = center + pc_mean
        data.append([center, quat])
    np.save(('/home/zibo/GPNet/output/grasp_pose_6.npy', np.array(data)))
