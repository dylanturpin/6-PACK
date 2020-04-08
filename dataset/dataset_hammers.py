import os
import pdb
import math
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from pyquaternion import Quaternion

def search_fit(points):
    min_x = points[:, 0].min()
    max_x = points[:, 0].max()
    min_y = points[:, 1].min()
    max_y = points[:, 1].max()
    min_z = points[:, 2].min()
    max_z = points[:, 2].max()

    return [min_x, max_x, min_y, max_y, min_z, max_z]

def divide_scale(scale, pts):
        pts[:, 0] = pts[:, 0] / scale[0]
        pts[:, 1] = pts[:, 1] / scale[1]
        pts[:, 2] = pts[:, 2] / scale[2]
        return pts

def get_anchor_box(ori_bbox):
    bbox = ori_bbox
    limit = torch.Tensor(search_fit(bbox))
    num_per_axis = 5
    gap_max = num_per_axis - 1

    small_range = [1, 3]

    gap_x = (limit[1] - limit[0]) / float(gap_max)
    gap_y = (limit[3] - limit[2]) / float(gap_max)
    gap_z = (limit[5] - limit[4]) / float(gap_max)

    ans = []
    scale = [max(limit[1], -limit[0]), max(limit[3], -limit[2]), max(limit[5], -limit[4])]

    for i in range(0, num_per_axis):
        for j in range(0, num_per_axis):
            for k in range(0, num_per_axis):
                ans.append([limit[0] + i * gap_x, limit[2] + j * gap_y, limit[4] + k * gap_z])

    ans = torch.Tensor(ans)
    scale = torch.Tensor(scale)

    ans = ans/scale

    return ans, scale


class Dataset(Dataset):
    def __init__(self, dataset_root=None, set_name='train', train_task_ids=[0], val_task_ids=[0], num_points=500):
        self.x = 0
        self.dataset_root = dataset_root
        data_path = os.path.join(dataset_root, 'data.npy')
        data = np.load(data_path, allow_pickle=True).item()
        self.state = data['state']
        self.quat = data['quat']
        self.pos = data['pos']
        self.rotation = data['rotation']
        self.vert = data['vert']
        self.faces = data['faces']
        self.xmap = None
        self.ymap = None

        if set_name == 'train':
            self.task_ids = train_task_ids
        elif set_name == 'val':
            self.task_ids = val_task_ids

        # num_frames_array snippet from Nvidia UnsupervisedLandmarkLearning
        self.num_frames_array = [0]
        for t_id in self.task_ids:
            dir_path = os.path.join(self.dataset_root, f'{t_id:06d}')
            # contains depth + rgb for each frame os divide by 2
            # subtract 1 since frames are returned as neighbouring pairs
            num_frames = len(os.listdir(dir_path))//2 - 1
            self.num_frames_array.append(num_frames)
        self.num_frames_array = np.array(self.num_frames_array).cumsum()

    def __len__(self):
        return self.num_frames_array[-1]

    def _get_point_cloud(self, depth):
        height = depth.shape[0]
        width = depth.shape[1]
        #val,counts = torch.unique(depth, return_counts=True)
        #table_val = val[counts.argmax()]
        table_val, _ = torch.mode(torch.flatten(depth),0)

        if self.xmap is None:
            # row
            self.xmap = torch.tensor([[j for i in range(height)] for j in range(height)], dtype=torch.float)
            # col
            self.ymap = torch.tensor([[i for i in range(width)] for j in range(width)], dtype=torch.float)

        choose = (torch.flatten(depth) < table_val).nonzero().flatten()

        # take random 500 points
        idx = np.random.choice(choose.shape[0], size=500)
        choose = choose[idx]

        depth_masked = depth.flatten()[choose]
        xmap_masked = self.xmap.flatten()[choose]
        ymap_masked = self.ymap.flatten()[choose]

        fovy = 35.0
        f = height / math.tan(fovy * math.pi / 360)
        cam = np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))
        cx = height/2
        cy = width/2
        cam_scale = 1.0

        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cx) * pt2 / f
        pt1 = (xmap_masked - cy) * pt2 / f

        pt2 = pt2/0.1
        cloud = torch.cat((pt1[:,None], pt0[:,None], pt2[:,None]), dim=1)

        # turn depths into coords by subtracting from camera
        # -0.13 0.6 0.6
        # 0 0.55 0.48
        cloud[:,0] = 0.575 + cloud[:,0]
        cloud[:,1] = 0.0125 + cloud[:,1]
        cloud[:,2] = (table_val/0.1) - cloud[:,2]
        cloud = cloud[:,[1,0,2]]


        choose = choose.view(1,-1)
        return choose, cloud

    def get_frame_index(self, global_idx):
        """maps global frame index to video index and local video frame index
        """
        vid_idx_idx = np.searchsorted(self.num_frames_array, global_idx, side='right')-1
        frame_idx = global_idx - self.num_frames_array[vid_idx_idx]
        vid_idx = self.task_ids[int(vid_idx_idx)]
        return vid_idx, frame_idx



    def __getitem__(self, index):
        vid_idx, frame_idx = self.get_frame_index(index)

        rgb_path_fr = os.path.join(self.dataset_root, f'{vid_idx:06d}', f'{frame_idx:06d}.jpg')
        depth_path_fr = os.path.join(self.dataset_root, f'{vid_idx:06d}', f'{frame_idx:06d}d.tiff')
        rgb_path_to = os.path.join(self.dataset_root, f'{vid_idx:06d}', f'{frame_idx+1:06d}.jpg')
        depth_path_to = os.path.join(self.dataset_root, f'{vid_idx:06d}', f'{frame_idx+1:06d}d.tiff')

        # img
        img_fr = np.array(Image.open(rgb_path_fr)).transpose((2,0,1))
        img_fr = torch.Tensor(img_fr)
        depth_fr = torch.Tensor(np.array(Image.open(depth_path_fr)))
        img_to = np.array(Image.open(rgb_path_to)).transpose((2,0,1))
        img_to = torch.Tensor(img_to)
        depth_to = torch.Tensor(np.array(Image.open(depth_path_to)))

        # get point cloud
        choose_fr, cloud_fr = self._get_point_cloud(depth_fr)
        choose_to, cloud_to = self._get_point_cloud(depth_to)
        choose_fr, cloud_fr = torch.LongTensor(choose_fr), torch.Tensor(cloud_fr)
        choose_to, cloud_to = torch.LongTensor(choose_to), torch.Tensor(cloud_to)

        # rotation and translation for each
        r_fr = torch.Tensor(self.rotation[vid_idx,frame_idx,:,:])
        t_fr = torch.Tensor(self.pos[vid_idx,frame_idx,:])
        r_to = torch.Tensor(self.rotation[vid_idx,frame_idx+1,:,:])
        t_to = torch.Tensor(self.pos[vid_idx,frame_idx+1,:])

        # joint anchor and scale
        joint_mesh = np.concatenate((self.vert[vid_idx][frame_idx,:,:], self.vert[vid_idx][frame_idx+1,:,:]),axis=0)
        anchor = torch.zeros(125,3, dtype=torch.float)
        scale = torch.zeros(3, dtype=torch.float)
        a,s = get_anchor_box(joint_mesh)
        anchor[:,:] = a
        scale[:] = s
        # get cloud to match anchor scale
        cloud_fr = cloud_fr / scale

        # mesh
        mesh = torch.Tensor(self.vert[vid_idx][frame_idx,:,:])
        # return mesh to canonical pose
        mesh = torch.mm(mesh - t_fr[None,:], r_fr)
        faces = torch.LongTensor(self.faces[vid_idx])

        # cate is 1
        cate = torch.LongTensor([1])

        # state
        state_fr = torch.Tensor(self.state[vid_idx,frame_idx,:])

        return img_fr, choose_fr, cloud_fr, r_fr, t_fr, img_to, choose_to, cloud_to, r_to, t_to, mesh, anchor, scale, cate
