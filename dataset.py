import torch.utils.data as data
import numpy as np
import os
from PIL import Image
import torch
import util

def read_data(lines):
    list_strs = lines.split(' ')
    num_frame = list_strs[-2]
    label = list_strs[-1]
    path = list_strs[0]
    for idx in range(1, len(list_strs)-2):
        path = path + ' ' + list_strs[idx]

    return [path, num_frame, label]

class Videolist_Parse(object):
	def __init__(self, row):
		self.row = row
	@property
	def path(self):
		return self.row[0]
	@property
	def num_frames(self):
		return int(self.row[1])
	@property
	def label(self):
		return int(self.row[2])

class VideoDataset(data.Dataset):
	def __init__(self, root, list, num_frames, time_stride, split='train'):
		self.list = list
		self.root = root
		self.num_frames = num_frames
		self.stride = time_stride
		self.split = split
		self.clip_transform = util.clip_transform(self.split, self.num_frames)

		self._parse_videolist()

	def __len__(self):
		return len(self.videolist)


	def __getitem__(self, idx):
		record = self.videolist[idx]
		if self.split == 'train':
			indices = self.get_indices(record)
			image_tensor = self.get_img(indices, record)
			return image_tensor.permute(1, 0, 2, 3), record.label

		elif self.split == 'val':
			indices = self.get_val_indices(record)
			image_tensor = self.get_img(indices, record)
			return image_tensor.permute(1, 0, 2, 3), record.label

		elif self.split == '3crop':
			indices = self.get_test_indices(record)
			image_tensor = self.get_test_img(indices, record)
			return image_tensor, record.label


	def _parse_videolist(self):
		lines = [read_data(x.strip()) for x in open(self.root + self.list)]
		self.videolist = [Videolist_Parse(item) for item in lines]


	def get_indices(self, record):
		if record.num_frames <= self.num_frames * self.stride:
			offsets = 0
		else:
			offsets = np.random.randint(0, record.num_frames - self.num_frames * self.stride)
		return offsets


	def get_val_indices(self, record):
		if record.num_frames <= self.num_frames * self.stride:
			offsets = 0
		else:
			offsets = record.num_frames // 2 - self.num_frames // 2 * self.stride
		return offsets


	def get_test_indices(self, record):
		if record.num_frames <= self.num_frames * self.stride:
			offsets = 0
		else:
			offsets = [int(idx) for idx in np.linspace(0, record.num_frames - self.num_frames * self.stride, 10)]
		return offsets




	def get_img(self, indices, record):
		imgs = []
		if indices == 0:
			for idx_img in range(self.num_frames):
				dir_img = os.path.join(self.root, record.path, str(idx_img % record.num_frames)+'.jpg')
				image = Image.open(dir_img).convert('RGB')
				imgs.append(image)
			frames = self.clip_transform(imgs)
			return frames
		else:
			for idx, idx_img in enumerate(range(indices, indices + self.num_frames)):
				dir_img = os.path.join(self.root, record.path, str(indices + idx * self.stride) + '.jpg')
				image = Image.open(dir_img).convert('RGB')
				imgs.append(image)
			frames = self.clip_transform(imgs)
			return frames


	def get_test_img(self, indices, record):
		if indices == 0:
			imgs = []
			for idx_img in range(self.num_frames):
				dir_img = os.path.join(self.root, record.path, str(idx_img % record.num_frames)+'.jpg')
				image = Image.open(dir_img).convert('RGB')
				imgs.append(image)
			frames = [self.clip_transform(imgs).permute(1, 0, 2, 3) for _ in range(3)]
			clip = torch.stack(frames, dim=0)
			return clip

		else:
			imgs_tensor = []
			for starts in indices:
				imgs = []
				for idx in range(self.num_frames):
					dir_img = os.path.join(self.root, record.path, str(starts + idx * self.stride) + '.jpg')
					image = Image.open(dir_img).convert('RGB')
					imgs.append(image)
				frames = [self.clip_transform(imgs).permute(1, 0, 2, 3) for _ in range(3)]
				clip = torch.stack(frames, dim=0)
				imgs_tensor.append(clip)
			imgs_tensor = torch.stack(imgs_tensor, dim=0)
			return imgs_tensor.view(-1, 3, self.num_frames, 112, 112)



if __name__ == '__main__':
	print ('haha')
	count = 0
	for x in open('../Datasets/list/kinetics_val.txt'):
		# count += 1
		lines = x.strip()
		print(lines)
		break
		# print (lines, type(lines), len(lines))
		# if count > 10:
		# 	break
