import os
import random
from datetime import datetime
import numpy as np
import torch.utils.data
import torchvision.transforms
import torchvision.transforms.functional
import PIL.Image

class ClipRandomCrop(torchvision.transforms.RandomCrop):
  def __init__(self, size):
    self.size = size
    self.i = None
    self.j = None
    self.th = None
    self.tw = None

  def __call__(self, img):
    if self.i is None:
      self.i, self.j, self.th, self.tw = self.get_params(img, output_size=self.size)
    return torchvision.transforms.functional.crop(img, self.i, self.j, self.th, self.tw)


class ClipRandomHorizontalFlip(object):
  def __init__(self, ratio=0.5):
    self.is_flip = random.random() < ratio

  def __call__(self, img):
    if self.is_flip:
      return torchvision.transforms.functional.hflip(img)
    else:
      return img


class ClipColorJitter(torchvision.transforms.ColorJitter):
  def __init__(self, brightness=0., contrast=0., saturation=0., hue=0.):
    super(ClipColorJitter, self).__init__(brightness=brightness, contrast=contrast,saturation=saturation, hue=hue)
    self.transform = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

  def __call__(self, img):
    return self.transform(img)


class ClipSubstractMean(object):
  def __init__(self, b=104, g=117, r=123):
    self.means = np.array((r, g, b))

  def __call__(self, img):
    img = np.array(img)
    return img - self.means


def pil_loader(path):
  with open(path, 'rb') as f:
    img = PIL.Image.open(f)
    return img.convert('RGB')


class VideoDataset(torch.utils.data.Dataset):
  def __init__(self, info_list_file, root_dir='', depth=16,
               image_height=160, image_width=160,
               crop_height=160, crop_width=160,
               rgb_means=(123, 117, 104),
               num_label=101, is_train=False, shuffle=False,
               image_loader=pil_loader):

    assert os.path.exists(info_list_file)
    self.root_dir = root_dir
    self.depth = depth
    self.image_height = image_height
    self.image_width = image_width
    self.crop_height = crop_height
    self.crop_width = crop_width
    self.rgb_means = np.array(rgb_means).astype(np.float32)
    self.num_label = num_label
    self.is_train = is_train
    self.shuffle = shuffle
    self.image_loader = image_loader
    self.vdirs, self.vframes, self.vlabels = self.read_info_list(info_list_file)

  def read_info_list(self, info_list_file):
    vdirs = []
    vframes = []
    vlabels = []
    with open(info_list_file) as f:
      for line in f:
        vdir, frames, labels = line.strip().split(':')
        vdirs.append(vdir)
        frame_start, frame_end = [int(frame) for frame in frames.split(' ')]
        assert frame_end - frame_start + 1 >= self.depth
        vframes.append((frame_start, frame_end))
        vlabels.append(tuple([int(label) for label in labels]))
    return vdirs, vframes, vlabels

  def __getitem__(self, index):
    clip = self.load_clip(index)  # D, H, W, C
    clip = np.transpose(clip, (3, 0, 1, 2))
    labels = self.vlabels[index]
    return clip, labels

  def load_clip(self, index):
    vdir = os.path.join(self.root_dir, self.vdirs[index])
    frame_begin, frame_end = self.vframes[index]
    frame_begin = np.random.randint(frame_begin, frame_end - self.depth + 1)
    frame_end = frame_begin + self.depth - 1
    image_filenames = [os.path.join(vdir, str(frame_id) + '.jpg')
                       for frame_id in range(frame_begin, frame_end + 1)]
    transforms = self.transforms()
    clip = np.stack([np.array(transforms(self.image_loader(image_filename)))
                     for image_filename in image_filenames]).astype(np.float32)
    clip -= self.rgb_means
    return clip

  def transforms(self):
    resize = torchvision.transforms.Resize((self.image_height, self.image_width), interpolation=2)
    if self.is_train:
      random_crop = ClipRandomCrop((self.crop_height, self.crop_width))
      flip = ClipRandomHorizontalFlip(ratio=0.5)
      return torchvision.transforms.Compose([resize, random_crop, flip])
    else:
      center_crop = torchvision.transforms.CenterCrop((self.crop_height, self.crop_width))
      return torchvision.transforms.Compose([resize, center_crop])

  def __len__(self):
    return len(self.vdirs)


if __name__ == '__main__':
  info_list_file = '/DATACENTER/2/rwduzhao/data/UCF101/info.lst'
  root_dir = '/DATACENTER/2/rwduzhao/data/UCF101/frame/'
  dataset = VideoDataset(info_list_file, root_dir=root_dir, depth=16,
                         image_height=182, image_width=242,
                         crop_height=160, crop_width=160,
                         is_train=True, shuffle=False,
                         image_loader=pil_loader)

  num_batch = 10
  batch_size = 10 * 4

  start_time = datetime.now()
  for batch_id in range(num_batch):
    print('batch:', batch_id + 1)
    for index in range(batch_size):
      print(dataset[index][0].shape)
  time_elapsed = datetime.now() - start_time
  print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

  start_time = datetime.now()
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, num_workers=16)
  for batch_id, batch in enumerate(dataloader):
    print('batch:', batch_id)
    if batch_id == num_batch:
      break
  time_elapsed = datetime.now() - start_time
  print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))