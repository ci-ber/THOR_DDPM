from core.DataLoader import DefaultDataset
import torchvision.transforms as transforms
from transforms.preprocessing import *
from dl_utils import get_data_from_csv
import glob


class RayLoader(DefaultDataset):
    def __init__(self, data_dir, file_type='', label_dir=None, mask_dir=None, target_size=(256, 256), test=False):
        self.target_size = target_size
        self.RES = transforms.Resize(self.target_size)
        print(target_size)
        self.mask_dir = mask_dir
        if mask_dir is not None:
            if 'csv' in mask_dir[0]:
                self.mask_files = get_data_from_csv(mask_dir)
            else:
                self.mask_files = [glob.glob(mask_dir_i + file_type) for mask_dir_i in mask_dir]
        super(RayLoader, self).__init__(data_dir, file_type, label_dir, mask_dir, target_size, test)

    def get_image_transform(self):
        default_t = transforms.Compose([ReadImage(), To01(65536), Norm98(cut_off=1),
                                        Pad((18,18)),
                                        AddChannelIfNeeded(),  self.RES,
                                        AssertChannelFirst()])
        # default_t = transforms.Compose([ReadImage(),To01(),
        #                                 # Norm98(), #Slice(),
        #                                 Pad((18, 18)),
        #                                 AddChannelIfNeeded(),
        #                                 AssertChannelFirst(), self.RES,
        #                                 #AdjustIntensity(),
        #                                # transforms.ToPILImage(), transforms.RandomAffine(10, (0.1, 0.1), (0.9, 1.1)), transforms.RandomHorizontalFlip(0.5),transforms.ToTensor()
        #                                 ])
        return default_t

    def get_image_transform_test(self):
        default_t = transforms.Compose([ReadImage(), To01(65536), Norm98(cut_off=1),
                                        Pad((18,18)),
                                        AddChannelIfNeeded(), self.RES,
                                        AssertChannelFirst()])
        # default_t = transforms.Compose([ReadImage(), To01(),
        #                                 # Norm98(), #Slice(),
        #                                 Pad((18, 18)),
        #                                 AddChannelIfNeeded(),
        #                                 AssertChannelFirst(), self.RES,
        #                                 #transforms.ToPILImage(), transforms.RandomAffine(20, (0.1, 0.1), (0.9, 1.1)),
        #                                 #transforms.RandomVerticalFlip(0.4),
        #                                 #transforms.ToTensor()
        #                                 ])
        return default_t

    def get_label_transform(self):
        default_t = transforms.Compose([ReadImage(), To01(255), Pad((18,18)),
                                        AddChannelIfNeeded(), self.RES,
                                        AssertChannelFirst()])
        # default_t = transforms.Compose([ReadImage(), To01(),
        #                                , #Slice(),
        #                                 Pad((18, 18)),
        #                                 AddChannelIfNeeded(),
                                        # AssertChannelFirst(), self.RES])
        return default_t

    def get_label(self, idx):
        return_label = 0
        if self.label_dir is not None:
            return_label = self.seg_t(self.label_files[idx])
        if self.mask_dir is not None:
            mask_label = self.seg_t(self.mask_files[idx])
            # print(mask_label.shape)
            return_label = torch.stack([return_label, mask_label], dim=0)
        return return_label