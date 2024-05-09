import cv2
from torch.utils.data import Dataset

class ChangeDetectionDiffDataset(Dataset):
    def __init__(self, diff_images, labels, transform_rgb=None, transform_label=None):
        self.transform_rgb = transform_rgb
        self.transform_label = transform_label
        self.diff_images = diff_images
        self.labels = labels

    def __len__(self):
        return len(self.diff_images)  # Return the number of samples

    def __getitem__(self, idx):
        diff_image = self.diff_images[idx]
        if len(self.labels) > 0:
            label = cv2.cvtColor(self.labels[idx], cv2.COLOR_BGR2GRAY)
        else:
            label = None
        if self.transform_rgb:
            diff_image = self.transform_rgb(diff_image)

        if self.transform_label:
            label = self.transform_label(label)

        return diff_image, label