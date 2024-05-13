import cv2
from torch.utils.data import Dataset

class ChangeDetectionDataset(Dataset):
    def __init__(self, before_images, after_images, labels, transform_rgb=None, transform_label=None):
        self.transform_rgb = transform_rgb
        self.transform_label = transform_label
        self.before_images = before_images
        self.after_images = after_images
        self.labels = labels

    def __len__(self):
        return len(self.before_images)  # Return the number of samples

    def __getitem__(self, idx):
        before_image = self.before_images[idx]
        after_images = self.after_images[idx]
        if len(self.labels) > 0:
            label = cv2.cvtColor(self.labels[idx], cv2.COLOR_BGR2GRAY)
        else:
            label = None

        if self.transform_rgb:
            before_img = self.transform_rgb(before_image)
            after_img = self.transform_rgb(after_images)

        if self.transform_label:
            label = self.transform_label(label)

        return before_img, after_img, label