import time
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchsummary import summary
from sklearn.metrics import jaccard_score
from tqdm import tqdm

def read_data(data_path, data_type, start_idx=0):
    if data_type == 'before':
        data_path = os.path.join(data_path, 'A')
    elif data_type == 'after':
        data_path = os.path.join(data_path, 'B')
    elif data_type == 'label':
        data_path = os.path.join(data_path, 'label')
    elif data_type == 'diff':
        data_path = os.path.join(data_path, 'Diff')
    elif data_type == 'label_reshaped':
        data_path = os.path.join(data_path, 'label_reshaped')
    else:
        print('data_type is not correct')
        return None

    data = []
    start_time = time.time()
    filenames = sorted(os.listdir(data_path))  # Sort filenames
    filenames = filenames[start_idx:]
    for filename in filenames:
        img = cv2.imread(os.path.join(data_path, filename))
        data.append(img)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time taken to read ",data_type," images: {:.2f} seconds".format(elapsed_time))
    return data


# display images {before, after, labels}
def display_images(imgs, labels):
    num_imgs = len(imgs)
    fig, axs = plt.subplots(1, num_imgs + 1, figsize=(15, 5))
    
    # Displaying the "Before" or "Diff" image
    axs[0].imshow(cv2.cvtColor(imgs[0], cv2.COLOR_BGR2RGB))
    if num_imgs > 1:
        axs[0].set_title('Before')
    else:
        axs[0].set_title('Diff')
    axs[0].axis('off')
    
    # Displaying the "After" image if available
    if num_imgs > 1:
        axs[1].imshow(cv2.cvtColor(imgs[1], cv2.COLOR_BGR2RGB))
        axs[1].set_title('After')
        axs[1].axis('off')
    
    # Displaying the label image
    axs[num_imgs].imshow(labels, cmap='gray')  # Use num_imgs instead of num_imgs - 1
    axs[num_imgs].set_title('Label')  # Use num_imgs instead of num_imgs - 1
    axs[num_imgs].axis('off')  # Use num_imgs instead of num_imgs - 1
    
    plt.show()

def preprocess(before_images, after_images, save_dir="./trainval/"):
    before_images = np.array(before_images)
    after_images = np.array(after_images)
    diff_images = []

    for idx, (before_img, after_img) in enumerate(zip(before_images, after_images)):
        # Compute absolute difference
        diff = cv2.absdiff(before_img, after_img)

        diff_resized = diff 
        diff_images.append(diff_resized)

        # Save the diff image
        save_path = os.path.join(save_dir,"Diff/", f"{idx:04d}.png")
        cv2.imwrite(save_path, diff_resized)

    return diff_images

def count_pure_black_masks(loader):
    pure_black_count = 0
    total_count = 0

    for data in loader:
        if len(data) == 3:
            _, _, labels = data
        # If loader returns 2 items
        elif len(data) == 2:
            _, labels = data
        else:
            raise ValueError("Unexpected number of items returned by the loader")
        for label in labels:
            if torch.all(label == 0):
                pure_black_count += 1
            total_count += 1

    return pure_black_count / total_count * 100



def evaluate(predictions, true_labels):
    jaccard_indices = []
    accuracies = []

    for i in range(len(predictions)):
        pred = predictions[i]
        true_label = true_labels[i]

        # Calculate Jaccard Index
        jaccard_index = jaccard_score(true_label, pred, zero_division=1, average="micro")
        jaccard_indices.append(jaccard_index)

        # Calculate accuracy
        correct_pixels = np.sum(pred == true_label)
        total_pixels = pred.size
        accuracy = correct_pixels / total_pixels
        accuracies.append(accuracy)

    mean_jaccard = np.mean(jaccard_indices)
    mean_accuracy = np.mean(accuracies)

    return mean_jaccard, mean_accuracy


def load_model(model, save_dir, model_name, device='cuda'):
    model_list = os.listdir(save_dir)
    if not model_list:
        return None
    
    # Load the model onto the specified device
    device = torch.device(device)
    model.load_state_dict(torch.load(os.path.join(save_dir, model_name), map_location=device))
    
    return model

def validate_diff(model, test_loader, device, threshold = 0.3):
    diff_images_list = []
    predicted_masks_list = []
    true_masks_list = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            diff_image, label = data

            diff_image = diff_image.to(device)
            label = label.to(device)

            outputs = model(diff_image)

            # Apply threshold to model outputs
            pred_masks = (outputs > threshold).float()

            for i, pred_mask in enumerate(pred_masks):
                diff_images_list.append(diff_image[i].cpu())

                # Save binary predicted mask
                pred_mask = pred_mask.squeeze().cpu().numpy()
                predicted_masks_list.append(pred_mask)

                true_mask = label[i].squeeze().cpu().numpy()
                true_masks_list.append(true_mask)

    return diff_images_list, predicted_masks_list, true_masks_list



def validate(model, test_loader, device, threshold = 0.3):
    before_images_list = []
    after_images_list = []
    predicted_masks_list = []
    true_masks_list = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            before_image, after_image, label = data

            before_image = before_image.to(device)
            after_image = after_image.to(device)
            label = label.to(device)

            outputs = model(before_image, after_image)

            # Apply threshold to model outputs
            pred_masks = (outputs > threshold).float()

            for i, pred_mask in enumerate(pred_masks):
                before_images_list.append(before_image[i].cpu())
                after_images_list.append(after_image[i].cpu())

                # Save binary predicted mask
                pred_mask = pred_mask.squeeze().cpu().numpy()
                predicted_masks_list.append(pred_mask)

                true_mask = label[i].squeeze().cpu().numpy()
                true_masks_list.append(true_mask)

    return before_images_list, after_images_list, predicted_masks_list, true_masks_list


def predict_diff(model, test_loader, device, threshold = 0.3):
    predicted_masks_list = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            diff_image,_ = data
            diff_image = diff_image.to(device)

            outputs = model(diff_image)

            # Apply threshold to model outputs
            pred_masks = (outputs > threshold).float()

            for i, pred_mask in enumerate(pred_masks):
                # Save binary predicted mask
                pred_mask = pred_mask.squeeze().cpu().numpy()
                predicted_masks_list.append(pred_mask)
    return predicted_masks_list

def predict(model, test_loader, device, threshold = 0.3):
    predicted_masks_list = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            bef_image, af_image, _ = data
            bef_image = bef_image.to(device)
            af_image = af_image.to(device)

            outputs = model(bef_image, af_image)

            # Apply threshold to model outputs
            pred_masks = (outputs > threshold).float()

            for i, pred_mask in enumerate(pred_masks):
                # Save binary predicted mask
                pred_mask = pred_mask.squeeze().cpu().numpy()
                predicted_masks_list.append(pred_mask)
    return predicted_masks_list


def display_predicted_images_diff(diff_images, predicted_masks, true_masks_list):
    num_images = min(10, len(diff_images))  # Display up to 10 images
    plt.figure(figsize=(12, 4 * num_images))

    for i in range(num_images):
        plt.subplot(num_images, 3, i * 3 + 1)
        plt.title("Difference Image")
        plt.imshow(TF.to_pil_image(diff_images[i])) 

        plt.subplot(num_images, 3, i * 3 + 2)
        plt.title("Predicted Mask")
        plt.imshow(TF.to_pil_image(predicted_masks[i]), cmap='gray') 

        plt.subplot(num_images, 3, i * 3 + 3)
        plt.title("True Mask")
        plt.imshow(TF.to_pil_image(true_masks_list[i]), cmap='gray') 

    plt.show()


def display_predicted_images(before_images, after_images, predicted_masks, true_masks_list):
    selected_indices = []  # Indices of images with non-completely black true masks
    for i, true_mask in enumerate(true_masks_list):
        # Check if the true mask is not completely black
        if true_mask.max() > 0:
            selected_indices.append(i)
            if len(selected_indices) >= 10:
                break  # Stop when 10 images are selected
    
    num_images = len(selected_indices)
    if num_images == 0:
        print("No images found with non-completely black true masks.")
        return

    plt.figure(figsize=(12, 4 * num_images))

    for idx, i in enumerate(selected_indices):
        plt.subplot(num_images, 4, idx * 4 + 1)
        plt.title("Before Image")
        plt.imshow(TF.to_pil_image(before_images[i]))

        plt.subplot(num_images, 4, idx * 4 + 2)
        plt.title("After Image")
        plt.imshow(TF.to_pil_image(after_images[i]))

        plt.subplot(num_images, 4, idx * 4 + 3)
        plt.title("Predicted Mask")
        plt.imshow(TF.to_pil_image(predicted_masks[i]), cmap='gray')

        plt.subplot(num_images, 4, idx * 4 + 4)
        plt.title("True Mask")
        plt.imshow(TF.to_pil_image(true_masks_list[i]), cmap='gray')

    plt.show()



def save_predictions(predicted_masks_list, output_dir = "./testset/Predictions/"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # Clear the directory if it already exists
        filelist = [f for f in os.listdir(output_dir)]
        for f in filelist:
            os.remove(os.path.join(output_dir, f))

    print("Saving predictions")
    for i, pred_mask in enumerate(predicted_masks_list):
        # Save predicted mask image
        cv2.imwrite(os.path.join(output_dir, f"predicted_mask_{i}.png"), (pred_mask * 255).astype(np.uint8))
    print("Saving Completed")