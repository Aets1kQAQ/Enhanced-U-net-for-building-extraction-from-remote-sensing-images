import os
import cv2
import numpy as np

def calculate_metrics(true, pred):
    # Flatten the arrays
    true = true.flatten()
    pred = pred.flatten()
    
    # True Positives, False Positives, True Negatives, False Negatives
    tp = np.sum((true == 1) & (pred == 1))
    fp = np.sum((true == 0) & (pred == 1))
    tn = np.sum((true == 0) & (pred == 0))
    fn = np.sum((true == 1) & (pred == 0))
    
    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

# Define paths
#这是预测结果的路径
images_path = 'outputs'
#这是修改结果/标签的路径
labels_path = 'label'

# Initialize lists to store results
f1_scores = []
precisions = []
recalls = []

# Get list of files in the images directory
image_files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]

for image_file in image_files:
    # Construct the full paths to the image and label files
    image_path = os.path.join(images_path, image_file)
    label_path = os.path.join(labels_path, image_file)

    # Check if the corresponding label file exists
    if os.path.exists(label_path):
        # Read the image and label files
        image = cv2.imread(image_path, 0)
        label = cv2.imread(label_path, 0)

        # Check if the image and label are read correctly
        if image is None:
            print(f'Failed to read image {image_path}')
            continue
        if label is None:
            print(f'Failed to read label {label_path}')
            continue

        # Convert images to binary (0 and 1)
        _, image_bin = cv2.threshold(image, 1, 1, cv2.THRESH_BINARY)
        _, label_bin = cv2.threshold(label, 1, 1, cv2.THRESH_BINARY)

        # Calculate metrics
        precision, recall, f1 = calculate_metrics(label_bin, image_bin)

        # Store the metrics
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        # Print individual metrics
        print(f'Metrics for {image_file}:')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

# Check if lists are not empty before calculating the average metrics
if f1_scores and precisions and recalls:
    # Compute the average metrics
    average_f1 = np.mean(f1_scores)
    average_precision = np.mean(precisions)
    average_recall = np.mean(recalls)

    # Print the average results
    print(f'Average F1 Score: {average_f1:.4f}')
    print(f'Average Precision: {average_precision:.4f}')
    print(f'Average Recall: {average_recall:.4f}')
else:
    print('No valid metrics were computed. Please check the input data.')
