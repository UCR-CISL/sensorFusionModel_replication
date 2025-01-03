import torch
from config import Config
from model.hydranet import HydraFusion
from demo import get_data_loaders

import matplotlib.pyplot as plt
import matplotlib.patches as patches

LABEL_DICT = {
    "car": 0,
    "van": 1,
    "truck": 2,
    "bus": 3,
    "motorbike": 4,
    "bicycle": 5,
    "pedestrian": 6,
    "group_of_pedestrians": 7
}

def initialize_model(config):
    model = HydraFusion(config)
    model = model.to(config.device)
    return model

def visualize_predictions(image, detections, title="Predictions"):
    """
    Visualize predictions by overlaying bounding boxes and labels on the image.
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image.permute(1, 2, 0).cpu().numpy())  # Convert tensor to image format (H, W, C)
    
    for detection in detections:
        bbox = detection["boxes"]
        labels = detection["labels"]
        for i in range(len(bbox)):
            x1, y1, x2, y2 = bbox[i]
            label = labels[i]
            
            # Draw bounding box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="r", facecolor="none"
            )
            ax.add_patch(rect)
            
            # Add label text
            ax.text(
                x1, y1 - 5, str(label), color="white", fontsize=12, bbox=dict(facecolor="red", alpha=0.5)
            )
    plt.title(title)
    plt.axis("off")
    plt.show()

def test_model(model, test_loader, config):
    """
    Test the model and visualize predictions for each sensor or fusion branch.
    """
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            radar = batch['radar'].to(config.device)
            lidar = batch['lidar'].to(config.device)
            camera_left = batch['camera_left'].to(config.device)
            camera_right = batch['camera_right'].to(config.device)
            bboxes = batch['bboxes']
            classes = batch['classes']

            targets = [
                {
                    "boxes": bbox,
                    "labels": torch.tensor(
                        [LABEL_DICT[label] for label in class_list],
                        device=config.device
                    )
                }
                for bbox, class_list in zip(bboxes, classes)
            ]

            # Perform inference
            _, detections = model(
                leftcamera_x=camera_left,
                rightcamera_x=camera_right,
                radar_x=radar,
                bev_lidar_x=lidar,
                radar_y=targets,
                cam_y=targets
            )
            
            # Iterate over the detections for each branch
            for branch, branch_detections in detections.items():
                for det in branch_detections:  # Iterate through detections
                    boxes = det["boxes"].cpu()
                    labels = det["labels"].cpu()
                    scores = det["scores"].cpu()

                    if boxes.size(0) > 0:  # Check if there are any detections
                        # Pass the corresponding image tensor based on the branch
                        if branch == "radar":
                            visualize_predictions(radar[0], boxes, labels, scores, title="Radar Predictions")
                        elif branch == "camera_left":
                            visualize_predictions(camera_left[0], boxes, labels, scores, title="Left Camera Predictions")
                        elif branch == "camera_right":
                            visualize_predictions(camera_right[0], boxes, labels, scores, title="Right Camera Predictions")
                        elif branch == "lidar":
                            visualize_predictions(lidar[0], boxes, labels, scores, title="Lidar Predictions")
                        else:
                            # Handle fused branches (e.g., fused1, fused2, fused3)
                            visualize_predictions(camera_left[0], boxes, labels, scores, title=f"{branch} Predictions")


def test_model_simply(model, test_loader, config):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in test_loader:
            radar = batch['radar'].to(config.device)
            lidar = batch['lidar'].to(config.device)
            camera_left = batch['camera_left'].to(config.device)
            camera_right = batch['camera_right'].to(config.device)

            losses, detections = model(
                leftcamera_x=camera_left,
                rightcamera_x=camera_right,
                radar_x=radar,
                bev_lidar_x=lidar
            )

            print(losses)
            total_loss += sum(loss for branch_loss in losses.values() for loss in branch_loss.values())
            num_batches += 1
            print(f'At batch: {num_batches}, Total loss = {total_loss}')

        avg_loss = total_loss / num_batches
        print(f"Average Test Loss: {avg_loss}")
        print(f"Example Detections: {detections}")

def main():
    args = []
    config = Config(args)
    model = initialize_model(config)

    # Load trained weights
    model.load_state_dict(torch.load("weights/model_weights_epoch_1_lr_5e3.pth"))

    # Load test data
    _, test_loader = get_data_loaders()

    # Test the model and visualize predictions
    print("Starting Testing and Visualization...")
    # test_model(model, test_loader, config)
    test_model_simply(model, test_loader, config)

if __name__ == "__main__":
    main()
