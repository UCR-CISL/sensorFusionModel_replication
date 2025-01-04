import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from model.hydranet import HydraFusion
from demo import get_data_loaders
import math

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

def train_model(model, train_loader, config):
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    model.train()
    for epoch in range(5):
        batch_num = 0
        for batch in train_loader:
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

            # Forward pass
            losses, detections = model(
                leftcamera_x=camera_left,
                rightcamera_x=camera_right,
                radar_x=radar,
                bev_lidar_x=lidar,
                radar_y=targets,
                cam_y=targets
            )

            # Normalize and combine losses
            normalized_losses = []
            for branch_loss in losses.values():
                branch_total = sum(branch_loss.values())
                normalized_losses.append(branch_total / len(branch_loss))
            total_loss = sum(normalized_losses)

            if not math.isnan(total_loss):
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                optimizer.step()
                print(f"Epoch {epoch}, Batch {batch_num+1}, Loss: {total_loss.item()}")

            batch_num += 1
        torch.save(model.state_dict(), f"model_weights_epoch_{epoch}.pth")
        scheduler.step()
    torch.save(model.state_dict(), "model_weights.pth")

def test_model(model, test_loader, config):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in test_loader:
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

            losses, detections = model(
                leftcamera_x=camera_left,
                rightcamera_x=camera_right,
                radar_x=radar,
                bev_lidar_x=lidar,
                radar_y=targets,
                cam_y=targets
            )

            total_loss += sum(loss for branch_loss in losses.values() for loss in branch_loss.values())
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Average Test Loss: {avg_loss}")
        print(f"Example Detections: {detections}")


def main():
    args = []
    config = Config(args)
    model = initialize_model(config)
    train_loader, test_loader = get_data_loaders()
    
    # Train the model
    print("Starting Training...")
    train_model(model, train_loader, config)

    # Test the model
    # print("Starting Testing...")
    # test_model(model, test_loader, config)

if __name__ == "__main__":
    main()
