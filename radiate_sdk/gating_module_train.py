import torch
from torch.utils.data import DataLoader
from model.gate import DeepGatingModule
from model.hydranet import HydraFusion
from config import Config
from demo import get_data_loaders

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

# Step 1: Load the pre-trained HydraFusion model
def load_hydrafusion_model(config, checkpoint_path):
    model = HydraFusion(config)
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(config.device)
    model.eval()
    return model

# Step 2: Extract stem outputs and branch losses
def extract_stem_outputs_and_losses(model, loader, config):
    stem_outputs, branch_losses = [], []
    with torch.no_grad():
        for batch in loader:
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

            losses, _ = model(
                leftcamera_x=camera_left,
                rightcamera_x=camera_right,
                radar_x=radar,
                bev_lidar_x=lidar,
                radar_y=targets,
                cam_y=targets
            )

            stem_outs = {
                "radar": model.radar_stem(radar),
                "camera_left": model.camera_stem(camera_left),
                "camera_right": model.camera_stem(camera_right),
                "lidar": model.lidar_stem(lidar)
            }

            stem_outputs.append(stem_outs)
            branch_losses.append({k: sum(v.values()) for k, v in losses.items()})

    return stem_outputs, branch_losses

# Step 3: Train the Gating Module
def train_gating_module(stem_outputs, branch_losses, config):
    gating_model = DeepGatingModule(input_channels=3, output_shape=7, dropout=0.5).to(config.device)
    optimizer = torch.optim.Adam(gating_model.parameters(), lr=5e-5)
    criterion = torch.nn.L1Loss()

    stem_tensors = [torch.cat([v.flatten() for v in outputs.values()], dim=0) for outputs in stem_outputs]
    loss_tensors = [torch.tensor([v for v in losses.values()], device=config.device) for losses in branch_losses]

    dataset = list(zip(stem_tensors, loss_tensors))
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    gating_model.train()
    for epoch in range(5):
        total_loss = 0
        for batch_stems, batch_losses in loader:
            batch_stems = batch_stems.to(config.device)
            batch_losses = batch_losses.to(config.device)

            predictions = gating_model(batch_stems)
            loss = criterion(predictions, batch_losses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        torch.save(gating_model.state_dict(), f"gating_model_{epoch}.pth")
    
    torch.save(gating_model.state_dict(), "gating_model.pth")

def main():
    args = []
    config = Config(args)

    train_loader, _ = get_data_loaders()

    # Load pre-trained HydraFusion model
    hydrafusion_model = load_hydrafusion_model(config, "weights/model_weights_epoch_1_lr_5e3.pth")

    # Extract stem outputs and branch losses
    print("Extracting stem outputs and branch losses...")
    stem_outputs, branch_losses = extract_stem_outputs_and_losses(hydrafusion_model, train_loader, config)

    # Train the Gating Module
    print("Training the Gating Module...")
    train_gating_module(stem_outputs, branch_losses, config)

if __name__ == "__main__":
    main()