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
    batch_num = 0
    with torch.no_grad():
        for batch in loader:
            batch_num += 1
            print(f"Processing batch : {batch_num}")
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

    torch.save({
        'stem_outputs': stem_outputs,
        'branch_losses': branch_losses
    }, 'stem_branch_data.pt')

    return stem_outputs, branch_losses

# Step 3: Train the Gating Module
def train_gating_module(stem_outputs, branch_losses, config):
    gating_model = DeepGatingModule(input_channels=64, output_shape=6, dropout=0.5).to(config.device)
    optimizer = torch.optim.Adam(gating_model.parameters(), lr=5e-5)
    criterion = torch.nn.L1Loss()

    downsampler = torch.nn.AdaptiveAvgPool2d((5, 5))
    stem_tensors = []
    loss_tensors = []

    for outputs, losses in zip(stem_outputs, branch_losses):
        radar = downsampler(outputs["radar"])
        lidar = downsampler(outputs["lidar"])
        camera_left = downsampler(outputs["camera_left"])
        camera_right = downsampler(outputs["camera_right"])

        combined = torch.cat([radar, lidar, camera_left, camera_right], dim=0)  # Shape: [4, 64, 5, 5]
        
        stem_tensors.append(combined)
        loss_tensors.append(torch.tensor([v for v in losses.values()], device=config.device))
    
    stem_tensors = torch.cat(stem_tensors, dim=0)
    loss_tensors = torch.stack(loss_tensors)

    dataset = list(zip(stem_tensors, loss_tensors))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    gating_model.train()
    for epoch in range(5):
        total_loss = 0
        for batch_stems, batch_losses in loader:
            batch_stems = batch_stems.to(config.device)
            batch_losses = batch_losses.to(config.device)
            # single_loss_sample = batch_losses[0].to(config.device)
            # print(len(batch_losses))

            predictions = gating_model(batch_stems)
            loss = 0
            for batch_loss in batch_losses:
                loss += criterion(predictions, batch_loss)
            loss /=32

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

    # train_loader, _ = get_data_loaders()

    # # Load pre-trained HydraFusion model
    # hydrafusion_model = load_hydrafusion_model(config, "weights/model_weights_epoch_1_lr_5e3.pth")

    # # Extract stem outputs and branch losses
    # print("Extracting stem outputs and branch losses...")
    # stem_outputs, branch_losses = extract_stem_outputs_and_losses(hydrafusion_model, train_loader, config)

    data = torch.load('stem_branch_data.pt')
    stem_outputs = data['stem_outputs']
    branch_losses = data['branch_losses']

    # print(branch_losses[0])
    # print("Stem output")
    # print(stem_outputs[0]["radar"].shape)
    # print(stem_outputs[0]["lidar"].shape)
    # print(stem_outputs[0]["camera_left"].shape)
    # print(stem_outputs[0]["camera_right"].shape)
    #print(branch_losses)

    # Train the Gating Module
    print("Training the Gating Module...")
    train_gating_module(stem_outputs, branch_losses, config)

if __name__ == "__main__":
    main()