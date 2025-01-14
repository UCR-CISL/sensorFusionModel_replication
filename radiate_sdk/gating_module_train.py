import torch
from torch.utils.data import DataLoader
from model.gate import DeepGatingModule
from model.hydranet import HydraFusion
from config import Config
from demo import get_data_loaders
import torch.multiprocessing as mp

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
    file_count = 0
    with torch.no_grad():
        for batch in loader:
            batch_num += 1
            print(f"Processing batch : {batch_num}")
            radar = batch['radar'].to(config.device)
            lidar = batch['lidar'].to(config.device)
            camera_left = batch['camera_left'].to(config.device)
            camera_right = batch['camera_right'].to(config.device)
            lidar_right = batch['lidar_right'].to(config.device)
            lidar_left = batch['lidar_left'].to(config.device)
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
                l_lidar_x=lidar_left,
                r_lidar_x=lidar_right,
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

            if batch_num % 4500 == 0:
                torch.save({
                     'stem_outputs': stem_outputs,
                     'branch_losses': branch_losses
                }, f'stem_branch_data{file_count}.pt')
                print(f"Saved files: stem_outputs_{file_count}.pt")
                
                stem_outputs = []
                branch_losses = []
                file_count += 1

                torch.cuda.empty_cache()

    if stem_outputs and branch_losses:
        torch.save({
            'stem_outputs': stem_outputs,
            'branch_losses': branch_losses
        }, f'stem_branch_data{file_count}.pt')
        print(f"Saved final files: stem_outputs_{file_count}.pt")

    return stem_outputs, branch_losses 

# Step 3: Train the Gating Module
def train_gating_module(stem_outputs, branch_losses, config):
    gating_model = DeepGatingModule(input_channels=64, output_shape=7, dropout=0.5).to(config.device)
    optimizer = torch.optim.Adam(gating_model.parameters(), lr=5e-5)
    criterion = torch.nn.L1Loss()

    downsampler = torch.nn.AdaptiveAvgPool2d((5, 5))
    # stem_tensors = []
    # loss_tensors = []

    num_samples = len(stem_outputs)
    combined_shape = (4, 64, 5, 5)  # Shape of each combined tensor

    stem_tensors = torch.empty((num_samples, *combined_shape))

    num_losses = 7 # len(branch_losses[0].values())
    loss_tensors = torch.empty((num_samples, num_losses))

    for i, (outputs, losses) in enumerate(zip(stem_outputs, branch_losses)):
        radar = downsampler(outputs["radar"])
        lidar = downsampler(outputs["lidar"])
        camera_left = downsampler(outputs["camera_left"])
        camera_right = downsampler(outputs["camera_right"])

        combined = torch.cat([radar, lidar, camera_left, camera_right], dim=0)  # Shape: [4, 64, 5, 5]

        stem_tensors[i] = combined
        loss_tensors[i] = torch.tensor([v for v in losses.values()])

    # for outputs, losses in zip(stem_outputs, branch_losses):
    #     radar = downsampler(outputs["radar"])
    #     lidar = downsampler(outputs["lidar"])
    #     camera_left = downsampler(outputs["camera_left"])
    #     camera_right = downsampler(outputs["camera_right"])

    #     # combined = torch.cat([radar, lidar, camera_left, camera_right], dim=0)  # Shape: [4, 64, 5, 5]
    #     combined = torch.stack([radar, lidar, camera_left, camera_right], dim=0)

    #     stem_tensors.append(combined)
    #     loss_tensors.append(torch.tensor([v for v in losses.values()], device=config.device))

    # loss_tensors = torch.stack(loss_tensors)

    stem_tensors_gpu = stem_tensors.to('cuda')
    loss_tensors_gpu = loss_tensors.to('cuda')

    dataset = list(zip(stem_tensors_gpu, loss_tensors_gpu))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    gating_model.train()
    for epoch in range(5):
        total_loss = 0
        for batch_stems, batch_losses in loader:
            # batch_stems = batch_stems.to(config.device)
            # batch_losses = batch_losses.to(config.device)
            # single_loss_sample = batch_losses[0].to(config.device)
            # print(len(batch_losses))

            # for i in range(4):  # batch_stems.size(1) = 4
            #     sensor_tensor = batch_stems[:, i, :, :, :]
            #     predictions = gating_model(sensor_tensor)
            #     print(predictions)
            #     return

            # loss = 0
            # for batch_loss in batch_losses:
            #     loss += criterion(predictions, batch_loss)
            # loss /=32


            sensor_tensors = []

            for i in range(4):  # batch_stems.size(1) = 4
                sensor_tensor = batch_stems[:, i, :, :, :]
                sensor_tensors.append(sensor_tensor)

            predictions_list = []

             # Forward pass for each sensor
            for sensor_tensor in sensor_tensors:
                predictions = gating_model(sensor_tensor)
                predictions_list.append(predictions)

            combined_predictions = torch.stack(predictions_list).mean(dim=0)
            aggregated_batch_losses = batch_losses.mean(dim=0)

            loss = criterion(combined_predictions, aggregated_batch_losses)
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
    # hydrafusion_model = load_hydrafusion_model(config, "model_weights.pth")

    # hydrafusion_model = load_hydrafusion_model(config, "model_weights.pth")

    # # Extract stem outputs and branch losses
    # print("Extracting stem outputs and branch losses...")
    # stem_outputs, branch_losses = extract_stem_outputs_and_losses(hydrafusion_model, train_loader, config)

    data1 = torch.load('stem_branch_data0.pt', map_location='cpu')
    stem_outputs1 = data1['stem_outputs']
    branch_losses1 = data1['branch_losses']

    data2 = torch.load('stem_branch_data1.pt', map_location='cpu')
    stem_outputs2 = data2['stem_outputs']
    branch_losses2 = data2['branch_losses']

    stem_outputs = stem_outputs1 + stem_outputs2
    branch_losses = branch_losses1 + branch_losses2

    # Train the Gating Module
    print("Training the Gating Module...")
    train_gating_module(stem_outputs, branch_losses,config)

if __name__ == "__main__":
    main()