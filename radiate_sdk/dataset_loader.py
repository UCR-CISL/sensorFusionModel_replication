import torch
import os
import numpy as np
import radiate
from torchvision.transforms import ToTensor, Resize

class RadiateDataset:
    def __init__(self, dataset_path, sequence_name, transform=None, input_size=(224, 224)):
        """
        Initialize the dataset loader with preprocessing for HydraFusion.
        :param dataset_path: Path to the RADIATE dataset.
        :param sequence_name: Specific driving sequence to load.
        :param transform: Optional transform for preprocessing the data.
        :param input_size: Expected input size for model stems.
        """
        self.dataset_path = dataset_path
        self.sequence_name = sequence_name
        self.transform = transform or ToTensor()  # Default to converting to tensors
        self.input_size = input_size

        # Load sequence
        sequence_path = os.path.join(self.dataset_path, self.sequence_name)
        self.sequence = radiate.Sequence(sequence_path)

        # Extract timestamps
        self.timestamps = np.arange(self.sequence.init_timestamp, self.sequence.end_timestamp, 1)

        # Initialize transformations
        self.resize = Resize(self.input_size)
    
    def __len__(self):
        """
        Return the total number of frames in the sequence.
        """
        return len(self.timestamps)

    def preprocess_radar(self, radar_data):
        """
        Preprocess radar data to match input requirements.
        :param radar_data: Radar image in Cartesian format.
        :return: Preprocessed radar image as a tensor.
        """
        radar_tensor = ToTensor()(radar_data)
        return self.resize(radar_tensor)

    def preprocess_lidar(self, lidar_data):
        """
        Preprocess LiDAR BEV data.
        :param lidar_data: LiDAR BEV image.
        :return: Preprocessed LiDAR image as a tensor.
        """
        lidar_tensor = ToTensor()(lidar_data)
        return self.resize(lidar_tensor)


    def preprocess_camera(self, camera_data):
        """
        Preprocess camera images to match input requirements.
        :param camera_data: Raw camera image.
        :return: Preprocessed camera image as a tensor.
        """
        camera_tensor = ToTensor()(camera_data)
        return self.resize(camera_tensor)
    
    def get_sensor_data(self, data, key, preprocess_func):
        """
        Safely fetch and preprocess sensor data.
        :param data: Dictionary of sensor data.
        :param key: Key to access the desired sensor data.
        :param preprocess_func: Function to preprocess the data.
        :return: Preprocessed sensor data or a zero tensor if key is missing or invalid.
        """
        if key in data and data[key] is not None:
            try:
                return preprocess_func(data[key])
            except Exception as e:
                print(f"Error processing sensor data for key '{key}': {e}")
        else:
            print(f"Missing sensor data for key '{key}'")
        # Return a zero tensor with the expected shape
        return torch.zeros((1, *self.input_size), dtype=torch.float32)

    def __getitem__(self, idx):
        """
        Fetch a single frame of synchronized sensor data.
        :param idx: Frame index.
        :return: Dictionary containing preprocessed sensor data and annotations.
        """
        timestamp = self.timestamps[idx]
        try:
            data = self.sequence.get_from_timestamp(timestamp)
        except Exception as e:
            print(f"Error retrieving data for timestamp {timestamp}: {e}. Skipping this frame.")
            return self.__getitem__((idx + 1) % len(self.timestamps))  # Retry with the next frame

        # Check if 'sensors' key exists
        if 'sensors' not in data:
            print(f"No sensor data available at timestamp {timestamp}. Skipping this frame.")
            return self.__getitem__((idx + 1) % len(self.timestamps))  # Skip to the next frame

        # Preprocess sensor data
        radar = self.get_sensor_data(data['sensors'], 'radar_cartesian', self.preprocess_radar)
        lidar = self.get_sensor_data(data['sensors'], 'lidar_bev_image', self.preprocess_lidar)
        left_camera = self.get_sensor_data(data['sensors'], 'camera_left_rect', self.preprocess_camera)
        right_camera = self.get_sensor_data(data['sensors'], 'camera_right_rect', self.preprocess_camera)

        # Extract and validate annotations
        annotations = data.get('annotations', {})
        if isinstance(annotations, dict):
            bboxes = []
            classes = []

            for modality, objects in annotations.items():
                for obj in objects:
                    # Extract 2D bounding boxes
                    if 'bbox' in obj and isinstance(obj['bbox'], dict):
                        position = obj['bbox'].get('position', [])
                        if len(position) == 4:  # Ensure valid format
                            x_min, y_min, width, height = position
                            bboxes.append([x_min, y_min, x_min + width, y_min + height])
                            classes.append(obj.get('class_name', 'unknown'))
        else:
            print(f"Invalid annotation format at timestamp {timestamp}. Skipping annotations.")

        # Convert bounding boxes to tensor
        bboxes = torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.empty((0, 4), dtype=torch.float32)

        return {
            'radar': radar,
            'lidar': lidar,
            'camera_left': left_camera,
            'camera_right': right_camera,
            'bboxes': bboxes,
            'classes': classes
        }
