import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset, random_split
from train import Trainer
from utils import TRAIN_FLAGS
import sklearn
from sklearn.model_selection import train_test_split

def get_device():
    print('-'*50)
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_id = torch.cuda.get_device_name(device_id)
        print(f"Device ID: {device_id}, GPU ID: {gpu_id}")
        print('-'*50)
    else:
        print(f"Device: CPU")
        print('-'*50)

class Controller:
    def __init__(self):
        self.args = TRAIN_FLAGS()
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        torch.cuda.set_device(self.args.gpu_index)
        generator = torch.Generator()
        generator.manual_seed(42)
        get_device()

        # Loading the data
        x_train_tensor, y_train_tensor = torch.load(self.args.trainpt_file)
        x_test_tensor, y_test_tensor = torch.load(self.args.testpt_file)

        # Create TensorDatasets and DataLoaders
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=False)
        self.val_dataloader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)


    def train_model(self) -> None:
        trainer = Trainer(self.args, len(self.train_dataloader))
        trainer.set_dataloaders(self.train_dataloader, self.val_dataloader,
                                self.val_dataloader)
        trainer.run()
        return None

if __name__ == "__main__":
    controller = Controller()
    if controller.args.train:
        controller.train_model()
