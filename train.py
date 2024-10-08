import argparse
from models.CNN2 import CNN2
from models.LSTM import LSTM
from models.CNN_LSTM import CNN_LSTM
from models.LMU import LMUModel
from run import Runner
import torch
from torchsummary import summary
from sklearn.metrics import confusion_matrix, classification_report

__all__ = ['Trainer']

def countParameters(model):
    """ Counts and prints the number of trainable and non-trainable parameters of a model """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable parameters: {trainable:,}\nFrozen parameters: {frozen:,}")

class Trainer(Runner):
    def __init__(self, 
                 args: argparse.Namespace, 
                 train_dataloader_length: int) -> None:
        super().__init__(args)
        self.meta_dir = args.train_dir
        self.epochs = args.train_epochs
        if args.model == "lmu":
            self.model = LMUModel(
                input_size = args.lmu_config[0], 
                output_size = args.lmu_config[1], 
                hidden_size = args.lmu_config[2], 
                memory_size = args.lmu_config[3], 
                seq_len = args.lmu_config[4], 
                theta = args.lmu_config[4],
                dropout = args.lmu_config[5], 
                learn_a = args.lmu_config[6], 
                learn_b = args.lmu_config[7], 
                FFT = args.lmu_config[8]
            ).to(torch.cuda.current_device(), non_blocking=True)
        elif args.model == "cnn":
            self.model = CNN2().to(torch.cuda.current_device(), non_blocking=True)
        elif args.model == "lstm":
            self.model = LSTM().to(torch.cuda.current_device(), non_blocking=True)
        elif args.model == "cnn_lstm":
            self.model = CNN_LSTM().to(torch.cuda.current_device(), non_blocking=True)
        else:
            print("Invalid model mode!")
            exit(0)


        countParameters(self.model)
        self.optimizer = self.get_optimizer(self.model, args.train_lr, args.train_alpha)
        # self.scheduler = self.get_scheduler(self.optimizer, train_dataloader_length, args.train_epochs)
    
    def save_checkpoint(self, 
                        fname: str = None) -> None:
        checkpoint = {'epoch': self.cur_epoch, 
                    'optimizer': self.optimizer,
                    'state_dict': self.model.state_dict(),
                    'best_loss': self.best_loss}
        super()._save_checkpoint(checkpoint, fname)
        return None

    def get_metrics(self, all_y_true, all_y_pred):
        print('-'*50)
        y_true_np = all_y_true.cpu().numpy() if isinstance(all_y_true, torch.Tensor) else all_y_true
        y_pred_np = all_y_pred.cpu().numpy() if isinstance(all_y_pred, torch.Tensor) else all_y_pred

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_true_np, y_pred_np)
        print("Confusion Matrix:\n", conf_matrix)
        print('-'*50)

        # Classification Report
        class_report = classification_report(y_true_np, y_pred_np)
        print("\nClassification Report:\n", class_report)
        print('-'*50)


    def run(self) -> None:
        self.run_init()
        self.reset_best_loss()
        while self.cur_epoch < self.epochs:
            super().run_epoch()
            self.cur_epoch += 1
        all_y_true, all_y_pred = self.evaluate()
        self.get_metrics(all_y_true, all_y_pred)
        # self.save_training_loss()
        # torch.save(self.model, "/home/satyapreets/Ajay/neurobench/mobilenet_training/experiments/vww/train/model.pth")
        return None
            