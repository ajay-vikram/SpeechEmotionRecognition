import os
import argparse
import json
import numpy as np
from typing import Union, List
import matplotlib.pyplot as plt


__all__ = ['TRAIN_FLAGS', 'EVAL_FLAGS', 'AverageMeter', 'ConfusionMatrix']


def TRAIN_FLAGS() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Speech Emotion Recognition")

    # Data config
    parser.add_argument(
        '--classes',
        default=['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise'],
        type=list,
        help='List of class names with labels 0 to 7'
    )

    parser.add_argument(
        '--trainpt_file',
        default='train_data.pt',
        type=str,
        help='Path of the training tensor file'
    )

    parser.add_argument(
        '--testpt_file',
        default='test_data.pt',
        type=str,
        help='Path of the testing tensor file'
    )

    # Model Config
    parser.add_argument(
        '--model',
        default='lmu',
        type=str,
        help='Model to train/val - lmu, cnn, lstm, cnn_lstm'
    )

    parser.add_argument(
        '--lmu_config',
        default=[1, 8, 128, 1024, 2376, 0.25, False, False, True],
        type=list,
        help='[input_size, output_size, hidden_size, memory_size, seq_len, dropout, learn_a, learn_b, FFT]'
    )

    

    # GPU Config
    parser.add_argument(
        '--gpu_index',
        default=0,
        type=int,
        help='GPU device to be used'
    )

    # Project
    parser.add_argument(
        '--seed',
        default=123,
        type=int,
        help='IMPORTANT:- Keep the seed same for all the stages'
    )
    parser.add_argument(
        '--proj',
        default='speech_emotion_recognition',
        type=str,
        help='Project name'
    )

    # Optimization
    parser.add_argument(
        '--train_lr',
        default=1e-3,
        type=float,
        help='Learning rate for the original F32 model'
    )
    parser.add_argument(
        '--train_alpha',
        default=0,
        type=float,
        help='Weight decay for training the original F32 model'
    )
    parser.add_argument(
        '--max_lr',
        default=0.01,
        type=float,
        help='Maximum learning rate for OneCycleLR'
    )

    # Training
    parser.add_argument(
        '--train_epochs',
        default=50,
        type=int,
        help='#epochs for training the original F32 model'
    )
    parser.add_argument(
        '--batch_size',
        default=64,
        type=int
    )
    parser.add_argument(
        '--workers',
        default=8,
        type=int,
        help='#processes for the dataloading'
    )
    parser.add_argument(
        '--eval_steps',
        default=10,
        type=int,
        help='#epoch steps during training for evaluating the model and determining \
        to modify the learning rate'
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        default=False,
        help='Whether to train a pretrained model'
    )

    # Stages
    parser.add_argument(
        '--train',
        action='store_true',
        default=False,
        help='Train the original F32 model'
    )

    args = parser.parse_args()
    args = set_sub_folder(args)
    os.makedirs(args.train_dir, exist_ok=True) if args.train else None
    with open(os.path.join(args.meta_dir, 'train_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    return args


def set_sub_folder(args: argparse.Namespace) -> argparse.Namespace:
    args.meta_dir = "/home/satyapreets/Ajay/Radar_HAR_multiple/meta"
    args.train_dir = os.path.join(args.meta_dir, 'lmu')
    return args


def EVAL_FLAGS() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Person/No person classification on VWW RGB dataset using RAMAN architecture inference")
    
    parser.add_argument(
        '--magnitude_prune', 
        action='store_true',
        default=False,
        help='Whether to load stochastic pruned or magnitude pruned model'
    )
    parser.add_argument(
        '--sparsity',
        default=0.,
        type=float,
        help='Sparsity level of quantized model'
    )
    parser.add_argument(
        '--q_type',
        default="int8",
        choices=['int8', 'ternary', 'mix', 'mix2', 'hybrid'],
        help='Type of quantized model'
    )
    parser.add_argument(
        '--ter_conv1',
        action="store_true",
        default=False,
        help='If true the conv1 layer of the mixed precision model has ternary weights, else 8-bit weights'
    )
    parser.add_argument(
        '--weight_bit',
        type=int,
        default=8,
        help='The bit precision of weights in a mixed precision model for the layers not ternarized'
    )
    
    args = parser.parse_args()

    args.stochastic_prune = not args.magnitude_prune

    return args


class AverageMeter:

    def __init__(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, 
               val: float, 
               n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


class ConfusionMatrix:
    """
    A class to create confusion matrix and display it or calculate precision, recall, F1 score.
    """

    def __init__(self, 
                 true_values: np.ndarray, 
                 predicted_values: np.ndarray, 
                 classes: List = None, 
                 from_one_hot: bool = False):
        """
        Creates an object of ConfusionMatrix.

        Parameters:
        -----------
        true_values: ndarray
            The target values.
        predicted_values: ndarray
            The predicted outcomes.
        classes: List, optional, default: None
            The List of labels corresponding to the encoded values.
        from_one_hot: bool, optional, default: False
            Whether `true_values` and `predicted_values` are one-hot encoded (True) or not (False).
        """
        self.matrix = self._get_matrix(true_values, predicted_values, from_one_hot)
        self.classes = classes if classes else [str(i) for i in range(self.matrix.shape[0])]
        self.weights = np.sum(self.matrix, axis=1)

    def __str__(self) -> str:
        return f"{self.matrix}"

    def plot(self, 
             block: bool = True, 
             colorbar: bool = True,
             show_percentage: bool = False,
             title: str = None,
             save_file: List[str] = None,
             fontsize: int = 14) -> None:
        """
        Plots the confusion matrix.

        Parameters:
        -----------
        block: bool, optional, default: True
            Whether the block the execution of the following codes (True) or not (False).
        colorbar: bool, optional, default: True
            Whether to show the colorbar (True) or not (False).
        show_percentage: bool, optional, default: False
            If True, the values shown are the percentages wrt the total number of predictions in the particular row.
        title: str, optional, default: None
            Title of the plot.
        save_file: List[str], optional, default: None
            Paths to save the plot.
        font_size: int, optional, default: 10
            The fontsize of texts
        """
        mat_to_show = self.matrix if not show_percentage \
            else self.matrix * 100 / self.matrix.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots()
        im = ax.imshow(mat_to_show)
        ax.figure.colorbar(im, ax=ax) if colorbar else None

        ax.set(xticks=np.arange(self.matrix.shape[1]), yticks=np.arange(self.matrix.shape[0]),
               xticklabels=self.classes, yticklabels=self.classes)
        ax.set_title("Confusion Matrix" if title is None else title, weight="bold", fontsize=fontsize + 2)
        ax.set_xlabel("Predicted label", weight="bold", fontsize=fontsize)
        ax.set_ylabel("True label", weight="bold", fontsize=fontsize)

        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor", fontsize=fontsize)
        plt.setp(ax.get_yticklabels(), fontsize=fontsize)

        for i in range(mat_to_show.shape[0]):
            for j in range(mat_to_show.shape[1]):
                if mat_to_show[i, j] == 0:
                    continue
                color = "w" if mat_to_show[i, j] < (np.max(mat_to_show) + np.min(mat_to_show)) / 2 else "k"
                ax.text(j, i, f"{mat_to_show[i, j]}" if not show_percentage else f"{mat_to_show[i, j]:.1f}%", 
                        ha="center", va="center", color=color, fontsize=fontsize, weight="bold")

        fig.tight_layout()
        [plt.savefig(path) for path in save_file]
        plt.show(block=block)

    def precision(self, 
                  average: Union[str, None] = "micro") -> Union[np.ndarray, float]:
        """
        Calculates precision.

        Parameters:
        -----------
        average: None or str, optional, default: 'micro'
            Type of averaging.

            None: Computes the precisions of individual classes.

            'micro': Computes the global average by counting the sums of the True Positives (TP), False Positives (FP).

            'macro': Computes the arithmatic mean of the per-class precisions treating all classes equally.

            'weighted': Computes the mean of all per-class precisions
            considering the number of actual occurrences of each class.
        
        Returns:
        --------
        precisions: ndarray or float
            The precision.
        """

        N = self.matrix.shape[0]
        precisions = np.zeros(N)
        for i in range(N):
            precisions[i] = self.matrix[i, i] / np.sum(self.matrix[:, i]) if np.sum(self.matrix[:, i]) != 0 else 0

        if not average or average is None:
            return precisions

        if average == "micro":
            return np.sum(self.matrix.diagonal()) / np.sum(self.matrix) if np.sum(self.matrix) != 0 else 0

        if average == "macro":
            return np.mean(precisions)

        if average == "weighted":
            return np.sum(self.weights * precisions) / np.sum(self.weights) if np.sum(self.weights) != 0 else 0

        raise Exception(f"Invalid type '{average}' for parameter 'average'. "
                        f"Possible types are: None, 'micro', 'macro', 'weighted'")

    def recall(self, 
               average: Union[str, None] = "micro") -> Union[np.ndarray, float]:
        """
        Calculates recall.

        Parameters:
        -----------
        average: None or str, optional, default: 'micro'
            Type of averaging.

            None: Computes the recalls of individual classes.

            'micro': Computes the global average by counting the sums of the True Positives (TP), False Negatives (FN).

            'macro': Computes the arithmatic mean of the per-class recalls treating all classes equally.

            'weighted': Computes the mean of all per-class recalls
            considering the number of actual occurrences of each class.

        Returns:
        --------
        recalls: ndarray or float
            The recall.
        """

        N = self.matrix.shape[0]
        recalls = np.zeros(N)
        for i in range(N):
            recalls[i] = self.matrix[i, i] / np.sum(self.matrix[i, :]) if np.sum(self.matrix[i, :]) != 0 else 0

        if not average:
            return recalls

        if average == "micro":
            return np.sum(self.matrix.diagonal()) / np.sum(self.matrix) if np.sum(self.matrix) != 0 else 0

        if average == "macro":
            return np.mean(recalls)

        if average == "weighted":
            return np.sum(self.weights * recalls) / np.sum(self.weights) if np.sum(self.weights) != 0 else 0

        raise Exception(f"Invalid type '{average}' for parameter 'average'. "
                        f"Possible types are: None, 'micro', 'macro', 'weighted'")

    def f1_score(self,
                 average: Union[str, None] = "micro") -> Union[np.ndarray, float]:
        """
        Calculates F1 score.

        Parameters:
        -----------
        average: None or str, optional, default: 'micro'
            Type of averaging.

            None: Computes the F1 scores of individual classes.

            'micro': Computes the global average by counting the sums of the
            True Positives (TP), False Positives (FP), False Negatives (FN).

            'macro': Computes the arithmatic mean of the per-class F1 scores treating all classes equally.

            'weighted': Computes the mean of all per-class F1 scores
            considering the number of actual occurrences of each class.

        Returns:
        --------
        precisions: ndarray or float
            The F1 score.
        """

        precisions = self.precision(average=None)
        recalls = self.recall(average=None)
        f1_scores = np.zeros_like(precisions)
        mask = (precisions != 0) * (recalls != 0)
        f1_scores[mask] = 2 * precisions[mask] * recalls[mask] / (precisions[mask] + recalls[mask])

        if not average:
            return f1_scores

        if average == "micro":
            return np.sum(self.matrix.diagonal()) / np.sum(self.matrix) if np.sum(self.matrix) != 0 else 0

        if average == "macro":
            return np.mean(f1_scores)

        if average == "weighted":
            return np.sum(self.weights * f1_scores) / np.sum(self.weights) if np.sum(self.weights) != 0 else 0

        raise Exception(f"Invalid type '{average}' for parameter 'average'. "
                        f"Possible types are: None, 'micro', 'macro', 'weighted'")

    @staticmethod
    def _get_matrix(true_values: np.ndarray, 
                    predicted_values: np.ndarray, 
                    from_one_hot: bool) -> np.ndarray:
        """
        Creates the confusion matrix.

        Parameters:
        -----------
        true_values: ndarray
            The target values.
        predicted_values: ndarray
            The predicted outcomes.
        from_one_hot: bool
            Whether true_values and predicted_values are one-hot encoded or not.

        Returns:
        --------
        matrix: ndarray
            The confusion matrix
        """

        if true_values.shape != predicted_values.shape:
            raise Exception("Different shapes of 'true_values' and 'predicted_values'")

        if from_one_hot:
            dim = true_values.shape[0]
            matrix = np.zeros((dim, dim))
            for j in range(true_values.shape[1]):
                matrix[np.argmax(true_values[:, j]), np.argmax(predicted_values[:, j])] += 1
            return matrix

        dim = int(np.max(np.concatenate((true_values, predicted_values)))) + 1
        matrix = np.zeros((dim, dim), dtype=int)
        for j in range(true_values.shape[0]):
            matrix[true_values[j], predicted_values[j]] += 1
        return matrix
