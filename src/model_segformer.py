from dataclasses import dataclass
from pathlib import Path

import cv2
import lightning as L
import numpy as np
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
from torchmetrics.wrappers import ClasswiseWrapper
from transformers.models.segformer import SegformerConfig, SegformerForSemanticSegmentation
from transformers.modeling_outputs import SemanticSegmenterOutput


@dataclass
class StepOutput:
    inputs: torch.Tensor
    targets: torch.Tensor
    outputs: torch.Tensor
    loss: torch.Tensor


class MulticlassSegFormer(L.LightningModule):

    def __init__(
        self,
        config: SegformerConfig,
        weights: str | None,
        num_classes: int,
        labels: list[str],
        criterion: torch.nn.Module,
        learning_rate: float,
        weight_decay: float = 0.,
        prediction_dir_path: Path = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.weights = weights
        self.num_classes = num_classes
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.prediction_dir_path = prediction_dir_path

        if self.weights is None:
            self._model = SegformerForSemanticSegmentation(config=self.config)
        else:
            self._model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_model_name_or_path=self.weights,
                config=self.config,
                ignore_mismatched_sizes=True,
            )

        metrics = MetricCollection(
            {
                'jaccard_index': ClasswiseWrapper(MulticlassJaccardIndex(
                    num_classes=self.num_classes,
                    average=None,
                ), labels),
                'jaccard_index_all': MulticlassJaccardIndex(
                    num_classes=self.num_classes,
                    average='macro',
                ),
                'accuracy': ClasswiseWrapper(MulticlassAccuracy(
                    num_classes=self.num_classes,
                    average=None,
                ), labels),
                'accuracy_all': MulticlassAccuracy(
                    num_classes=self.num_classes,
                    average='macro',
                ),
            }
        )

        self._train_metrics = metrics.clone(prefix='train/')
        self._val_metrics = metrics.clone(prefix='val/')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> SemanticSegmenterOutput:
        return self._model(inputs)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        step_output = self._step(batch)
        self.log(
            name='train_loss',
            value=step_output.loss,
            sync_dist=True,
            prog_bar=True,
        )

        predictions = step_output.outputs.argmax(dim=1)
        targets = step_output.targets.argmax(dim=1)

        self._train_metrics(predictions, targets)

        metrics_dict = self._train_metrics.compute()
        for metric_name, metric_value in metrics_dict.items():
            self.log(metric_name, metric_value)

        return step_output.loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        step_output = self._step(batch)
        self.log(
            name='val_loss',
            value=step_output.loss,
            sync_dist=True,
            prog_bar=True,
        )

        predictions = step_output.outputs.argmax(dim=1)
        targets = step_output.targets.argmax(dim=1)

        self._val_metrics(predictions, targets)

        metrics_dict = self._val_metrics.compute()
        for metric_name, metric_value in metrics_dict.items():
            self.log(metric_name, metric_value)

    def predict_step(
        self,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        inputs = batch
        outputs = self.forward(inputs).logits
        outputs = self._postprocess(outputs)
        return outputs.argmax(dim=1)

    def on_predict_batch_end(
        self,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        output_np = outputs.detach().cpu().numpy()

        reverse_class_map = {
            0: (255, 0, 0),  # clutter/ background
            1: (255, 255, 255),  # impervious surfaces
            2: (0, 0, 255),  # building
            3: (0, 255, 255),  # low vegetation
            4: (0, 255, 0),  # tree
            5: (255, 255, 0),  # car
        }

        for i, output in enumerate(output_np):
            output_rgb = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)

            for label, rgb in reverse_class_map.items():
                output_rgb[output == label] = rgb

            output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
            save_path = self.prediction_dir_path / f'image_{batch_idx}_{i}.png'
            cv2.imwrite(str(save_path), output_bgr)

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> StepOutput:
        inputs, targets = batch

        outputs = self.forward(inputs).logits
        outputs = self._postprocess(outputs)

        loss = self.criterion(outputs, targets)

        return StepOutput(
            inputs=inputs,
            targets=targets,
            outputs=outputs,
            loss=loss,
        )

    @staticmethod
    def _postprocess(logits: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            logits,
            scale_factor=4,
            mode='bilinear',
        )
