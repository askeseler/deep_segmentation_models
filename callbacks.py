from matplotlib.pyplot import draw
import torch
import pytorch_lightning as pl
from torchvision.utils import make_grid, draw_segmentation_masks
from torchvision.transforms.functional import convert_image_dtype

class VisualizationCallback(pl.Callback):
    """Callback to compare predicted with groundtruth segmentations for validation samples.
    """

    def __init__(self):
        super(VisualizationCallback, self).__init__()
        self.images = []
        self.preds = []
        self.targets = []

    def on_validation_start(self, trainer, pl_module):
        self.images = []
        self.preds = []
        self.targets = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,  batch_idx, dataloader_idx):
        if len(self.images) < 10:
            x, target, pred = outputs
            if pred.ndim == 4:
                pred = pred.argmax(1)
            self.images += [*torch.sigmoid(x).unbind()]
            self.preds += [*pred.unbind()]
            self.targets += [*target.unbind()]
    
    def on_validation_end(self, trainer, pl_module):
        images = [convert_image_dtype(image, torch.uint8) for image in self.images]
        preds = [self.one_hot(pred, num_classes=pl_module.n_classes) for pred in self.preds]

        targets = self.targets#[self.one_hot(target, num_classes=pl_module.hparams.num_classes) for target in self.targets]

        colors = ["black", "yellow", "darkgreen", "blue", "brown", "cyan", "yellow", "red", "orange"]
        #pl_module.logger.experiment.add_image("Avg. Mask", make_grid([255*torch.sum(p, axis=0).unsqueeze(0)/pl_module.n_classes for p in preds], nrow=5), trainer.global_step)
        preds = [draw_segmentation_masks(image.cpu().detach()[:3], masks.cpu().detach().type(torch.bool), alpha=0.5, colors=colors) for image, masks in zip(images, preds)]
        targets = [draw_segmentation_masks(image.cpu().detach()[:3], masks.cpu().detach().type(torch.bool), alpha=0.5, colors=colors) for image, masks in zip(images, targets)]
        images = make_grid(images, nrow=5)
        preds = make_grid(preds, nrow=5)
        targets = make_grid(targets, nrow=5)
        pl_module.logger.experiment.add_image("Images", images, trainer.global_step)
        pl_module.logger.experiment.add_image("Masked Images", targets, trainer.global_step)
        pl_module.logger.experiment.add_image("Predictions", preds, trainer.global_step)

    def on_test_start(self, trainer, pl_module):
        self.test_images = []
        self.test_preds = []
        self.test_targets = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch,  batch_idx, dataloader_idx):
        if len(self.images) < 10:
            x, target, pred = outputs
            self.test_images += [*torch.sigmoid(x).unbind()]
            self.test_preds += [*pred.unbind()]
            self.test_targets += [*target.unbind()]
    
    def on_test_end(self, trainer, pl_module):
        images = [convert_image_dtype(image, torch.uint8) for image in self.test_images]
        preds = [self.one_hot(pred, num_classes=pl_module.n_classes) for pred in self.test_preds]
        targets = [self.one_hot(target, num_classes=pl_module.n_classes) for target in self.test_targets] 
        colors = ["black", "yellow", "darkgreen", "blue", "brown", "cyan", "yellow", "red", "orange"]
        preds = [draw_segmentation_masks(image.cpu().detach()[:3], masks.cpu().detach(), alpha=0.5, colors=colors) for image, masks in zip(images, preds)]
        targets = [draw_segmentation_masks(image.cpu().detach()[:3], masks.cpu().detach(), alpha=0.5, colors=colors) for image, masks in zip(images, targets)]
        images = make_grid(images, nrow=5)
        preds = make_grid(preds, nrow=5)
        targets = make_grid(targets, nrow=5)
        pl_module.logger.experiment.add_image("Test Images", images, trainer.global_step)
        pl_module.logger.experiment.add_image("Masked Test Images", targets, trainer.global_step)
        pl_module.logger.experiment.add_image("Test Predictions", preds, trainer.global_step)

    def one_hot(self, pred, num_classes=10):
        masks = [pred == i for i in range(num_classes)]
        masks = torch.stack(masks)
        return masks