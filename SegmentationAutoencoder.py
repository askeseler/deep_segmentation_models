import torch
import pytorch_lightning as pl
from torchvision.models.segmentation import fcn_resnet50, deeplabv3_mobilenet_v3_large
def precision(pred, target):
    #pred = pred.argmax(1)
    tp = torch.logical_and(pred, target).sum()
    fp = torch.logical_and(pred, torch.logical_not(target)).sum()
    return tp/(tp+fp)

def recall(pred, target):
    #pred = pred.argmax(1)
    tp = torch.logical_and(pred, target).sum()
    fn = torch.logical_and(torch.logical_not(pred), target).sum()
    return tp/(tp+fn)

def accuracy(pred, target):
    assert str(pred.shape) == str(target.shape)
    #pred = pred.argmax(1)
    accuracy = (pred == target).sum() / torch.prod(torch.tensor(target.shape))
    return accuracy

class SegmentationNet(pl.LightningModule):
    """Lightning module for segmentation tasks.
    The module prepares segmentation models from torchvision. The backbone is pretrained on
    the ImageNet classification task by default. If the option 'weights' is used with 'pretrained'
    then a decoder pretrained on a subset of the COCO train2017 dataset with 20 (+background) classes.
    In this case only the last classification submodule is trained while the weights of all other
    layers are freezed.

    Args:
        model_name: A string specifying the name of the model (existing implementations for: fcn_resnet50
            and deeplabv3_mobilenet_v3_large).
        num_classes: An integer counting the number of classes (including background) to segment.
        weights: A string either specifying a path to a saved checkpoint or 'pretrained' indicating to use the
            pretrained model as explained above.
        pos_weight: A float specifying the weight for the foreground class in a foreground-background segmentation.
            If more than 2 classes are segmented all classes get unit weight for now.
        lr: A float specifying the learning rate to be used in the optimizer.
    """

    def __init__(self, model_name="fcn_resnet50", num_classes=2, num_input_channel = 3, weights=None, pos_weight=1, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        # check for model name
        if model_name == "fcn_resnet50":
            model_loader = fcn_resnet50
        elif model_name == "deeplabv3_mobilenet_v3_large":
            model_loader = deeplabv3_mobilenet_v3_large
        # load model
        if weights == "pretrained":
            self.model = model_loader(pretrained=True)
            if model_name == "fcn_resnet50":
                self.model.aux_classifier[-1] = torch.nn.Conv2d(256, num_classes, kernel_size=1, device=self.device)
            elif model_name == "deeplabv3_mobilenet_v3_large":
                self.model.classifier[-1] = torch.nn.Conv2d(256, num_classes, kernel_size=1, device=self.device)
        elif weights is not None:
            self.model = model_loader(num_classes=num_classes, pretrained_backbone=False)
            self.model.load_state_dict(torch.load(weights, map_location=self.device))
        else:
            self.model = model_loader(num_classes=num_classes)
        # define loss function and metrics
        if num_classes == 2:
            weight = torch.tensor([1, pos_weight], device=self.device)
        else:
            weight = None
        if num_input_channel != 3:
            if model_name == "fcn_resnet50":
                self.model.backbone.conv1 = torch.nn.Conv2d(num_input_channel, 64, kernel_size=7, stride=2, padding=3,bias=False)
            else:
                print(self.model.backbone.__dict__)
                sys.exit()
                self.model.backbone.conv1 = torch.nn.Conv2d(num_input_channel, 16, kernel_size=7, stride=2, padding=3,bias=False)

        self.n_classes = num_classes
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        self.metric_fns = {"accuracy": accuracy, "precision": precision, "recall": recall} 

    def forward(self, x):
        y = self.model(x)['out']
        return y

    def training_step(self, batch, batch_idx):
        x, target = batch[0], batch[1]
        y = self.model(x)['out']
        loss = self.loss_fn(y, target)
        self.log("loss/train", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch[0], batch[1]
        y = self.model(x)["out"]
        loss = self.loss_fn(y, target)
        metrics = {metric: metric_fn(y, target) for metric, metric_fn in self.metric_fns.items()}
        self.log("loss/val", loss, on_step=False, on_epoch=True)
        self.log_dict(metrics, on_step=False, on_epoch=True)
        pred = y.argmax(1)
        return x, target, pred

    def test_step(self, batch, batch_idx):
        x, target = batch[0], batch[1]
        y = self.model(x)["out"]
        loss = self.loss_fn(y, target)
        metrics = {metric: metric_fn(y, target) for metric, metric_fn in self.metric_fns.items()}
        self.log("loss/test", loss, on_step=False, on_epoch=True)
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss, metrics

    def configure_optimizers(self):
        if self.hparams.weights == "pretrained":
            self.freeze_parameters(without="classifier")
        params = []
        for param in self.model.parameters():
            if param.requires_grad:
                params += [param]
        optimizer = torch.optim.Adam(params, lr=self.hparams.lr)
        return optimizer

    def freeze_parameters(self, without="classifier"):
        for param in self.model.parameters():
            param.requires_grad = False
        if without == "classifier":
            if self.hparams.model_name == "fcn_resnet50":
                self.model.aux_classifier.requires_grad_(True)
            elif self.hparams.model_name == "deeplabv3_mobilenet_v3_large":
                self.model.classifier.requires_grad_(True)