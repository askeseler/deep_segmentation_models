import os
from argparse import ArgumentParser

import numpy as np
import torch

from Unet import Unet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from dataset import DirDataset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import callbacks
from callbacks import *
import pandas as pd
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn.functional import one_hot
from PIL import Image


def label_images(images, preds, targets):
    images = [convert_image_dtype(image, torch.uint8) for image in images]
    colors = ["black", "yellow", "darkgreen", "blue",
              "brown", "cyan", "yellow", "red", "orange"]
    # pl_module.logger.experiment.add_image("Avg. Mask", make_grid([255*torch.sum(p, axis=0).unsqueeze(0)/pl_module.n_classes for p in preds], nrow=5), trainer.global_step)
    preds = [draw_segmentation_masks(image.cpu(), masks.cpu().type(
        torch.bool), alpha=0.5, colors=colors) for image, masks in zip(images, preds)]
    targets = [draw_segmentation_masks(image.cpu(), masks.cpu().type(
        torch.bool), alpha=0.5, colors=colors) for image, masks in zip(images, targets)]
    # images = make_grid(images, nrow=5)
    # preds = make_grid(preds, nrow=5)
    # targets = make_grid(targets, nrow=5)
    return images, preds, targets


def main(hparams):
    if hparams.resnet50:
        from UnetResNet50 import UNetWithResnet50Encoder
        import UnetResNet50
        model = UNetWithResnet50Encoder(
            n_classes=hparams.n_classes, num_input_channel=hparams.img_channels)
    elif hparams.resnet18:
        from UnetResNet18 import UNetWithResnet18Encoder
        import UnetResNet18
        model = UNetWithResnet18Encoder(
            n_classes=hparams.n_classes, num_input_channel=hparams.img_channels)
    elif hparams.resnet_autoencoder:
        from SegmentationAutoencoder import SegmentationNet
        import SegmentationAutoencoder
        model = SegmentationNet(
            num_classes=hparams.n_classes, num_input_channel=hparams.img_channels, lr=0.01)
    elif hparams.mobilnet_autoencoder:
        from SegmentationAutoencoder import SegmentationNet
        import SegmentationAutoencoder
        model = SegmentationNet(num_classes=hparams.n_classes, lr=0.01, model_name="deeplabv3_mobilenet_v3_large",
                                num_input_channel=hparams.img_channels)
    elif hparams.unet:
        from Unet import Unet
        model = Unet(n_classes=hparams.n_classes,
                     n_channels=hparams.img_channels)
    else:
        print("Specify a model: --resnet18, --resnet50, --resnet_autoencoder, --unet or --mobilnet_autoencoder")

    os.makedirs(hparams.log_dir, exist_ok=True)

    try:
        log_dir = os.path.join(hparams.log_dir, "lightning_logs",
                               "version_" + str(len(os.listdir(hparams.log_dir))))
    except IndexError:
        log_dir = os.path.join(hparams.log_dir, "lightning_logs", 'version_0')

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, 'checkpoints'),
        every_n_train_steps=1000,)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5,
        verbose=True,)

    callbacks = [VisualizationCallback(), checkpoint_callback]

    logger = TensorBoardLogger(hparams.log_dir)

    trainer = Trainer(
        default_root_dir=hparams.log_dir,
        gpus=1,
        callbacks=callbacks,
        max_epochs=int(hparams.n_epochs),
        log_every_n_steps=10, logger=logger)

    dataset = hparams.dataset

    if hparams.predict:
        # python train.py --dataset /media/gerstenberger/Data1/datasets/tree_segmentation/ --log_dir ./tree_segmentation_autoencoder_1/ --n_epochs 100 --resnet50 0 --resnet18 0 --resnet_autoencoder 1 --pretrained 0 --img_channels 3 --n_classes 4 --predict 1
        val_ds = DirDataset(f'{dataset}/images/test/', f'{dataset}/pixelwise_annotations/test/', scale=hparams.scale,
                            grayscale=hparams.grayscale, n_channels=hparams.img_channels, n_classes=hparams.n_classes)
        val_loader = DataLoader(val_ds, batch_size=1,
                                pin_memory=True, shuffle=False, num_workers=20)
        checkpoint = os.path.join(
            hparams.log_dir, "lightning_logs", "version_0", "checkpoints")
        checkpoint = os.path.join(checkpoint, os.listdir(checkpoint)[0])
        os.makedirs(os.path.join(hparams.log_dir,
                    "predictions"), exist_ok=True)
        model.load_state_dict(torch.load(checkpoint)["state_dict"])
        model.freeze()
        device = "cuda"
        model.to(device)
        from tqdm import tqdm

        tps = np.zeros(hparams.n_classes)  # per class
        fps = np.zeros(hparams.n_classes)
        fns = np.zeros(hparams.n_classes)
        tns = np.zeros(hparams.n_classes)

        global_i = 0  # image
        for batch in tqdm(val_loader):
            x, target = batch[0], batch[1]
            y = torch.sigmoid(model(x.to(device)))  # model(x.to(device))
            y = one_hot(y.argmax(1), hparams.n_classes).permute(
                0, 3, 1, 2)  # assign 1 to max class

            # annotate images
            imgs_out, annotated_preds, annotated_targets = label_images(
                x, y, target)
            for i, p, t in zip(imgs_out, annotated_preds, annotated_targets):
                p = Image.fromarray(
                    np.array(p.permute(1, 2, 0), dtype=np.uint8))
                outdir = os.path.join(hparams.log_dir, "annotated_imgs")
                os.makedirs(outdir, exist_ok=True)
                p.save(os.path.join(outdir, str(global_i)+".jpg"))
                global_i += 1

            # compute evaluation
            y = y.detach().cpu().numpy()
            for i_batch in range(target.shape[0]):
                for i_label in range(target.shape[1]):
                    a = y[i_batch, i_label] > .5
                    b = target[i_batch, i_label].cpu(
                    ).detach().numpy().astype(bool)

                    # import matplotlib.pyplot as plt
                    # fig, ax = plt.subplots(2)
                    # print(a)
                    # ax[0].imshow(a)
                    # ax[1].imshow(b)
                    # plt.show()
                    tp = np.sum(np.logical_and(a, b))
                    fn = np.sum(np.logical_and(~a, b))
                    fp = np.sum(np.logical_and(a, ~b))
                    tn = np.sum(np.logical_and(~a, ~b))
                    tps[i_label] += tp
                    fns[i_label] += fn
                    tns[i_label] += tn
                    fps[i_label] += fp
                # import matplotlib.pyplot as plt
                # plt.imshow(np.argmax(y[i_batch], axis = 0))
                # plt.show()
        tpr = tps / (tps+fns)
        tnr = tns / (tns + fps)
        f1 = 2*tps/(2*tps+fps+fns)
        acc = (tps + tns) / (tps+fps+fns+tns)
        tpr = list(tpr)
        tnr = list(tnr)
        f1 = list(f1)
        acc = list(acc)
        tpr.append(np.mean(tpr))
        tnr.append(np.mean(tnr))
        f1.append(np.mean(f1))
        acc.append(np.mean(acc))
        headers = ["feature_" + str(i) for i in range(hparams.n_classes)]
        headers.append("Macro. Avg.")
        results = pd.DataFrame([tpr, tnr, f1, acc], columns=headers)
        results["Metric"] = ["TPR", "TNR", "F1", "ACC"]
        results.to_csv(os.path.join(
            hparams.log_dir, "results.csv"), index=False)
    elif hparams.predict_single:
        pass
    else:
        train_ds = DirDataset(f'{dataset}/images/train/', f'{dataset}/pixelwise_annotations/train/', scale=hparams.scale,
                              grayscale=hparams.grayscale, n_channels=hparams.img_channels, n_classes=hparams.n_classes)
        val_ds = DirDataset(f'{dataset}/images/test/', f'{dataset}/pixelwise_annotations/test/', scale=hparams.scale,
                            grayscale=hparams.grayscale, n_channels=hparams.img_channels, n_classes=hparams.n_classes)
        train_loader = DataLoader(
            train_ds, batch_size=4, pin_memory=True, shuffle=True, num_workers=20)
        val_loader = DataLoader(val_ds, batch_size=1,
                                pin_memory=True, shuffle=False, num_workers=20)
        # python train.py --dataset /media/gerstenberger/Data1/datasets/tree_segmentation/ --log_dir ./tree_segmentation_unet18/ --n_epochs 100 --resnet50 0 --resnet18 0 --resnet_autoencoder 1 --pretrained 0 --img_channels 3 --n_classes 4
        trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--dataset', required=True)
    parent_parser.add_argument('--log_dir', default='./lightning_logs')
    parent_parser.add_argument('--n_epochs', default=10)
    parent_parser.add_argument('--n_classes', type=int, default=1)
    parent_parser.add_argument('--grayscale', type=int, default=0)
    parent_parser.add_argument('--resnet50', action='store_true')
    parent_parser.add_argument('--resnet18', action='store_true')
    parent_parser.add_argument('--resnet_autoencoder', action='store_true')
    parent_parser.add_argument('--unet', action='store_true')
    parent_parser.add_argument('--mobilnet_autoencoder', action='store_true')
    parent_parser.add_argument('--predict', action='store_true')
    parent_parser.add_argument('--predict_single', action='store_true')

    parent_parser.add_argument('--scale', type=float, default=1.0)

    parent_parser.add_argument('--pretrained', type=int, default=1)
    # parent_parser.add_argument('--binarize_masks', type=int, default=0)
    parent_parser.add_argument('--img_channels', type=int, default=3)

    hparams = parent_parser.parse_args()

    main(hparams)
