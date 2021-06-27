import time
import torch
import matplotlib.pyplot as plt

from utils.bbox_utils import coord_trans


def detection_solver(detector: torch.nn.Module, data_loader, cfg):
    """
    Optimize detector.
    """
    detector = detector.to(cfg.device)

    # optimizer
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, detector.parameters()),
                                cfg.init_lr, cfg.momentum, weight_decay=cfg.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: cfg.lr_decay ** epoch)

    loss_history = []
    epoch_loss_history = []
    detector.train()

    # start training
    print("Start Training...")
    for epoch in range(cfg.epochs):
        tik = time.time()
        epoch_loss = 0.
        print(f"\nEpoch:{epoch+1} / {cfg.epochs}")
        print(f"Learning Rate:{lr_scheduler.get_last_lr()[0]:.6f}")
        for i, data_batch in enumerate(data_loader):
            imgs, bboxes, h_list, w_list, _ = data_batch
            bboxes = coord_trans(bboxes, h_list, w_list, mode="p2a")
            bboxes = bboxes.to(dtype=torch.float32, device=cfg.device)
            imgs = imgs.to(dtype=torch.float32, device=cfg.device)

            loss = detector(imgs, bboxes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            epoch_loss += loss.item() * imgs.shape[0]

            if (i+1) % 5 == 0:
                print('\r(Iter {} / {}) loss: {:.4f}'.format(i+1, len(data_loader), loss.item()), end='')

        tok = time.time()
        epoch_loss /= len(data_loader)
        epoch_loss_history.append(epoch_loss)
        print('\r(Iter {} / {}) loss: {:.4f} - time: {:.2f}'.format(
            len(data_loader), len(data_loader), epoch_loss, tok-tik))

        lr_scheduler.step()

    # plot the training losses
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss history')
    plt.subplot(1, 2, 2)
    plt.plot(epoch_loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


