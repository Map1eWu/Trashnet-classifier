# 实验分支：尝试替换backbone结构

import argparse, time, csv, logging, os
import itertools
import torch
from   torch import nn
from   torchvision import datasets, transforms, models
from   tqdm.auto import tqdm
from   torch.utils.tensorboard import SummaryWriter

from   initialize import dataset_initialize
from   val_trashnet import val_test

def parse_args():
    
    parser = argparse.ArgumentParser(description="Train a model on the TrashNet dataset.")
    parser.add_argument("--dataset_train_path",         type=str,   default="F:/python_envs/datasets/TrashNet/dataset-split/train")
    parser.add_argument("--dataset_val_path",           type=str,   default="F:/python_envs/datasets/TrashNet/dataset-split/val"  )
    
    parser.add_argument("--num_epochs",                 type=int,   default=20      )
    parser.add_argument("--batch_size",                 type=int,   default=32      )
    parser.add_argument("--learning_rate",              type=float, default=1e-4    )
    parser.add_argument("--device",                     type=str,   default="cuda"  )

    parser.add_argument("--resume_weights",             type=str,   default=None    )
    parser.add_argument("--output_path",                type=str,   default="./run/")

    args = parser.parse_args()
    return args



def model_inition(model, num_classes):

    for param in model.parameters():
        param.requires_grad = False

    for param in model.features[5].parameters():
        param.requires_grad = True
    for param in model.features[6].parameters():
        param.requires_grad = True
    for param in model.features[7].parameters():
        param.requires_grad = True
    for param in model.norm.parameters():
        param.requires_grad = True

    model.head = nn.Linear(model.head.in_features, num_classes)
    model.head.weight.requires_grad = True
    return model

def denormalize(images):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(images.device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(images.device)
    return (images * std + mean).clamp(0, 1)

class Updata():
    def __init__(self, output_path, num_epochs):
        
        self.num_epochs = num_epochs

        os.makedirs(output_path, exist_ok=True)

        # --- 1. 文本日志 (train.log) ---
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(os.path.join(output_path, "train.log"), mode='a', encoding='utf-8'),
                logging.StreamHandler() # 同时输出到控制台
            ]
        )
        self.logger = logging.getLogger()
        
        # --- 2. CSV 日志 (metrics.csv) ---
        csv_path = os.path.join(output_path, "metrics.csv")
        is_new_file = not os.path.exists(csv_path)
        
        self.csv_file = open(csv_path, mode='a', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        
        # 如果是新文件，写入表头
        if is_new_file:
            self.csv_writer.writerow(['Timestamp', 'Epoch', 'Step', 'Loss', 'Accuracy'])
            self.csv_file.flush()
        
        # --- 3. TensorBoard ---
        self.writer       = SummaryWriter(output_path + "tensorboard_logs/")

    def bar_init(self, epoch, num_epochs, train_loader):
        self.progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        return self.progress_bar
    
    def val(self, loss, acc, epoch):
        self.writer.add_scalar("Loss/val",    loss, epoch)
        self.writer.add_scalar("Accuracy/val", acc, epoch)

    def __call__(self, loss, acc, step, epoch, imgs):

        if step % 10 == 0:
            self.writer.add_images('Train/imgs', denormalize(imgs[:8]), step)

        self.progress_bar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "acc": f"{acc:.3f}"
        })
        
        self.writer.add_scalar("Loss/train", loss.item(), step)
        self.writer.add_scalar("Accuracy/train", acc,     step)

        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.csv_writer.writerow([
            current_time,
            epoch + 1,
            step,
            f"{float(loss):.3f}",
            f"{float(acc):.3f}",
        ])
        self.csv_file.flush()

        if step % 100 == 0 :
            self.logger.info(
                f"Epoch [{epoch+1}/{self.num_epochs}] "
                f"Step [{step}] "
                f"Loss: {float(loss):.4f} "
                f"Acc: {float(acc):.4f}"
            )

def train(args, train_loader, val_loader, device, num_classes):

    num_epochs = args.num_epochs
    updata = Updata(args.output_path, num_epochs)
    logger = updata.logger

    # 模型初始化
    if args.resume_weights is not None:
        logger.info(f"Resuming training from weights: {args.resume_weights}")
        model = models.swin_t()
        model = model_inition(model, num_classes)
        model.load_state_dict(torch.load(args.resume_weights))
        start_epochs = int(args.resume_weights.split("/")[-1].split("_")[0])  # 从文件名解析出起始 epoch
        logger.info(f"Resuming from epoch: {start_epochs}")
    else:
        logger.info("Initializing model with Swin-Transformer pre-trained weights.")
        model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        model = model_inition(model, num_classes)
        start_epochs = 0
    model.to(device)

    learning_rate = args.learning_rate
    optimizer = torch.optim.AdamW(
        itertools.chain(
            model.features[5].parameters(),
            model.features[6].parameters(),
            model.features[7].parameters(),
            model.norm.parameters(),
            model.head.parameters(),
        ),
        lr=learning_rate,
    )

    criterion = nn.CrossEntropyLoss()
    
    global_step = start_epochs * len(train_loader)
    best_val_acc = 0.0

    logger.info(f"Training started. Config: BS={args.batch_size}, LR={learning_rate}")
    logger.info(f"Total epochs: {num_epochs}, Starting epoch: {start_epochs+1}")

    for epoch in range(start_epochs, num_epochs):
        model.train()

        total_loss, total_acc = 0, 0

        progress_bar = updata.bar_init(epoch, num_epochs, train_loader)

        for imgs, labels in progress_bar:

            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(imgs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            acc = (outputs.argmax(dim=1) == labels).float().mean().item()
            total_acc += acc
            global_step += 1

            updata(loss, acc, global_step, epoch, imgs)

        avg_loss = total_loss / len(train_loader)
        avg_acc  = total_acc  / len(train_loader)

        logger.info(f"Epoch [{epoch+1}/{num_epochs}] Enter Val Test...")
        with torch.no_grad():
            val_loss, val_acc = val_test(model, val_loader, device, criterion)
        updata.val(val_loss, val_acc, epoch)
        
        logger.info(f"Epoch [{epoch+1}/{num_epochs}]:")
        logger.info(f"Train_Loss : {avg_loss:.3f}, Val_Loss : {val_loss:.3f}, "
                    f"Train Acc. : {avg_acc :.3f}, Val Acc  : {val_acc :.3f}")

        weights_dir = os.path.join(args.output_path, "weights")
        all_dir     = os.path.join(weights_dir, "all")
        os.makedirs(all_dir, exist_ok=True)

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 
                       os.path.join(weights_dir, "best_model.pth"))
            logger.info(f"New best model saved with val acc: {val_acc:.4f}")

        torch.save(model.state_dict(), 
                   os.path.join(weights_dir, "latest_model.pth"))
        torch.save(model.state_dict(), 
                   os.path.join(all_dir, f"{epoch + 1}_model.pth"))

    logger.info(f"Best model saved as 'best_model.pth' with val acc: {best_val_acc:.4f}")

def main(args):

    device = args.device
    if device == "cuda" and torch.cuda.is_available():
        print("Using GPU for training.")
        torch.cuda.set_device(0)
    else:
        print("Using CPU for training.")
        device = "cpu"

    # 数据集增强和预处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform   = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_loader, class_names = dataset_initialize(args.dataset_train_path, train_transform, args.batch_size)
    val_loader, _             = dataset_initialize(args.dataset_val_path,   val_transform,   args.batch_size)

    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")

    # 训练
    train(
        args=args, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        device=device, 
        num_classes=num_classes
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)
