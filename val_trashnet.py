
import torch, argparse, os
from   tqdm.auto import tqdm
from   PIL import Image
from   torchvision import datasets, transforms, models
from   torch.nn import Linear
from initialize import dataset_initialize
def parse_args():
    
    parser = argparse.ArgumentParser(description="Train a model on the TrashNet dataset.")
    parser.add_argument("--dataset_val_path",           type=str,   default="./TrashNet/dataset-split/val"  )
    
    parser.add_argument("--num_classes",                type=int,   default=6       )
    parser.add_argument("--device",                     type=str,   default="cuda"  )
    parser.add_argument("--img_path",                   type=str,   default=None    )

    parser.add_argument("--output_path",                type=str,   default=None    )
    parser.add_argument("--weights_path",               type=str,   default=r".\run\weights\best_model.pth")

    args = parser.parse_args()
    args.img_path = r".\test_img\test_img4.jpg"
    return args

def val_test(model, val_loader, device, loss_func=None):
    model.eval()

    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if loss_func is not None:
                loss = loss_func(outputs, labels)
                total_loss += loss.item()

    accuracy = correct / total

    if loss_func is not None:
        return total_loss / len(val_loader), accuracy

    print(f"Validation Accuracy: {accuracy:.4f}")

def main(args):
    
    device = args.device
    if device == "cuda" and torch.cuda.is_available():
        print("Using GPU for validation.")
        torch.cuda.set_device(0)
    else:
        print("Using CPU for validation.")
        device = "cpu"

    model = models.swin_t()
    model.head = Linear(model.head.in_features, args.num_classes)
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model.to(device)

    if args.img_path is None:
        val_loader, _ = dataset_initialize(args.data_path, args.batch_size, "val")
        val_test(model, val_loader, device)
        return
    
    img = Image.open(args.img_path).convert("RGB")
    img = transforms.Resize((224, 224))(img)
    img = transforms.ToTensor()(img).unsqueeze(0).to(device)
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    outputs = model(img)
    _, predicted = torch.max(outputs.data, 1)
    topk_values, topk_indices = torch.topk(outputs.data, k=3)

    probs = torch.softmax(topk_values, dim=1)[0].tolist()
    print(f"Predicted: {class_names[predicted.item()]}")
    print("Top-3:")
    for i, (idx, prob) in enumerate(zip(topk_indices[0].tolist(), probs)):
        print(f"  {i+1}. {class_names[idx]:<12} {prob:.2%}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
