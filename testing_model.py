import torch
import matplotlib.pyplot as plt
from PIL import Image
from saving_loading import loading_model
from inception import GoogLeNet
import random
import torchvision
from dataset import get_class_names
from torchvision import transforms

torch.cuda.manual_seed(42)
random.seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize(size = (224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def pred_plot_image(model, image_path, class_names, transform = None, device = device):
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    target_image = target_image / 255.
    if transform:
        target_image = transform(target_image)
    model.to(device)

    model.eval()
    with torch.inference_mode():
        target_image = target_image.usqueeze(dim = 0)
        target_image_pred = model(target_image.to(device))
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    plt.imshow(target_image.squeeze().permute(1, 2, 0)) # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else: 
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)

def main():
    model = loading_model(model = GoogLeNet, model_save_path="models/MyGoogLeNet", device = device)
    class_names = get_class_names()
    pred_plot_image(model = model, image_path="Sports data/valid/basketball/4.jpg", class_names = class_names, transform=transform)


if __name__ == "__main__":
    main()