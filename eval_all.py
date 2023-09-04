import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
from PIL import ImageFont, ImageDraw


def create_images_dir_if_not_exists(model_name):
    if 'images' not in os.listdir():
        os.mkdir('images')

    if model_name not in os.listdir('images'):
        os.mkdir(os.path.join('images', model_name))

def get_create_images_dir_if_not_exists(model_name):
    create_images_dir_if_not_exists(model_name)
    return os.path.join('images', model_name)

def convert_tensor_to_PIL_image(tensor):
    transform = transforms.ToPILImage()
    return transform(tensor)

def save_image(image_name, predicted_class, image_matrix, model_name):
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )

    image_path = os.path.join(get_create_images_dir_if_not_exists(model_name), image_name)
    im = convert_tensor_to_PIL_image(inv_normalize(image_matrix))

    mf = ImageFont.truetype('arial.ttf', 8)
    ImageDraw.Draw(im).text((0,0), predicted_class, (255,0,0), font=mf)

    im.save(image_path)


def evaluate_model(model, model_name, val_loader, idx_to_class):
    num_correct = 0
    total_images = 0

    distribution = {idx : 5 for idx in range(10)}
    images_saved = 0
    

    with torch.no_grad():
        for (input, target) in val_loader:
            preds = torch.argmax(torch.nn.functional.softmax(model(input.cuda()), dim=1), dim=1)
            num_correct += (preds == target.cuda()).sum().item()
            
            total_images += len(input)
            
            if images_saved >= 50:
                continue

            for i, idx in enumerate(target):
                idx = idx.item()
                if distribution[idx] < 0:
                    continue
                
                image_file_name = idx_to_class[idx] + '_' + str(distribution[idx]) + '.png'
                distribution[idx] -= 1

                save_image(image_file_name, idx_to_class[preds[i].item()], input[i], model_name)
                images_saved += 1



    accuracy = num_correct / total_images
    classification_error = (1 - accuracy) * 100

    print('--------------------------------------------------------------------------')
    print('MODEL:', model_name)
    print(f"Total images: {total_images}")
    print(f"Correct predictions: {num_correct}")
    print(f"Classification error: {classification_error:.2f}%")


def main():
    model_names = sorted(name for name in resnet.__dict__
        if name.islower() and not name.startswith("__")
            and name.startswith("resnet")
            and callable(resnet.__dict__[name]))
    
    all_files = os.listdir('pretrained_models')
    model_paths = {}
    models = {}


    for model_file_name in all_files:
        model_paths[model_file_name.split('-')[0]] = os.path.join('pretrained_models', model_file_name)

    for model_name in model_names:
        model_ = torch.nn.DataParallel(resnet.__dict__[model_name]())
        model_.cuda()
        model_.eval()
        model_.load_state_dict(torch.load(model_paths[model_name])['state_dict'])
        models[model_name] = model_

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    
    if 'data' not in os.listdir():
        print('Downloading CIFAR10 dataset. This may take a couple of minutes depending on your internet connection.' + 
              'May need to download manually if torchvision takes eternity. If you download manually, make sure the directory structure is as follows:\n' +
              './data\n' +
              '|---- /cifar-10-batches-py/\n' +
              '|---- |---- data_batch_1\n' +
              '|---- |---- data_batch_2\n' +
              '|---- |---- data_batch_3\n' +
              '|---- |---- data_batch_4\n' +
              '|---- |---- data_batch_5\n' +
              '|---- |---- batches.meta'
            )

    cifar_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))

    val_loader = torch.utils.data.DataLoader(cifar_dataset, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

    idx_to_class = {cifar_dataset.class_to_idx[k]:k for k in cifar_dataset.class_to_idx.keys()}

    for model_name in models.keys():
        evaluate_model(models[model_name], model_name, val_loader, idx_to_class)
            

if __name__ == '__main__':
    main()