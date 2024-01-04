import torch
import os
import numpy as np

def make_dataset(path):
    
    images = []
    targets = []
    for file in os.listdir(path+"raw/"):
        if 'images' in file:
            image = torch.load(path+"raw/"+file)
            images.append(image)
        
        elif 'target' in file:
            target = torch.load(path+"raw/"+file)
            targets.append(target)

    images = torch.cat(images,dim = 0)
    targets = torch.cat(targets,dim = 0)

    temp_img = images.view(images.shape[0],-1)
    img_mean = torch.mean(temp_img,dim = 1).reshape(temp_img.shape[0],1,1)
    img_std = torch.std(temp_img,dim = 1).reshape(temp_img.shape[0],1,1)

    normalized_images = (images - img_mean)/img_std

    torch.save(normalized_images,path + 'processed/processed_images_train.pt')
    torch.save(normalized_images,path + 'processed/processed_targets_train.pt')

def make_examples(path):
    test_im = torch.load(path+"raw/test_images.pt")
    randoms = torch.randint(0,test_im.shape[0],(10,))
    imgs = [test_im[i].numpy() for i in randoms]

    np.save('data/example_images.npy',np.array(imgs))

if __name__=="__main__":
    path = "/mnt/c/Users/s184364/DTU/MLOps/repos/Oskars_cookiecutter/data/"
    make_dataset(path)

    print("making examples for inference...")
    make_examples(path)

    