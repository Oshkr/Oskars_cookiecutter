import torch
import argparse
import numpy as np

from models import model

from visualizations import visualize

def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    return torch.cat([model(batch) for batch in dataloader], 0)


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="Predict using a pretrained model.")
    parser.add_argument("model_path", help="Path to the pretrained model file")
    parser.add_argument("input_data_path", help="Path to the input data file for prediction")

    args = parser.parse_args()
    path = args.model_path
    data = args.input_data_path

    cur_model = model.Network(784, 10, [512, 256, 128])

    # loads the first model
    state_dict = torch.load(path)
    cur_model.load_state_dict(state_dict)
    
    imgs = np.load(data)
    imgs = torch.from_numpy(imgs)

    #print(imgs.view(imgs.shape[0],-1).shape)
    
    predictions = cur_model(imgs.view(imgs.shape[0],-1))    

    visualize.visualize_images(imgs,torch.exp(predictions))
    

