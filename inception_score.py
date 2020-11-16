import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

import torchvision
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy


class Inception_Score():
    def __init__(self, dataset: data.Dataset):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Dataset & DataLoader
        self.N = len(dataset)
        self.batch_size = 32

        self.dataset = dataset
        self.dataloader = data.DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=1)
        self.transform = nn.Upsample(size=(299, 299), mode='bilinear').to(self.device)

        # Inception Model
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.inception_model.eval()


    def compute_score(self, splits=1):
        preds = np.zeros((self.N, 1000))

        # Compute the mean KL-divergence
        # You have to calculate the inception score.
        # The logit values from inception model are already stored in 'preds'.

        inception_score = 0.0

        ### YOUR CODE HERE (~ 20 lines)


        ### END YOUR CODE

        return inception_score


#############################################
# Testing functions below.                  #
#############################################

def test_inception_score():
    print("======Inception Score Test Case======")

    # CIFAR10 Datset without Label
    class CIFAR10woLabel(data.Dataset):
        def __init__(self):
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
            self.dataset = torchvision.datasets.CIFAR10(root='./data/', download=True, transform=transform)

        def __getitem__(self, index):
            return self.dataset[index][0]

        def __len__(self):
            return len(self.dataset)

    print("Calculating Inception Score...")

    Inception = Inception_Score(CIFAR10woLabel())
    score = Inception.compute_score(splits=1)

    assert np.allclose(score, 9.719672, atol=1e-3), \
        "Your inception score does not match expected result."

    print("All test passed!")


if __name__ == '__main__':
    test_inception_score()
