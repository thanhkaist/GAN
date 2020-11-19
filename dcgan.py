import os
import time
import datetime
import random
import argparse
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from dataloader import create_dataloader, FolderDataset
from inception_score import Inception_Score
from util import conv, deconv, denorm, save_checkpoint, load_checkpoint
import numpy as np


class DCGAN_Generator(nn.Module):
    def __init__(self):
        super(DCGAN_Generator, self).__init__()
        model = []

        ### DCGAN Generator
        # You have to implement 4-layers generator.
        # For more details on the generator architecture, please check the homework description.
        # Note 1: Recommend to use 'deconv' function implemented in 'util.py'.

        ### YOUR CODE HERE (~ 4 lines)
        model.append(deconv(256,256,4,1,0,norm='bn',activation='relu'))
        model.append(deconv(256,128,4,2,1,norm='bn',activation='relu'))
        model.append(deconv(128,64,4,2,1,norm='bn',activation='relu'))
        model.append(deconv(64,3,4,2,1,norm=None,activation='tanh'))
        ### END YOUR CODE

        self.model = nn.Sequential(*model)

    def forward(self, z: torch.Tensor):
        # Input (z) size : [Batch, 256, 1, 1]
        # Output (Image) size : [Batch, 3, 32, 32]

        ### YOUR CODE HERE (~ 2 lines)
        input = z.reshape(z.shape[0],256,1,1)
        output = self.model(input)
        ### END YOUR CODE

        return output


class DCGAN_Discriminator(nn.Module):
    def __init__(self, type: str='gan'):
        """
        Parameters
        type: gan loss type: 'gan' or 'lsgan' or 'wgan' or 'wgan-gp'
        """
        super(DCGAN_Discriminator, self).__init__()
        model = []

        ### DCGAN Discriminator
        # You have to implement 4-layers generator.
        # For more details on the generator architecture, please check the homework description
        # Note 1: Recommend to use 'conv' function implemented in 'util.py'
        # Note 2: Don't forget that the discriminator architecture depends on the type of gan loss.

        ### YOUR CODE HERE (~ 4 lines)
        model.append(conv(3,64,4,2,1,norm='bn',activation='lrelu'))
        model.append(conv(64,128,4,2,1,norm='bn',activation='lrelu'))
        model.append(conv(128,256,4,2,1,norm='bn',activation='lrelu'))
        model.append(conv(256,1,4,1,0,norm=None,activation=None))

        ### END YOUR CODE

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor):
        # Input (z) size : [Batch, 3, 32, 32]
        # Output (Image) size : [Batch, 1]

        ### YOUR CODE HERE (~ 1 lines)
        output = self.model(x)

        ### END YOUR CODE

        return output


class DCGAN_Solver():
    def __init__(self, type: str='gan', lr: float=0.0002, batch_size: int=64, num_workers: int=1, device=None):
        """
        Parameters
        type: gan loss type: 'gan' or 'lsgan' or 'wgan' or 'wgan-gp'
        lr: learning rate
        batch_size: batch size
        num_workers: the number of workers for train dataloader
        """

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Declare Generator and Discriminator
        self.type = type
        self.netG = DCGAN_Generator()
        self.netD = DCGAN_Discriminator(type=type)

        # Declare the Criterion for GAN loss
        # Doc for Binary Cross Entropy Loss: https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
        # Doc for MSE Loss: https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
        # Note1: Implement 'GPLoss' function before using WGAN-GP loss.
        # Note2: It is okay not to implement the criterion for WGAN.



        ### YOUR CODE HERE (~ 8 lines)
        if self.type == 'gan':
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.type == "lsgan":
            self.criterion = nn.MSELoss()
        elif self.type == "wgan-gp":
            self.penalty = 10
            self.gploss = GPLoss(self.device)
        ### END YOUR CODE

        # Declare the Optimizer for training
        # Doc for Adam optimizer: https://pytorch.org/docs/stable/optim.html#torch.optim.Adam

        ### YOUR CODE HERE (~ 2 lines)
        self.optimizerG = optim.Adam(self.netG.parameters(),lr=lr,betas=(0.5,0.999))
        self.optimizerD = optim.Adam(self.netD.parameters(),lr=lr,betas=(0.5,0.999)) #

        ### END YOUR CODE

        # Declare the DataLoader
        # Note1: Use 'create_dataloader' function implemented in 'dataloader.py'

        ### YOUR CODE HERE (~ 1 lines)
        self.trainloader, self.testloader = create_dataloader('cifar10',batch_size,num_workers)

        ### END YOUR CODE

        # Make directory
        os.makedirs(os.path.join('./results/', self.type, 'images'), exist_ok=True)
        os.makedirs(os.path.join('./results/', self.type, 'checkpoints'), exist_ok=True)


    def train(self, epochs: int=100):

        self.netG.to(self.device)
        self.netD.to(self.device)

        start_time = time.time()

        print("=====Train Start======")

        for epoch in range(epochs):
            for iter, (real_img, _) in enumerate(self.trainloader):
                self.netG.train()
                self.netD.train()

                batch_size = real_img.size(0)
                real_label = torch.ones(batch_size).to(self.device)
                fake_label = torch.zeros(batch_size).to(self.device)
                real_img = real_img.to(self.device)
                z = torch.randn(real_img.size(0), 256).to(self.device)

                ###################################################################################
                # (1) Update Discriminator
                # Compute the discriminator loss. You have to implement 4 types of loss functions ('gan', 'lsgan', 'wgan', 'wgan-gp').
                # You can implement 'wgan' loss function wihtout using self.criterion.
                # Note1 : Use self.criterion and self.type which is declared in the init function.
                # Note2 : Use the 'detach()' function appropriately.
                ###################################################################################

                ### YOUR CODE HERE (~ 15 lines)
                D_out = self.netD(real_img).squeeze()
                if self.type == 'lsgan':
                    D_real_loss = 1/2*self.criterion(D_out,real_label)
                elif (self.type == 'wgan') or (self.type == 'wgan-gp') :
                    D_real_loss = -D_out.mean()
                else:
                    D_real_loss = self.criterion(D_out,real_label)

                fake_img = self.netG(z)

                if self.type == 'wgan-gp':
                    #alpha = torch.Tensor(np.random.random((real_img.size(0), 1, 1, 1))).to(self.device)
                    alpha = torch.rand(real_img.size(0), 1, 1, 1)
                    #alpha = torch.FloatTensor(real_img.size(0), 1, 1, 1).uniform_(0, 1)
                    alpha = alpha.expand(real_img.size(0), real_img.size(1), real_img.size(2), real_img.size(3)).to(self.device)
                    fake_img = (alpha * real_img.detach() + ((1 - alpha) * fake_img.detach())).requires_grad_(True)

                D_out = self.netD(fake_img).squeeze()

                if self.type == 'lsgan':
                    D_fake_loss = 1/2*self.criterion(D_out, fake_label)
                elif self.type == 'wgan':
                    D_fake_loss = D_out.mean()
                elif self.type == 'wgan-gp':
                    D_fake_loss = D_out.mean() + self.penalty*self.gploss(D_out,fake_img)
                else:
                    D_fake_loss = self.criterion(D_out,fake_label)

                lossD =  D_real_loss + D_fake_loss

                ### END YOUR CODE

                # Test code
                if epoch == 0 and iter == 0:
                    test_lossD_function(self.type, lossD)

                self.netD.zero_grad()
                lossD.backward()
                self.optimizerD.step()

                ### Clipping the weights of Discriminator
                clip_value = 0.01

                if self.type == 'wgan':
                    ### YOUR CODE HERE (~2 lines)
                    # weight cliping
                    for p in self.netD.parameters():
                        p.data.clamp_(-clip_value, clip_value)
                    ### END YOUR CODE

                ###################################################################################
                # (2) Update Generator
                # Compute the generator loss. You have to implement 4 types of loss functions ('gan', 'lsgan', 'wgan', 'wgan-gp').
                # You can implement 'wgan' and 'wgan-gp' loss functions without using self.criterion.
                ###################################################################################



                ### YOUR CODE HERE (~ 10 lines)

                fake_img = self.netG(z)
                D_out = self.netD(fake_img).squeeze()

                if self.type == 'lsgan':
                    lossG = self.criterion(D_out,real_label)
                elif self.type == 'wgan' or self.type == 'wgan-gp':
                    lossG = -D_out.mean()
                else:
                    lossG = self.criterion(D_out,real_label)

                ### END YOUR CODE

                # Test code
                if epoch == 0 and iter == 0:
                    test_lossG_function(self.type, lossG)

                self.netG.zero_grad()
                lossG.backward()
                self.optimizerG.step()

                if (iter + 1) % 100 == 0:
                    end_time = time.time() - start_time
                    end_time = str(datetime.timedelta(seconds=end_time))[:-7]
                    print('Time [%s], Epoch [%d/%d], Step[%d/%d], lossD: %.4f, lossG: %.4f'
                          % (end_time, epoch+1, epochs, iter+1, len(self.trainloader), lossD.item(), lossG.item()))

            # Save Images
            fake_img = fake_img.reshape(fake_img.size(0), 3, 32, 32)
            save_image(denorm(fake_img), os.path.join('./results/', self.type, 'images', 'fake_image-{:03d}.png'.format(epoch+1)))

            if (epoch + 1) % 50 == 0:
                save_checkpoint(self.netG, os.path.join('./results', self.type, 'checkpoints', 'netG_{:02d}.pth'.format(epoch+1)), self.device)
                save_checkpoint(self.netD, os.path.join('./results', self.type, 'checkpoints', 'netD_{:02d}.pth'.format(epoch+1)), self.device)

        # Save Checkpoints
        save_checkpoint(self.netG, os.path.join('./results', self.type, 'checkpoints', 'netG_final.pth'), self.device)
        save_checkpoint(self.netD, os.path.join('./results', self.type, 'checkpoints', 'netD_final.pth'), self.device)

    def test(self):
        load_checkpoint(self.netG, os.path.join('./results', self.type, 'checkpoints', 'netG_final.pth'), self.device)
        self.netG.eval()

        os.makedirs(os.path.join('./results/', self.type, 'evaluation'), exist_ok=True)

        print("=====Test Start======")

        with torch.no_grad():
            for iter in range(1000):
                z = torch.randn(1, 256).to(self.device)
                fake_img = self.netG(z)
                save_image(denorm(fake_img), os.path.join('./results/', self.type, 'evaluation', 'fake_image-{:05d}.png'.format(iter+1)))

        # Compute the Inception score
        dataset = FolderDataset(folder = os.path.join('./results/', self.type, 'evaluation'))
        Inception = Inception_Score(dataset)
        score = Inception.compute_score(splits=1)

        print('Inception Score : ', score)


class GPLoss(nn.Module):
    def __init__(self, device):
        """
        Parameters
        device: device type: 'cpu' or 'cuda'
        """
        super(GPLoss, self).__init__()
        self.device = device

    def forward(self, y: torch.Tensor, x: torch.Tensor):
        """
        Parameters
        y: interpolate logits
        x: interpolate images
        """
        ### Gradient Penalty Loss
        # Penalize the norm of gradient of the critic with respect to its input. Calculate the L2 norm of gradient dy/dx.
        # Doc for torch.autograd.grad: https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
        # Doc for torch.norm: https://pytorch.org/docs/stable/generated/torch.norm.html#torch.norm

        ### YOUR CODE HERE (~ 5 lines)
        fake = torch.autograd.Variable(torch.Tensor(x.shape[0]).fill_(1.0), requires_grad=False).to(self.device)
        #
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        loss = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        ### END YOUR CODE

        return loss


#############################################
# Testing functions below.                  #
#############################################

def test_initializer_and_forward():

    print("=====Model Initializer Test Case======")

    gan_type = ['gan', 'lsgan', 'wgan', 'wgan-gp']

    netG = DCGAN_Generator()

    # the first test
    try:
        netG.load_state_dict(torch.load("sanity_check/sanity_check_dcgan_netG.pth", map_location='cpu'))
    except Exception as e:
        print("Your DCGAN generator initializer is wrong. Check the handout and comments in details and implement the model precisely.")
        raise e
    print("The first test passed!")

    # the second test
    for i, type in enumerate(gan_type):
        netD = DCGAN_Discriminator(type=type)
        try:
            if i == 0:
                netD.load_state_dict(torch.load("sanity_check/sanity_check_dcgan_netD1.pth", map_location='cpu'))
            else:
                netD.load_state_dict(torch.load("sanity_check/sanity_check_dcgan_netD2.pth", map_location='cpu'))
        except Exception as e:
            print("Your DCGAN discriminator initializer is wrong. Check the handout and comments in details and implement the model precisely.")
            raise e
    print("The second test passed!")

    print("All 2 tests passed!")


def test_lossG_function(gan_type, lossG):
    print("=====Generator Loss Function Test Case======")
    expected_lossG = [1.5416, 0.3207, 0.0017, 0.3192] #0.5849
    # the first test
    if gan_type == 'gan':
        assert lossG.detach().allclose(torch.tensor(expected_lossG[0]), atol=1e-2), \
            f"Generator ({gan_type}) Loss of the model does not match expected result."
    # the second test
    elif gan_type == 'lsgan':
        assert lossG.detach().allclose(torch.tensor(expected_lossG[1]), atol=1e-2), \
            f"Generator ({gan_type}) Loss of the model does not match expected result."
    # the third test
    elif gan_type == 'wgan':
        assert lossG.detach().allclose(torch.tensor(expected_lossG[2]), atol=1e-2), \
            f"Generator ({gan_type}) Loss of the model does not match expected result."
    # the fourth test
    elif gan_type == 'wgan-gp':
        assert lossG.detach().allclose(torch.tensor(expected_lossG[3]), atol=1e-2), \
            f"Generator ({gan_type}) Loss of the model does not match expected result."

    print(f"Generator {gan_type} loss function test passed!")


def test_lossD_function(gan_type, lossD):
    print("=====Discriminator Loss Function Test Case======")
    expected_lossD = [1.3483, 0.3373, -0.1794,1.0421 ] ##0.9908
    # the first test
    if gan_type == 'gan':
        assert lossD.detach().allclose(torch.tensor(expected_lossD[0]), atol=1e-2), \
            f"Discriminator ({gan_type}) Loss of the model does not match expected result."
    # the second test
    elif gan_type == 'lsgan':
        assert lossD.detach().allclose(torch.tensor(expected_lossD[1]), atol=1e-2), \
            f"Discriminator ({gan_type}) Loss of the model does not match expected result."
    # the third test
    elif gan_type == 'wgan':
        assert lossD.detach().allclose(torch.tensor(expected_lossD[2]), atol=1e-2), \
            f"Discriminator ({gan_type}) Loss of the model does not match expected result."
    # the fourth test
    elif gan_type == 'wgan-gp':
        assert lossD.detach().allclose(torch.tensor(expected_lossD[3]), atol=1e-2), \
            f"Discriminator ({gan_type}) Loss of the model does not match expected result."

    print(f"Generator {gan_type} loss function test passed!")


if __name__ == "__main__":
    torch.set_printoptions(precision=4)
    random.seed(1234)
    torch.manual_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("--gan_type", default='gan', choices=['gan', 'lsgan', 'wgan', 'wgan-gp'], help='Select GAN Loss function')
    opt = parser.parse_args()

    # Test Code
    test_initializer_and_forward()

    # Hyper-parameters
    gan_type = opt.gan_type
    epochs = 200
    lr = 0.0002
    batch_size = 64
    num_workers = 8
    train = True # train : True / test : False (Compute the Inception Score)

    # Train or Test
    solver = DCGAN_Solver(type=gan_type, lr=lr, batch_size=batch_size, num_workers=num_workers)
    if train:
        solver.train(epochs=epochs)
    else:
        solver.test()