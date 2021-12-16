import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

"""
Loading data
"""
data_input = np.load('data.npy')
data_torch = torch.tensor(data_input).float()   # transform to tensor
data = data_torch.permute(0,3,1,2)              # dim:(1281, 3, 26, 26)
label_input = np.load('label.npy')

"""
A Convolutional Variational Autoencoder
"""
class VAE(nn.Module):
    def __init__(self, imgChannels=3, featureDim=32*22*22, zDim=256):   # 26-3+1-3+1 = 22
        super(VAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, 16, 3)
        self.encConv2 = nn.Conv2d(16, 32, 3)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(32, 16, 3)
        self.decConv2 = nn.ConvTranspose2d(16, imgChannels, 3)

    def encoder(self, x):
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        x = x.view(-1, 32*22*22)    # reshape in order to do linear transformation
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    # adding Gaussian noise (by mean & log variance)
    def reparameterize(self, mu, logVar):
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        x = F.relu(self.decFC1(z))
        x = x.view(-1, 32, 22, 22)
        x = F.relu(self.decConv1(x))
        x = torch.sigmoid(self.decConv2(x))
        return x

    def forward(self, x):
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar

"""
Loss function (Binary Cross Entropy + KL Divergence)
"""
def loss_function(output, input, mu, logVar):
    bce = F.binary_cross_entropy(output, input, reduction='sum')
    kld = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
    return bce + kld


"""
Initialization
"""
batch_size = 256
learning_rate = 0.001
num_epochs = 100

data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)

net = VAE()
net = net.float()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


"""
Training
"""
print('start training !')
plot_train_x = []
plot_train_y = []
for epoch in range(num_epochs):
    for batch_idx, input in enumerate(data_loader):
        
        output, mu, logVar = net(input.float())

        loss = loss_function(output, input, mu, logVar)

        # Backpropagation based on the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch {}: Loss {}'.format(epoch+1, loss))
    loss = loss.detach().numpy()
    plot_train_x.append(epoch+1)
    plot_train_y.append(loss)

print('end of training !')
plots = plt.plot(plot_train_x, plot_train_y)
plt.show()


"""
Get the outputs of each class (5 outputs per each)
"""
print("-")
print("start showing output images !")
net.eval()
with torch.no_grad():
    label_class = 0
    end = 0
    while end != 1:
        for i in range(1281):
            if label_class == 9:    # have already found all of the classes
                end = 1
                break
            if label_input[i][0] == label_class:
                print("showing class: {}".format(label_class))
                label_class +=1
                for j in range(5):
                    output, mu, logVar = net(data[i].unsqueeze(0).float())
                    img = output.reshape((3,26,26))
                    plt.imshow(img.permute(1, 2, 0))
                    plt.show()
    print("end of the program !")


# # store the output into gen_data.npy
# output_list = []
# with torch.no_grad():
#     for i in range(1281):
#         for j in range(5):
#             output, mu, logVar = net(data[i].unsqueeze(0).float())
#             output_list.append(output.numpy())   
#     np.save('gen_data.npy', output_list, allow_pickle=True)
                

    