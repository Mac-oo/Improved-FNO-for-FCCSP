import sys
import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
import scipy.io
from utilities3 import *
from Adam import Adam
from scipy import fft

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        """
        Frequency domain transformation of Fourier neural operator layer.
        It performs three operations: forward FFT, learnable linear transform in Fourier space, and inverse FFT.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to retain in each spatial dimension <=floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.randn(
            in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.randn(
            in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.randn(
            in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.randn(
            in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x, y, z), (in_channel, out_channel, x, y, z) -> (batch, out_channel, x, y, z)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # Step1: Transform x to Fourier space
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Step2: Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Step3: Return x to physical space, output x [batch, channels, nx, ny, nz]
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNO3d(nn.Module):
    def __init__(self, modes, width, n_layers=4):
        super(FNO3d, self).__init__()
        """
        The overall network:
        1. Lifting layer: Lift the input to the desire channel dimension by self.fc0.
        2. 4 Fourier neural operator layers of the integral operators u' = (W + K)(u).
            W defined by self.w, K defined by self.conv.
        3. Projection layer: Project from the channel space to the output space by self.fc1 and self.fc2.

            input: the solution of the coefficient function and locations (a(x, y, z), x, y, z)
            input shape: (batchsize, x=s, y=s, z=s, c=6)
            output: the solution 
            output shape: (batchsize, x=s, y=s, z=s, c=12)
        """
        self.modes1 = int(modes)
        self.modes2 = int(modes)
        self.modes3 = int(modes / 2)
        self.width = int(width)

        # Layer1: Lifting layer, [x, y, z, E, v, CTE] + 3 grid coordinates = 9
        # self.fc0 = nn.Conv3d(9, self.width, 1)  # FCNN replaces the linear layer
        self.fc0 = nn.Conv3d(6, self.width, 1)  # FCNN replaces the linear layer

        # Layer2-5: 4 Fourier neural operator layers
        self.convs = nn.ModuleList()
        self.ws = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3))
            self.ws.append(nn.Conv3d(self.width, self.width, 1))

        # Layer6: Projection layer, FCNN replaces the linear layer
        self.fc1 = nn.Conv3d(self.width, 128, 1)
        self.fc2 = nn.Conv3d(128, 12, 1)

    def get_grid(self, shape, device):
        """
        The get_grid method generates normalized coordinate grids [0, 1] for each spatial dimension.
        """
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

    def forward(self, x):
        # Step 1: Input Preparation and Grid Concatenation
        # grid = self.get_grid(x.shape, x.device)  # (batch, nx, ny, nz, 3)
        # x = torch.cat((x, grid), dim=-1)  # (batch, nx, ny, nz, 9)

        # Step 2: Lifting and Padding
        x = x.permute(0, 4, 1, 2, 3)    # (batch, nx, ny, nz, 6 or 9)
        pad_z = fft.next_fast_len(x.shape[-1], real=True) - x.shape[-1]
        pad_y = fft.next_fast_len(x.shape[-2], real=True) - x.shape[-2]
        pad_x = fft.next_fast_len(x.shape[-3], real=True) - x.shape[-3]
        x = F.pad(x, [0, pad_z, 0, pad_y, 0, pad_x])
        x = self.fc0(x)

        # Step 3: Fourier Block Processing
        for i in range(len(self.convs)):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            x = F.gelu(x)

        # Step 4: Projection to Output
        x = x[..., 0:x.shape[-3] - pad_x, 0:x.shape[-2] - pad_y, 0:x.shape[-1] - pad_z]
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.permute(0, 2, 3, 4, 1)


# ######### Configurations ##########
ndata = 220
n_train = 170
n_test = 50
batch_size = 2
learning_rate = 0.0037
epochs = 500
modes = 16
width = 48
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# ########## Load data and normalization ##########
save_path = '../train/'
train_results = '../data/'
dataset_path = "D:/MAC/Neural_operator/Data/interpol_fccsp_data_steps.npz"
data = np.load(dataset_path)

x_data = torch.tensor(data['interpolated_input'], dtype=torch.float32)  # (ndata, gx, gy, gz, 6)
x_mask = torch.ones_like(x_data)  # The default input is globally valid, the masks are set to 1, (ndata, gx, gy, gz, 6)

y_data_raw = torch.tensor(data['interpolated_output'], dtype=torch.float32)  # (ndata, gx, gy, gz, 3, 4)
step_active = (torch.max(torch.abs(y_data_raw), dim=-1, keepdim=True)[0] > 1e-10).float()
step_active = torch.cummax(step_active, dim=-2)[0]
geom_mask_raw = step_active.expand(-1, -1, -1, -1, -1, 4)   # (N, gx, gy, gz, 3, 4)
y_data = y_data_raw.flatten(start_dim=-2)   # Flatten (ndata, gx, gy, gz, 12)
geom_mask = geom_mask_raw.flatten(start_dim=-2)  # (ndata, gx, gy, gz, 12)

indices = np.random.permutation(ndata)
train_idx = indices[:n_train]
test_idx = indices[n_train:ndata]
np.savez(os.path.join(save_path, "dataset_split.npz"), train_idx=train_idx, test_idx=test_idx)

x_train = x_data[train_idx]
y_train = y_data[train_idx]
xmask_train = x_mask[train_idx]
mask_train = geom_mask[train_idx]

x_test = x_data[test_idx]
y_test = y_data[test_idx]
xmask_test = x_mask[test_idx]
mask_test = geom_mask[test_idx]

x_indices = [0, 1, 2, 3, 4, 5]
y_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

x_normalizer = MaskedUnitGaussianNormalizer(x_train, xmask_train, indices_to_normalize=x_indices)
x_train = x_normalizer.encode(x_train, xmask_train)
x_test = x_normalizer.encode(x_test, xmask_test)
x_normalizer.to(device)

y_normalizer = MaskedUnitGaussianNormalizer(y_train, mask_train, indices_to_normalize=y_indices)
y_train = y_normalizer.encode(y_train, mask_train)
y_test = y_normalizer.encode(y_test, mask_test)
y_normalizer.to(device)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train, mask_train),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test, mask_test),
                                          batch_size=batch_size, shuffle=False)

# ########## Training and evaluation ##########
model = FNO3d(modes=modes, width=width, n_layers=4).to(device)
print(count_params(model))

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# myloss = LpLoss(size_average=False)
criterion = MaskedMSELoss(channel_average=False)

train_losses = []
test_losses = []

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_loss = 0

    for x, y, mask in train_loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)  # x: (batch, H, W, D, 6), y: (batch, H, W, D, 4)

        optimizer.zero_grad()
        out = model(x)  # out: [batch, H, W, D, 4]
        loss = criterion(out, y, mask)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y, mask in test_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            out = model(x)

            loss = criterion(out, y, mask)
            test_loss += loss.item()

    train_loss /= n_train
    test_loss /= n_test

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    t2 = default_timer()
    print(f"Epoch: {ep}, Time: {t2 - t1:.2f}s, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    if ep % 50 == 0:
        torch.save(model.state_dict(), save_path + "/fourier.pt")


#########
torch.save(model.state_dict(), save_path + "/fourier_fccsp_final.pt")
scipy.io.savemat(save_path + '/loss_fccsp.mat', {'train_losses': np.array(train_losses), 'test_losses': np.array(test_losses)})

plt.clf()
plt.figure(1)
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')

plt.plot(np.arange(len(train_losses)), train_losses, label="Train loss")
plt.plot(np.arange(len(test_losses)), test_losses, label="Test loss")
# plt.yscale('log')
plt.legend()

plt.tight_layout()
plt.savefig(train_results + "/3Steps_MSE_loss.png", dpi=600)
plt.show()
plt.close()

