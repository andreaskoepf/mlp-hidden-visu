
import os
from copy import deepcopy
import argparse
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


class SequentialCapture(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.layers = nn.ModuleList(modules)

    def forward(self, x):
        outputs = [x.detach()]
        for l in self.layers:
            x = l(x)
            outputs.append(x.detach())
        return x, outputs


class Activation(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


def create_model(num_layers=4, num_hidden=5, non_linearity=F.relu, normalize=nn.Identity(), num_in=2, num_out=1):
    net = []
    for i in range(num_layers):
        net.append(
            nn.Sequential(
                nn.Linear(num_in, num_hidden), 
                deepcopy(normalize),
                Activation(non_linearity)
            )
        )
        num_in = num_hidden
    net.append(nn.Linear(num_hidden, num_out))
    return SequentialCapture(net)


def checkerboard(x, board_size=4):
    x = x.mul(board_size).floor().clamp(0, board_size-1)
    x = x % 2
    mask = x[:,0] == x[:,1]
    y = torch.zeros(x.size(0), device=x.device)
    y[mask] = 1
    y.requires_grad = True
    return y


def grid_coordinates(h, w, normalize=False):
    x = torch.zeros(h*w, 2)
    i = torch.arange(h*w)
    x[:,0] = i.div(w, rounding_mode='trunc')
    x[:,1] = i % w
    if normalize:
        x[:,0] = x[:,0] / h + 1/(2*h)
        x[:,1] = x[:,1] / w + 1/(2*w)
    return x


def evaluate_model(model, h, w, color_map='jet'):
    cm = get_cmap(color_map)
    norm = Normalize()

    g = grid_coordinates(h, w, True)

    _, y = model.forward(g)
    max_hiddens = max([l.size(1) for l in y])
    layers = len(y)

    grid = torch.zeros(max_hiddens * layers, 4, h, w)

    for i,l in enumerate(y):
        for j in range(l.size(1)):
            x = l[:,j].view(h, w)
            x = torch.from_numpy(cm(norm(x.numpy())))
            grid[i*max_hiddens+j] = x.permute(2,0,1)
    
    return torchvision.utils.make_grid(grid, nrow=max_hiddens, normalize=True, scale_each=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', type=str, help='device to use')
    parser.add_argument('--device-index', default=0, type=int, help='device index')
    parser.add_argument('--name', default='visu', type=str, help='prefix for output image files')
    parser.add_argument('--manual_seed', default=42, type=int, help='initialization of pseudo-RNG')
    parser.add_argument('--batch_size', default=1000, type=int, help='batch size')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')

    parser.add_argument('--optimizer', default='Adam', type=str, help='Optimizer to use (Adam, AdamW)')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--loss_fn', default='MSE', type=str)
    parser.add_argument('--target', default='checkerboard', type=str)

    # network topology
    parser.add_argument('--activation', default='tanh', type=str)
    parser.add_argument('--max_steps', default=3000, type=int)
    parser.add_argument('--eval_interval', default=100, type=int)
    parser.add_argument('--num_layers', default=5, type=int, help='number of hidden layers')
    parser.add_argument('--num_hiddens', default=8, type=int, help='number of neurons in hidden layers')

    parser.add_argument('--output_path', default='./results', type=str)

    opt = parser.parse_args()
    return opt


def main():
    print('Using pytorch version {}'.format(torch.__version__))
    opt = parse_args()
    print('Options:', opt)
    device = torch.device(opt.device, opt.device_index)
    print('Device:', device)
    torch.manual_seed(opt.manual_seed)

    activation_name = opt.activation
    optimizer_name = opt.optimizer
    loss_fn_name = opt.loss_fn
    generator_name = opt.target
    max_steps = opt.max_steps
    batch_size = opt.batch_size
    eval_interval = opt.eval_interval
    name = opt.name

    activation_fn_map = {
        'sigmoid': F.sigmoid,
        'tanh': torch.tanh,
        'cos': torch.cos,
        'relu': F.relu,
        'leakyrelu': F.leaky_relu,
        'elu': F.elu,
        'selu': F.selu,
        'celu': F.celu,
        'silu': F.silu,
        'mish': F.mish,        
        'gelu': F.gelu
    }

    activation_name = activation_name.lower()
    if activation_name in activation_fn_map:
        non_linearity = activation_fn_map[activation_name]
    else:
        raise RuntimeError('Unsupported activation function type specified. Supported are: ' + ', '.join(activation_fn_map.keys()))

    loss_fn_name = loss_fn_name.lower()
    if loss_fn_name == 'smoothl1':
        loss_fn = torch.nn.SmoothL1Loss(reduction='mean')
    elif loss_fn_name == 'mse' or loss_fn_name == 'l2':
        loss_fn = torch.nn.MSELoss(reduction='mean')
    elif loss_fn_name== 'mae' or loss_fn_name == 'l1':
        loss_fn = torch.nn.L1Loss(reduction='mean')
    else:
        raise RuntimeError('Unsupported loss function type specified.')

    if generator_name == 'checkerboard':
        generator = checkerboard
    else:
        raise RuntimeError('Unsupported generator specified.')

    if not os.path.isdir(opt.output_path):
        print('creating ', opt.output_path)
        os.makedirs(opt.output_path)

    # create model
    model = create_model(num_layers=opt.num_layers, num_hidden=opt.num_hiddens, non_linearity=non_linearity)
    print('Number of parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # create optimizer
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, momentum=opt.momentum)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, amsgrad=False)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, amsgrad=False)
    else:
        raise RuntimeError('Unsupported optimizer specified.')

    for i in range(0, max_steps):
        batch = torch.rand(batch_size, 2, requires_grad=False)
        target = generator(batch)

        y,_ = model(batch)
        loss = loss_fn(y.view(-1), target.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(i, loss.item())

        if i % eval_interval == 0:
            result = evaluate_model(model, 100, 100)
            fn = '{}_{:06d}.png'.format(name, i)
            fn = os.path.join(opt.output_path, fn)

            torchvision.utils.save_image(result, fn)


if __name__ == '__main__':
    main()