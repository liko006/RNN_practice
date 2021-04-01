import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn, Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
from torch import Tensor
from torchinfo import summary
from tqdm import tqdm

labels_list = ['airplane', 'bus', 'cat',
               'dolphin', 'guitar', 'hurricane',
               'laptop', 'mountain', 'sheep',
               'The_Eiffel_Tower', 'tree' , 'umbrella']

labels_dict = {
    'airplane': np.array([1,0,0,0,0,0,0,0,0,0,0,0]),
    'bus' : np.array([0,1,0,0,0,0,0,0,0,0,0,0]),
    'cat' : np.array([0,0,1,0,0,0,0,0,0,0,0,0]),
    'dolphin' : np.array([0,0,0,1,0,0,0,0,0,0,0,0]),
    'guitar' : np.array([0,0,0,0,1,0,0,0,0,0,0,0]),
    'hurricane' : np.array([0,0,0,0,0,1,0,0,0,0,0,0]),
    'laptop' : np.array([0,0,0,0,0,0,1,0,0,0,0,0]),
    'mountain' : np.array([0,0,0,0,0,0,0,1,0,0,0,0]),
    'sheep' : np.array([0,0,0,0,0,0,0,0,1,0,0,0]),
    'The_Eiffel_Tower' : np.array([0,0,0,0,0,0,0,0,0,1,0,0]),
    'tree' : np.array([0,0,0,0,0,0,0,0,0,0,1,0]),
    'umbrella' : np.array([0,0,0,0,0,0,0,0,0,0,0,1])
}

data_dir = os.path.abspath('data')

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_max_len(strokes):
    """Return the maximum length of an array of strokes."""
    max_len = 0
    for stroke in strokes:
        ml = len(stroke)
        if ml > max_len:
            max_len = ml
    return max_len

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        pass
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    else:
        raise Exception('input must be a tensor or ndarray.')
    return x.float()

def init_orthogonal_(weight, hsize):
    assert weight.size(0) == 4*hsize
    for i in range(4):
        nn.init.orthogonal_(weight[i*hsize:(i+1)*hsize])

def load_strokes(data_dir):
    
    """Loads the .npz file, 
    and splits the set into train/valid/test."""

    train_strokes = None
    valid_strokes = None
    test_strokes = None
    
    for label in labels_list:
        
        data = np.load(os.path.join(data_dir, f'{str(label)}.npz'), 
                       encoding='latin1', allow_pickle=True)

        if train_strokes is None:
            train_strokes = data['train']
            valid_strokes = data['valid']
            test_strokes = data['test']
            train_label = np.broadcast_to(labels_dict[label], (len(data['train']), len(labels_dict[label])))
            valid_label = np.broadcast_to(labels_dict[label], (len(data['valid']), len(labels_dict[label])))
            test_label = np.broadcast_to(labels_dict[label], (len(data['test']), len(labels_dict[label])))
        else:
            train_strokes = np.concatenate((train_strokes, data['train']))
            valid_strokes = np.concatenate((valid_strokes, data['valid']))
            test_strokes = np.concatenate((test_strokes, data['test']))
            train_label = np.concatenate((train_label, np.broadcast_to(labels_dict[label], (len(data['train']), len(labels_dict[label])))))
            valid_label = np.concatenate((valid_label, np.broadcast_to(labels_dict[label], (len(data['valid']), len(labels_dict[label])))))
            test_label = np.concatenate((test_label, np.broadcast_to(labels_dict[label], (len(data['test']), len(labels_dict[label])))))              
        
    all_strokes = np.concatenate((train_strokes, valid_strokes, test_strokes))
    
    num_points = 0
    for stroke in all_strokes:
        num_points += len(stroke)
    avg_len = num_points / len(all_strokes)
    
    print('Dataset combined: {} ({}/{}/{}), avg len {}'.format(
        len(all_strokes), len(train_strokes), len(valid_strokes),
        len(test_strokes), int(avg_len)))
    
    max_seq_len = get_max_len(all_strokes)
    
    return train_strokes, valid_strokes, test_strokes, train_label, valid_label, test_label, max_seq_len


class QuickDrawDataset(Dataset):
    
    def __init__(self, strokes, labels, scale_factor=None, max_len=250, limit=1000):
        strokes = [to_tensor(stk) for stk in strokes]
        self.labels = [to_tensor(lbl) for lbl in labels]
        self.max_len = max_len
        self.limit = limit
        self.preprocess(strokes) # list of drawings in stroke-3 format, sorted by size
        self.normalize(scale_factor)
        

    def preprocess(self, strokes):
        """Remove entries from strokes having > max_len points.
        Clamp x-y values to (-limit, limit)
        """
        raw_data = []
        seq_len = []
        count_data = 0
        for i in range(len(strokes)):
            data = strokes[i]
            if len(data) <= (self.max_len):
                count_data += 1
                data = data.clamp(-self.limit, self.limit)
                raw_data.append(data)
                seq_len.append(len(data))
        self.sort_idx = np.argsort(seq_len)
        self.strokes = [raw_data[ix] for ix in self.sort_idx]
        print("total drawings <= max_seq_len is %d" % count_data)

    def calculate_normalizing_scale_factor(self):
        """Calculate the normalizing factor explained in appendix of sketch-rnn."""
        strokes = [elt for elt in self.strokes if len(elt) <= self.max_len]
        data = torch.cat(strokes)
        return data[:,:2].std()

    def normalize(self, scale_factor=None):
        """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
        if scale_factor is None:
            scale_factor = self.calculate_normalizing_scale_factor()
        self.scale_factor = scale_factor
        for i in range(len(self.strokes)):
            self.strokes[i][:,:2] /= self.scale_factor

    def __len__(self):
        return len(self.strokes)

    def __getitem__(self, idx):
        
        data = self.strokes[idx]
        target = self.labels[idx]

        return data, target


class QDRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(QDRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, dropout=0.2, bidirectional=True)
        self.classifier = nn.Linear(2*hidden_dim, 12)
       
        nn.init.xavier_normal_(self.classifier.weight)
        
        
    def forward(self, data, length):
        data = rnn_utils.pack_padded_sequence(data, length.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.rnn(data) # [2,batch,hid]
        x, _ = rnn_utils.pad_packed_sequence(x, batch_first=True)
        # x = x.permute(1,0,2).flatten(1).contiguous() # [batch,2*hid]
        output = self.classifier(x)
        # output = F.softmax(output)
        return output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_strokes, valid_strokes, test_strokes, train_label, valid_label, test_label, max_seq_len = load_strokes(data_dir)

trainset = QuickDrawDataset(train_strokes, train_label, max_len=max_seq_len)
validset = QuickDrawDataset(valid_strokes, valid_label, max_len=max_seq_len)

def collate_fn(dataset):
    train_data = list(list(zip(*dataset))[0])
    labels = list(zip(*dataset))[1]
    target = torch.Tensor([np.array(l) for l in labels])
    train_data.sort(key=lambda data: len(data), reverse=True)
    data_length = [len(data) for data in train_data]
    data_length = torch.Tensor(data_length)
    train_data = rnn_utils.pad_sequence(train_data, batch_first=True, padding_value=0)
    return train_data, target, data_length

train_loader = DataLoader(trainset, batch_size=128, collate_fn=collate_fn, shuffle=True, num_workers=0)
valid_loader = DataLoader(validset, batch_size=128, collate_fn=collate_fn, shuffle=False, num_workers=0)

num_epoch = 10
device = 'cuda'

model = QDRNN(3, 128, 8).to(device)

grad_clip = 1
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epoch):
    
    model.train()
    loss_meter = AverageMeter()
    with tqdm(total=len(train_loader.dataset)) as progress_bar:
        
        for idx, (data, target, length) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            length = length.to(device, non_blocking=True)
            target = target.to(device, dtype=torch.long, non_blocking=True)
            # training step
            optimizer.zero_grad()
            output = model(data, length)
#             loss = criterion(output, target)
            # reshape output to [Batch, NumClass] and target accordingly
            reshaped_output = output[:, -1, :].squeeze()
            reshaped_target = torch.argmax(target, dim=1)
            loss = criterion(reshaped_output, reshaped_target)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            # update loss meter and progbar
            loss_meter.update(loss.item(), data.size(0))
            progress_bar.set_postfix(loss=loss_meter)
            progress_bar.update(data.size(0))
        
    train_loss = loss_meter.avg
            
    print(f'Epoch {epoch} | L: {train_loss:.7f}')

            
    model.eval()
    loss_meter = AverageMeter()
    with tqdm(total=len(valid_loader.dataset)) as progress_bar:
        
        for idx, (data, target, length) in enumerate(valid_loader):
            data = data.to(device, non_blocking=True)
            length = length.to(device, non_blocking=True)
            target = target.to(device, dtype=torch.long, non_blocking=True)

            output = model(data, length)
#             loss = criterion(output, target)
            # reshape output to [Batch, NumClass] and target accordingly
            reshaped_output = output[:, -1, :].squeeze()
            reshaped_target = torch.argmax(target, dim=1)
            loss = criterion(reshaped_output, reshaped_target)

            loss_meter.update(loss.item(), data.size(0))

    val_loss = loss_meter.avg
        
    print(f'Epoch {epoch} | L: {val_loss:.7f}\n')
    
    print('Epoch %0.2i, Train Loss: %0.5f, Valid Loss: %0.5f' %
              (epoch+1, train_loss, val_loss))

torch.save(model.state_dict(), f'models/GRU_[Batch,Class]-e{num_epoch}_reshaped_output.pth')


def load_model(epoch: int, device: torch.device = 'cuda'):
    
    model = QDRNN(3, 128, 8).to(device)
    model.load_state_dict(torch.load(f'models/GRU_[Batch,Class]-e{epoch}_reshaped_output.pth'))

    return model


testset = QuickDrawDataset(test_strokes, test_label, max_len=max_seq_len)

def collate_fn(dataset):
    train_data = list(list(zip(*dataset))[0])
    labels = list(zip(*dataset))[1]
    target = torch.Tensor([np.array(l) for l in labels])
    train_data.sort(key=lambda data: len(data), reverse=True)
    data_length = [len(data) for data in train_data]
    data_length = torch.Tensor(data_length)
    train_data = rnn_utils.pad_sequence(train_data, batch_first=True, padding_value=0)
    return train_data, target, data_length

test_loader = DataLoader(testset, batch_size=128, collate_fn=collate_fn, shuffle=False, num_workers=0)

device = 'cuda'

model = load_model(10).to(device)


criterion = nn.CrossEntropyLoss()

model.eval()
loss_meter = AverageMeter()
test_acc = 0.0
for idx, (data, target, length) in enumerate(test_loader):
    data = data.to(device, non_blocking=True)
    length = length.to(device, non_blocking=True)
    target = target.to(device, dtype=torch.long, non_blocking=True)

    output = model(data, length)
#     loss = criterion(output, target)
    # reshape output to [Batch, NumClass] and target accordingly
    reshaped_output = output[:, -1, :].squeeze()
    reshaped_target = torch.argmax(target, dim=1)
    loss = criterion(reshaped_output, reshaped_target)
    
    loss_meter.update(loss.item(), data.size(0))
    
    outputs = reshaped_output > 0.0
    test_acc += (outputs == target).float().mean()
    
test_loss = loss_meter.avg

print(f'test loss:{test_loss:.5}, test_acc:{test_acc/(idx+1):.5}')