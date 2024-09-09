import os
import sys
sys.path.append(os.path.dirname(__file__))

from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch.utils.data
from torch.autograd import Variable
from pathlib import Path

# Self-defined library
from dataset import DriveDataset
from transform import RotateDiscreteAngle
from metrics.loss import DiscriminativeLoss
from ModelLoader import EmbeddingModel

# Configurations

# # for model UNet
# model_name = 'UNet_clean.pth'
# loss record during training
# txt_name = 'UNet_clean.txt'
# result_folder = '../results/patches/UNet/'
# model_type = 'InSegNN' # 'InSegNN' is correct there
# embeddingModel = EmbeddingModel(model_type = model_type)
# # model config
# config = dict()
# config['in_channels'] = 3
# config['out_channels_instance'] = 12
# config['init_features'] = 16
# config['downsample_time'] = 4
# config['use_temporal'] = False
# config['semantic_path'] = False
# dataset_time = 1

# for model InSegNN
model_name = 'InSegNN_new.pth'
# loss record during training
txt_name = 'InSegNN_new.txt'
result_folder = '../results/patches/InSegNN/'
model_type = 'InSegNN' # 'InSegNN' is correct there
embeddingModel = EmbeddingModel(model_type = model_type)
# model config
config = dict()
config['in_channels'] = 3
config['out_channels_instance'] = 12
config['init_features'] = 16
config['downsample_time'] = 3
config['use_temporal'] = True
config['temporal_model'] = 'improved'
config['time'] = 5 # this is for the model architecture when temporal_model is 'improved'
config['semantic_path'] = False
dataset_time = config['time'] # can be different with config['time'], this is for the dataset


# Other configuration
# dataset
data_dir = '../datasets/temporal_test_dataset'
usebackground = False
remove_overlay = False
time = dataset_time # can be different with config['time'], this is for the dataset
max_ins_num = 15

# transform
mean = 0
std = 1

# loss
    # disc lose hyperparameter
norm=2
disc_alpha=1
disc_beta=1
disc_gamma=0.001
delta_var=0.5
delta_dist=3

use_whole_temporal_sequence = True # Last time or whole time sequence

# train
best_loss =  np.inf
    # use continue learning rate when continue training
use_continue_lr = False
learning_rate = 0.001
batch_size = 32

use_cuda = True
continue_train = False
epoch_start = 0
epoch_end = 300

train_dataset = DriveDataset(
    root_dir=data_dir,
    start=0,
    end=7824,
    time=time,
    max_instance_number=max_ins_num,
    remove_overlay=remove_overlay,
    usebackground = usebackground,
    transform_all=
    transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        RotateDiscreteAngle(angle = [0, 90, 180, 270])
    ]),
    transform_image=
    transforms.Compose([
        transforms.Normalize(mean, std),
    ])
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

# Create model
embeddingModel.CreateModel(config, use_cuda)

# Load model and parameters if continue training
if continue_train:
    model_dir = Path('../results/checkpoints')
    model_path = model_dir.joinpath(model_name)
    param = torch.load(model_path)
    if isinstance(param, dict):
        keys_list = list(param.keys())
        if 'loss' in keys_list:
            best_loss = param['loss']
        if 'epoch' in keys_list:
            epoch_start = param['epoch'] + 1
        if 'learning_rate' in keys_list:
            if use_continue_lr:
                learning_rate = param['learning_rate']
        if  'temporal_encoder' in keys_list:
            embeddingModel.model['temporal_encoder'].load_state_dict(param['temporal_encoder'])
            embeddingModel.model['instance_decoder'].load_state_dict(param['instance_decoder'])
        else:
            embeddingModel.model.load_state_dict(param['model'])
    else:
        embeddingModel.model = param

# Loss function hyper-parameter
norm=2
disc_alpha=1
disc_beta=1
disc_gamma=0.001
delta_var=0.5
delta_dist=3
# Create loss function
criterion_disc = DiscriminativeLoss(norm=norm,
                                    alpha = disc_alpha, beta = disc_beta, gamma = disc_gamma,
                                    delta_var=delta_var,delta_dist=delta_dist,
                                    use_cuda = use_cuda)

if use_cuda:
    criterion_disc = criterion_disc.cuda()


# Optimizer
parameters_to_train = []
parameters_to_train = embeddingModel.model.parameters()

params_with_grad = filter(lambda p: p.requires_grad, parameters_to_train)

print('Create Optimzier with learning rate : ', learning_rate)
optimizer = optim.Adam(params_with_grad, lr=learning_rate)
if continue_train:
    if isinstance(param, dict):
        if 'optimizer_state_dict' in keys_list:
            optimizer.load_state_dict(param['optimizer_state_dict'])

# Create scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 mode='min',
                                                  factor=0.1,
                                                 patience=10,
                                                 verbose=True)


# Train
for epoch in range(epoch_start,epoch_end):
    print(f'epoch : {epoch}')

    # Create loss list, disc_losses are for discriminative loss, semantic_losses are for semantic loss
    disc_losses = [0.0] * len(train_loader)
    semantic_losses = [0.0] * len(train_loader)

    loss_var = [0.0] * len(train_loader)
    loss_dist = [0.0] * len(train_loader)
    loss_reg = [0.0] * len(train_loader)

    # for batched in train_dataloader:
    for batch_idx, (images, sem_labels, ins_labels, ins_numbers) in enumerate(train_loader):

        images = Variable(images)
        sem_labels = Variable(sem_labels)
        ins_labels = Variable(ins_labels)

        if use_cuda:
            images = images.cuda()
            sem_labels = sem_labels.cuda()
            ins_labels = ins_labels.cuda()
            ins_numbers = ins_numbers.cuda()

        output = embeddingModel.ForwardModel(images, type='train')

        if config['semantic_path']:
            sem_predicts = output[0]
            ins_predicts = output[1]
        else:
            ins_predicts = output

        # select last patch of temporal sequence (the latest one in timeline)
        if not use_whole_temporal_sequence:
            ins_predict = ins_predicts[:, time - 1, :, :, :]
            ins_label = ins_labels[:, time-1]
            ins_number = ins_numbers[:, time-1]
            disc_loss, l_var, l_dist, l_reg = criterion_disc(ins_predict,
                                                            ins_label,
                                                            ins_number)
        # use whole temporal sequence patches to calculate discriminative loss
        else:
            disc_loss = 0.0
            l_var = 0.0
            l_dist = 0.0
            l_reg = 0.0

            # batch, time, channel, height, width
            for t in range(time):
                disc_loss_s, l_var_s, l_dist_s, l_reg_s = criterion_disc(ins_predicts[:,t,:,:,:],
                                                                    ins_labels[:,t,:,:,:],
                                                                    ins_numbers[:,t])
                disc_loss += disc_loss_s
                l_var += l_var_s
                l_dist += l_dist_s
                l_reg += l_reg_s

            disc_loss = disc_loss / time
            l_var = l_var / time
            l_dist = l_dist / time
            l_reg = l_reg / time

        disc_loss.backward()
        optimizer.step()

        disc_losses[batch_idx] = disc_loss

        loss_var[batch_idx] = l_var
        loss_dist[batch_idx] = l_dist
        loss_reg[batch_idx] = l_reg

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx, len(train_loader),
            100. * batch_idx / len(train_loader), disc_loss.data.item()))

    disc_loss = sum(disc_losses) / len(disc_losses)

    scheduler.step(disc_loss)

    l_var = sum(loss_var) / len(loss_var)
    l_dist = sum(loss_dist) / len(loss_dist)
    l_reg = sum(loss_reg) / len(loss_reg)

    print(f'Discriminative Loss: {disc_loss:.4f}')
    print(f'l_var: {l_var:.4f}')
    print(f'l_dist: {l_dist:.4f}')
    print(f'l_reg: {l_reg:.4f}')


    model_dir = Path('../results/checkpoints')
    # append mode
    file = open(model_dir.joinpath(txt_name), "a")
    file.write('{} {:.4f} {:.4f} {:.4f} {:.4f} \n'.format(epoch, disc_loss, l_var, l_dist, l_reg))

    file.close()

    if disc_loss < best_loss:
        best_loss = disc_loss
        print('Best Model!')
        torch.save({
            'epoch': epoch,
            'model': embeddingModel.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'loss': disc_loss,
        }, model_dir.joinpath(model_name))