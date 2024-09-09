import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import openpyxl
import torch.nn.functional
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

np.random.seed(0)

from post_processing.post import PostProcess, Coloring, ChannelToLabel
from metrics import MatchedDiceScore, ScaledSymmetricBestDice, DIC

from dataset import DriveDataset
from ModelLoader import EmbeddingModel

# # for model UNet
# model_name = 'UNet_clean.pth'
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
model_name = 'InSegNN_clean.pth'
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

# other configurations for both models
if torch.cuda.is_available():
    use_cuda = True
    device = torch.device('cuda')
else:
    use_cuda = False
    device = torch.device('cpu')
result_dir = Path(result_folder)
if not result_dir.exists():
    result_dir.mkdir()

# config list
# dataset
data_dir = '../patches_input'
usebackground = False
remove_overlay = False
time = dataset_time # can be different with config['time'], this is for the dataset

patch_list = np.loadtxt(data_dir + '/' + 'test.txt')
_,branch_id = np.unique(patch_list[:,2:4],return_index=True,axis=0)

max_ins_num = 18
batch_size = 32

# explanation of predict_target_only_in_latest_frame
# if true, only calculate the SBD of target in the latest frame
# if false, put target in all frames then find best SBD
predict_target_only_in_latest_frame = True

# record the ID of patches and the position in temporal sequqnce
# that you want to draw figrues or scatter plot
# draw_id_list and draw_time_list should have the same length
# if no anything in the list, all patches will be drawn
draw_id_list = []
# previous time is negative, latest time is 0
draw_time_list = []
# draw figures containing 1. images, 2. ground truth, 3. prediction
draw_figure = True
# draw scatter plot of the embedding (marked color according to the ground 1. truth and 2. prediction)
draw_scatter = False

# transform
mean = 0
std = 1

# post_processing parameters
# MDS TSNE UMAP LLE
manifold_method = 'MDS'
cluster_method = 'meanshift'
bandwidth_ratio = 1 / 4
# cluster_method = 'AffinityPropagation'
# cluster_method = 'DBSCAN'
# cluster_method = 'OPTICS'
# cluster_method = 'Birch'
# cluster_method = 'AgglomerativeClustering_ward'


model = embeddingModel.CreateModel(config, use_cuda = use_cuda)
# Load model
model_dir = Path('../results/checkpoints/')
model_path = model_dir.joinpath(model_name)
print('Load model from {}'.format(model_path))
param = torch.load(model_path,map_location=device)

if isinstance(param, dict):
    keysList = list(param.keys())
    embeddingModel.model.load_state_dict(param['model'])
else:
    embeddingModel.model = param
    
# create excel
workbook = openpyxl.Workbook()
#workbook = openpyxl.load_workbook(result_dir.joinpath('statistics result of patches.xlsx'))
sheet = workbook.active
sheet['A1'] = 'Patch ID'
sheet['B1'] = 'Position in temporal sequence'
sheet['C1'] = 'Instance number'
sheet['D1'] = 'DIC'
sheet['E1'] = 'SBD'
sheet['F1'] = 'Dice_pred'
sheet['G1'] = 'Dice_true'
sheet['H1'] = 'Scaled SBD'
sheet['I1'] = 'Scaled dice_pred'
sheet['J1'] = 'Scaled dice_true'

fig = plt.figure()
# %%
for branch in range(423,len(branch_id)-1):
    start = branch_id[branch]
    end = branch_id[branch+1] # to test
    # end = 2388
    # start = 2389
    # end = 4828
    # start = 4829
    # end = 7543
    # start = 7544
    # end = 10153
    
    # start = 0
    # end = 889
    # start = 890
    # end = 1862
    # start = 1863
    # end = 3882
    # start = 3883
    # end = 5882
    # start = 5883
    # end = 8059
    # start = 8060
    # end = 10153
    
    test_dataset = DriveDataset(
        root_dir = data_dir,
        start = start,
        end = end,
        time = time,
        max_instance_number = max_ins_num,
        usebackground = usebackground,
        remove_overlay = remove_overlay,
        transform_all =
        transforms.Compose([
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomVerticalFlip(0.5),
            # RotateDiscreteAngle(angle = [0, 90, 180, 270])
        ]),
        transform_image =
        transforms.Compose([
            transforms.Normalize(mean, std),
        ])
    )
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    
    # Inference
    k = 0
    for batch_idx, (image, sem_label, ins_label, ins_number) in enumerate(test_loader):
        if use_cuda:
            image = image.cuda()
        ins_pred_ = embeddingModel.ForwardModel(image, type = 'test')
        if k == 0:
            images = image.cpu().numpy()
            sem_labels = sem_label.numpy()
            ins_labels = ins_label.numpy()
            ins_numbers = ins_number.numpy()
            ins_pred = ins_pred_.detach().cpu().numpy()
        images = np.concatenate([images, image.cpu().numpy()], axis = 0)
        sem_labels = np.concatenate([sem_labels,sem_label.numpy()], axis = 0)
        ins_labels = np.concatenate([ins_labels,ins_label.numpy()], axis = 0)
        ins_numbers = np.concatenate([ins_numbers,ins_number.numpy()], axis = 0)
        ins_pred = np.concatenate([ins_pred,ins_pred_.detach().cpu().numpy()], axis = 0)
        k += 1
    # for each layer, 0: background, 1: target
    # batch, time, channel, height, width
    mask = sem_labels[:,:,1,:,:].astype(bool)

    
    if predict_target_only_in_latest_frame:
        circulation = 1
    else:
        circulation = time # from latest time (0) to most previous time (-t)
    
    for i in range(test_dataset.__len__()):
        for j in range(circulation):
            print('[', i, '/', test_dataset.__len__(),
                  '], Patch ID:', test_dataset.list[i] - j,
                  ', Position in temporal sequence: ', -j)
            image_piece_post = PostProcess(mask[i, time - (j + 1)], ins_pred[i, time - (j + 1)])
    
            # need to be known if use K-Means
            # image_piece_post.ins_number = ins_numbers[i, time - (j + 1)]
    
            try:
                label_predict = image_piece_post.Clustering(method = cluster_method,
                                                        embedding_type = 'original',
                                                        ratio = bandwidth_ratio)
            except:
                print('Clustering error')
                continue
            instance_collection = image_piece_post.InstanceLayer()
            img_ins = image_piece_post.InstanceMap()
            ins_number_estimate = image_piece_post.ins_number
    
            SBD, SBD_scaled, \
            dice_pred, dice_true, dice_pred_scale, dice_true_scale, \
            best_dice_pred, best_dice_true, \
            pred_area_p, pred_area_t, true_area_t, true_area_p \
                = ScaledSymmetricBestDice(instance_collection,
                                          ins_labels[i,time - (j + 1), :ins_numbers[i, time - (j + 1)]])
    
            dic = DIC(ins_number_estimate, ins_numbers[i, time - (j + 1)])
    
            sheet.append([test_dataset.list[i] - j, -j, ins_number_estimate,
                          dic, SBD, dice_pred, dice_true,
                          SBD_scaled, dice_pred_scale, dice_true_scale])
    
            if draw_id_list.__len__() == 0 or (test_dataset.list[i] - j in draw_id_list and -j in draw_time_list):
                dice_instance, ins_result_order, ins_labels_order = MatchedDiceScore(instance_collection,
                                                                                     ins_labels[i, time - (j + 1),
                                                                                     :ins_numbers[i, time - (j + 1)]],
                                                                                     match=False)
                colormap = plt.cm.Set2
                img_ins_color, ins_colormap = Coloring(ins_result_order, colormap, 'image_channel')
                img_gt_color, gt_colormap = Coloring(ins_labels_order, colormap, 'image_channel')
    
                if draw_figure:
                    # pixel_no * instance layer
                    plt.clf()
    
                    plt.subplot(1, 3, 1)
                    # (RGB channel, height, width) to (height, width, RGB channel)
                    plt.imshow(images[i, time - (j + 1), :].transpose(1, 2, 0))
                    plt.axis('off')
                    plt.title("images")
    
                    plt.subplot(1, 3, 2)
                    plt.imshow(img_gt_color)
                    plt.axis('off')
                    plt.title("Ground Truth, SBD = " + str(round(SBD, 4)))
    
                    plt.subplot(1, 3, 3)
                    plt.imshow(img_ins_color)
                    plt.axis('off')
                    plt.title("Prediction, DIC = " + str(dic))
                    path = result_dir.joinpath(
                        str(test_dataset.list[i] - j) + '_' + str(-j) + '_figure.png')
                    plt.savefig(path, dpi = 300)
    
                if draw_scatter:
                    ins_labels_order_list = ChannelToLabel(ins_labels_order)
                    ins_result_order_list = ChannelToLabel(ins_result_order)
                    # embedding_2d normalization to -1 ~ 1
                    embedding_2d = image_piece_post.Visualization(method=manifold_method)
                    embedding_2d[:, 0] = (embedding_2d[:, 0] - np.min(embedding_2d[:, 0])) / (
                                np.max(embedding_2d[:, 0]) - np.min(embedding_2d[:, 0]))
                    embedding_2d[:, 1] = (embedding_2d[:, 1] - np.min(embedding_2d[:, 1])) / (
                                np.max(embedding_2d[:, 1]) - np.min(embedding_2d[:, 1]))
    
                    common_instance_number = np.max([ins_number_estimate, ins_numbers[i, time - (j + 1)]])
    
                    plt.clf()
                    # scatter plot marked by ground truth colormap
                    ax1 = plt.subplot(1, 2, 1)
                    for k in range(common_instance_number):
                        plt.scatter(embedding_2d[ins_labels_order_list == k + 1, 0],
                                    embedding_2d[ins_labels_order_list == k + 1, 1],
                                    s = 0.5,
                                    c = [np.array(gt_colormap[k][:3])] * len(embedding_2d[ins_labels_order_list == k + 1, 0]))
                    ax1.set_aspect(1)
                    plt.title("Ground Truth")
                    # hide x y axis
                    plt.xticks([])
                    plt.yticks([])
    
                    # scatter plot marked by prediction colormap (which is the result of clustering)
                    ax2 = plt.subplot(1, 2, 2)
                    for k in range(common_instance_number):
                        plt.scatter(embedding_2d[ins_result_order_list == k + 1, 0],
                                    embedding_2d[ins_result_order_list == k + 1, 1],
                                    s=0.5, c=[np.array(ins_colormap[k][:3])] * len(
                                embedding_2d[ins_result_order_list == k + 1, 0]))
                    ax2.set_aspect(1)
                    plt.title("Prediction")
                    # hide x y axis
                    plt.xticks([])
                    plt.yticks([])
    
                    path = result_dir.joinpath(
                        str(test_dataset.list[i] - j) + '_' + str(-j) + '_scattering.png')
                    plt.savefig(path, dpi=300)
    
    workbook.save(result_dir.joinpath('statistics result of patches.xlsx'))