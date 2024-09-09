import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from pathlib import Path
import torch
import pandas as pd
from skimage import io
import scipy.ndimage as ndi
from copy import copy
import matplotlib.lines as mlines
from ModelLoader import EmbeddingModel
from tracing.vesselTracing import VesselTracing
from post_processing.post import Coloring
from tracing.traceUtils import FixLeftLabel, DrawCross
np.random.seed(0)
# %%

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


if torch.cuda.is_available():
    use_cuda = True
    device = torch.device('cuda')
else:
    use_cuda = False
    device = torch.device('cpu')
embeddingModel.CreateModel(config, use_cuda)
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

# save result
result_dir = Path('../results/tracing/')
if not result_dir.exists():
    result_dir.mkdir()
# %%
use_list = True
# test subject index
# subject = 2,7,9,10,11,20 has problem
for subject in range(21):
    # Tracing Config
    config_trace = dict()

    # read Image and Mask
    config_trace['image_name'] = 'C:/Users/Administrator/OneDrive - Johns Hopkins/2023 Siyi Chen vessel tracing/whole_image/input/test/'+str(subject)+'_test.tif'
    config_trace['mask_name'] = 'C:/Users/Administrator/OneDrive - Johns Hopkins/2023 Siyi Chen vessel tracing/whole_image/mask/'+str(subject)+'_manual_main_branch.png'

    # manual set start point list 0
    # read the point list
    # read by default 1st sheet of an excel file

    config_trace['startpoint_list'] = []
    config_trace['secondpoint_list'] = []

    if use_list:
        dataframe = pd.read_excel('C:/Users/Administrator/OneDrive - Johns Hopkins/2023 Siyi Chen vessel tracing/whole_image/input/multiple start point/summary_2.xlsx')
        subject_list = np.array(dataframe['Subject'])
        subject_index = np.argwhere(subject_list == subject)
        if subject_index.size == 0:
            continue
        sp_row_list = np.array(dataframe['Row_sp'])[subject_index]
        sp_col_list = np.array(dataframe['Column_sp'])[subject_index]
        op_row_list = np.array(dataframe['Row_other'])[subject_index]
        op_col_list = np.array(dataframe['Column_other'])[subject_index]

        collect = np.array(range(len(sp_row_list)))
        vessel_no = np.array(dataframe['Vessel'][subject_index[:,0]])
        sample = np.array(dataframe['Sample'][subject_index[:,0]])
        for i in range(len(sp_row_list)):
            config_trace['startpoint_list'].append([sp_row_list[i][0].astype(int),sp_col_list[i][0].astype(int)])
            config_trace['secondpoint_list'].append([op_row_list[i][0].astype(int),op_col_list[i][0].astype(int)])

    else:

        image = io.imread(config_trace['image_name'])
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True
        point = []

        def mouse_event(event):
            print('x: {} and y: {}'.format(event.xdata, event.ydata))
            point.append([int(event.ydata),int(event.xdata)])
            plt.plot(range(int(event.xdata - 2),int(event.xdata + 2)), range(int(event.ydata - 2),int(event.ydata + 2)), '.', color = 'red')
            plt.plot(range(int(event.xdata + 2), int(event.xdata - 2),-1),range(int(event.ydata - 2), int(event.ydata + 2)), '.', color = 'red')
            fig.canvas.draw()


        fig, ax = plt.subplots(figsize = (15, 15))
        cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
        plt.axis('off')
        ax.imshow(image)
        plt.show()
        for n in range(len(point) // 2):
            config_trace['startpoint_list'].append(point[2 * n])
            config_trace['secondpoint_list'].append(point[2 * n + 1])
        collect = np.array(range(len(config_trace['startpoint_list'])))
        # draw start point annotation images
    image = io.imread(config_trace['image_name'])

    fig, ax = plt.subplots(figsize = (15, 15),num=0,clear=True)
    plt.axis('off')
    ax.imshow(image)

    print('startpoint_list', config_trace['startpoint_list'])
    print('secondpoint_list', config_trace['secondpoint_list'])

    one_side_length = 7
    for c in range(len(collect)):

        angle = np.arctan2(config_trace['secondpoint_list'][c][0]-config_trace['startpoint_list'][c][0],
                               config_trace['secondpoint_list'][c][1]-config_trace['startpoint_list'][c][1]) + np.pi/4

        xmin, xmax, ymin, ymax = DrawCross(config_trace['startpoint_list'][c][1],config_trace['startpoint_list'][c][0],one_side_length,angle)
        l = mlines.Line2D([xmin, xmax], [ymin, ymax], c = 'b',linewidth = 1)
        ax.add_line(l)
        xmin, xmax, ymin, ymax = DrawCross(config_trace['startpoint_list'][c][1], config_trace['startpoint_list'][c][0],
                                           one_side_length, angle + np.pi/2)
        l = mlines.Line2D([xmin, xmax], [ymin, ymax], c = 'b', linewidth = 1)
        ax.add_line(l)

        plt.arrow(config_trace['startpoint_list'][c][1], config_trace['startpoint_list'][c][0],
                    1.5*(config_trace['secondpoint_list'][c][1]-config_trace['startpoint_list'][c][1]),
                    1.5*(config_trace['secondpoint_list'][c][0]-config_trace['startpoint_list'][c][0]),
                    shape='full',color = 'b', length_includes_head=True, head_width=one_side_length, head_length=one_side_length,linewidth=1)
    config_trace['frame_folder'] = result_dir.joinpath(str(subject) + '/frame/')
    if not config_trace['frame_folder'].exists():
        config_trace['frame_folder'].mkdir(parents=True)
    path = result_dir.joinpath(str(subject) + '/startpoint/')
    if not path.exists():
        path.mkdir(parents = True)
    path = path.joinpath(str(subject) + '_startpoint.tif')
    plt.savefig(path, dpi = 300)

# 16
# config_trace['startpoint_list'] = [[166, 453],[331, 444],[161, 464],[325, 468]]
# config_trace['secondpoint_list'] = [[134, 427],[354, 425],[141, 453],[355, 453]]
# config_trace['startpoint_list'] = [[312,418],[289,504],[134, 427],[331, 444],[161, 464],[325, 468]]
# config_trace['secondpoint_list'] = [[323,489],[308,514],[127, 411],[354, 425],[141, 453],[355, 453]]
# config_trace['startpoint_list'] = [[208,496],[134, 427],[161, 464],[331, 444],[339,483],[304,512],[325, 468],[302,458]]
# config_trace['secondpoint_list'] = [[182,498],[127, 411],[141, 453],[354, 425],[373,470],[341,526],[355, 453],[312,426]]

# 01,2,345,67

# 1
# config_trace['startpoint_list'] = [[300,79],[306,64],[206,100],[202,110]]
# config_trace['secondpoint_list'] = [[307,87],[317,69],[194,109],[192,122]]


    # Dataset
    config_trace['crop_size'] = 96
    mean = 0
    std = 1
    config_trace['transforms'] =  transforms.Compose([transforms.Normalize(mean, std)])

    # Control Temporal Learning
    config_trace['time'] = 5
    # Control global_iteration_option, False means control group
    config_trace['dynamic_probability_map'] = True
    # Control spatial shift
    config_trace['spatial_shift_multisampling'] = True

    # configuration about drawing figure
    # Draw scatter (empty means no scatter)
    config_trace['select_frame'] = []
    config_trace['manifold_method'] = 'MDS'

    config_trace['arrow_line_width'] = 2
    config_trace['frame_line_width'] = 2.5
    config_trace['cross_one_side_length'] = 7

    # DL model
    config_trace['model'] = embeddingModel
    config_trace['use_cuda'] = use_cuda

    # Post
    config_trace['cluster_method'] = 'meanshift'
    config_trace['ratio'] = 1/4

    # Binary Vessel Map
    config_trace['binary threshold'] = 0.6 # the probabilty of one pixel predicted as instance of one tree
    # first dilation then erosion as post process
    config_trace['dilation size'] = 16
    config_trace['erosion size'] = 14

    # Other
    config_trace['count_limit'] = 1000 # iteration count threshold, stop after xx times iterations
    config_trace['step'] = 10 # sample movement step

    # Save frame
    # config_trace['frame_folder'] = result_dir.joinpath(str(subject) + '/frame/')
    # if not config_trace['frame_folder'].exists():
    #     config_trace['frame_folder'].mkdir(parents = True)
    #
    # config_trace['weight_folder'] = result_dir.joinpath(str(subject) + '/weight/')
    # if not config_trace['weight_folder'].exists():
    #     config_trace['weight_folder'].mkdir(parents = True)


    config_trace['step_folder'] = result_dir.joinpath(str(subject) + '/step/')
    if not config_trace['step_folder'].exists():
        config_trace['step_folder'].mkdir(parents = True)

    colormap = plt.cm.Set2

    # Construct Tracing Class
    vesselTrace = VesselTracing(config_trace)
    # Start tracing
    global_binary_map, global_binary_map_post = vesselTrace.TracingSelectedVessel()
    vessel_information = vesselTrace.vessel
    image_class = vesselTrace.image_class

    # Generate vessel mask from instance segmentation
    vessel_mask = np.zeros((global_binary_map.shape[1], global_binary_map.shape[2]))
    for i in range(len(vessel_information['value'])):
        vessel_mask[global_binary_map[i] > 0] = 1

    # Compare Vessel Mask with GT
    mask = io.imread(config_trace['mask_name'])
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    mask[mask > 0] = 1

    mask_difference = mask - vessel_mask
    global_binary_map_fill = copy(global_binary_map)
    # assign region in mask difference to the nearest vessel
    labeled_array, num_features = ndi.label(mask_difference)
    for i in range(1, num_features + 1):
        target_map = np.zeros_like(mask_difference)
        target_map[labeled_array == i] = 1
        global_binary_map_fill = FixLeftLabel(global_binary_map_fill, target_map)

    img_ins_color, ins_colormap = Coloring(global_binary_map, colormap, 'image_channel')
    img_ins_color_post, ins_colormap_post = Coloring(global_binary_map_post, colormap, 'image_channel')
    img_ins_color_fill, ins_colormap_fill = Coloring(global_binary_map_fill, colormap, 'image_channel')

    row = 1
    column = 2
    fig = plt.figure(num=0,clear=True)
    plt.subplot(row, column, 1)
    plt.imshow(image_class.image)
    plt.axis('on')
    plt.title('Input Intensity')

    plt.subplot(row, column, 2)
    plt.imshow(img_ins_color_fill)
    plt.axis('off')
    plt.title('Output Instance Segmentation')

    path1 = result_dir.joinpath(str(subject) + '/output/')
    if not path1.exists():
        path1.mkdir(parents = True)
    path_whole = path1.joinpath('whole_image.png')
    plt.imsave(path_whole, img_ins_color_fill)
    path_display = path1.joinpath('display.png')
    plt.savefig(path_display, dpi = 300)

    path2 = result_dir.joinpath(str(subject) + '/map/')
    if not path2.exists():
        path2.mkdir(parents=True)
    for i in range(len(collect)):
        nameid = str(vessel_no[i]) + '_' + str(sample[i]) # str(collect[i]) for single start point # 
        path_instance1 = path2.joinpath(nameid + '.png')
        path_instance2 = path2.joinpath(nameid + '_post.png')
        path_instance3 = path2.joinpath(nameid + '_fill.png')
        plt.imsave(path_instance1, global_binary_map[i], cmap = 'gray')
        plt.imsave(path_instance2, global_binary_map_post[i], cmap = 'gray')
        plt.imsave(path_instance3, global_binary_map_fill[i], cmap = 'gray')
    print("save results of ", str(subject))



