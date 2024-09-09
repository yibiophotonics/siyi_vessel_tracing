import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from pathlib import Path
import cv2
import os
import pandas as pd
np.random.seed(0)

from treelib import Tree
from skimage.morphology import skeletonize

from tracing.traceUtils import BuildTreeStructure, DyeGradientMap, gradient_coloring

#%%
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if len(img.shape) == 3:
            img = img[:, :, 0]
        if img is not None:
            images.append(img)
    return images

def load_images_from_folder_instance(folder):
    images = []
    file = [filename for filename in os.listdir(folder)]
    for c in range(len(file)):
        filename = str(c) + '.png'
        img = cv2.imread(os.path.join(folder, filename))
        if len(img.shape) == 3:
            img = img[:, :, 0]
        if img is not None:
            images.append(img)
    return images

def read_input_points(subject):
    config_trace = dict()
    dataframe = pd.read_excel('D:/siyi_vessle/ji_amir_resize/whole_image/summary.xlsx')
    subject_list = np.array(dataframe['Subject'])
    subject_index = np.argwhere(subject_list == subject)
    sp_row_list = np.array(dataframe['Row_sp'])[subject_index]
    sp_col_list = np.array(dataframe['Column_sp'])[subject_index]
    op_row_list = np.array(dataframe['Row_other'])[subject_index]
    op_col_list = np.array(dataframe['Column_other'])[subject_index]

    config_trace['startpoint_list'] = []
    config_trace['secondpoint_list'] = []

    for i in range(len(sp_row_list)):
    # for i in range(2,4):
        config_trace['startpoint_list'].append([sp_row_list[i][0].astype(int),sp_col_list[i][0].astype(int)])
        config_trace['secondpoint_list'].append([op_row_list[i][0].astype(int),op_col_list[i][0].astype(int)])
    return config_trace



image_dir = Path('D:/siyi_vessle/ji_amir_resize/whole_image/test/')
mask_dir = Path('D:/siyi_vessle/ji_amir_resize/whole_image/mask_2/')

predict_dir = Path('D:/siyi_vessle/ji_amir_resize/whole_image/vessel_match - Copy/prediction/')
gt_dir = Path('D:/siyi_vessle/ji_amir_resize/whole_image/vessel_match/ground_truth/')

result_dir = Path('D:/siyi_vessle/ji_amir_resize/whole_image/vessel_match - Copy/prediction/')

save_digit = 6

# index = range(4,7)
# index = range(16,17)
# index = range(1,2)
index = range(21,22)

dice_instance = np.zeros(len(index))
dice_semantic = np.zeros(len(index))
SBD = np.zeros(len(index))
number = np.zeros(len(index))

for i in range(len(index)):

    # Load Image
    image_path = image_dir.joinpath(str(index[i]) + '.tif')
    image = io.imread(image_path)

    # Load Mask
    mask_path = mask_dir.joinpath(str(index[i]) + '.png')
    mask = io.imread(mask_path)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    # Load Predict (min:0, max:255)
    instance_collection = np.array(load_images_from_folder_instance(predict_dir.joinpath(str(index[i]) + '/'))) / 255
    print('predicted instance number',instance_collection.shape[0])
    ins_number_estimate = instance_collection.shape[0]
    # Load GT (min:0, max:255, 0:foreground, 1:background)
    ins_labels = 1 - np.array(load_images_from_folder(gt_dir.joinpath(str(index[i]) + '/'))) / 255
    print('gt instance number',ins_labels.shape[0])
    ins_number = ins_labels.shape[0]

    # input start point
    config_trace = read_input_points(index[i])


    # Skeletonization
    instance_collection_skeleton = np.zeros(instance_collection.shape)
    tree_collection = []
    for j in range(ins_number_estimate):
        dilation_size = 5
        kernel1 = np.ones((dilation_size, dilation_size), np.uint8)
        vessel_dilation = cv2.dilate(instance_collection[j], kernel1, iterations=1)
        instance_collection_skeleton[j] = skeletonize(vessel_dilation)
        # build tree structure
        tree_structure = BuildTreeStructure(instance_collection_skeleton[j], config_trace['startpoint_list'][j])
        tree_collection.append(tree_structure)

    print("tree structure built")
    # add gradient color to skeleton
    length = ins_number_estimate
    gradient_map = np.zeros((length,instance_collection.shape[1], instance_collection.shape[2])).astype(float)
    centroid_map = np.zeros((length, instance_collection.shape[1], instance_collection.shape[2])).astype(float)
    dye_map = np.zeros((length, instance_collection.shape[1], instance_collection.shape[2])).astype(float)
    for j in range(len(tree_collection)):
        node_generator = tree_collection[j].tree.expand_tree(nid = 0, mode=Tree.WIDTH)
        node_list = list(node_generator)
        node_list = node_list[1:] # remove root
        for k in range(len(node_list)):
            idx = node_list[k]
            point_list_centroid = np.array(tree_collection[j].dict[str(idx)]['centroid'])
            if len(point_list_centroid) != 0:
                centroid_map[j, point_list_centroid[:, 0], point_list_centroid[:, 1]] = 1
        connect_list = np.argwhere(tree_collection[j].graph == 1)
        # print(connect_list)
        weight_graph = np.zeros(tree_collection[j].graph.shape[0]).astype(int)
        weight_graph[1] = 100
        for k in range(connect_list.shape[0]):
            parent = connect_list[k, 0]
            child = connect_list[k, 1]
            point_list_follow =np.array(tree_collection[j].connection[str(parent) + ',' + str(child)])[0] # decrease dimension
            # print(str(parent)+','+str(child))
            if len(point_list_follow) != 0:
                weight_graph[child] = weight_graph[parent] + len(point_list_follow)
                for l in range(len(point_list_follow)):
                    gradient_map[j, point_list_follow[l, 0], point_list_follow[l, 1]] = weight_graph[parent] + l + 1
        # plt.subplot(1, 2, 1)
        # plt.imshow(instance_collection[j], cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(gradient_map[j], cmap='hot')
        # plt.colorbar()
        # plt.show()
        print("gradient map of " + str(j) + " built")

        dye_map[j] = DyeGradientMap(gradient_map[j],instance_collection[j])


    # Instance Number x Height x Width
    '''
    dice_semantic[i] = 0
    dice_instance[i], ins_result_order, ins_labels_order = dice_coef(instance_collection, ins_labels,instance=True)
    SBD[i] = SymmetricBestDice(instance_collection, ins_labels)
    number[i] = np.abs(ins_number - ins_number_estimate)

    print("test:", index[i])
    print(f'semantic dice:{dice_semantic[i]:.4f}')
    print(f'instance dice:{dice_instance[i]:.4f}')
    print(f'SBD:{SBD[i]:.4f}')
    print(f'DIC:{number[i]}')

    colormap = plt.cm.Set2
    img_ins_color, ins_colormap = Coloring(ins_result_order, colormap, 'image_channel')
    img_gt_color, gt_colormap = Coloring(ins_labels_order, colormap, 'image_channel')

    # pixel_no * instance layer
    plt.clf()

    plt.subplot(1, 3, 1)

    plt.imshow(image)
    plt.axis('off')
    plt.title("dice_instance = " + str(round(dice_instance[i], 4)))

    plt.subplot(1, 3, 2)
    plt.imshow(img_gt_color)
    plt.axis('off')
    plt.title("SBD = " + str(round(SBD[i], 4)))

    plt.subplot(1, 3, 3)
    plt.imshow(img_ins_color)
    plt.axis('off')
    plt.title("DIC = " + str(number[i]))

    plt.tight_layout()

    path = result_dir.joinpath(str(index[i]) + '_results/' + str(index[i]) + '.png')
    plt.savefig(path, dpi=300)

    lines = ["test idx: " + str(i),
             "semantic dice = " + str(round(dice_semantic[i], save_digit)),
             "instance dice = " + str(round(dice_instance[i], save_digit)),
             "SBD = " + str(round(SBD[i], save_digit)),
             "DIC = " + str(round(number[i], save_digit))]

    with open(result_dir.joinpath('results.txt'), 'a') as f:
        for line in lines:
            f.write(line)
            f.write('\n')
        f.write('\n')


    plt.clf()
'''

    # root_color = \
    #     [[3, 109, 68],
    #     [222, 69, 0],
    #     [0, 40, 152],
    #     [131, 25, 96],
    #     [83, 123, 0],
    #     [190, 161, 0],
    #     [99, 46, 134],
    #     [18, 73, 108]]
    root_color = \
    [
        [0, 37, 196],
        [230, 0, 65]
    ]

    root_color = np.array(root_color) / 255

    import colorsys

    root_color_hsv = np.zeros(root_color.shape)
    for color in range(root_color.shape[0]):
        r, g, b = root_color[color]
        (h, s, v) = colorsys.rgb_to_hsv(r, g, b)
        root_color_hsv[color] = np.array([h, s, v])
        # adjust saturation
        root_color_hsv[color, 1] = root_color_hsv[color, 1] * 2
        if root_color_hsv[color, 1] > 1:
            root_color_hsv[color, 1] = 1
        # adjust value
        # root_color_hsv[color, 2] = 0.8 * root_color_hsv[color, 2]
        # return to rgb
        root_color[color] = np.array(colorsys.hsv_to_rgb(root_color_hsv[color, 0], root_color_hsv[color, 1], root_color_hsv[color, 2]))

    # tail_color = \
    #     [[255, 255, 255],
    #     [255, 255, 255],
    #     [255, 255, 255],
    #     [255, 255, 255],
    #     [255, 255, 255],
    #     [255, 255, 255],
    #     [255, 255, 255],
    #     [255, 255, 255]]

    tail_color = \
    [
        [2, 216, 255],
        [255, 195, 11]
    ]

    tail_color = np.array(tail_color) / 255


    # tail_color = \
    #     [[0, 255, 157],
    #     [255, 169, 129],
    #     [77, 124, 255],
    #     [255, 77, 196],
    #     [190, 255, 52],
    #     [255, 250, 219],
    #     [192, 92, 255],
    #     [2, 216, 255]]
    # tail_color = np.array(tail_color) / 255
    #
    # tail_color_hsv = np.zeros(tail_color.shape)
    # for color in range(tail_color.shape[0]):
    #     r, g, b = tail_color[color]
    #     (h, s, v) = colorsys.rgb_to_hsv(r, g, b)
    #     tail_color_hsv[color] = np.array([h, s, v])
    #     # adjust hue
    #     tail_color_hsv[color, 0] = tail_color_hsv[color, 0] + 0.1
    #     if tail_color_hsv[color, 0] > 1:
    #         tail_color_hsv[color, 0] = tail_color_hsv[color, 0] - 1
    #
    #     # # adjust saturation
    #     # tail_color_hsv[color, 1] = tail_color_hsv[color, 1] * 2
    #     # adjust value
    #     tail_color_hsv[color, 2] = tail_color_hsv[color, 2] * 2
    #     if tail_color_hsv[color, 2] > 1:
    #         tail_color_hsv[color, 2] = 1
    #     # return to rgb
    #     tail_color[color] = np.array(colorsys.hsv_to_rgb(tail_color_hsv[color, 0], tail_color_hsv[color, 1], tail_color_hsv[color, 2]))

    # ins_gradient_color_img = gradient_coloring(dye_map, plt.cm.hot, plt.cm.spring,plt.cm.cool , plt.cm.summer, plt.cm.autumn,plt.cm.winter)
    ins_gradient_color_img = gradient_coloring(dye_map,root_color,tail_color,background = 'white')
    # ins_gradient_color_img = gradient_coloring(dye_map,plt.cm.winter, plt.cm.spring, plt.cm.summer)
    plt.imshow(ins_gradient_color_img)
    # path = result_dir.joinpath(str(index[i]) + '_results/' + str(index[i]) + '_gradient.png')
    # plt.savefig(path, dpi=300)
    plt.imsave(result_dir.joinpath(str(index[i]) + '.png'), ins_gradient_color_img)
    plt.show()
'''
instance_mean = np.mean(dice_instance)
instance_std = np.std(dice_instance)
semantic_mean = np.mean(dice_semantic)
semantic_std = np.std(dice_semantic)
SBD_mean = np.mean(SBD)
SBD_std = np.std(SBD)
number_mean = np.mean(number)
number_std = np.std(number)


lines= ["idx list: " + str(index),
        "semantic dice = " + str(round(semantic_mean,save_digit)) + " ± " + str(round(semantic_std,save_digit)),
        "instance dice = " + str(round(instance_mean,save_digit)) + " ± " + str(round(instance_std,save_digit)),
        "SBD = " + str(round(SBD_mean,save_digit)) + " ± " + str(round(SBD_std,save_digit)),
        "DIC = " + str(round(number_mean,save_digit)) + " ± " + str(round(number_std,save_digit))]

with open(result_dir.joinpath('readme.txt'), 'a') as f:
    for line in lines:
        print(line)
        f.write(line)
        f.write('\n')
    f.write('\n')
'''



