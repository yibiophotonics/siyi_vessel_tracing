import numpy as np
import pandas as pd
np.random.seed(0)
from skimage.morphology import skeletonize
from scipy import signal

import scipy.ndimage as ndi
from .vesselTree import VesselTree

def MaxConnectedRegion(image):
    labeled_array, num_features = ndi.label(image)
    if num_features > 1:
        single_layer = np.zeros_like(image)
        max_area = 0
        for j in range(1, num_features + 1):
            single_layer[labeled_array == j] = 1
            area = np.sum(single_layer)
            if area > max_area:
                max_area = area
                label = j
            single_layer = np.zeros_like(image)
        image = np.zeros_like(image)
        image[labeled_array == label] = 1
    return image

def BuildTreeStructure(skeleton, sp):
    vessel_tree = VesselTree()  # root node, no anything in it
    conv = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    conv_skeleton = signal.convolve2d(skeleton, conv, boundary = 'fill', mode = 'same')
    map = conv_skeleton * skeleton

    # find end tail of tree strcture
    end_point_list = np.argwhere(map == 2)
    start_point = np.expand_dims(sp, axis = 0)
    start_point = np.repeat(start_point, end_point_list.shape[0], axis = 0)

    difference = start_point - end_point_list
    # find the distance between start point and end point
    distance = np.sqrt(np.sum(difference ** 2, axis=1))
    index = np.argsort(distance)
    end_point_list = end_point_list[index, :] # ascending order

    tree_start_point = end_point_list[0]

    centroid, cluster = MergeCentroid(map)
    done_map = np.zeros_like(map)
    done_map[tree_start_point[0], tree_start_point[1]] = 1
    vessel_tree.TraceTree(tree_start_point, 0, map, done_map, centroid, cluster) # parent = 0
    #vessel_tree.tree.show()

    return vessel_tree

def MergeCentroid(map):
    # Find Centroid of bifurcation point
    centroid_map = np.zeros_like(map)
    centroid_map[map >= 4] = 1
    labeled_array, num_features = ndi.label(centroid_map)
    cluster = []
    centroid = []
    for i in range(num_features):
        this_cluster = np.argwhere(labeled_array == i + 1)
        cluster.append(this_cluster)
        mean = np.mean(this_cluster, axis = 0)
        mean = np.expand_dims(mean, axis = 0)
        mean = np.repeat(mean, this_cluster.shape[0], axis = 0)

        difference = mean - this_cluster
        distance = np.sqrt(np.sum(difference ** 2, axis = 1))
        index = np.argsort(distance)
        this_cluster = this_cluster[index, :]  # ascending order

        centroid.append(this_cluster[0])
    # centroid num, x, y
    # cluster num, pixel num in cluster, x, y
    return centroid, cluster

def DrawCross(x, y, one_side_length, angle):
    # Draw figure use
    xmin = x - one_side_length * np.cos(angle)
    xmax = x + one_side_length * np.cos(angle)
    ymin = y - one_side_length * np.sin(angle)
    ymax = y + one_side_length * np.sin(angle)
    return xmin, xmax, ymin, ymax

def SelectedConnectedRegion(image, bias_point, sp_point, size):
    labeled_array, num_features = ndi.label(image)
    point_local = GlobalToLocal(sp_point, bias_point, size) # order； global_point, global_reference
    if num_features > 1:
        label = labeled_array[point_local[0], point_local[1]]
        image = np.zeros_like(image)
        image[labeled_array == label] = 1
    return image

def GlobalToLocal(global_point, global_reference, size):
    local_point = np.array([global_point[0] - global_reference[0] + round(size/2),
                            global_point[1] - global_reference[1] + round(size/2)])
    return local_point

def MergeProbabilityMap(option, whole, partial, slicer_row, slicer_column):

    size = slicer_row.stop - slicer_row.start
    whole_enlarge = np.pad(whole, ((size, size), (size, size)), 'constant', constant_values = 0)
    slicer_row = slice(slicer_row.start + size, slicer_row.stop + size, 1)
    slicer_column = slice(slicer_column.start + size, slicer_column.stop + size, 1)

    if option == False:
        whole_enlarge[slicer_row, slicer_column] = partial
    else:
        whole_enlarge[slicer_row, slicer_column] = whole_enlarge[slicer_row,slicer_column] + partial
    whole_merge = whole_enlarge[size:-size, size:-size]
    return whole_merge

def FindEndPoint(global_binary_map_post, start_point, stop_point_list): #startpoint is initial sp
    skeleton = skeletonize(global_binary_map_post)
    conv = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    conv_skeleton = signal.convolve2d(skeleton, conv, boundary = 'fill', mode = 'same')
    map = conv_skeleton * skeleton
    end_point_list = np.argwhere(map == 2)

    start_point = np.expand_dims(start_point, axis=0)
    start_point = np.repeat(start_point, end_point_list.shape[0], axis= 0)


    difference = start_point - end_point_list
    distance = np.sqrt(np.sum(difference ** 2, axis=1))
    index = np.argsort(distance)
    distance_sort = distance[index]
    end_point_list = end_point_list[index, :] #ascending order

    if stop_point_list:
        delete_list = []
        for i in range(len(end_point_list)):
            end_point = end_point_list[i, :]
            end_point = np.expand_dims(end_point, axis = 0)
            end_point = np.repeat(end_point, len(stop_point_list), axis = 0)
            difference = end_point - stop_point_list
            distance_to_stop = np.sqrt(np.sum(difference ** 2, axis = 1))
            # delete end point close to stop point list
            # (including detected point, initial start point and so on)
            if np.min(distance_to_stop) < 10:
                delete_list.append(i)
        end_point_list = np.delete(end_point_list, delete_list, axis = 0)
        distance_sort = np.delete(distance_sort, delete_list, axis = 0)

    return end_point_list, distance_sort

def FixLeftLabel(ins_collection, target_map):
    for i in range(ins_collection.shape[0]):
        reference = ins_collection[i,:,:]
        reference_end_point_num = EndPointNum(reference)
        merge = reference + target_map
        merge = merge.astype(int)
        merge_end_point_num = EndPointNum(merge)
        if merge_end_point_num < reference_end_point_num:
            reference_fill_holes = ndi.binary_fill_holes(reference, structure=np.ones((3, 3))).astype(int)
            merge_fill_holes = ndi.binary_fill_holes(merge, structure=np.ones((3, 3))).astype(int)

            if np.max(np.unique(merge_fill_holes - (reference_fill_holes + target_map))) == 0:
                ins_collection[i,:,:] = merge
        return ins_collection











def read_input_points(subject):
    config_trace = dict()
    dataframe = pd.read_excel('../results/Tracing_algorithm/input/summary.xlsx')
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

def EndPointNum(image):
    from skimage.morphology import skeletonize
    from scipy import signal
    skeleton = skeletonize(image)
    conv = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    conv_skeleton = signal.convolve2d(skeleton, conv, boundary='fill', mode='same')
    map = conv_skeleton * skeleton
    end_point_num = len(np.argwhere(map == 2))
    return end_point_num


def DyeGradientMap(gradient_map,segmentation_map):
    dye_map = np.zeros_like(gradient_map)
    point_list = np.argwhere( np.invert(gradient_map.astype(bool)) * segmentation_map.astype(bool) > 0)
    for i in range(point_list.shape[0]):
        nearest_point = FindNearestPoint(point_list[i],gradient_map)
        value = gradient_map[nearest_point[0],nearest_point[1]]
        dye_map[point_list[i][0],point_list[i][1]] = value
    dye_map = dye_map + gradient_map
    return dye_map

def FindNearestPoint(point, map):
    radius = -1
    list = np.array([])
    while list.shape[0] == 0:
        radius += 2
        region_row = slice(point[0] - radius, point[0] + radius, 1)
        region_col = slice(point[1] - radius, point[1] + radius, 1)
        region = map[region_row, region_col]
        list = np.argwhere(region > 0)

    list_global = Local2Global_list(list, point, 1 + 2 * radius)
    nearest_point = lambda x, X: np.argmin(np.sum((x - X) ** 2, axis=1))
    return list_global[nearest_point(list_global, point)]

def Local2Global(local_point, global_reference, size): # single point
    gloabl_point = np.array([local_point[0] + global_reference[0] - round(size/2), local_point[1] + global_reference[1] - round(size/2)])
    return gloabl_point

def Local2Global_list(local_point_list, global_reference, size): # single point

    gloabl_point = local_point_list + global_reference - np.floor(size/2)
    return gloabl_point.astype(int)

def gradient_coloring_old(mask,colormap1, colormap2):
    # mask Channel * H * W
    ins_color_img = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
    n_ins = mask.shape[0]
    mask = mask.transpose(1,2,0)
    # mask is numpy and H * W * Channel
    for i in range(n_ins):
        max_value = int(np.max(mask[:,:,i]))
        if max_value != 0:
            if i % 2 == 0:
                colors = [colormap1(each/max_value) for each in range(0,max_value + 1)]
            elif i % 2 == 1:
                colors = [colormap2(each/max_value) for each in range(0,max_value + 1)]
            # elif i % 6 == 2:
            #     colors = [colormap3(each/max_value) for each in range(0,max_value + 1)]
            # elif i % 6 == 3:
            #     colors = [colormap4(each/max_value) for each in range(0,max_value + 1)]
            # elif i % 6 == 4:
            #     colors = [colormap5(each/max_value) for each in range(0,max_value + 1)]
            # elif i % 6 == 5:
            #     colors = [colormap6(each/max_value) for each in range(0,max_value + 1)]
            point_list = np.argwhere(mask[:,:,i] != 0)
            corresponding_value = mask[point_list[:,0],point_list[:,1],i]
            for j in range(point_list.shape[0]):
                ins_color_img[point_list[j,0],point_list[j,1],:] =\
                    (np.array(colors[corresponding_value[j].astype(int)][:3]) * 255).astype(np.uint8)
    return ins_color_img

def gradient_coloring(mask,root_color,tail_color,background = 'black'):
    # mask Channel * H * W
    if background == 'black':
        ins_color_img = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
    elif background == 'white':
        ins_color_img = np.ones((mask.shape[1], mask.shape[2], 3), dtype=np.uint8) * 255
    n_ins = mask.shape[0]
    mask = mask.transpose(1,2,0)
    # mask is numpy and H * W * Channel
    n_color = np.array(root_color).shape[0]
    for i in range(n_ins):
        max_value = int(np.max(mask[:,:,i]))
        if max_value != 0:
            index = i % n_color
            colors = [intercolor(np.array(root_color[index])[0:3],tail_color[index][0:3],each/max_value) for each in range(0,max_value + 1)]
            point_list = np.argwhere(mask[:,:,i] != 0)
            corresponding_value = mask[point_list[:,0],point_list[:,1],i]
            for j in range(point_list.shape[0]):
                ins_color_img[point_list[j,0],point_list[j,1],:] =\
                    (np.array(colors[corresponding_value[j].astype(int)][:3]) * 255).astype(np.uint8)
    return ins_color_img

def intercolor(color1, color2, ratio):
    # gamma = 1
    # gamma = 1/4
    gamma = 1/2
    color1 = np.power(color1,gamma)
    color2 = np.power(color2,gamma)
    return np.power(color1 * (1 - ratio) + color2 * ratio,1/gamma)