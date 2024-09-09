import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import numpy as np
import torch
import cv2
import os
from treelib import Tree
from post_processing.post import PostProcess, Coloring, ChannelToLabel
from skimage.morphology import skeletonize
from .inputs import Image
from copy import copy
from .traceUtils import MergeProbabilityMap, FindEndPoint, BuildTreeStructure,\
                        DrawCross, MaxConnectedRegion, SelectedConnectedRegion
from metrics import MatchedDiceScore

class VesselTracing():
    def __init__(self,config):
        self.config = config
        self.size = self.config['crop_size']
        self.step = self.config['step'] # movement length of select frame
        self.threshold = self.config['binary threshold'] # vessel value map
        self.dilation_size = self.config['dilation size']
        self.erosion_size = self.config['erosion size']

        self.count = 0
        self.stop_list = []
        self.frame_count = 0
        self.Initialize()

    def Initialize(self): # Initialize the image and start point
        keysList = list(self.config.keys())
        if 'mask_name' in keysList:
            mask_name = self.config['mask_name']
        else:
            mask_name = None
        self.image_class = Image(self.config['image_name'], mask_name = mask_name, transforms = self.config['transforms'])

    def TracingSelectedVessel(self):
        vessel = dict()
        vessel['value'] = np.zeros((len(self.config['startpoint_list']), self.image_class.image.shape[0], self.image_class.image.shape[1]))
        vessel['weight'] = np.zeros((len(self.config['startpoint_list']), self.image_class.image.shape[0], self.image_class.image.shape[1]))
        global_binary_map = np.zeros((len(self.config['startpoint_list']), self.image_class.image.shape[0], self.image_class.image.shape[1]))
        global_binary_map_post = np.zeros((len(self.config['startpoint_list']), self.image_class.image.shape[0], self.image_class.image.shape[1]))

        for i in range(len(self.config['startpoint_list'])):
            # self.count = 0
            self.stop_list = []
            self.is_root = True
            start_point = np.asarray(self.config['startpoint_list'][i])
            second_point = np.asarray(self.config['secondpoint_list'][i])
            direction_vector = second_point - start_point
            # Normalize the direction vector
            direction_vector = direction_vector / np.sqrt(np.sum(direction_vector ** 2))
            vessel['value'][i], vessel['weight'][i], = self.TracingSingleVessel(vessel['value'][i], vessel['weight'][i],
                                                                            start_point, direction_vector, start_point, i)
            global_probability_map, global_binary_map[i], global_binary_map_post[i] = \
                self.BinarizeVesselMap(vessel['value'][i], vessel['weight'][i])

        self.vessel = vessel
        return global_binary_map, global_binary_map_post

    def TracingSingleVessel(self, vessel_value, vessel_weight, sp,
                            direction_vector, initial_start_point, vessel_index):

        # defence: avoid large iteration times
        #print("count:", self.count)
        if self.count > self.config['count_limit']:
            print("time limit")
            return vessel_value, vessel_weight

        # temporal memory learning
        if not self.is_root:
            # acuiqre global binary map with connected region
            global_probability_map, global_binary_map, global_binary_map_post = self.BinarizeVesselMap(vessel_value, vessel_weight)
            skeleton = skeletonize(global_binary_map_post)
            # build the tree by
            current_tree = BuildTreeStructure(skeleton, initial_start_point)

            # search point list in the tree, relevant_point_list without input sp
            if self.config['time'] > 1:
                relevant_point_list = self.SearchPointInTreeForTemporalSequence(current_tree, sp)
            else:
                relevant_point_list = []
        else:
            relevant_point_list = []

        # create temporal sequence
        region_row, region_column, label_predict, instance_collection, img_ins, ins_number_estimate, postProcess \
            = self.AccquireResults(sp, relevant_point_list)

        # add spatial shift
        if self.config['spatial_shift_multisampling']:
            # - upper, + down; - left, + right
            # upper, down, left, right
            sp_shift = []
            row = [-self.step, self.step, 0, 0]
            column = [0, 0, -self.step, self.step]
            sp_shift.append([sp[0] - self.step, sp[1]])
            sp_shift.append([sp[0] + self.step, sp[1]])
            sp_shift.append([sp[0], sp[1] - self.step])
            sp_shift.append([sp[0], sp[1] + self.step])

            region_row_shift = []
            region_column_shift = []
            instance_collection_shift = []
            ins_number_estimate_shift = []
            if self.frame_count in self.config['select_frame']:
                postProcessShift = []

            img_ins_shift = []
            # Through deep learning to acquire prediction
            for i in range(0, 4):
                region_row_shift_single, region_column_shift_single, label_predict_shift_single, \
                instance_collection_shift_single, img_ins_shift_single, ins_number_estimate_shift_single, postProcessShiftSingle \
                    = self.AccquireResults(sp_shift[i], relevant_point_list)

                region_row_shift.append(region_row_shift_single)
                region_column_shift.append(region_column_shift_single)
                instance_collection_shift.append(instance_collection_shift_single)
                ins_number_estimate_shift.append(ins_number_estimate_shift_single)

                img_ins_shift.append(img_ins_shift_single)
                if self.frame_count in self.config['select_frame']:
                    postProcessShift.append(postProcessShiftSingle)

        # select target instance from all instances
        if not self.is_root:
            # Method: max overlay region with known instance mask
            target_instance = self.ExtractTargetInstance(global_binary_map, region_row, region_column,
                          instance_collection, ins_number_estimate)

            if self.config['spatial_shift_multisampling']:
                target_instance_shift = []
                for i in range(0, 4):
                    target_instance_shift_single = self.ExtractTargetInstance(global_binary_map, region_row_shift[i],
                                                                              region_column_shift[i], instance_collection_shift[i],
                                                                              ins_number_estimate_shift[i])
                    target_instance_shift.append(target_instance_shift_single)

        else:
            # parent == 0
            # Method: centor point's instance
            core_label = img_ins[int(self.size / 2), int(self.size / 2)]
            target_instance = np.zeros_like(img_ins)
            target_instance[img_ins == core_label] = 1

            if self.config['spatial_shift_multisampling']:
                target_instance_shift = []
                for i in range(0, 4):
                    core_label_shift_single = img_ins_shift[i][int(self.size / 2 - row[i]), int(self.size / 2 - column[i])]
                    target_instance_shift_single = np.zeros_like(img_ins_shift[i])
                    target_instance_shift_single[img_ins_shift[i] == core_label_shift_single] = 1
                    target_instance_shift.append(target_instance_shift_single)

        # Draw step1 and 2
        # Step 1: Show current branch tree and three kinds of patch frames in different color
        print("frame id:", self.frame_count)
        if not self.is_root:
            mask = global_binary_map_post
        else:
            mask = np.zeros_like(vessel_value)

        #self.DrawStep1(mask, vessel_index, sp, relevant_point_list, sp_shift)

        embedding2d_collection = []
        if self.frame_count in self.config['select_frame']:
            embedding2d_collection.append(postProcess.Visualization(method = self.config['manifold_method']))
            for i in range(0, 4):
                embedding2d_collection.append(postProcessShift[i].Visualization(method = self.config['manifold_method']))

        #self.DrawStep2(instance_collection, instance_collection_shift, embedding2d_collection)

        if self.config['spatial_shift_multisampling']:
            # Merge the spatial shift result to probability map
            for i in range(0, 4):
                mask_region = self.image_class.ExtractMask(slicer_row = region_row_shift[i], slicer_column = region_column_shift[i])
                mask_selected = SelectedConnectedRegion(mask_region, sp_shift[i], sp, self.size)

                # Important: only select high sample-time region
                # otherwise, boundary region must lack sample
                # detail not cover in algorithm explanation
                if i == 0:
                    mask_selected[:self.step, :] = 0
                elif i == 1:
                    mask_selected[-self.step:, :] = 0
                elif i == 2:
                    mask_selected[:, :self.step] = 0
                elif i == 3:
                    mask_selected[:, -self.step:] = 0

            vessel_value = MergeProbabilityMap(self.config['dynamic_probability_map'],
                                 vessel_value, target_instance_shift[i] * mask_selected,
                                 region_row_shift[i], region_column_shift[i])
            vessel_weight = MergeProbabilityMap(self.config['dynamic_probability_map'],
                                  vessel_weight, mask_selected,
                                  region_row_shift[i], region_column_shift[i])


        # Merge the center frame result to probability map
        mask_region = self.image_class.ExtractMask(slicer_row = region_row, slicer_column = region_column)
        mask_selected = SelectedConnectedRegion(mask_region, sp, sp, self.size)
        vessel_value = MergeProbabilityMap(self.config['dynamic_probability_map'],
                             vessel_value, target_instance * mask_selected,
                             region_row, region_column)
        vessel_weight = MergeProbabilityMap(self.config['dynamic_probability_map'],
                              vessel_weight,  mask_selected,
                              region_row, region_column)

        # Binarize the probability map as the new instance map
        global_probability_map, global_binary_map, global_binary_map_post = self.BinarizeVesselMap(vessel_value,
                                                                                                   vessel_weight)

        # Detect undetected end point with an order (close first)
        end_point, distance = FindEndPoint(global_binary_map_post, sp, self.stop_list)

        # Delete the possible root point when it is first iteration
        if self.is_root:
            direction_vector_branch = np.zeros((end_point.shape[0], 2))
            dot_product = np.zeros(end_point.shape[0])
            record = []
            for k in range(end_point.shape[0]):
                direction_vector_branch[k] = end_point[k] - sp

                # Normalize the direction vector
                direction_vector_branch[k] = direction_vector_branch[k] / np.sqrt(
                np.sum(direction_vector_branch[k] ** 2))
                dot_product[k] = np.dot(direction_vector_branch[k], direction_vector)
                # end points that are almost opposite to start direction vector, indicating it is root node
                if (dot_product[k] - (-1)) < 0.1:
                    record.append(k)

            for l in range(len(record)):
                self.stop_list.append(end_point[record[l]])
            end_point = np.delete(end_point, record, axis = 0)

        # Draw step 3,4,5
        #self.DrawStep3(global_probability_map, sp)
        #self.DrawStep4(global_binary_map, global_binary_map_post, sp)

        if self.is_root:
            current_tree = None
        #self.DrawStep5(global_binary_map_post, sp, vessel_index, end_point, current_tree)
        self.frame_count += 1 # use for the id of step files

        # find next point as start point to trace
        if end_point is None or len(end_point) == 0:
            print("No point needs to track")
            return vessel_value, vessel_weight
        else:
            k = 0
            while k < end_point.shape[0]:
                if self.count > self.config['count_limit']:
                    print("return, too much iteration")
                    return vessel_value, vessel_weight

                sp_branch = end_point[k]
                self.stop_list.append(sp_branch)
                region_row_branch, region_column_branch = self.image_class.GenerateRegion(sp_branch, self.size)
                region_row_branch, region_column_branch = self.image_class.ExtractBase(region_row_branch, region_column_branch)
                area = vessel_weight[region_row_branch, region_column_branch]

                # TODO
                # make sure there is some update in map
                if np.count_nonzero(area == 0) > 0:
                # if np.count_nonzero(area == 0) > self.size ** 2 * 0:
                    self.count = self.count + 1
                    if self.is_root:
                        self.is_root = False
                    vessel_value, vessel_weight= self.TracingSingleVessel(vessel_value, vessel_weight,
                                                                                     sp_branch,
                                                                                     direction_vector,
                                                                                     initial_start_point, vessel_index)

                    global_probability_map, global_binary_map, global_binary_map_post = self.BinarizeVesselMap(vessel_value, vessel_weight)

                    end_point, distance = FindEndPoint(global_binary_map_post, sp, self.stop_list)
                    k = 0
                else:
                    print("Repeated region")
                    k = k + 1

        return vessel_value, vessel_weight

    def BinarizeVesselMap(self, vessel_value, vessel_weight):

        vessel_weight_adjust = copy(vessel_weight)
        # avoid zero divide
        vessel_weight_adjust[vessel_weight_adjust == 0] = 0.1
        global_probability_map = vessel_value / vessel_weight_adjust

        global_binary_map = np.zeros_like(global_probability_map)
        global_binary_map[global_probability_map >= self.threshold] = 1

        kernel1 = np.ones((self.dilation_size, self.dilation_size), np.uint8)
        vessel_dilation = cv2.dilate(global_binary_map, kernel1, iterations = 1)
        kernel2 = np.ones((self.erosion_size, self.erosion_size), np.uint8)
        vessel_erosion = cv2.erode(vessel_dilation, kernel2, iterations = 1)
        global_binary_map_post = vessel_erosion * self.image_class.mask
        global_binary_map_post = MaxConnectedRegion(global_binary_map_post)
        global_binary_map_post[global_binary_map_post > 0] = 1

        return global_probability_map, global_binary_map, global_binary_map_post

    def SearchPointInTreeForTemporalSequence(self, vesseltree, sp):
        # sp must be end tail of current tree
        min_distance = np.inf
        min_index = None
        for i in range(1, len(vesseltree.dict) + 1):
            node = vesseltree.dict[str(i)]['coordinate']
            distance = np.sqrt(np.sum(np.square(node - sp)))
            if distance < min_distance:
                min_distance = distance
                min_index = i

        current_child = min_index
        current_parent = vesseltree.tree.parent(current_child).identifier

        pointlist = []
        if current_parent == 0:
            return pointlist

        key = str(current_parent) + ',' + str(current_child)
        # list of points from parent node to child node
        current_list = np.array(vesseltree.connection[key])[0]
        current_position = len(current_list) - 1

        circulation = 1
        while current_parent != 0 and circulation < self.config['time']:
            current_position = current_position - self.config['step']
            while current_position < 0:
                current_child = current_parent
                current_parent = vesseltree.tree.parent(current_parent).identifier
                if current_parent == 0:
                    break
                current_list = np.array(vesseltree.connection[str(current_parent) + ',' + str(current_child)])[0]
                current_position = len(current_list) + current_position
            if current_parent == 0:
                break

            current_coordinate = current_list[current_position]
            current_coordinate = np.expand_dims(current_coordinate, axis = 0)

            if circulation == 1:
                pointlist = current_coordinate
            else:
                pointlist = np.concatenate((pointlist, current_coordinate), axis = 0)

            circulation += 1

        if pointlist != []:
            pointlist = pointlist[::-1]
        return pointlist

    def AccquireResults(self, sp, point_list):

        # Accqurie region around sp
        #print("sp", sp)
        region_row, region_column = self.image_class.GenerateRegion(sp, self.size)
        sp = np.array(sp)
        # length x coordinate
        if point_list == []:
            point_list = sp
            point_list = np.expand_dims(point_list, axis = 0)
        else:
            sp = np.expand_dims(sp, axis = 0)
            point_list = np.concatenate((point_list, sp), axis=0)

        images, time = self.image_class.LoadSequence(point_list, self.size)  # Load the image sequence and time is an updated value
        mask_piece = self.image_class.ExtractMask(region_row, region_column)

        if mask_piece.any():
            # Run DL model
            images = torch.unsqueeze(images, 0)  # add a dimension for batch
            if self.config['use_cuda']:
                images = images.cuda()
            output = self.config['model'].ForwardModel(images, type = 'test')
            ins_pred = output.cpu().data.numpy()
            ins_pred = np.squeeze(ins_pred, axis = 0)  # remove the dimension for batch
            postProcess = PostProcess(mask_piece, ins_pred[time - 1])

            label_predict = postProcess.Clustering(method = self.config['cluster_method'], embedding_type = 'original',
                                                        ratio = self.config['ratio'])
            instance_collection = postProcess.InstanceLayer()
            img_ins = postProcess.InstanceMap()
            ins_number_estimate = postProcess.ins_number

            return region_row, region_column, label_predict, instance_collection, img_ins, ins_number_estimate, postProcess
        else:
            return None

    def ExtractTargetInstance(self, global_binary_map, region_row, region_column,
                      instance_collection, ins_number_estimate):

        Vessel_enlarge = np.pad(global_binary_map, (self.size, self.size), 'constant', constant_values = (0, 0))
        slicer_row = slice(region_row.start + self.size, region_row.stop + self.size, 1)
        slicer_column = slice(region_column.start + self.size, region_column.stop + self.size, 1)
        mask = Vessel_enlarge[slicer_row, slicer_column]

        max_overlay_area = 0
        max_overlay_index = None
        for i in range(0, ins_number_estimate):
            overlay_area = np.sum(mask * instance_collection[i])
            if overlay_area > max_overlay_area:
                max_overlay_area = overlay_area
                max_overlay_index = i

        if max_overlay_index is not None:
            target_instance = instance_collection[max_overlay_index]
        else:
            target_instance = np.zeros((self.size, self.size)) # return var was child_single_label
        return target_instance

    def DrawStep1(self, mask, vessel_index, sp, temporal_sequence_point_list, sp_spatial):
        # Show current branch tree and three kinds of patch frames in different color
        opacity_rate = 0.6
        image = copy(self.image_class.image)
        image[mask > 0,:] = (1 - opacity_rate) * image[mask > 0,:] + 255 * opacity_rate

        # only frame no result (last segmentation result)
        fig, ax = plt.subplots(figsize = (15, 15))

        image_x, image_y = image.shape[0], image.shape[1]
        ax.set_xlim(0 - self.size, image_x + self.size)
        ax.set_ylim(0 - self.size, image_y + self.size)
        ax.imshow(image)
        plt.axis('off')

        # add root arrow
        if 'arrow_line_width' in self.config:
            arrow_line_width = self.config['arrow_line_width']
        else:
            arrow_line_width = 2

        if 'cross_one_side_length' in self.config:
            one_side_length = self.config['cross_one_side_length']
        else:
            one_side_length = 7

        angle = np.arctan2(self.config['secondpoint_list'][vessel_index][0] - self.config['startpoint_list'][vessel_index][0],
                           self.config['secondpoint_list'][vessel_index][1] - self.config['startpoint_list'][vessel_index][1]) + np.pi / 4

        xmin, xmax, ymin, ymax = DrawCross(self.config['startpoint_list'][vessel_index][1], self.config['startpoint_list'][vessel_index][0],
                                           one_side_length, angle)
        l = mlines.Line2D([xmin, xmax], [ymin, ymax], c = 'b', linewidth = arrow_line_width)
        ax.add_line(l)
        xmin, xmax, ymin, ymax = DrawCross(self.config['startpoint_list'][vessel_index][1], self.config['startpoint_list'][vessel_index][0],
                                           one_side_length, angle + np.pi / 2)
        l = mlines.Line2D([xmin, xmax], [ymin, ymax], c = 'b', linewidth = arrow_line_width)
        ax.add_line(l)

        plt.arrow(self.config['startpoint_list'][vessel_index][1], self.config['startpoint_list'][vessel_index][0],
                  1.5*(self.config['secondpoint_list'][vessel_index][1] - self.config['startpoint_list'][vessel_index][1]),
                  1.5*(self.config['secondpoint_list'][vessel_index][0] - self.config['startpoint_list'][vessel_index][0]),
                  shape = 'full', color = 'b', length_includes_head = True, head_width = one_side_length,
                  head_length = one_side_length, linewidth = arrow_line_width)


        # Draw frame
        if 'frame_line_width' in self.config:
            frame_line_width = self.config['frame_line_width']
        else:
            frame_line_width = 2.5

        # Create Rectangle patches to represent the temporal sequence YELLOW
        if temporal_sequence_point_list != []:
            for c in range(len(temporal_sequence_point_list)):
                rect = patches.Rectangle((temporal_sequence_point_list[c][1] - int(self.size / 2),
                                          temporal_sequence_point_list[c][0] - int(self.size / 2)),
                                         self.size, self.size, linewidth = frame_line_width, edgecolor = '#FFF000',
                                         facecolor = 'none')
                ax.add_patch(rect)

        # Create the spatial-shift multi-sampling Rectangle patch MAGENTA
        for j in range(0,4):
            rect = patches.Rectangle((sp_spatial[j][1] - int(self.size / 2), sp_spatial[j][0] - int(self.size / 2)),
                                 self.size, self.size,
                                 linewidth = frame_line_width, edgecolor = '#F000FF', facecolor = 'none')
            # Add the patch to the Axes
            ax.add_patch(rect)

        # Create the target Rectangle patch BLUE
        rect = patches.Rectangle((sp[1] - int(self.size / 2), sp[0] - int(self.size / 2)),
                                 self.size, self.size,
                                 linewidth = frame_line_width, edgecolor = '#00FFF0', facecolor = 'none')
        # Add the patch to the Axes
        ax.add_patch(rect)

        sub_folder = os.path.join(self.config['step_folder'], 'step1/')
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
        path_frame = os.path.join(sub_folder, str(self.frame_count) + '.png')
        plt.savefig(path_frame, dpi = 300)
        plt.close()

    def DrawStep2(self, instance_collection, instance_collection_multi, embedding2d_collection):
        colormap = plt.cm.Set2
        # first predict, second gt(base), ins_labels_order actually no change
        img_ins_color_list = []

        # for scatter plot
        ins_order_collection = []
        colormap_collection = []

        for i in range(0, 4):
            dice, ins_multi_order, ins_labels_order = MatchedDiceScore(instance_collection_multi[i], instance_collection, match = False)
            img_ins_color, ins_colormap = Coloring(ins_multi_order, colormap, 'image_channel')
            img_ins_color_list.append(img_ins_color)

            ins_order_collection.append(ins_multi_order)
            colormap_collection.append(ins_colormap)

        img_target_color, target_colormap = Coloring(ins_labels_order, colormap, 'image_channel')
        ins_order_collection.insert(0, ins_labels_order)
        colormap_collection.insert(0, target_colormap)

        # pixel_no * instance layer
        plt.clf()

        plt.subplot(2, 3, 1)
        plt.imshow(img_target_color)
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(img_ins_color_list[0])
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.imshow(img_ins_color_list[1])
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.imshow(img_ins_color_list[2])
        plt.axis('off')

        plt.subplot(2, 3, 6)
        plt.imshow(img_ins_color_list[3])
        plt.axis('off')

        plt.tight_layout()

        sub_folder = os.path.join(self.config['step_folder'], 'step2/list/')
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
        path_frame = os.path.join(sub_folder, str(self.frame_count) + '.png')
        plt.savefig(path_frame, dpi =300)
        plt.close()

        plt.clf()

        # scatter plot
        if self.frame_count in self.config['select_frame']:
            sub_folder = os.path.join(self.config['step_folder'], 'step2/scatter/')
            if not os.path.exists(sub_folder):
                os.makedirs(sub_folder)
            for i in range(0, 5):
                ins_result_order_list = ChannelToLabel(ins_order_collection[i])
                ins_colormap = colormap_collection[i]
                embedding_2d = embedding2d_collection[i]
                # normalize to make sure aspect ratio is 1:1
                embedding_2d[:, 0] = (embedding_2d[:, 0] - np.min(embedding_2d[:, 0])) / (
                            np.max(embedding_2d[:, 0]) - np.min(embedding_2d[:, 0]))
                embedding_2d[:, 1] = (embedding_2d[:, 1] - np.min(embedding_2d[:, 1])) / (
                            np.max(embedding_2d[:, 1]) - np.min(embedding_2d[:, 1]))

                ins_number_estimate = ins_order_collection[i].shape[0]

                plt.clf()
                ax1 = plt.subplot(1, 1, 1)
                for k in range(ins_number_estimate):
                    plt.scatter(embedding_2d[ins_result_order_list == k + 1, 0],
                                embedding_2d[ins_result_order_list == k + 1, 1], \
                            s=0.5, c=[np.array(ins_colormap[k][:3])] * len(embedding_2d[ins_result_order_list == k + 1, 0]))
                ax1.set_aspect(1)
                # hide x y axis
                plt.xticks([])
                plt.yticks([])
                path_frame = os.path.join(sub_folder, str(self.frame_count) + '_' + str(i) + '.png')
                plt.savefig(path_frame, dpi=300)
                plt.clf()

        plt.close()

    def DrawStep3(self, weight, sp):
        # weight map with blue frame or not
        image_x, image_y = weight.shape[0], weight.shape[1]
        fig, ax = plt.subplots(figsize = (15, 15))
        ax.set_xlim(0 - self.size, image_x + self.size)
        ax.set_ylim(0 - self.size, image_y + self.size)

        if 'frame_line_width' in self.config:
            frame_line_width = self.config['frame_line_width']
        else:
            frame_line_width = 2.5

        if self.config['dynamic_probability_map']:
            im = ax.imshow(weight, cmap = 'hot')
        else:
            im = ax.imshow(weight, cmap = 'gray')
        # fig.colorbar(im) # show color bar
        plt.axis('off')

        # without frame mark
        sub_folder = os.path.join(self.config['step_folder'], 'step3/no/')
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
        path_frame = os.path.join(sub_folder, str(self.frame_count) + '.png')
        plt.savefig(path_frame, dpi = 300)

        # Create a Rectangle patch
        rect = patches.Rectangle((sp[1] - int(self.size / 2), sp[0] - int(self.size / 2)),
                                 self.size, self.size,
                                 linewidth = frame_line_width, edgecolor = '#00FFF0', facecolor = 'none')
        # Add the patch to the Axes
        ax.add_patch(rect)

        sub_folder = os.path.join(self.config['step_folder'], 'step3/with/')
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
        path_frame = os.path.join(sub_folder, str(self.frame_count) + '.png')
        plt.savefig(path_frame, dpi = 300)

        plt.close()

    def DrawStep4(self, mask, mask_post, sp):
        opacity_rate = 0.6
        image = copy(self.image_class.image)
        image[mask > 0,:] = (1 - opacity_rate) * image[mask > 0,:] + 255 * opacity_rate

        image_x, image_y = image.shape[0], image.shape[1]
        fig, ax = plt.subplots(figsize = (15, 15))
        ax.set_xlim(0 - self.size, image_x + self.size)
        ax.set_ylim(0 - self.size, image_y + self.size)

        if 'frame_line_width' in self.config:
            frame_line_width = self.config['frame_line_width']
        else:
            frame_line_width = 2.5

        ax.imshow(image)
        plt.axis('off')

        # without frame mark
        sub_folder = os.path.join(self.config['step_folder'], 'step4/pre/no/')
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
        path_frame = os.path.join(sub_folder, str(self.frame_count) + '.png')
        plt.savefig(path_frame, dpi = 300)

        # Create the target Rectangle patch BLUE
        rect = patches.Rectangle((sp[1] - int(self.size / 2), sp[0] - int(self.size / 2)),
                                 self.size, self.size,
                                 linewidth = frame_line_width, edgecolor = '#00FFF0', facecolor = 'none')
        # Add the patch to the Axes
        ax.add_patch(rect)

        sub_folder = os.path.join(self.config['step_folder'], 'step4/pre/all/')
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
        path_frame = os.path.join(sub_folder, str(self.frame_count) + '.png')
        plt.savefig(path_frame, dpi = 300)

        plt.close()

        # mask_post image
        opacity_rate = 0.6
        image = copy(self.image_class.image)
        image[mask_post > 0,:] = (1 - opacity_rate) * image[mask_post > 0,:] + 255 * opacity_rate

        image_x, image_y = image.shape[0], image.shape[1]
        fig, ax = plt.subplots(figsize = (15, 15))
        ax.set_xlim(0 - self.size, image_x + self.size)
        ax.set_ylim(0 - self.size, image_y + self.size)

        if 'frame_line_width' in self.config:
            frame_line_width = self.config['frame_line_width']
        else:
            frame_line_width = 2.5

        ax.imshow(image)
        plt.axis('off')

        # without frame mark
        sub_folder = os.path.join(self.config['step_folder'], 'step4/post/no/')
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
        path_frame = os.path.join(sub_folder, str(self.frame_count) + '.png')
        plt.savefig(path_frame, dpi = 300)

        # Create the target Rectangle patch BLUE
        rect = patches.Rectangle((sp[1] - int(self.size / 2), sp[0] - int(self.size / 2)),
                                 self.size, self.size,
                                 linewidth = frame_line_width, edgecolor = '#00FFF0', facecolor = 'none')
        # Add the patch to the Axes
        ax.add_patch(rect)

        sub_folder = os.path.join(self.config['step_folder'], 'step4/post/all/')
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
        path_frame = os.path.join(sub_folder, str(self.frame_count) + '.png')
        plt.savefig(path_frame, dpi = 300)

        plt.close()


    def DrawStep5(self, mask, sp, vessel_index, end_point, tree_structure):
        opacity_rate = 0.6
        image = copy(self.image_class.image)
        image[mask > 0,:] = (1-opacity_rate) * image[mask > 0,:] + 255 * opacity_rate

        fig, ax = plt.subplots()
        ax.imshow(image)
        plt.axis('off')

        if tree_structure is not None:
            # add tree structure
            node_generator = tree_structure.tree.expand_tree(nid = 0, mode = Tree.WIDTH)
            node_list = list(node_generator)
            node_list = node_list[1:] # remove root
            # for k in range(len(node_list)):
            #     idx = node_list[k]
            #     point_list_coordinate = np.array(tree_structure.dict[str(idx)]['coordinate'])

            connect_list = np.argwhere(tree_structure.graph == 1)
            # add edge
            for k in range(connect_list.shape[0]):
                parent = connect_list[k, 0]
                child = connect_list[k, 1]
                parent_point = np.array(tree_structure.dict[str(parent)]['coordinate'])
                child_point = np.array(tree_structure.dict[str(child)]['coordinate'])
                if parent_point != [] and child_point != []:
                    l = mlines.Line2D([parent_point[0], child_point[0]], [parent_point[1], child_point[1]],
                                      c = 'k', linewidth = 0.5)
                    ax.add_line(l)

            # # add node
            # for k in range(len(node_list)):
            #     idx = node_list[k]
            #     point_list_coordinate = np.array(tree_structure.dict[str(idx)]['coordinate'])
            #     circle = patches.Circle((point_list_coordinate[0], point_list_coordinate[1]), radius=5, facecolor='r',
            #                         edgecolor='r')
            #     ax.add_patch(circle)
            #
            # # add number mark
            # for k in range(len(node_list)):
            #     idx = node_list[k]
            #     point_list_coordinate = np.array(tree_structure.dict[str(idx)]['coordinate'])
            #     ax.text(point_list_coordinate[0], point_list_coordinate[1], str(idx), fontsize=5, color='w')

        # add stop cross
        if 'cross_one_side_length' in self.config:
            one_side_length = self.config['cross_one_side_length']
        else:
            one_side_length = 7
        angle = np.pi/4
        for c in range(1,len(self.stop_list)): # skip the first one (origin source)
            # circle = patches.Circle((self.stop_list[c][1],self.stop_list[c][0]), radius=5,facecolor='none',edgecolor='g')
            # ax.add_patch(circle)

            xmin,xmax,ymin,ymax = DrawCross(self.stop_list[c][1],self.stop_list[c][0],one_side_length,angle)
            l = mlines.Line2D([xmin, xmax], [ymin, ymax], c = 'r',linewidth = 1)
            ax.add_line(l)
            xmin, xmax, ymin, ymax = DrawCross(self.stop_list[c][1],self.stop_list[c][0],
                                           one_side_length, angle + np.pi/2)
            l = mlines.Line2D([xmin, xmax], [ymin, ymax], c = 'r',  linewidth = 1)
            ax.add_line(l)

        # add possible circle
        if end_point is not None and len(end_point) != 0:
           for c in range(len(end_point)):
                circle = patches.Circle((end_point[c][1], end_point[c][0]), radius=5, facecolor='none', edgecolor='#00FF00',linewidth=1)
                ax.add_patch(circle)
           # add selected circle (fill)
           circle = patches.Circle((end_point[0][1], end_point[0][0]), radius=5, facecolor='#00FF00', edgecolor='#00FF00',linewidth=1)
           ax.add_patch(circle)

        # add root arrow
        angle = np.arctan2(self.config['secondpoint_list'][vessel_index][0] - self.config['startpoint_list'][vessel_index][0],
                           self.config['secondpoint_list'][vessel_index][1] - self.config['startpoint_list'][vessel_index][1]) + np.pi / 4

        xmin, xmax, ymin, ymax = DrawCross(self.config['startpoint_list'][vessel_index][1], self.config['startpoint_list'][vessel_index][0],
                                           one_side_length, angle)
        l = mlines.Line2D([xmin, xmax], [ymin, ymax], c = 'b', linewidth = 1)
        ax.add_line(l)
        xmin, xmax, ymin, ymax = DrawCross(self.config['startpoint_list'][vessel_index][1], self.config['startpoint_list'][vessel_index][0],
                                           one_side_length, angle + np.pi / 2)
        l = mlines.Line2D([xmin, xmax], [ymin, ymax], c = 'b', linewidth = 1)
        ax.add_line(l)

        plt.arrow(self.config['startpoint_list'][vessel_index][1], self.config['startpoint_list'][vessel_index][0],
                  1.5*(self.config['secondpoint_list'][vessel_index][1] - self.config['startpoint_list'][vessel_index][1]),
                  1.5*(self.config['secondpoint_list'][vessel_index][0] - self.config['startpoint_list'][vessel_index][0]),
                  shape='full', color='b', length_includes_head=True, head_width=one_side_length,
                  head_length=one_side_length, linewidth=1)


        # without frame mark
        sub_folder = os.path.join(self.config['step_folder'], 'step5/no/')
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
        path_frame = os.path.join(sub_folder, str(self.frame_count) + '.png')
        plt.savefig(path_frame, dpi=300)

        # Create the target Rectangle patch BLUE
        rect = patches.Rectangle((sp[1] - int(self.size / 2 ), sp[0] - int(self.size / 2)),
                                 self.size, self.size,
                                 linewidth=1.5, edgecolor='#00FFF0', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

        # cover patch : add possible circle
        if end_point is not None and len(end_point) != 0:
           for c in range(len(end_point)):
                circle = patches.Circle((end_point[c][1], end_point[c][0]), radius=5, facecolor='none', edgecolor='#00FF00',linewidth=1)
                ax.add_patch(circle)
           # add selected circle (fill)
           circle = patches.Circle((end_point[0][1], end_point[0][0]), radius=5, facecolor='#00FF00', edgecolor='#00FF00',linewidth=1)
           ax.add_patch(circle)

        sub_folder = os.path.join(self.config['step_folder'], 'step5/all/')
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
        path_frame = os.path.join(sub_folder, str(self.frame_count) + '.png')
        plt.savefig(path_frame, dpi = 300)
        plt.close()

