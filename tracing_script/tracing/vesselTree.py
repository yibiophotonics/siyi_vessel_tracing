import numpy as np
from treelib import Tree

class VesselTree():
    def __init__(self):
        self.tree = Tree()
        self.dict = dict()
        self.largest_node = 1
        self.tree.create_node('0', 0)  # root node, no anything in it
        self.graph = np.zeros((128, 128))
        self.connection = dict()

    def InitializeVessel(self):
        self.idx += 1
        self.history[str(self.idx)] = []

    def CreateNode(self, coordinate, parent):
        # if it is a new vessel, set parent as 0
        new_node = self.largest_node
        self.tree.create_node(str(coordinate), new_node, parent=parent)
        self.largest_node += 1
        self.dict[str(new_node)] = dict()
        self.dict[str(new_node)]['coordinate'] = coordinate
        self.dict[str(new_node)]['follow'] = []
        self.dict[str(new_node)]['centroid'] = []

        return new_node

    def DeleteNode(self, node):
        self.tree.remove_node(node)
        del self.dict[str(node)]

    def ReadNode(self, node):
        # return a dictionary containing coordinate and segmentation
        return self.dict[str(node)]

    def PushResult(self, node, result):
        self.dict[str(node)]['segmentation'] = result

    def MakeConnection(self, parent, child):
        self.graph[parent][child] = 1
        self.connection[str(parent) + ',' + str(child)] = []
        self.connection[str(parent) + ',' + str(child)].append(np.array(self.dict[str(parent)]['follow']))
        self.dict[str(parent)]['follow'].clear()

    def TraceTree(self, sp, parent, map, done_map, centroid, cluster):
        if map[sp[0], sp[1]] == 2:
            new_node = self.CreateNode(sp, parent)
            if parent != 0:
                self.MakeConnection(parent, new_node)
            parent = new_node

        if map[sp[0], sp[1]] == 2 or map[sp[0], sp[1]] == 3:
            near_point, point_level = FindNearPoint(sp, map, done_map)

        elif map[sp[0], sp[1]] > 3: # == 4 or map[sp[0], sp[1]] == 5 or map[sp[0], sp[1]] == 6:
            near_centroid, near_cluster = FindNearCentroid(sp, centroid, cluster)
            new_node = self.CreateNode(near_centroid, parent)
            self.MakeConnection(parent, new_node)
            parent = new_node
            done_map[near_centroid[0], near_centroid[1]] = 1
            for i in range(near_cluster.shape[0]):
                self.dict[str(parent)]['centroid'].append(near_cluster[i])
                done_map[near_cluster[i][0], near_cluster[i][1]] = 1
            near_point, point_level = FindPointAfterCentroidPoint(near_cluster, map, done_map)
        else:
            print('error, map value is not correct',end=": ")
            print(map[sp[0], sp[1]])

        if near_point is None:
            return done_map
        else:
            for i in range(near_point.shape[0]):
                done_map[near_point[i][0], near_point[i][1]] = 1
                if point_level[i] == 2 or point_level[i] > 3: # == 4 or point_level[i] == 5 or point_level[i] == 6:
                    done_map = self.TraceTree(near_point[i], parent, map, done_map, centroid, cluster)
                elif point_level[i] == 3:
                    self.dict[str(parent)]['follow'].append(near_point[i])
                    done_map = self.TraceTree(near_point[i], parent, map, done_map, centroid, cluster)
                else:
                    print('error, point level is not correct',end=": ")
                    print(point_level[i])
        return done_map

def FindNearPoint(center_point, map, done_map): # for value = 2 or 3
    row_slicer = slice(center_point[0] - 1,center_point[0] + 2)
    col_slicer = slice(center_point[1] - 1,center_point[1] + 2)
    target = map[row_slicer,col_slicer]
    target_done_map = done_map[row_slicer, col_slicer]
    point_local = np.argwhere(target * (1 - target_done_map) != 0)
    point_level = target[point_local[:,0],point_local[:,1]]
    near_point = point_local + np.array([row_slicer.start, col_slicer.start])

    order = np.argsort(point_level)
    near_point = near_point[order,:] # point level from small to large (larger one is bifurcation)
    point_level = point_level[order]

    return near_point, point_level

def FindNearCentroid(center_point, centroid, cluster): # for value = 4 or 5
    centroid = np.array(centroid)
    center_point = np.expand_dims(center_point, axis = 0)
    center_point = np.repeat(center_point, centroid.shape[0], axis = 0)

    difference = center_point - centroid
    distance = np.sqrt(np.sum(difference ** 2, axis=1))
    index = np.argsort(distance)
    this_cluster = centroid[index, :]  # ascending order
    near_centroid = this_cluster[0]
    near_cluster = cluster[index[0]]
    return near_centroid, near_cluster

def FindPointAfterCentroidPoint(near_cluster, map, done_map):
    for i in range(near_cluster.shape[0]):
        each_near_point, each_point_level = FindNearPoint(near_cluster[i], map, done_map)

        if i == 0:
            near_point = each_near_point
            point_level = each_point_level
        else:
            near_point = np.concatenate((near_point, each_near_point), axis = 0)
            point_level = np.concatenate((point_level, each_point_level), axis = None)

    near_point, index = np.unique(near_point, axis = 0, return_index = True)
    point_level = point_level[index]

    order = np.argsort(point_level)
    near_point = near_point[order,:] # from small to large
    point_level = point_level[order]
    return near_point, point_level



