import numpy as np
import torch
import torch.utils.data
from sympy.utilities.iterables import multiset_permutations

def MatchedDiceScore(y_pred, y_true, match = False):
    """
    This functions sort the prediction and ground truth in a way that the matched dice score is maximized.
    match(bool): True if input y_pred is already matched with y_true.
    Usually use False so the function will find the best order for y_pred to match y_true.
    """

    # Convert the input to a numpy array if it is a torch Tensor
    if isinstance(y_true, torch.Tensor):
        y_true = bool(y_true.detach().numpy())

    if isinstance(y_pred, torch.Tensor):
        y_pred = bool(y_pred.detach().numpy())

    y_pred = y_pred.astype(bool)
    y_true = y_true.astype(bool)

    # make the channel size same
    channel_size = np.max([y_true.shape[0], y_pred.shape[0]])

    if y_true.shape[0] < channel_size:
        zero_pad = np.zeros((channel_size - y_true.shape[0], y_true.shape[1], y_true.shape[2]))
        y_true = np.concatenate((y_true, zero_pad), axis=0)
    else:
        zero_pad = np.zeros((channel_size - y_pred.shape[0], y_pred.shape[1], y_pred.shape[2]))
        y_pred = np.concatenate((y_pred, zero_pad), axis=0)

    # y_pred: instance_number * height * width
    if not match:
        # the boundary of permutation, you can change it to a larger number if you have a powerful computer
        if y_pred.shape[0] <= 6:
            dice_best = 0
            # p is the permutation of the prediction index
            for p in multiset_permutations(range(y_pred.shape[0])):
                y_pred_p = y_pred[p]
                TP = np.sum(y_true * y_pred_p)
                FP = np.sum((1-y_true) * y_pred_p)
                FN = np.sum(y_true * (1 - y_pred_p))
                smooth = 0.0001
                dice = (2. * TP + smooth) / (2. * TP + FP + FN + smooth)
                if dice > dice_best:
                    dice_best = dice
                    order = p

            # y_pred is sorted tp match the y_true
            return dice_best, y_pred[order], y_true
        # if the number of instances is too big, use a greedy algorithm to find an approximate solution
        else:
            # descending order of the size of the instance in ground truth
            rank = np.argsort(-np.sum(y_true,axis=(1,2)))
            # position: pred in order, content: corresponding truth instance
            rank_pair = np.zeros(y_pred.shape[0])
            # the pred instance that has been selected by one truth instance will be marked as 1
            selected = np.zeros(y_pred.shape[0])

            for i in range(len(rank)):
                # descending order of the similarity between all the pred instance and one truth instance
                reference = np.argsort(-np.sum(y_true[rank[i]] * y_pred, axis=(1,2))) # descending
                count = 0
                while selected[reference[count]] == 1:
                    count += 1
                rank_pair[rank[i]] = reference[count]
                selected[reference[count]] = 1
            y_pred = y_pred[rank_pair.astype(int)]

            # calculate the dice score
            TP = np.sum(y_true * y_pred)
            FP = np.sum((1 - y_true) * y_pred)
            FN = np.sum(y_true * (1 - y_pred))
            smooth = 0.0001
            dice = (2. * TP + smooth) / (2. * TP + FP + FN + smooth)

            # y_pred is sorted tp match the y_true
            return dice, y_pred, y_true
    else:
        # just flatten the instances to a big map and calculate the dice of all instances
        # so it a dice score weighted by size
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        TP = np.sum(y_true * y_pred)
        FP = np.sum((1 - y_true) * y_pred)
        FN = np.sum(y_true * (1 - y_pred))
        smooth = 0.0001

        dice = (2. * TP + smooth) / (2. * TP + FP + FN + smooth)

        return dice

def SymmetricBestDice(y_pred, y_true):
    """
    Compute Symmetric Best Dice coefficient between two instance segmentation masks.
    """

    # Convert the input to a numpy array if it is a torch Tensor
    if isinstance(y_true, torch.Tensor):
        y_true = bool(y_true.detach().numpy())

    if isinstance(y_pred, torch.Tensor):
        y_pred = bool(y_pred.detach().numpy())

    y_pred = y_pred.astype(bool)
    y_true = y_true.astype(bool)

    best_dice_pred = np.zeros(y_pred.shape[0])
    for i in range(y_pred.shape[0]):
        for j in range(y_true.shape[0]):
            dice = PureDiceScore(y_pred[i], y_true[j])
            if dice > best_dice_pred[i]:
                best_dice_pred[i] = dice
    dice_pred = np.mean(best_dice_pred)

    best_dice_true = np.zeros(y_true.shape[0])
    for i in range(y_true.shape[0]):
        for j in range(y_pred.shape[0]):
            dice = PureDiceScore(y_true[i], y_pred[j])
            if dice > best_dice_true[i]:
                best_dice_true[i] = dice
    dice_true = np.mean(best_dice_true)

    return np.min([dice_pred, dice_true])

def ScaledSymmetricBestDice(y_pred, y_true):
    """
    Compute Scaled Symmetric Best Dice coefficient between two instance segmentation masks.
    outputs are array of components coming from calculating dice between each pair of instances.

    This function also provide basic SBD result as well as the scaled version of SBD.
    """

    # Convert the input to a numpy array if it is a torch Tensor
    if isinstance(y_true, torch.Tensor):
        y_true = bool(y_true.detach().numpy())

    if isinstance(y_pred, torch.Tensor):
        y_pred = bool(y_pred.detach().numpy())

    y_pred = y_pred.astype(bool)
    y_true = y_true.astype(bool)

    best_dice_pred = np.zeros(y_pred.shape[0])
    pred_area_p = np.zeros(y_pred.shape[0])
    pred_area_t = np.zeros(y_pred.shape[0])

    for i in range(y_pred.shape[0]):
        pred_area_p[i] = y_pred[i].sum()
        for j in range(y_true.shape[0]):
            dice = PureDiceScore(y_pred[i], y_true[j])
            if dice > best_dice_pred[i]:
                best_dice_pred[i] = dice
                pred_area_t[i] = y_true[j].sum()

    dice_pred = np.mean(best_dice_pred)
    dice_pred_scale = np.sum(best_dice_pred * pred_area_p)/np.sum(pred_area_p)

    best_dice_true = np.zeros(y_true.shape[0])
    true_area_p = np.zeros(y_true.shape[0])
    true_area_t = np.zeros(y_true.shape[0])

    for i in range(y_true.shape[0]):
        true_area_t[i] = y_true[i].sum()
        for j in range(y_pred.shape[0]):
            dice = PureDiceScore(y_true[i], y_pred[j])
            if dice > best_dice_true[i]:
                best_dice_true[i] = dice
                true_area_p[i] = y_pred[j].sum()

    dice_true = np.mean(best_dice_true)
    dice_true_scale = np.sum(best_dice_true * true_area_t) / np.sum(true_area_t)

    SBD = np.min([dice_pred, dice_true])
    SBD_scaled = np.min([dice_pred_scale, dice_true_scale])

    # narray
    return  SBD, SBD_scaled, \
            dice_pred, dice_true, dice_pred_scale, dice_true_scale, \
            best_dice_pred, best_dice_true, \
            pred_area_p, pred_area_t, true_area_t, true_area_p

def PureDiceScore(y_true, y_pred):
    """
    Compute the Dice coefficient between two binary masks.
    """
    TP = np.sum(y_true * y_pred)
    FP = np.sum((1 - y_true) * y_pred)
    FN = np.sum(y_true * (1 - y_pred))
    smooth = 0.0001

    dice = (2. * TP + smooth) / (2. * TP + FP + FN + smooth)
    return dice


def DIC(count_pred, count_true):
    """
    Compute the absolute difference in count between the predicted and true counts.
    """
    if isinstance(count_true, torch.Tensor):
        count_true = count_true.cpu().detach().numpy()

    if isinstance(count_pred, torch.Tensor):
        count_pred = count_pred.cpu().detach().numpy()
    difference_mean = np.mean(abs(count_pred - count_true))

    return difference_mean