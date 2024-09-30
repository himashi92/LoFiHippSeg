import pprint
import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
from medpy import metric
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType, Activations
from numpy import logical_and as l_and, logical_not as l_not
from scipy.spatial.distance import directed_hausdorff
from skimage.measure import label
from surface_distance.metrics import compute_surface_distances, compute_average_surface_distance, compute_dice_coefficient, compute_robust_hausdorff


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def pad_batch1_to_compatible_size(batch):
    print(batch.shape)
    shape = batch.shape
    zyx = list(shape[-3:])
    for i, dim in enumerate(zyx):
        max_stride = 16
        if dim % max_stride != 0:
            # Make it divisible by 16
            zyx[i] = ((dim // max_stride) + 1) * max_stride
    zmax, ymax, xmax = zyx
    zpad, ypad, xpad = zmax - batch.size(2), ymax - batch.size(3), xmax - batch.size(4)
    assert all(pad >= 0 for pad in (zpad, ypad, xpad)), "Negative padding value error !!"
    pads = (0, xpad, 0, ypad, 0, zpad)
    batch = F.pad(batch, pads)
    return batch, (zpad, ypad, xpad)


post_trans = Compose(
    [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)

post_trans2 = Compose(
    [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)
device = torch.device("cuda:0")
VAL_AMP = True
voxel_sz = [1,1,1]

def DSC(gt, pred):
    gt = gt.astype(dtype=bool)
    pred = pred.astype(dtype=bool)
    dice_coeff = compute_dice_coefficient(gt, pred)
    return dice_coeff


def HD(gt, pred, voxel_sz):
    gt = gt.astype(dtype=bool)
    pred = pred.astype(dtype=bool)
    surface_dist = compute_surface_distances(pred, gt, spacing_mm=voxel_sz)
    Haus = compute_robust_hausdorff(surface_dist, 100)
    return Haus


def HD95(gt, pred, voxel_sz):
    gt = gt.astype(dtype=bool)
    pred = pred.astype(dtype=bool)
    surface_dist = compute_surface_distances(pred, gt, spacing_mm=voxel_sz)
    Haus_95 = compute_robust_hausdorff(surface_dist, 95)
    return Haus_95


def ASSD(gt, pred):
    gt = gt.astype(dtype=bool)
    pred = pred.astype(dtype=bool)
    surface_dist = compute_surface_distances(pred, gt, spacing_mm=[1, 1, 1])
    mean_surface_dis = compute_average_surface_distance(surface_dist)
    return np.mean(mean_surface_dis)


def RVE(gt, pred):  # Relative Volume Error (RVE)
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()

    gt_volume = np.sum(gt_flat)
    pred_volume = np.sum(pred_flat)
    rve_value = abs((pred_volume - gt_volume) / gt_volume)

    return rve_value


# define inference method
def inference(input, model, patch_size):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=patch_size,
            sw_batch_size=4,
            predictor=model,
            overlap=0.5
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)

def inference2(input, model, patch_size):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=patch_size,
            sw_batch_size=4,
            predictor=model,
            overlap=0.85
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)


def calculate_metrics(preds, targets, patient, tta=False):
    """
    Parameters
    ----------
    preds:
        torch tensor of size 1*C*Z*Y*X
    targets:
        torch tensor of same shape
    patient :
        The patient ID
    tta:
        is tta performed for this run
    """
    pp = pprint.PrettyPrinter(indent=4)
    assert preds.shape == targets.shape, "Preds and targets do not have the same size"

    metrics_list = []
    dice_total = 0.0
    labels = [1,2]
    for i, label in enumerate(labels):
        metrics = dict(
            patient_id=patient,
            label=str(label),
            tta=tta,
        )

        # if np.sum(targets[i]) == 0:
        #     iou = np.nan
        #     dice = 1 if np.sum(preds[i]) == 0 else 0
        #     tn = np.sum(l_and(l_not(preds[i]), l_not(targets[i])))
        #     fp = np.sum(l_and(preds[i], l_not(targets[i])))
        #     asd = np.nan
        #     haussdorf_dist = np.nan
        #
        # else:
        #     preds_coords = np.argwhere(preds[i])
        #     targets_coords = np.argwhere(targets[i])
        #     haussdorf_dist = directed_hausdorff(preds_coords, targets_coords)[0]
        #     #asd = metric.binary.asd(preds[i], targets[i])
        #
        #     tp = np.sum(l_and(preds[i], targets[i]))
        #     tn = np.sum(l_and(l_not(preds[i]), l_not(targets[i])))
        #     fp = np.sum(l_and(preds[i], l_not(targets[i])))
        #     fn = np.sum(l_and(l_not(preds[i]), targets[i]))
        #
        #     iou = tp / (tp + fp + fn)
        #     #asd = tn / (tn + fp)
        #
        #     dice = 2 * tp / (2 * tp + fp + fn)
        #     dice_total += dice
        dsc_value = DSC(targets[i], preds[i])
        hd_value = HD(targets[i], preds[i], voxel_sz)
        hd95_value = HD95(targets[i], preds[i], voxel_sz)
        assd_value = ASSD(targets[i], preds[i])
        rve_value = RVE(targets[i], preds[i])

        dice_total += dsc_value

        metrics[HAUSSDORF95] = hd95_value
        metrics[HAUSSDORF] = hd_value
        metrics[DICE] = dsc_value
        metrics[ASD] = assd_value
        metrics[RVEV] = rve_value
        pp.pprint(metrics)
        metrics_list.append(metrics)

    dice_avg = dice_total / 2

    return metrics_list, dice_avg


def var_all_case_monai(model, testloader, patch_size=(128, 128, 96), save_result=True, test_save_path=None):
    ith = 0
    total_metric = 0.0
    total_metric_average = 0.0
    metrics_list = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(testloader):
            volume_batch, volume_batch_t, label_batch, seg_path, crops_idx = batch['image'], batch['image_t'], batch['label'],  batch['seg_path'][0], \
                                                             batch['crop_indexes']
            val_inputs, val_labels = volume_batch.cuda(), label_batch.cuda()

            val_inputs = val_inputs.cuda()

            ref_seg_img = sitk.ReadImage(seg_path)
            ref_seg = sitk.GetArrayFromImage(ref_seg_img)

            # inputs, pads = pad_batch1_to_compatible_size(val_inputs)
            # inputs = inputs.cuda()

            with torch.no_grad():
                val_outputs_1 = inference(val_inputs, model, patch_size)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs_1)]

            segs = torch.zeros((1, 2, ref_seg.shape[0], ref_seg.shape[1], ref_seg.shape[2]))
            segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = val_outputs[0]
            print(f"SEGS : {segs.shape}")
            segs = segs[0].numpy() > 0.5
            #
            #
            # with torch.no_grad():
            #     val_outputs_1 = model(inputs)
            #     pre_segs = torch.sigmoid(val_outputs_1)
            #
            # maxz, maxy, maxx = pre_segs.size(2) - pads[0], pre_segs.size(3) - pads[1], pre_segs.size(4) - pads[2]
            # pre_segs = pre_segs[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()
            #
            # segs = torch.zeros((1, 2, ref_seg.shape[0], ref_seg.shape[1], ref_seg.shape[2]))
            # segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs[0]

            left = segs[0]
            right = segs[1]
            labelmap = np.zeros(segs[0].shape, dtype=np.uint8)

            labelmap[left] = 1
            labelmap[right] = 2

            print(np.unique(labelmap))

            labelmap = sitk.GetImageFromArray(labelmap)
            labelmap.CopyInformation(ref_seg_img)
            prediction = labelmap

            patient_id = seg_path.split('/')[-1]
            ref_seg = sitk.GetArrayFromImage(ref_seg_img)
            refmap_left, refmap_right = [np.zeros_like(ref_seg) for i in range(2)]

            refmap_left = ref_seg == 1
            refmap_right = ref_seg == 2

            refmap = np.stack([refmap_left, refmap_right])

            print(f"PRED: {segs.shape}, REF: {refmap.shape}")

            patient_metric_list, dice_per_case = calculate_metrics(segs, refmap, patient_id)
            metrics_list.append(patient_metric_list)

            total_metric += dice_per_case

            if save_result:
                sitk.WriteImage(prediction, f"{test_save_path}/{patient_id}")
            ith += 1

    avg_metric = total_metric / len(testloader)
    print('average metric is decoder 1 {}'.format(avg_metric))

    return avg_metric


def var_all_case_cotrain(model, testloader, patch_size=(128, 128, 96), save_result=True, test_save_path=None):
    ith = 0
    total_metric = 0.0
    total_metric_average = 0.0
    metrics_list = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(testloader):
            volume_batch, volume_batch_t, label_batch, seg_path, crops_idx = batch['image'], batch['image_t'], batch['label'],  batch['seg_path'][0], \
                                                             batch['crop_indexes']
            val_inputs, val_labels = volume_batch.cuda(), label_batch.cuda()

            val_inputs = val_inputs.cuda()

            ref_seg_img = sitk.ReadImage(seg_path)
            ref_seg = sitk.GetArrayFromImage(ref_seg_img)

            inputs, pads = pad_batch1_to_compatible_size(val_inputs)
            inputs = inputs.cuda()

            with torch.no_grad():
                val_outputs_1 = model(inputs)
                pre_segs = torch.sigmoid(val_outputs_1)

            maxz, maxy, maxx = pre_segs.size(2) - pads[0], pre_segs.size(3) - pads[1], pre_segs.size(4) - pads[2]
            pre_segs = pre_segs[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()

            segs = torch.zeros((1, 2, ref_seg.shape[0], ref_seg.shape[1], ref_seg.shape[2]))
            segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs[0]

            segs = segs[0].numpy() > 0.5

            left = segs[0]
            right = segs[1]
            labelmap = np.zeros(segs[0].shape, dtype=np.uint8)

            labelmap[left] = 1
            labelmap[right] = 2

            print(np.unique(labelmap))

            labelmap = sitk.GetImageFromArray(labelmap)
            labelmap.CopyInformation(ref_seg_img)
            prediction = labelmap

            patient_id = seg_path.split('/')[-1]
            ref_seg = sitk.GetArrayFromImage(ref_seg_img)
            refmap_left, refmap_right = [np.zeros_like(ref_seg) for i in range(2)]

            refmap_left = ref_seg == 1
            refmap_right = ref_seg == 2

            refmap = np.stack([refmap_left, refmap_right])

            print(f"PRED: {segs.shape}, REF: {refmap.shape}")

            patient_metric_list, dice_per_case = calculate_metrics(segs, refmap, patient_id)
            metrics_list.append(patient_metric_list)

            total_metric += dice_per_case

            if save_result:
                sitk.WriteImage(prediction, f"{test_save_path}/{patient_id}")
            ith += 1

    avg_metric = total_metric / len(testloader)
    print('average metric is decoder 1 {}'.format(avg_metric))

    return avg_metric


def var_all_case_cotrain_t(model, testloader, patch_size=(128, 128, 96), save_result=True, test_save_path=None):
    ith = 0
    total_metric = 0.0
    total_metric_average = 0.0
    metrics_list = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(testloader):
            volume_batch, volume_batch_t, label_batch, seg_path, crops_idx = batch['image'], batch['image_t'], batch['label'],  batch['seg_path'][0], \
                                                             batch['crop_indexes']
            val_inputs, val_labels = volume_batch_t.cuda(), label_batch.cuda()

            val_inputs = val_inputs.cuda()

            ref_seg_img = sitk.ReadImage(seg_path)
            ref_seg = sitk.GetArrayFromImage(ref_seg_img)

            inputs, pads = pad_batch1_to_compatible_size(val_inputs)
            inputs = inputs.cuda()

            with torch.no_grad():
                val_outputs_1 = model(inputs)
                pre_segs = torch.sigmoid(val_outputs_1)

            maxz, maxy, maxx = pre_segs.size(2) - pads[0], pre_segs.size(3) - pads[1], pre_segs.size(4) - pads[2]
            pre_segs = pre_segs[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()

            segs = torch.zeros((1, 2, ref_seg.shape[0], ref_seg.shape[1], ref_seg.shape[2]))
            segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs[0]

            segs = segs[0].numpy() > 0.5

            left = segs[0]
            right = segs[1]
            labelmap = np.zeros(segs[0].shape, dtype=np.uint8)

            labelmap[left] = 1
            labelmap[right] = 2

            print(np.unique(labelmap))

            labelmap = sitk.GetImageFromArray(labelmap)
            labelmap.CopyInformation(ref_seg_img)
            prediction = labelmap

            patient_id = seg_path.split('/')[-1]
            ref_seg = sitk.GetArrayFromImage(ref_seg_img)
            refmap_left, refmap_right = [np.zeros_like(ref_seg) for i in range(2)]

            refmap_left = ref_seg == 1
            refmap_right = ref_seg == 2

            refmap = np.stack([refmap_left, refmap_right])

            print(f"PRED: {segs.shape}, REF: {refmap.shape}")

            patient_metric_list, dice_per_case = calculate_metrics(segs, refmap, patient_id)
            metrics_list.append(patient_metric_list)

            total_metric += dice_per_case

            if save_result:
                sitk.WriteImage(prediction, f"{test_save_path}/{patient_id}")
            ith += 1

        avg_metric = total_metric / len(testloader)
        print('average metric is decoder 1 {}'.format(avg_metric))

    return avg_metric


def test_all_case_single(model, model2, testloader, patch_size=(128, 128, 96), save_result=True, test_save_path_1=None, test_save_path_2=None, th=0.5):
    ith = 0
    total_metric = 0.0
    total_metric_average = 0.0
    metrics_list = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(testloader):
            volume_batch, volume_batch_t, seg_path, crops_idx = batch['image'], batch['image_t'],batch['seg_path'][0], \
                                                             batch['crop_indexes']
            val_inputs, val_inputs_t= volume_batch.cuda(), volume_batch_t.cuda()

            val_inputs = val_inputs.cuda()

            ref_seg_img = sitk.ReadImage(seg_path)
            ref_seg = sitk.GetArrayFromImage(ref_seg_img)

            inputs, pads = pad_batch1_to_compatible_size(val_inputs)
            inputs = inputs.cuda()

            inputs_t, pads_t = pad_batch1_to_compatible_size(val_inputs_t)
            inputs_t = inputs_t.cuda()

            with torch.no_grad():
                val_outputs_1 = model(inputs)
                pre_segs1 = torch.sigmoid(val_outputs_1)

                val_outputs_2 = model2(inputs_t)
                pre_segs2 = torch.sigmoid(val_outputs_2)

            maxz, maxy, maxx = pre_segs1.size(2) - pads[0], pre_segs1.size(3) - pads[1], pre_segs1.size(4) - pads[2]
            pre_segs1 = pre_segs1[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()

            segs1 = torch.zeros((1, 2, ref_seg.shape[0], ref_seg.shape[1], ref_seg.shape[2]))
            segs1[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs1[0]

            segs1 = segs1[0].numpy() > th

            maxz, maxy, maxx = pre_segs2.size(2) - pads_t[0], pre_segs2.size(3) - pads_t[1], pre_segs2.size(4) - pads_t[2]
            pre_segs2 = pre_segs2[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()

            segs2 = torch.zeros((1, 2, ref_seg.shape[0], ref_seg.shape[1], ref_seg.shape[2]))
            segs2[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs2[0]

            segs2 = segs2[0].numpy() > th


            left1 = segs1[0]
            right1 = segs1[1]
            labelmap1 = np.zeros(segs1[0].shape, dtype=np.uint8)

            left2 = segs2[0]
            right2 = segs2[1]
            labelmap2 = np.zeros(segs2[0].shape, dtype=np.uint8)

            labelmap1[left1] = 1
            labelmap1[right1] = 2

            labelmap2[left2] = 1
            labelmap2[right2] = 2

            labelmap1 = sitk.GetImageFromArray(labelmap1)
            labelmap1.CopyInformation(ref_seg_img)
            prediction1 = labelmap1

            labelmap2 = sitk.GetImageFromArray(labelmap2)
            labelmap2.CopyInformation(ref_seg_img)
            prediction2 = labelmap2

            patient_id = seg_path.split('/')[-1]

            if save_result:
                # LISA_HF_12345_hipp_prediction
                case_name = patient_id.replace('LISA_VALIDATION', 'LISA_HF')
                case_name = case_name.replace('_ciso', '_hipp_prediction')
                sitk.WriteImage(prediction1, f"{test_save_path_1}/{case_name}")
                sitk.WriteImage(prediction2, f"{test_save_path_2}/{case_name}")

            ith += 1

    return None


def test_all_case(model, model2, testloader, patch_size=(128, 128, 96), save_result=True, test_save_path=None, th=0.5):
    ith = 0
    total_metric = 0.0
    total_metric_average = 0.0
    metrics_list = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(testloader):
            volume_batch, volume_batch_t, seg_path, crops_idx = batch['image'], batch['image_t'],batch['seg_path'][0], \
                                                             batch['crop_indexes']
            val_inputs, val_inputs_t= volume_batch.cuda(), volume_batch_t.cuda()

            val_inputs = val_inputs.cuda()

            ref_seg_img = sitk.ReadImage(seg_path)
            ref_seg = sitk.GetArrayFromImage(ref_seg_img)

            inputs, pads = pad_batch1_to_compatible_size(val_inputs)
            inputs = inputs.cuda()

            inputs_t, pads_t = pad_batch1_to_compatible_size(val_inputs_t)
            inputs_t = inputs_t.cuda()

            with torch.no_grad():
                val_outputs_1 = model(inputs)
                pre_segs1 = torch.sigmoid(val_outputs_1)

                val_outputs_2 = model2(inputs_t)
                pre_segs2 = torch.sigmoid(val_outputs_2)

            maxz, maxy, maxx = pre_segs1.size(2) - pads[0], pre_segs1.size(3) - pads[1], pre_segs1.size(4) - pads[2]
            pre_segs1 = pre_segs1[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()

            segs1 = torch.zeros((1, 2, ref_seg.shape[0], ref_seg.shape[1], ref_seg.shape[2]))
            segs1[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs1[0]

            segs1 = segs1[0].numpy() > th

            maxz, maxy, maxx = pre_segs2.size(2) - pads_t[0], pre_segs2.size(3) - pads_t[1], pre_segs2.size(4) - pads_t[2]
            pre_segs2 = pre_segs2[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()

            segs2 = torch.zeros((1, 2, ref_seg.shape[0], ref_seg.shape[1], ref_seg.shape[2]))
            segs2[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs2[0]

            segs2 = segs2[0].numpy() > th

            segs = np.logical_or(segs1, segs2)

            left = segs[0]
            right = segs[1]
            labelmap = np.zeros(segs[0].shape, dtype=np.uint8)

            labelmap[left] = 1
            labelmap[right] = 2

            print(np.unique(labelmap))

            labelmap = sitk.GetImageFromArray(labelmap)
            labelmap.CopyInformation(ref_seg_img)
            prediction = labelmap

            patient_id = seg_path.split('/')[-1]
            ref_seg = sitk.GetArrayFromImage(ref_seg_img)
            refmap_left, refmap_right = [np.zeros_like(ref_seg) for i in range(2)]

            refmap_left = ref_seg == 1
            refmap_right = ref_seg == 2

            refmap = np.stack([refmap_left, refmap_right])

            print(f"PRED: {segs.shape}, REF: {refmap.shape}")

            if save_result:
                # LISA_HF_12345_hipp_prediction
                case_name = patient_id.replace('LISA_VALIDATION', 'LISA_HF')
                case_name = case_name.replace('_ciso', '_hipp_prediction')
                sitk.WriteImage(prediction, f"{test_save_path}/{case_name}")

            ith += 1

    return None


def test_all_case_unet(model, testloader, patch_size=(128, 128, 96), save_result=True, test_save_path=None):
    ith = 0
    total_metric = 0.0
    total_metric_average = 0.0
    metrics_list = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(testloader):
            volume_batch, volume_batch_t, seg_path, crops_idx = batch['image'], batch['image_t'],batch['seg_path'][0], \
                                                             batch['crop_indexes']
            val_inputs, val_inputs_t= volume_batch.cuda(), volume_batch_t.cuda()

            val_inputs = val_inputs.cuda()

            ref_seg_img = sitk.ReadImage(seg_path)
            ref_seg = sitk.GetArrayFromImage(ref_seg_img)

            inputs, pads = pad_batch1_to_compatible_size(val_inputs)
            inputs = inputs.cuda()

            with torch.no_grad():
                val_outputs_1 = model(inputs)
                pre_segs1 = torch.sigmoid(val_outputs_1)

            maxz, maxy, maxx = pre_segs1.size(2) - pads[0], pre_segs1.size(3) - pads[1], pre_segs1.size(4) - pads[2]
            pre_segs1 = pre_segs1[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()

            segs1 = torch.zeros((1, 2, ref_seg.shape[0], ref_seg.shape[1], ref_seg.shape[2]))
            segs1[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs1[0]

            segs = segs1[0].numpy() > 0.5

            left = segs[0]
            right = segs[1]
            labelmap = np.zeros(segs[0].shape, dtype=np.uint8)

            labelmap[left] = 1
            labelmap[right] = 2

            print(np.unique(labelmap))

            labelmap = sitk.GetImageFromArray(labelmap)
            labelmap.CopyInformation(ref_seg_img)
            prediction = labelmap

            patient_id = seg_path.split('/')[-1]
            ref_seg = sitk.GetArrayFromImage(ref_seg_img)
            refmap_left, refmap_right = [np.zeros_like(ref_seg) for i in range(2)]

            refmap_left = ref_seg == 1
            refmap_right = ref_seg == 2

            refmap = np.stack([refmap_left, refmap_right])

            print(f"PRED: {segs.shape}, REF: {refmap.shape}")

            if save_result:
                # LISA_HF_12345_hipp_prediction
                case_name = patient_id.replace('LISA_VALIDATION', 'LISA_HF')
                case_name = case_name.replace('_ciso', '_hipp_prediction')
                sitk.WriteImage(prediction, f"{test_save_path}/{case_name}")

            ith += 1

    return None

def visualize(testloader, patch_size=(128, 128, 96), save_result=True, test_save_path=None):
    ith = 0
    total_metric = 0.0
    total_metric_average = 0.0
    metrics_list = []
    with torch.no_grad():
        for step, batch in enumerate(testloader):
            val_inputs, val_inputs_t, val_labels, seg_path, crops_idx, amp, phase, f_mask, f_amp = batch['image'], batch['image_t'], batch['label'],  batch['seg_path'][0], \
                                                             batch['crop_indexes'], batch['amplitude'], batch['phase'], batch['f_mask'], batch['f_amp']

            ref_seg_img = sitk.ReadImage(seg_path)
            ref_seg = sitk.GetArrayFromImage(ref_seg_img)

            patient_id = seg_path.split('/')[-1]
            ref_seg = sitk.GetArrayFromImage(ref_seg_img)

            image = sitk.GetImageFromArray(val_inputs)
            #image.CopyInformation(ref_seg_img)

            image_t = sitk.GetImageFromArray(val_inputs_t)
            #image_t.CopyInformation(ref_seg_img)

            amplitude = sitk.GetImageFromArray(amp)
            #amplitude.CopyInformation(ref_seg_img)

            phase_o = sitk.GetImageFromArray(phase)
            #phase_o.CopyInformation(phase)

            freq_m = sitk.GetImageFromArray(f_mask)
            #freq_m.CopyInformation(ref_seg_img)

            filtered_amp = sitk.GetImageFromArray(f_amp)

            if save_result:
                sitk.WriteImage(image, f"{test_save_path}/Original_input_{patient_id}")
                sitk.WriteImage(image_t, f"{test_save_path}/Freq_mask_input_{patient_id}")
                sitk.WriteImage(amplitude, f"{test_save_path}/Amplitude_{patient_id}")
                sitk.WriteImage(phase_o, f"{test_save_path}/Phase_{patient_id}")
                sitk.WriteImage(freq_m, f"{test_save_path}/Freq_mask{patient_id}")
                sitk.WriteImage(filtered_amp, f"{test_save_path}/Filtered_amp{patient_id}")
            ith += 1

    return avg_metric


def test_all_case_monai(model, model2, testloader, patch_size=(128, 128, 96), save_result=True, test_save_path=None):
    ith = 0
    total_metric = 0.0
    total_metric_average = 0.0
    metrics_list = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(testloader):
            volume_batch, volume_batch_t, seg_path, crops_idx = batch['image'], batch['image_t'],batch['seg_path'][0], \
                                                             batch['crop_indexes']
            val_inputs, val_inputs_t= volume_batch.cuda(), volume_batch_t.cuda()

            val_inputs = val_inputs.cuda()

            ref_seg_img = sitk.ReadImage(seg_path)
            ref_seg = sitk.GetArrayFromImage(ref_seg_img)

            # inputs, pads = pad_batch1_to_compatible_size(val_inputs)
            # inputs = inputs.cuda()
            #
            # inputs_t, pads_t = pad_batch1_to_compatible_size(val_inputs_t)
            # inputs_t = inputs_t.cuda()

            with torch.no_grad():
                val_outputs_1 = inference(val_inputs, model, patch_size)
                val_outputs_1 = [post_trans(i) for i in decollate_batch(val_outputs_1)]

                val_outputs_2 = inference(val_inputs_t, model2, patch_size)
                val_outputs_2 = [post_trans(i) for i in decollate_batch(val_outputs_2)]

            segs1 = torch.zeros((1, 2, ref_seg.shape[0], ref_seg.shape[1], ref_seg.shape[2]))
            segs1[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = val_outputs_1[0]

            segs1 = segs1[0].numpy() > 0.5

            segs2 = torch.zeros((1, 2, ref_seg.shape[0], ref_seg.shape[1], ref_seg.shape[2]))
            segs2[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = val_outputs_2[0]

            segs2 = segs2[0].numpy() > 0.5

            segs = np.logical_or(segs1, segs2)

            left = segs[0]
            right = segs[1]
            labelmap = np.zeros(segs[0].shape, dtype=np.uint8)

            labelmap[left] = 1
            labelmap[right] = 2

            print(np.unique(labelmap))

            labelmap = sitk.GetImageFromArray(labelmap)
            labelmap.CopyInformation(ref_seg_img)
            prediction = labelmap

            patient_id = seg_path.split('/')[-1]
            ref_seg = sitk.GetArrayFromImage(ref_seg_img)
            refmap_left, refmap_right = [np.zeros_like(ref_seg) for i in range(2)]

            refmap_left = ref_seg == 1
            refmap_right = ref_seg == 2

            refmap = np.stack([refmap_left, refmap_right])

            print(f"PRED: {segs.shape}, REF: {refmap.shape}")

            if save_result:
                # LISA_HF_12345_hipp_prediction
                case_name = patient_id.replace('LISA_VALIDATION', 'LISA_HF')
                case_name = case_name.replace('_ciso', '_hipp_prediction')
                sitk.WriteImage(prediction, f"{test_save_path}/{case_name}")

            ith += 1

    return None


def test_all_case_monai2(model, model2, testloader, patch_size=(128, 128, 96), save_result=True, test_save_path=None):
    ith = 0
    total_metric = 0.0
    total_metric_average = 0.0
    metrics_list = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(testloader):
            volume_batch, volume_batch_t, seg_path, crops_idx = batch['image'], batch['image_t'],batch['seg_path'][0], \
                                                             batch['crop_indexes']
            val_inputs, val_inputs_t= volume_batch.cuda(), volume_batch_t.cuda()

            val_inputs = val_inputs.cuda()

            ref_seg_img = sitk.ReadImage(seg_path)
            ref_seg = sitk.GetArrayFromImage(ref_seg_img)

            with torch.no_grad():
                val_outputs_1 = inference2(val_inputs, model, patch_size)
                val_outputs_1 = [post_trans(i) for i in decollate_batch(val_outputs_1)]

                val_outputs_2 = inference2(val_inputs_t, model2, patch_size)
                val_outputs_2 = [post_trans(i) for i in decollate_batch(val_outputs_2)]

            segs1 = torch.zeros((1, 2, ref_seg.shape[0], ref_seg.shape[1], ref_seg.shape[2]))
            segs1[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = val_outputs_1[0]

            segs1 = segs1[0].numpy() > 0.5

            segs2 = torch.zeros((1, 2, ref_seg.shape[0], ref_seg.shape[1], ref_seg.shape[2]))
            segs2[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = val_outputs_2[0]

            segs2 = segs2[0].numpy() > 0.5

            segs = np.logical_or(segs1, segs2)

            left = segs[0]
            right = segs[1]
            labelmap = np.zeros(segs[0].shape, dtype=np.uint8)

            labelmap[left] = 1
            labelmap[right] = 2

            print(np.unique(labelmap))

            labelmap = sitk.GetImageFromArray(labelmap)
            labelmap.CopyInformation(ref_seg_img)
            prediction = labelmap

            patient_id = seg_path.split('/')[-1]
            ref_seg = sitk.GetArrayFromImage(ref_seg_img)
            refmap_left, refmap_right = [np.zeros_like(ref_seg) for i in range(2)]

            refmap_left = ref_seg == 1
            refmap_right = ref_seg == 2

            refmap = np.stack([refmap_left, refmap_right])

            print(f"PRED: {segs.shape}, REF: {refmap.shape}")

            if save_result:
                # LISA_HF_12345_hipp_prediction
                case_name = patient_id.replace('LISA_VALIDATION', 'LISA_HF')
                case_name = case_name.replace('_ciso', '_hipp_prediction')
                sitk.WriteImage(prediction, f"{test_save_path}/{case_name}")

            ith += 1

    return None

HAUSSDORF = "hd"
HAUSSDORF95 = "hd95"
ACC = "accuracy"
DICE = "dice"
RVEV = "rve"
ASD = "asd"
METRICS = [DICE, RVEV, HAUSSDORF, HAUSSDORF95, ASD]