import os
import time
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from ..utils.registry import Registry
from ..utils.logger import get_root_logger
from ..utils.misc import AverageMeter, intersection_and_union, intersection_and_union_gpu, make_dirs
from ..datasets.utils import collate_fn

TEST = Registry("test")


@TEST.register_module()
class SegmentationTest(object):
    """SegmentationTest
    for large outdoor point cloud
    """

    def __call__(self, cfg, test_loader, model):
        test_dataset = test_loader.dataset
        logger = get_root_logger()
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        model.eval()

        save_path = os.path.join(cfg.save_path, "result", "test_epoch{}".format(cfg.epochs))
        make_dirs(save_path)
        if "ScanNet" in cfg.dataset_type:
            sub_path = os.path.join(save_path, "submit")
            make_dirs(sub_path)
        pred_save, label_save = [], []
        for idx in range(len(test_dataset)):
            end = time.time()
            data_name = test_dataset.get_data_name(idx)
            pred_save_path = os.path.join(save_path, '{}_pred.npy'.format(data_name))
            label_save_path = os.path.join(save_path, '{}_label.npy'.format(data_name))
            if os.path.isfile(pred_save_path) and os.path.isfile(label_save_path):
                logger.info('{}/{}: {}, loaded pred and label.'.format(idx + 1, len(test_dataset), data_name))
                pred, label = np.load(pred_save_path), np.load(label_save_path)
            else:
                data_dict_list, label = test_dataset[idx]
                pred = torch.zeros((label.size, cfg.data.num_classes)).cuda()
                batch_num = int(np.ceil(len(data_dict_list) / cfg.batch_size_test))
                for i in range(batch_num):
                    s_i, e_i = i * cfg.batch_size_test, min((i + 1) * cfg.batch_size_test, len(data_dict_list))
                    input_dict = collate_fn(data_dict_list[s_i:e_i])
                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    idx_part = input_dict["index"]
                    with torch.no_grad():
                        pred_part = model(input_dict)["seg_logits"]  # (n, k)
                        pred_part = F.softmax(pred_part, -1)
                    if cfg.empty_cache:
                        torch.cuda.empty_cache()
                    bs = 0
                    for be in input_dict["offset"]:
                        pred[idx_part[bs: be], :] += pred_part[bs: be]
                        bs = be
                    logger.info('Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}'.format(
                        idx + 1, len(test_dataset), data_name=data_name, batch_idx=i, batch_num=batch_num))
                pred = pred.max(1)[1].data.cpu().numpy()
                np.save(pred_save_path, pred)
                np.save(label_save_path, label)
            intersection, union, target = intersection_and_union(pred, label, cfg.data.num_classes,
                                                                 cfg.data.ignore_index)
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - end)
            logger.info('Test: {} [{}/{}]-{} '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Accuracy {acc:.4f} ({m_acc:.4f}) '
                        'mIoU {iou:.4f} ({m_iou:.4f})'.format(data_name, idx + 1, len(test_dataset), label.size,
                                                              batch_time=batch_time, acc=acc, m_acc=m_acc,
                                                              iou=iou, m_iou=m_iou))
            pred_save.append(pred)
            label_save.append(label)
            if "ScanNet" in cfg.dataset_type:
                np.savetxt(os.path.join(save_path, "submit", '{}.txt'.format(data_name)),
                           test_dataset.class2id[pred].reshape([-1, 1]), fmt="%d")

        with open(os.path.join(save_path, "pred.pickle"), 'wb') as handle:
            pickle.dump({'pred': pred_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(save_path, "label.pickle"), 'wb') as handle:
            pickle.dump({'label': label_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}'.format(mIoU, mAcc, allAcc))
        for i in range(cfg.data.num_classes):
            logger.info('Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}'.format(
                idx=i, name=cfg.data.names[i], iou=iou_class[i], accuracy=accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
