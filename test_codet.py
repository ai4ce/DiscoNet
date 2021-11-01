'''
/************************************************************************
 MIT License
 Copyright (c) 2021 AI4CE Lab@NYU, MediaBrain Group@SJTU
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 *************************************************************************/
/**
 *  @file    test_codet.py
 *  @author  YIMING LI (https://roboticsyimingli.github.io/)
 *  @date    10/10/2021
 *  @version 1.0
 *
 *  @brief Test Code of Collaborative BEV Detection
 *
 *  @section DESCRIPTION
 *
 *  This is official implementation for: NeurIPS 2021 Learning Distilled Collaboration Graph for Multi-Agent Perception
 *
 */
'''
import matplotlib
matplotlib.use('Agg')
import random
import torch
import torch.optim as optim
from copy import deepcopy
import argparse
from utils.CoDetModule import *
from utils.CoDetModel import *
from utils.loss import *
from data.Dataset import V2XSIMDataset
from data.config import Config, ConfigGlobal
from utils.mean_ap import eval_map

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


def main(args):
    config = Config('train', binary=True, only_det=True)
    config_global = ConfigGlobal('train', binary=True, only_det=True)

    need_log = args.log
    num_workers = args.nworker

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    config.inference = args.inference
    if args.bound == 'upperbound':
        flag = 'upperbound'
    else:
        if args.com == 'when2com':
            flag = 'when2com'
            if args.inference == 'argmax_test':
                flag = 'who2com'
            if args.warp_flag:
                flag = flag + '_warp'
        elif args.com == 'v2v':
            flag = 'v2v'
        elif args.com == 'disco':
            flag = 'disco'
        elif args.com == 'sum':
            flag = 'sum'
        elif args.com == 'mean':
            flag = 'mean'
        elif args.com == 'max':
            flag = 'max'
        elif args.com == 'cat':
            flag = 'cat'
        elif args.com == 'agent':
            flag = 'agent'
        else:
            flag = 'lowerbound'
            if args.box_com:
                flag += '_box_com'

    print('flag', flag)
    config.flag =  flag
    config.split = 'test'
    valset = V2XSIMDataset(dataset_roots=[f'{args.data}/agent{i}' for i in range(5)], config=config, config_global=config_global, split='val', val=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=num_workers)
    print("Validation dataset size:", len(valset))

    logger_root = args.logpath if args.logpath != '' else 'logs'

    if flag == 'upperbound' or flag.startswith('lowerbound'):
        model = FaFNet(config)
    elif flag.startswith('when2com') or flag.startswith('who2com'):
        # model = PixelwiseWeightedFusionSoftmax(config, layer=args.layer)
        model = When2com(config, layer=args.layer, warp_flag=args.warp_flag)
    elif args.com == 'disco':
        model = DiscoNet(config, layer=args.layer, kd_flag=args.kd_flag)
    elif args.com == 'sum':
        model = SumFusion(config, layer=args.layer, kd_flag=args.kd_flag)
    elif args.com == 'mean':
        model = MeanFusion(config, layer=args.layer, kd_flag=args.kd_flag)
    elif args.com == 'max':
        model = MaxFusion(config, layer=args.layer, kd_flag=args.kd_flag)
    elif args.com == 'cat':
        model = CatFusion(config, layer=args.layer, kd_flag=args.kd_flag)
    elif args.com == 'agent':
        model = AgentwiseWeightedFusion(config, layer=args.layer, kd_flag=args.kd_flag)
    else:
        model = V2VNet(config, gnn_iter_times=args.gnn_iter_times, layer=args.layer, layer_channel=256)

    model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = {'cls': SoftmaxFocalClassificationLoss(), 'loc': WeightedSmoothL1LocalizationLoss()}

    fafmodule = FaFModule(model, model, config, optimizer, criterion, args.kd_flag)

    model_save_path = args.resume[:args.resume.rfind('/')]

    # if args.inference == 'argmax_test':
    #     model_save_path = model_save_path.replace('when2com', 'who2com')
    log_file_name = os.path.join(model_save_path, 'log.txt')
    saver = open(log_file_name, "a")
    saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
    saver.flush()

    # Logging the details for this experiment
    saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
    saver.write(args.__repr__() + "\n\n")
    saver.flush()

    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch'] + 1
    fafmodule.model.load_state_dict(checkpoint['model_state_dict'])
    fafmodule.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    fafmodule.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))

    #  ===== eval =====
    fafmodule.model.eval()
    save_fig_path = [check_folder(os.path.join(model_save_path, f'vis{i}')) for i in range(5)]
    tracking_path = [check_folder(os.path.join(model_save_path, f'tracking{i}')) for i in range(5)]

    # for local and global mAP evaluation
    det_results_local = [[] for i in range(5)]
    annotations_local = [[] for i in range(5)]

    tracking_file = [set()] * 5
    for cnt, sample in enumerate(valloader):
        t = time.time()
        padded_voxel_point_list, padded_voxel_points_teacher_list, label_one_hot_list, reg_target_list, reg_loss_mask_list, anchors_map_list, vis_maps_list, gt_max_iou, filenames, \
                target_agent_id_list, num_agent_list, trans_matrices_list = zip(*sample)

        filename0 = filenames[0]
        trans_matrices = torch.stack(tuple(trans_matrices_list), 1)
        target_agent_ids = torch.stack(tuple(target_agent_id_list), 1)
        num_agent = torch.stack(tuple(num_agent_list), 1)
        if flag == 'upperbound':
            padded_voxel_points = torch.cat(tuple(padded_voxel_points_teacher_list), 0)
        else:
            padded_voxel_points = torch.cat(tuple(padded_voxel_point_list), 0)
        
        label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
        reg_target = torch.cat(tuple(reg_target_list), 0)
        reg_loss_mask = torch.cat(tuple(reg_loss_mask_list), 0)
        anchors_map = torch.cat(tuple(anchors_map_list), 0)
        vis_maps = torch.cat(tuple(vis_maps_list), 0)

        data = {}
        data['bev_seq'] = padded_voxel_points.to(device)
        data['labels'] = label_one_hot.to(device)
        data['reg_targets'] = reg_target.to(device)
        data['anchors'] = anchors_map.to(device)
        data['vis_maps'] = vis_maps.to(device)
        data['reg_loss_mask'] = reg_loss_mask.to(device).type(dtype=torch.bool)

        data['target_agent_ids'] = target_agent_ids.to(device)
        data['num_agent'] = num_agent.to(device)
        data['trans_matrices'] = trans_matrices.to(device)

        if flag == 'lowerbound_box_com':
            loss, cls_loss, loc_loss, result = fafmodule.predict_all_with_box_com(data, data['trans_matrices'])
        else:
            loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1)

        # local qualitative evaluation
        num_sensor = num_agent_list[0][0].numpy()
        for k in range(num_sensor):
            data_agents = {}
            data_agents['bev_seq'] = torch.unsqueeze(padded_voxel_points[k, :, :, :, :], 1)
            data_agents['reg_targets'] = torch.unsqueeze(reg_target[k, :, :, :, :, :], 0)
            data_agents['anchors'] = torch.unsqueeze(anchors_map[k, :, :, :, :], 0)
            temp = gt_max_iou[k]
            data_agents['gt_max_iou'] = temp[0]['gt_box'][0, :, :]
            result_temp = result[k]

            temp = {'bev_seq': data_agents['bev_seq'][0, -1].cpu().numpy(), 'result': result_temp[0][0],
                    'reg_targets': data_agents['reg_targets'].cpu().numpy()[0],
                    'anchors_map': data_agents['anchors'].cpu().numpy()[0],
                    'gt_max_iou': data_agents['gt_max_iou']}
            det_results_local[k], annotations_local[k] = cal_local_mAP(config, temp, det_results_local[k], annotations_local[k])

            filename = str(filename0[0][0])
            cut = filename[filename.rfind('agent') + 7:]
            seq_name = cut[:cut.rfind('_')]
            idx = cut[cut.rfind('_') + 1:cut.rfind('/')]
            seq_save = os.path.join(save_fig_path[k], seq_name)
            check_folder(seq_save)
            idx_save = str(idx) + '.png'
            temp_ = deepcopy(temp)
            if args.visualization:
              visualization(config, temp, os.path.join(seq_save, idx_save))

            # == tracking ==
            if args.tracking:
                scene, frame = filename.split('/')[-2].split('_')
                det_file = os.path.join(tracking_path[k], f'det_{scene}.txt')
                if scene not in tracking_file[k]:
                    det_file = open(det_file, 'w')
                    tracking_file[k].add(scene)
                else:
                    det_file = open(det_file, 'a')
                det_corners = get_det_corners(config, temp_)
                for ic, c in enumerate(det_corners):
                    det_file.write(','.join([
                        str(frame),
                        '-1',
                        '{:.2f}'.format(c[0]),
                        '{:.2f}'.format(c[1]),
                        '{:.2f}'.format(c[2]),
                        '{:.2f}'.format(c[3]),
                        str(result_temp[0][0][0]['score'][ic]),
                        '-1', '-1', '-1'
                    ]) + '\n')
                    det_file.flush()

                det_file.close()


        print("Validation scene {}, at frame {}".format(seq_name, idx))
        print("Takes {} s\n".format(str(time.time() - t)))

    mean_ap_local = []
    # local mAP evaluation
    for k in range(4):
        print("Local mAP@0.5 from agent {}".format(k))
        mean_ap, _ = eval_map(det_results_local[k], annotations_local[k], scale_ranges=None, iou_thr=0.5, dataset=None, logger=None)
        mean_ap_local.append(mean_ap)
        print("Local mAP@0.7 from agent {}".format(k))
        ean_ap, _ = eval_map(det_results_local[k], annotations_local[k], scale_ranges=None, iou_thr=0.7, dataset=None, logger=None)
        mean_ap_local.append(mean_ap)

    # average local mAP evaluation
    det_results_all_local = det_results_local[0] + det_results_local[1] + det_results_local[2] + det_results_local[3]
    annotations_all_local = annotations_local[0] + annotations_local[1] + annotations_local[2] + annotations_local[3]
    print("Average Local mAP@0.5")
    mean_ap_local_average, _ = eval_map(det_results_all_local, annotations_all_local, scale_ranges=None, iou_thr=0.5, dataset=None, logger=None)
    mean_ap_local.append(mean_ap_local_average)

    print("Average Local mAP@0.7")
    mean_ap_local_average, _ = eval_map(det_results_all_local, annotations_all_local, scale_ranges=None, iou_thr=0.7, dataset=None, logger=None)
    mean_ap_local.append(mean_ap_local_average)

    print("Quantitative evaluation results of model from {}, at epoch {}".format(args.resume, start_epoch - 1))
    print("agent0 mAP@0.5 is {} and mAP@0.7 is {}".format(mean_ap_local[0], mean_ap_local[1]))
    print("agent1 mAP@0.5 is {} and mAP@0.7 is {}".format(mean_ap_local[2], mean_ap_local[3]))
    print("agent2 mAP@0.5 is {} and mAP@0.7 is {}".format(mean_ap_local[4], mean_ap_local[5]))
    print("agent3 mAP@0.5 is {} and mAP@0.7 is {}".format(mean_ap_local[6], mean_ap_local[7]))

    print("average local mAP@0.5 is {} and average local mAP@0.7 is {}".format(mean_ap_local[8], mean_ap_local[9]))

    if need_log:
        saver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default=None, type=str, help='The path to the preprocessed sparse BEV training data')
    parser.add_argument('--batch', default=4, type=int, help='Batch size')
    parser.add_argument('--nepoch', default=100, type=int, help='Number of epochs')
    parser.add_argument('--nworker', default=1, type=int, help='Number of workers')
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--log', action='store_true', help='Whether to log')
    parser.add_argument('--logpath', default='./log', help='The path to the output log file')
    parser.add_argument('--resume', default='', type=str, help='The path to the saved model that is loaded to resume training')
    parser.add_argument('--resume_teacher', default='', type=str, help='The path to the saved teacher model that is loaded to resume training')
    parser.add_argument('--layer', default=3, type=int, help='Communicate which layer in the single layer com mode')
    parser.add_argument('--warp_flag', action='store_true', help='Whether to use pose info for Ｗhen2com')
    parser.add_argument('--kd_flag', default=0, type=int, help='Whether to enable distillation (only DiscNet is 1 )')
    parser.add_argument('--kd_weight', default=100000, type=int, help='KD loss weight')
    parser.add_argument('--gnn_iter_times', default=3, type=int, help='Number of message passing for V2VNet')
    parser.add_argument('--visualization', default=True, help='Visualize validation result')
    parser.add_argument('--com', default='', type=str, help='disco/when2com/v2v/sum/mean/max/cat/agent')
    parser.add_argument('--bound', type=str, help='The input setting: lowerbound -> single-view or upperbound -> multi-view')
    parser.add_argument('--inference', type=str)
    parser.add_argument('--tracking',action='store_true')
    parser.add_argument('--box_com', action='store_true')
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parser.parse_args()
    print(args)
    main(args)