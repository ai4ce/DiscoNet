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
 *  @file    CoDetModel.py
 *  @author  YIMING LI (https://roboticsyimingli.github.io/)
 *  @date    10/10/2021
 *  @version 1.0
 *
 *  @brief Co-det Models of Collaborative BEV Detection
 *
 *  @section DESCRIPTION
 *
 *  This is official implementation for: NeurIPS 2021 Learning Distilled Collaboration Graph for Multi-Agent Perception
 *
 */
'''
import torch.nn.functional as F
import torch.nn as nn
import torch
from utils.model import *
import numpy as np
import copy
import torchgeometry as tgm
import random
import convolutional_rnn as convrnn


class DiscoNet(nn.Module):
    def __init__(self, config, layer=3, in_channels=13, kd_flag=True):
        super(DiscoNet, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)
        self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.agent_num = 5
        self.kd_flag = kd_flag
        self.layer = layer
        self.ModulationLayer3 = ModulationLayer3(config)
        if self.layer ==3:
            self.PixelWeightedFusion = PixelWeightedFusionSoftmax(256)
        elif self.layer ==2:
            self.PixelWeightedFusion = PixelWeightedFusionSoftmax(128)

        # Detection decoder
        self.decoder = lidar_decoder(height_feat_size=in_channels)

    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat

    def forward(self, bevs, trans_matrices, num_agent_tensor, batch_size=1):

        bevs = bevs.permute(0, 1, 4, 2, 3) # (Batch, seq, z, h, w)
        x_0,x_1,x_2,x_3,x_4 = self.u_encoder(bevs)
        device = bevs.device

        if self.layer ==4:
            feat_maps = x_4
            size = (1, 512, 16, 16)
        elif self.layer ==3:
            feat_maps = x_3
            size = (1, 256, 32, 32)
        elif self.layer == 2:
            feat_maps = x_2
            size = (1, 128, 64, 64)
        elif self.layer == 1:
            feat_maps = x_1
            size = (1, 64, 128, 128)
        elif self.layer == 0:
            feat_maps = x_0
            size = (1, 32, 256, 256)

        # print(feat_maps.shape, x_3.shape, x_2.shape, x_1.shape)

        # get feat maps for each agent [10 512 16 16] -> [2 5 512 16 16]
        feat_map = {}
        feat_list = []

        for i in range(self.agent_num):
            feat_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            feat_list.append(feat_map[i])

        local_com_mat = torch.cat(tuple(feat_list), 1) # [2 5 512 16 16] [batch, agent, channel, height, width]
        local_com_mat_update = torch.cat(tuple(feat_list), 1) # to avoid the inplace operation


        save_agent_weight_list = list()
        p = np.array([1.0, 0.0])

        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]
            for i in range(num_agent):
                tg_agent = local_com_mat[b, i]
                all_warp = trans_matrices[b, i] # transformation [2 5 5 4 4]

                neighbor_feat_list = list()
                neighbor_feat_list.append(tg_agent)

                #com_outage = random.randint(0,1)
                p_com_outage = np.random.choice([0, 1], p=p.ravel())

                if p_com_outage==1:
                    agent_wise_weight_feat = neighbor_feat_list[0]
                else:
                    for j in range(num_agent):
                        if j != i:
                            nb_agent = torch.unsqueeze(local_com_mat[b, j], 0) # [1 512 16 16]
                            nb_warp = all_warp[j] # [4 4]
                            # normalize the translation vector
                            x_trans = (4*nb_warp[0, 3])/128
                            y_trans = -(4*nb_warp[1, 3])/128

                            theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
                            theta_rot = torch.unsqueeze(theta_rot, 0)
                            grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))  # for grid sample

                            theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
                            theta_trans = torch.unsqueeze(theta_trans, 0)
                            grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))  # for grid sample

                            #first rotate the feature map, then translate it
                            warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                            warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                            warp_feat = torch.squeeze(warp_feat_trans)
                            neighbor_feat_list.append(warp_feat)

                    # agent-wise weighted fusion
                    tmp_agent_weight_list =list()
                    sum_weight = 0
                    for k in range(num_agent):
                        cat_feat = torch.cat([tg_agent, neighbor_feat_list[k]], dim=0)
                        cat_feat = cat_feat.unsqueeze(0)
                        AgentWeight = torch.squeeze(self.PixelWeightedFusion(cat_feat))
                        tmp_agent_weight_list.append(torch.exp(AgentWeight))
                        sum_weight = sum_weight + torch.exp(AgentWeight)

                    agent_weight_list = list()
                    for k in range(num_agent):
                        AgentWeight = torch.div(tmp_agent_weight_list[k], sum_weight)
                        AgentWeight.expand([256, -1, -1])
                        agent_weight_list.append(AgentWeight)

                    agent_wise_weight_feat = 0
                    for k in range(num_agent):
                        agent_wise_weight_feat = agent_wise_weight_feat + agent_weight_list[k]*neighbor_feat_list[k]

                # feature update
                local_com_mat_update[b, i] = agent_wise_weight_feat

                #save_agent_weight_list.append(agent_weight_list)

        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(local_com_mat_update)

        if self.kd_flag == 1:
            if self.layer ==4:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,x_2,x_3,feat_fuse_mat,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 3:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,x_2,feat_fuse_mat,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 2:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,feat_fuse_mat,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 1:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,feat_fuse_mat,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 0:
                x_8, x_7, x_6, x_5 = self.decoder(feat_fuse_mat,x_1,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            x = x_8
        else:
            if self.layer ==4:
                x = self.decoder(x_0,x_1,x_2,x_3,feat_fuse_mat,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 3:
                x = self.decoder(x_0,x_1,x_2,feat_fuse_mat,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 2:
                x = self.decoder(x_0,x_1,feat_fuse_mat,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 1:
                x = self.decoder(x_0,feat_fuse_mat,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 0:
                x = self.decoder(feat_fuse_mat,x_1,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)

        # vis = vis.permute(0, 3, 1, 2)
        # if not maps is None:
        #     x = torch.cat([x,maps],axis=-1)
        # if not vis is None:
        #     x = torch.cat([x,vis],axis=1)

        # Cell Classification head
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)

        # Detection head
        loc_preds =self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        #loc_pred (N * T * W * H * loc)
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        #MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0],-1,motion_cat)
            result['state'] = motion_cls_preds

        if self.kd_flag == 1:
            return result, x_8, x_7, x_6, x_5, feat_fuse_mat
        else:
            return result


class V2VNet(nn.Module):
    def __init__(self, config, gnn_iter_times, layer, layer_channel, in_channels=13):
        super(V2VNet, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)
        self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.agent_num = 5
        self.layer = layer
        self.layer_channel = layer_channel

        # Detection decoder
        self.decoder = lidar_decoder(height_feat_size=in_channels)

        self.gnn_iter_num = gnn_iter_times
        self.convgru = convrnn.Conv2dGRU(in_channels=self.layer_channel * 2,
                                         out_channels=self.layer_channel,
                                         kernel_size=3,
                                         num_layers=1,
                                         bidirectional=False,
                                         dilation=1,
                                         stride=1)

    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat

    def forward(self, bevs, trans_matrices, num_agent_tensor, batch_size=1):
        # trans_matrices [batch 5 5 4 4]
        # num_agent_tensor, shape: [batch, num_agent]; how many non-empty agent in this scene

        bevs = bevs.permute(0, 1, 4, 2, 3) # (Batch, seq, z, h, w)
        x_0,x_1,x_2,x_3,x_4 = self.u_encoder(bevs)
        device = bevs.device

        if self.layer ==4:
            feat_maps = x_4
            size = (1, 512, 16, 16)
        elif self.layer ==3:
            feat_maps = x_3
            size = (1, 256, 32, 32)
        elif self.layer == 2:
            feat_maps = x_2
            size = (1, 128, 64, 64)
        elif self.layer == 1:
            feat_maps = x_1
            size = (1, 64, 128, 128)
        elif self.layer == 0:
            feat_maps = x_0
            size = (1, 32, 256, 256)

        # get feat maps for each agent [10 512 16 16] -> [2 5 512 16 16]
        feat_map = {}

        feat_list = []


        for i in range(self.agent_num):
            feat_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            feat_list.append(feat_map[i])


        local_com_mat = torch.cat(tuple(feat_list), 1) # [2 5 512 16 16] [batch, agent, channel, height, width]

        local_com_mat_update = torch.cat(tuple(feat_list), 1) # to avoid the inplace operation
        p = np.array([1.0, 0.0])

        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]

            agent_feat_list = list()
            for nb in range(self.agent_num):  # self.agent_num = 5
                agent_feat_list.append(local_com_mat[b, nb])

            for _ in range(self.gnn_iter_num):

                updated_feats_list = list()

                for i in range(num_agent):
                    neighbor_feat_list = list()
                    all_warp = trans_matrices[b, i] # transformation [2 5 5 4 4]
                    com_outage = np.random.choice([0, 1], p=p.ravel())

                    if com_outage == 0:
                        for j in range(num_agent):
                            if j != i:
                                nb_agent = torch.unsqueeze(agent_feat_list[j], 0) # [1 512 16 16]
                                nb_warp = all_warp[j] # [4 4]
                                # normalize the translation vector
                                x_trans = (4*nb_warp[0, 3])/128
                                y_trans = -(4*nb_warp[1, 3])/128

                                theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
                                theta_rot = torch.unsqueeze(theta_rot, 0)
                                grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))  # 得到grid 用于grid sample

                                theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
                                theta_trans = torch.unsqueeze(theta_trans, 0)
                                grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))  # 得到grid 用于grid sample

                                #first rotate the feature map, then translate it
                                warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                                warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                                warp_feat = torch.squeeze(warp_feat_trans)

                                neighbor_feat_list.append(warp_feat)

                        mean_feat = torch.mean(torch.stack(neighbor_feat_list), dim=0)  # [c, h, w]
                        cat_feat = torch.cat([agent_feat_list[i], mean_feat], dim=0)
                        cat_feat = cat_feat.unsqueeze(0).unsqueeze(0)  # [1, 1, c, h, w]
                        updated_feat, _ = self.convgru(cat_feat, None)
                        updated_feat = torch.squeeze(torch.squeeze(updated_feat, 0), 0)  # [c, h, w]
                        updated_feats_list.append(updated_feat)

                    else:
                        updated_feats_list.append(agent_feat_list[i])

                agent_feat_list = updated_feats_list

            for k in range(num_agent):
                local_com_mat_update[b, k] = agent_feat_list[k]

        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(local_com_mat_update)

        if self.layer ==4:
            x = self.decoder(x_0,x_1,x_2,x_3,feat_fuse_mat,batch_size)
        elif self.layer == 3:
            x = self.decoder(x_0,x_1,x_2,feat_fuse_mat,x_4,batch_size)
        elif self.layer == 2:
            x = self.decoder(x_0,x_1,feat_fuse_mat,x_3,x_4,batch_size)
        elif self.layer == 1:
            x = self.decoder(x_0,feat_fuse_mat,x_2,x_3,x_4,batch_size)
        elif self.layer == 0:
            x = self.decoder(feat_fuse_mat,x_1,x_2,x_3,x_4,batch_size)

        # vis = vis.permute(0, 3, 1, 2)
        # if not maps is None:
        #     x = torch.cat([x,maps],axis=-1)
        # if not vis is None:
        #     x = torch.cat([x,vis],axis=1)

        # Cell Classification head
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)

        # Detection head
        loc_preds =self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        #loc_pred (N * T * W * H * loc)
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        return result


class When2com(nn.Module):
    def __init__(self, config, n_classes=21, in_channels=13, feat_channel=512, feat_squeezer=-1, attention='additive',
                 has_query=True, sparse=False, layer=3, warp_flag=1, image_size=512,
                 shared_img_encoder='unified', key_size=1024, query_size=32):
        super(When2com, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)

        self.sparse = sparse
        self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.agent_num = 5
        self.key_size = key_size
        self.query_size = query_size
        self.shared_img_encoder = shared_img_encoder
        self.has_query = has_query
        self.warp_flag = warp_flag
        self.layer = layer

        self.key_net = km_generator(out_size=self.key_size, input_feat_sz=image_size / 32)
        self.attention_net = MIMOGeneralDotProductAttention(self.query_size, self.key_size, self.warp_flag)
        # # Message generator
        self.query_key_net = policy_net4(in_channels=in_channels)
        if self.has_query:
            self.query_net = km_generator(out_size=self.query_size, input_feat_sz=image_size / 32)

        # Detection decoder
        self.decoder = lidar_decoder(height_feat_size=in_channels)

        # List the parameters of each modules
        self.attention_paras = list(self.attention_net.parameters())
        if self.shared_img_encoder == 'unified':
            self.img_net_paras = list(self.u_encoder.parameters()) + list(self.decoder.parameters())

        self.policy_net_paras = list(self.query_key_net.parameters()) + list(
            self.key_net.parameters()) + self.attention_paras
        if self.has_query:
            self.policy_net_paras = self.policy_net_paras + list(self.query_net.parameters())

        self.all_paras = self.img_net_paras + self.policy_net_paras

        if self.motion_state:
            self.motion_cls = MotionStateHead(config)

    def argmax_select(self, warp_flag, val_mat, prob_action):
        # v(batch, query_num, channel, size, size)
        cls_num = prob_action.shape[1]

        coef_argmax = F.one_hot(prob_action.max(dim=1)[1],  num_classes=cls_num).type(torch.cuda.FloatTensor)
        coef_argmax = coef_argmax.transpose(1, 2)
        attn_shape = coef_argmax.shape
        bats, key_num, query_num = attn_shape[0], attn_shape[1], attn_shape[2]
        coef_argmax_exp = coef_argmax.view(bats, key_num, query_num, 1, 1, 1)


        if warp_flag==1:
            v_exp = val_mat
        else:
            v_exp = torch.unsqueeze(val_mat, 2)
            v_exp = v_exp.expand(-1, -1, query_num, -1, -1, -1)


        output = coef_argmax_exp * v_exp  # (batch,4,channel,size,size)
        feat_argmax = output.sum(1)  # (batch,1,channel,size,size)


        # compute connect
        count_coef = copy.deepcopy(coef_argmax)
        ind = np.diag_indices(self.agent_num)
        count_coef[:, ind[0], ind[1]] = 0
        num_connect = torch.nonzero(count_coef).shape[0] / (self.agent_num * count_coef.shape[0])

        return feat_argmax, coef_argmax, num_connect

    def activated_select(self, warp_flag, val_mat, prob_action, thres=0.2):

        coef_act = torch.mul(prob_action, (prob_action > thres).float())
        attn_shape = coef_act.shape
        bats, key_num, query_num = attn_shape[0], attn_shape[1], attn_shape[2]
        coef_act_exp = coef_act.view(bats, key_num, query_num, 1, 1, 1)

        if warp_flag==1:
            v_exp = val_mat
        else:
            v_exp = torch.unsqueeze(val_mat, 2)
            v_exp = v_exp.expand(-1, -1, query_num, -1, -1, -1)

        output = coef_act_exp * v_exp  # (batch,4,channel,size,size)
        feat_act = output.sum(1)  # (batch,1,channel,size,size)

        # compute connect
        count_coef = coef_act.clone()
        ind = np.diag_indices(self.agent_num)
        count_coef[:, ind[0], ind[1]] = 0
        num_connect = torch.nonzero(count_coef).shape[0] / (self.agent_num * count_coef.shape[0])
        return feat_act, coef_act, num_connect

    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat

    def forward(self, bevs, trans_matrices, num_agent_tensor, maps=None, vis=None, training=True, MO_flag=True, inference='activated', batch_size=1):

        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)
        # vis = vis.permute(0, 3, 1, 2)
        # pass encoder
        x,x_1,x_2,x_3,x_4 = self.u_encoder(bevs)
        device = bevs.device

        if self.layer ==4:
            feat_maps = x_4
            if self.warp_flag:
                size = (1, 512, 16, 16)
                val_mat = torch.zeros(batch_size, 5, 5, 512, 16, 16).to(device)
        elif self.layer ==3:
            feat_maps = x_3
            if self.warp_flag:
                size = (1, 256, 32, 32)
                val_mat = torch.zeros(batch_size, 5, 5, 256, 32, 32).to(device)
        elif self.layer == 2:
            feat_maps = x_2
            if self.warp_flag:
                size = (1, 128, 64, 64)
                val_mat = torch.zeros(batch_size, 5, 5, 128, 64, 64).to(device)

        # get feat maps for each agent
        feat_map = {}
        feat_list = []
        for i in range(self.agent_num):
            feat_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            feat_list.append(feat_map[i])

        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''
         generate value matrix for each agent, Yiming, 2021.4.22
         
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        if self.warp_flag==1:
            local_com_mat = torch.cat(tuple(feat_list), 1) # [2 5 512 16 16] [batch, agent, channel, height, width]
            for b in range(batch_size):
                num_agent = num_agent_tensor[b, 0]
                for i in range(num_agent):
                    tg_agent = local_com_mat[b, i]
                    all_warp = trans_matrices[b, i] # transformation [2 5 5 4 4]
                    for j in range(num_agent):
                        if j==i:
                            val_mat[b, i, j] = tg_agent
                        else:
                            nb_agent = torch.unsqueeze(local_com_mat[b, j], 0) # [1 512 16 16]
                            nb_warp = all_warp[j] # [4 4]
                            # normalize the translation vector
                            x_trans = (4*nb_warp[0, 3])/128
                            y_trans = -(4*nb_warp[1, 3])/128

                            theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
                            theta_rot = torch.unsqueeze(theta_rot, 0)
                            grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))  # 得到grid 用于grid sample

                            theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
                            theta_trans = torch.unsqueeze(theta_trans, 0)
                            grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))  # 得到grid 用于grid sample

                            #first rotate the feature map, then translate it
                            warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                            warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                            warp_feat = torch.squeeze(warp_feat_trans)

                            val_mat[b, i, j] = warp_feat
        else:
            val_mat = torch.cat(tuple(feat_list), 1)

        # pass feature maps through key and query generator
        query_key_maps = self.query_key_net(bevs)
        keys = self.key_net(query_key_maps)
        
        if self.has_query:
            querys = self.query_net(query_key_maps)
        # get key and query
        key = {}
        query = {}
        key_list = []
        query_list = []

        for i in range(self.agent_num):
            key[i] = torch.unsqueeze(keys[batch_size * i:batch_size * (i + 1)], 1)
            key_list.append(key[i])
            if self.has_query:
                query[i] = torch.unsqueeze(querys[batch_size * i:batch_size * (i + 1)], 1)
            else:
                query[i] = torch.ones(batch_size, 1, self.query_size).to('cuda')
            query_list.append(query[i])

        key_mat = torch.cat(tuple(key_list), 1)
        query_mat = torch.cat(tuple(query_list), 1)
        if MO_flag:
            query_mat = query_mat
        else:
            query_mat = torch.unsqueeze(query_mat[:,0,:],1)

        feat_fuse, prob_action = self.attention_net(query_mat, key_mat, val_mat, sparse=self.sparse)
        #print(query_mat.shape, key_mat.shape, val_mat.shape, feat_fuse.shape)
        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(feat_fuse)
        if self.layer ==4:
            x = self.decoder(x,x_1,x_2,x_3,feat_fuse_mat,batch_size)
        elif self.layer ==3:
            x = self.decoder(x,x_1,x_2,feat_fuse_mat,x_4,batch_size)
        elif self.layer == 2:
            x = self.decoder(x,x_1,feat_fuse_mat,x_3,x_4,batch_size)

        # not related to how we combine the feature (prefer to use the agnets' own frames: to reduce the bandwidth)
        small_bis = torch.eye(prob_action.shape[1])*0.001
        small_bis = small_bis.reshape((1, prob_action.shape[1], prob_action.shape[2]))
        small_bis = small_bis.repeat(prob_action.shape[0], 1, 1).cuda()
        prob_action = prob_action + small_bis

        if training:
            action = torch.argmax(prob_action, dim=1)
            num_connect = self.agent_num - 1
        else:
            if inference == 'softmax':
                action = torch.argmax(prob_action, dim=1)
                num_connect = self.agent_num - 1

            elif inference == 'argmax_test':
                print('argmax_test')
                feat_argmax, connect_mat, num_connect = self.argmax_select(self.warp_flag, val_mat, prob_action)
                feat_argmax_mat = self.agents2batch(feat_argmax)  # (batchsize*agent_num, channel, size, size)
                feat_argmax_mat = feat_argmax_mat.detach()
                pred_argmax = self.decoder(x, x_1, x_2, feat_argmax_mat, x_4, batch_size)

                action = torch.argmax(connect_mat, dim=1)
                #return pred_argmax, prob_action, action, num_connect
                x=pred_argmax
            elif inference == 'activated':
                print('activated')
                feat_act, connect_mat, num_connect = self.activated_select(self.warp_flag, val_mat, prob_action)
                feat_act_mat = self.agents2batch(feat_act)  # (batchsize*agent_num, channel, size, size)
                feat_act_mat = feat_act_mat.detach()
                if self.layer ==4:
                    pred_act = self.decoder(x, x_1, x_2, x_3, feat_act_mat,batch_size)
                elif self.layer == 3:
                    pred_act = self.decoder(x, x_1, x_2, feat_act_mat, x_4, batch_size)
                elif self.layer == 2:
                    pred_act = self.decoder(x, x_1, feat_act_mat, x_3, x_4, batch_size)
                action = torch.argmax(connect_mat, dim=1)
                #return pred_act, prob_action, action, num_connect
                x=pred_act
            else:
                raise ValueError('Incorrect inference mode')

        # if not maps is None:
        #     x = torch.cat([x,maps],axis=-1)
        # if not vis is None:
        #     x = torch.cat([x,vis],axis=1)

        # Cell Classification head
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)

        # Detection head
        loc_preds =self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        #loc_pred (N * T * W * H * loc)
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        #MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0],-1,motion_cat)          
            result['state'] = motion_cls_preds
        return result


class SumFusion(nn.Module):
    def __init__(self, config, layer=3, in_channels=13, kd_flag=True):
        super(SumFusion, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)
        self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.agent_num = 5
        self.layer = layer
        self.kd_flag = kd_flag
        # Detection decoder
        self.decoder = lidar_decoder(height_feat_size=in_channels)

    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat

    def forward(self, bevs, trans_matrices, num_agent_tensor, batch_size=1):

        bevs = bevs.permute(0, 1, 4, 2, 3) # (Batch, seq, z, h, w)
        x_0,x_1,x_2,x_3,x_4 = self.u_encoder(bevs)
        device = bevs.device

        if self.layer ==4:
            feat_maps = x_4
            size = (1, 512, 16, 16)
        elif self.layer ==3:
            feat_maps = x_3
            size = (1, 256, 32, 32)
        elif self.layer == 2:
            feat_maps = x_2
            size = (1, 128, 64, 64)
        elif self.layer == 1:
            feat_maps = x_1
            size = (1, 64, 128, 128)
        elif self.layer == 0:
            feat_maps = x_0
            size = (1, 32, 256, 256)

        # print(feat_maps.shape, x_3.shape, x_2.shape, x_1.shape)

        # get feat maps for each agent [10 512 16 16] -> [2 5 512 16 16]
        feat_map = {}
        feat_list = []

        for i in range(self.agent_num):
            feat_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            feat_list.append(feat_map[i])

        local_com_mat = torch.cat(tuple(feat_list), 1) # [2 5 512 16 16] [batch, agent, channel, height, width]
        local_com_mat_update = torch.cat(tuple(feat_list), 1) # to avoid the inplace operation

        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]
            for i in range(num_agent):
                tg_agent = local_com_mat[b, i]
                all_warp = trans_matrices[b, i] # transformation [2 5 5 4 4]

                neighbor_feat_list = list()
                neighbor_feat_list.append(tg_agent)
                for j in range(num_agent):
                    if j != i:
                        nb_agent = torch.unsqueeze(local_com_mat[b, j], 0) # [1 512 16 16]
                        nb_warp = all_warp[j] # [4 4]
                        # normalize the translation vector
                        x_trans = (4*nb_warp[0, 3])/128
                        y_trans = -(4*nb_warp[1, 3])/128

                        theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
                        theta_rot = torch.unsqueeze(theta_rot, 0)
                        grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))  # 得到grid 用于grid sample

                        theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
                        theta_trans = torch.unsqueeze(theta_trans, 0)
                        grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))  # 得到grid 用于grid sample

                        #first rotate the feature map, then translate it
                        warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                        warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                        warp_feat = torch.squeeze(warp_feat_trans)
                        neighbor_feat_list.append(warp_feat)

                # mean fusion
                sum_feat = torch.sum(torch.stack(neighbor_feat_list), dim=0)  # [c, h, w]
                # feature update
                local_com_mat_update[b, i] = sum_feat

        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(local_com_mat_update)

        if self.kd_flag == 1:
            if self.layer ==4:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,x_2,x_3,feat_fuse_mat,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 3:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,x_2,feat_fuse_mat,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 2:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,feat_fuse_mat,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 1:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,feat_fuse_mat,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 0:
                x_8, x_7, x_6, x_5 = self.decoder(feat_fuse_mat,x_1,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            x = x_8
        else:
            if self.layer ==4:
                x = self.decoder(x_0,x_1,x_2,x_3,feat_fuse_mat,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 3:
                x = self.decoder(x_0,x_1,x_2,feat_fuse_mat,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 2:
                x = self.decoder(x_0,x_1,feat_fuse_mat,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 1:
                x = self.decoder(x_0,feat_fuse_mat,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 0:
                x = self.decoder(feat_fuse_mat,x_1,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)

        # vis = vis.permute(0, 3, 1, 2)
        # if not maps is None:
        #     x = torch.cat([x,maps],axis=-1)
        # if not vis is None:
        #     x = torch.cat([x,vis],axis=1)

        # Cell Classification head
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)

        # Detection head
        loc_preds =self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        #loc_pred (N * T * W * H * loc)
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        #MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0],-1,motion_cat)
            result['state'] = motion_cls_preds

        if self.kd_flag == 1:
            return result, x_8, x_7, x_6, x_5, feat_fuse_mat
        else:
            return result


class MeanFusion(nn.Module):
    def __init__(self, config, layer=3, in_channels=13, kd_flag=True):
        super(MeanFusion, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)
        self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.agent_num = 5
        self.kd_flag = kd_flag
        self.layer = layer

        # Detection decoder
        self.decoder = lidar_decoder(height_feat_size=in_channels)

    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat

    def forward(self, bevs, trans_matrices, num_agent_tensor, batch_size=1):

        bevs = bevs.permute(0, 1, 4, 2, 3) # (Batch, seq, z, h, w)
        x_0,x_1,x_2,x_3,x_4 = self.u_encoder(bevs)
        device = bevs.device

        if self.layer ==4:
            feat_maps = x_4
            size = (1, 512, 16, 16)
        elif self.layer ==3:
            feat_maps = x_3
            size = (1, 256, 32, 32)
        elif self.layer == 2:
            feat_maps = x_2
            size = (1, 128, 64, 64)
        elif self.layer == 1:
            feat_maps = x_1
            size = (1, 64, 128, 128)
        elif self.layer == 0:
            feat_maps = x_0
            size = (1, 32, 256, 256)

        # print(feat_maps.shape, x_3.shape, x_2.shape, x_1.shape)

        # get feat maps for each agent [10 512 16 16] -> [2 5 512 16 16]
        feat_map = {}
        feat_list = []

        for i in range(self.agent_num):
            feat_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            feat_list.append(feat_map[i])

        local_com_mat = torch.cat(tuple(feat_list), 1) # [2 5 512 16 16] [batch, agent, channel, height, width]
        local_com_mat_update = torch.cat(tuple(feat_list), 1) # to avoid the inplace operation

        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]
            for i in range(num_agent):
                tg_agent = local_com_mat[b, i]
                all_warp = trans_matrices[b, i] # transformation [2 5 5 4 4]

                neighbor_feat_list = list()
                neighbor_feat_list.append(tg_agent)
                for j in range(num_agent):
                    if j != i:
                        nb_agent = torch.unsqueeze(local_com_mat[b, j], 0) # [1 512 16 16]
                        nb_warp = all_warp[j] # [4 4]
                        # normalize the translation vector
                        x_trans = (4*nb_warp[0, 3])/128
                        y_trans = -(4*nb_warp[1, 3])/128

                        theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
                        theta_rot = torch.unsqueeze(theta_rot, 0)
                        grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))  # 得到grid 用于grid sample

                        theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
                        theta_trans = torch.unsqueeze(theta_trans, 0)
                        grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))  # 得到grid 用于grid sample

                        #first rotate the feature map, then translate it
                        warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                        warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                        warp_feat = torch.squeeze(warp_feat_trans)
                        neighbor_feat_list.append(warp_feat)

                # mean fusion
                mean_feat = torch.mean(torch.stack(neighbor_feat_list), dim=0)  # [c, h, w]
                # feature update
                local_com_mat_update[b, i] = mean_feat

        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(local_com_mat_update)

        if self.kd_flag == 1:
            if self.layer ==4:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,x_2,x_3,feat_fuse_mat,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 3:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,x_2,feat_fuse_mat,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 2:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,feat_fuse_mat,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 1:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,feat_fuse_mat,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 0:
                x_8, x_7, x_6, x_5 = self.decoder(feat_fuse_mat,x_1,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            x = x_8
        else:
            if self.layer ==4:
                x = self.decoder(x_0,x_1,x_2,x_3,feat_fuse_mat,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 3:
                x = self.decoder(x_0,x_1,x_2,feat_fuse_mat,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 2:
                x = self.decoder(x_0,x_1,feat_fuse_mat,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 1:
                x = self.decoder(x_0,feat_fuse_mat,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 0:
                x = self.decoder(feat_fuse_mat,x_1,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)

        # vis = vis.permute(0, 3, 1, 2)
        # if not maps is None:
        #     x = torch.cat([x,maps],axis=-1)
        # if not vis is None:
        #     x = torch.cat([x,vis],axis=1)

        # Cell Classification head
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)

        # Detection head
        loc_preds =self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        #loc_pred (N * T * W * H * loc)
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        #MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0],-1,motion_cat)
            result['state'] = motion_cls_preds

        if self.kd_flag == 1:
            return result, x_8, x_7, x_6, x_5, feat_fuse_mat
        else:
            return result


class MaxFusion(nn.Module):
    def __init__(self, config, layer=3, in_channels=13, kd_flag=True):
        super(MaxFusion, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)
        self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.agent_num = 5
        self.kd_flag = kd_flag
        self.layer = layer

        # Detection decoder
        self.decoder = lidar_decoder(height_feat_size=in_channels)

    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat

    def forward(self, bevs, trans_matrices, num_agent_tensor, batch_size=1):

        bevs = bevs.permute(0, 1, 4, 2, 3) # (Batch, seq, z, h, w)
        x_0,x_1,x_2,x_3,x_4 = self.u_encoder(bevs)
        device = bevs.device

        if self.layer ==4:
            feat_maps = x_4
            size = (1, 512, 16, 16)
        elif self.layer ==3:
            feat_maps = x_3
            size = (1, 256, 32, 32)
        elif self.layer == 2:
            feat_maps = x_2
            size = (1, 128, 64, 64)
        elif self.layer == 1:
            feat_maps = x_1
            size = (1, 64, 128, 128)
        elif self.layer == 0:
            feat_maps = x_0
            size = (1, 32, 256, 256)

        # print(feat_maps.shape, x_3.shape, x_2.shape, x_1.shape)

        # get feat maps for each agent [10 512 16 16] -> [2 5 512 16 16]
        feat_map = {}
        feat_list = []

        for i in range(self.agent_num):
            feat_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            feat_list.append(feat_map[i])

        local_com_mat = torch.cat(tuple(feat_list), 1) # [2 5 512 16 16] [batch, agent, channel, height, width]
        local_com_mat_update = torch.cat(tuple(feat_list), 1) # to avoid the inplace operation

        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]
            for i in range(num_agent):
                tg_agent = local_com_mat[b, i]
                all_warp = trans_matrices[b, i] # transformation [2 5 5 4 4]

                neighbor_feat_list = list()
                neighbor_feat_list.append(tg_agent)
                for j in range(num_agent):
                    if j != i:
                        nb_agent = torch.unsqueeze(local_com_mat[b, j], 0) # [1 512 16 16]
                        nb_warp = all_warp[j] # [4 4]
                        # normalize the translation vector
                        x_trans = (4*nb_warp[0, 3])/128
                        y_trans = -(4*nb_warp[1, 3])/128

                        theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
                        theta_rot = torch.unsqueeze(theta_rot, 0)
                        grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))  # 得到grid 用于grid sample

                        theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
                        theta_trans = torch.unsqueeze(theta_trans, 0)
                        grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))  # 得到grid 用于grid sample

                        #first rotate the feature map, then translate it
                        warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                        warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                        warp_feat = torch.squeeze(warp_feat_trans)
                        neighbor_feat_list.append(warp_feat)

                # mean fusion
                max_feat = torch.max(torch.stack(neighbor_feat_list), dim=0)  # [c, h, w]
                # feature update
                local_com_mat_update[b, i] = max_feat.values

        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(local_com_mat_update)

        if self.kd_flag == 1:
            if self.layer ==4:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,x_2,x_3,feat_fuse_mat,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 3:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,x_2,feat_fuse_mat,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 2:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,feat_fuse_mat,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 1:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,feat_fuse_mat,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 0:
                x_8, x_7, x_6, x_5 = self.decoder(feat_fuse_mat,x_1,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            x = x_8
        else:
            if self.layer ==4:
                x = self.decoder(x_0,x_1,x_2,x_3,feat_fuse_mat,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 3:
                x = self.decoder(x_0,x_1,x_2,feat_fuse_mat,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 2:
                x = self.decoder(x_0,x_1,feat_fuse_mat,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 1:
                x = self.decoder(x_0,feat_fuse_mat,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 0:
                x = self.decoder(feat_fuse_mat,x_1,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)

        # vis = vis.permute(0, 3, 1, 2)
        # if not maps is None:
        #     x = torch.cat([x,maps],axis=-1)
        # if not vis is None:
        #     x = torch.cat([x,vis],axis=1)

        # Cell Classification head
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)

        # Detection head
        loc_preds =self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        #loc_pred (N * T * W * H * loc)
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        #MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0],-1,motion_cat)
            result['state'] = motion_cls_preds

        if self.kd_flag == 1:
            return result, x_8, x_7, x_6, x_5, feat_fuse_mat
        else:
            return result


class CatFusion(nn.Module):
    def __init__(self, config, layer=3, in_channels=13, kd_flag=True):
        super(CatFusion, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)
        self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.agent_num = 5
        self.kd_flag = kd_flag
        self.layer = layer
        self.ModulationLayer3 = ModulationLayer3(config)

        # Detection decoder
        self.decoder = lidar_decoder(height_feat_size=in_channels)

    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat

    def forward(self, bevs, trans_matrices, num_agent_tensor, batch_size=1):

        bevs = bevs.permute(0, 1, 4, 2, 3) # (Batch, seq, z, h, w)
        x_0,x_1,x_2,x_3,x_4 = self.u_encoder(bevs)
        device = bevs.device

        if self.layer ==4:
            feat_maps = x_4
            size = (1, 512, 16, 16)
        elif self.layer ==3:
            feat_maps = x_3
            size = (1, 256, 32, 32)
        elif self.layer == 2:
            feat_maps = x_2
            size = (1, 128, 64, 64)
        elif self.layer == 1:
            feat_maps = x_1
            size = (1, 64, 128, 128)
        elif self.layer == 0:
            feat_maps = x_0
            size = (1, 32, 256, 256)

        # print(feat_maps.shape, x_3.shape, x_2.shape, x_1.shape)

        # get feat maps for each agent [10 512 16 16] -> [2 5 512 16 16]
        feat_map = {}
        feat_list = []

        for i in range(self.agent_num):
            feat_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            feat_list.append(feat_map[i])

        local_com_mat = torch.cat(tuple(feat_list), 1) # [2 5 512 16 16] [batch, agent, channel, height, width]
        local_com_mat_update = torch.cat(tuple(feat_list), 1) # to avoid the inplace operation

        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]
            for i in range(num_agent):
                neighbor_feat_list = list()
                tg_agent = local_com_mat[b, i]
                all_warp = trans_matrices[b, i] # transformation [2 5 5 4 4]
                for j in range(num_agent):
                    if j != i:
                        nb_agent = torch.unsqueeze(local_com_mat[b, j], 0) # [1 512 16 16]
                        nb_warp = all_warp[j] # [4 4]
                        # normalize the translation vector
                        x_trans = (4*nb_warp[0, 3])/128
                        y_trans = -(4*nb_warp[1, 3])/128

                        theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
                        theta_rot = torch.unsqueeze(theta_rot, 0)
                        grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))  # 得到grid 用于grid sample

                        theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
                        theta_trans = torch.unsqueeze(theta_trans, 0)
                        grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))  # 得到grid 用于grid sample

                        #first rotate the feature map, then translate it
                        warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                        warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                        warp_feat = torch.squeeze(warp_feat_trans)
                        neighbor_feat_list.append(warp_feat)
                        # sum fusion
                        # tg_agent = tg_agent + warp_feat.type(dtype=torch.float32)

                # cat fusion
                mean_feat = torch.mean(torch.stack(neighbor_feat_list), dim=0)  # [c, h, w]
                cat_feat = torch.cat([tg_agent, mean_feat], dim=0)
                cat_feat = cat_feat.unsqueeze(0)  # [1, 1, c, h, w]
                modulation_feat = self.ModulationLayer3(cat_feat)

                # feature update
                local_com_mat_update[b, i] = modulation_feat

        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(local_com_mat_update)

        if self.kd_flag == 1:
            if self.layer ==4:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,x_2,x_3,feat_fuse_mat,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 3:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,x_2,feat_fuse_mat,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 2:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,feat_fuse_mat,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 1:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,feat_fuse_mat,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 0:
                x_8, x_7, x_6, x_5 = self.decoder(feat_fuse_mat,x_1,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            x = x_8
        else:
            if self.layer ==4:
                x = self.decoder(x_0,x_1,x_2,x_3,feat_fuse_mat,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 3:
                x = self.decoder(x_0,x_1,x_2,feat_fuse_mat,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 2:
                x = self.decoder(x_0,x_1,feat_fuse_mat,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 1:
                x = self.decoder(x_0,feat_fuse_mat,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 0:
                x = self.decoder(feat_fuse_mat,x_1,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)

        # vis = vis.permute(0, 3, 1, 2)
        # if not maps is None:
        #     x = torch.cat([x,maps],axis=-1)
        # if not vis is None:
        #     x = torch.cat([x,vis],axis=1)

        # Cell Classification head
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)

        # Detection head
        loc_preds =self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        #loc_pred (N * T * W * H * loc)
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        #MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0],-1,motion_cat)
            result['state'] = motion_cls_preds

        if self.kd_flag == 1:
            return result, x_8, x_7, x_6, x_5, feat_fuse_mat
        else:
            return result


class AgentwiseWeightedFusion(nn.Module):
    def __init__(self, config, layer=3, in_channels=13, kd_flag=True):
        super(AgentwiseWeightedFusion, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)
        self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.agent_num = 5
        self.kd_flag = kd_flag
        self.layer = layer
        self.ModulationLayer3 = ModulationLayer3(config)
        self.AgentWeightedFusion = AgentWeightedFusion(config)

        # Detection decoder
        self.decoder = lidar_decoder(height_feat_size=in_channels)

    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat

    def forward(self, bevs, trans_matrices, num_agent_tensor, batch_size=1):

        bevs = bevs.permute(0, 1, 4, 2, 3) # (Batch, seq, z, h, w)
        x_0,x_1,x_2,x_3,x_4 = self.u_encoder(bevs)
        device = bevs.device

        if self.layer ==4:
            feat_maps = x_4
            size = (1, 512, 16, 16)
        elif self.layer ==3:
            feat_maps = x_3
            size = (1, 256, 32, 32)
        elif self.layer == 2:
            feat_maps = x_2
            size = (1, 128, 64, 64)
        elif self.layer == 1:
            feat_maps = x_1
            size = (1, 64, 128, 128)
        elif self.layer == 0:
            feat_maps = x_0
            size = (1, 32, 256, 256)

        # print(feat_maps.shape, x_3.shape, x_2.shape, x_1.shape)

        # get feat maps for each agent [10 512 16 16] -> [2 5 512 16 16]
        feat_map = {}
        feat_list = []

        for i in range(self.agent_num):
            feat_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            feat_list.append(feat_map[i])

        local_com_mat = torch.cat(tuple(feat_list), 1) # [2 5 512 16 16] [batch, agent, channel, height, width]
        local_com_mat_update = torch.cat(tuple(feat_list), 1) # to avoid the inplace operation

        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]
            for i in range(num_agent):
                tg_agent = local_com_mat[b, i]
                all_warp = trans_matrices[b, i] # transformation [2 5 5 4 4]

                neighbor_feat_list = list()
                neighbor_feat_list.append(tg_agent)
                for j in range(num_agent):
                    if j != i:
                        nb_agent = torch.unsqueeze(local_com_mat[b, j], 0) # [1 512 16 16]
                        nb_warp = all_warp[j] # [4 4]
                        # normalize the translation vector
                        x_trans = (4*nb_warp[0, 3])/128
                        y_trans = -(4*nb_warp[1, 3])/128

                        theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
                        theta_rot = torch.unsqueeze(theta_rot, 0)
                        grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))  # 得到grid 用于grid sample

                        theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
                        theta_trans = torch.unsqueeze(theta_trans, 0)
                        grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))  # 得到grid 用于grid sample

                        #first rotate the feature map, then translate it
                        warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                        warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                        warp_feat = torch.squeeze(warp_feat_trans)
                        neighbor_feat_list.append(warp_feat)

                # agent-wise weighted fusion
                agent_weight_list =list()
                for k in range(num_agent):
                    cat_feat = torch.cat([tg_agent, neighbor_feat_list[k]], dim=0)
                    cat_feat = cat_feat.unsqueeze(0)
                    AgentWeight = self.AgentWeightedFusion(cat_feat)
                    agent_weight_list.append(AgentWeight)

                soft_agent_weight_list = torch.squeeze(F.softmax(torch.tensor(agent_weight_list).unsqueeze(0), dim=1))

                agent_wise_weight_feat = 0
                for k in range(num_agent):
                    agent_wise_weight_feat = agent_wise_weight_feat + soft_agent_weight_list[k]*neighbor_feat_list[k]

                # feature update
                local_com_mat_update[b, i] = agent_wise_weight_feat

        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(local_com_mat_update)

        if self.kd_flag == 1:
            if self.layer ==4:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,x_2,x_3,feat_fuse_mat,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 3:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,x_2,feat_fuse_mat,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 2:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,feat_fuse_mat,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 1:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,feat_fuse_mat,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 0:
                x_8, x_7, x_6, x_5 = self.decoder(feat_fuse_mat,x_1,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            x = x_8
        else:
            if self.layer ==4:
                x = self.decoder(x_0,x_1,x_2,x_3,feat_fuse_mat,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 3:
                x = self.decoder(x_0,x_1,x_2,feat_fuse_mat,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 2:
                x = self.decoder(x_0,x_1,feat_fuse_mat,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 1:
                x = self.decoder(x_0,feat_fuse_mat,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 0:
                x = self.decoder(feat_fuse_mat,x_1,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)

        # vis = vis.permute(0, 3, 1, 2)
        # if not maps is None:
        #     x = torch.cat([x,maps],axis=-1)
        # if not vis is None:
        #     x = torch.cat([x,vis],axis=1)

        # Cell Classification head
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)

        # Detection head
        loc_preds =self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        #loc_pred (N * T * W * H * loc)
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        #MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0],-1,motion_cat)
            result['state'] = motion_cls_preds

        if self.kd_flag == 1:
            return result, x_8, x_7, x_6, x_5, feat_fuse_mat
        else:
            return result


class ModulationLayer3(nn.Module):
    def __init__(self,config):
        super(ModulationLayer3, self).__init__()

        self.conv1_1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))

        return x_1


class PixelWeightedFusionSoftmax(nn.Module):
    def __init__(self,channel):
        super(PixelWeightedFusionSoftmax, self).__init__()

        self.conv1_1 = nn.Conv2d(channel*2, 128, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(128)

        self.conv1_2 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.conv1_3 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)
        self.bn1_3 = nn.BatchNorm2d(8)

        self.conv1_4 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        # self.bn1_4 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = F.relu(self.bn1_3(self.conv1_3(x_1)))
        x_1 = F.relu(self.conv1_4(x_1))

        return x_1


class AgentWeightedFusion(nn.Module):
    def __init__(self,config):
        super(AgentWeightedFusion, self).__init__()

        self.conv1_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(128)

        self.conv1_2 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.conv1_3 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)
        self.bn1_3 = nn.BatchNorm2d(8)

        self.conv1_4 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)

        # self.conv1_1 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
        # self.bn1_1 = nn.BatchNorm2d(1)
        self.conv1_5 = nn.Conv2d(1, 1, kernel_size=32, stride=1, padding=0)
        # # self.bn1_2 = nn.BatchNorm2d(1)

    def forward(self, x):
        # x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        # x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        # x_1 = F.sigmoid(self.conv1_2(x_1))
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = F.relu(self.bn1_3(self.conv1_3(x_1)))
        x_1 = F.relu(self.conv1_4(x_1))
        x_1 = F.relu(self.conv1_5(x_1))

        return x_1


class ClassificationHead(nn.Module):

    def __init__(self, config):

        super(ClassificationHead, self).__init__()
        category_num = config.category_num
        channel = 32
        if config.use_map:
            channel += 6
        if config.use_vis:
            channel += 13

        anchor_num_per_loc = len(config.anchor_size)

        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channel, category_num*anchor_num_per_loc, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(channel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


class SingleRegressionHead(nn.Module):
    def __init__(self,config):
        super(SingleRegressionHead,self).__init__()
        category_num = config.category_num
        channel = 32
        if config.use_map:
            channel += 6
        if config.use_vis:
            channel += 13

        anchor_num_per_loc = len(config.anchor_size)
        box_code_size = config.box_code_size
        if config.only_det:
            out_seq_len = 1
        else:
            out_seq_len = config.pred_len
    
        if config.binary:
            if config.only_det:
                self.box_prediction = nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(channel),
                    nn.ReLU(),
                    nn.Conv2d(channel, anchor_num_per_loc * box_code_size * out_seq_len, kernel_size=1, stride=1, padding=0))        
            else:      
                self.box_prediction = nn.Sequential(
                    nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, anchor_num_per_loc * box_code_size * out_seq_len, kernel_size=1, stride=1, padding=0))
                

    def forward(self,x):
        box = self.box_prediction(x)

        return box


class TeacherNet(nn.Module):
    def __init__(self, config):
        super(TeacherNet, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        # self.RegressionList = nn.ModuleList([RegressionHead for i in range(seq_len)])
        self.regression = SingleRegressionHead(config)
        # self.fusion_method = config.fusion_method

        # if self.use_map:
        #     if self.fusion_method == 'early_fusion':
        #         self.stpn = STPN(height_feat_size=config.map_dims[2]+config.map_channel)
        #     elif self.fusion_method == 'middle_fusion':
        #         self.stpn = STPN(height_feat_size=config.map_dims[2],use_map=True)
        #     elif self.fusion_method == 'late_fusion':
        #         self.map_encoder = MapExtractor(map_channel=config.map_channel)
        #         self.stpn = STPN(height_feat_size=config.map_dims[2])
        # else:
        self.stpn = STPN_KD(height_feat_size=config.map_dims[2])

        # if self.motion_state:
        #     self.motion_cls = MotionStateHead(config)

    def forward(self, bevs, maps=None, vis=None):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)
        # vis = vis.permute(0, 3, 1, 2)
        x_8, x_7, x_6, x_5, x_3, x_2 = self.stpn(bevs)
        return x_8, x_7, x_6, x_5, x_3, x_2


class FaFNet(nn.Module):
    def __init__(self, config):
        super(FaFNet, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)

        self.stpn = STPN_KD(height_feat_size=config.map_dims[2])

    def forward(self, bevs, maps=None, vis=None, batch_size=None):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)
        # vis = vis.permute(0, 3, 1, 2)
        x_8, x_7, x_6, x_5, x_3, x_2 = self.stpn(bevs)

        x = x_8
        # if not maps is None:
        #     x = torch.cat([x,maps],axis=-1)
        # if not vis is None:
        #     x = torch.cat([x,vis],axis=1)

        #Cell Classification head
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)

        # Detection head
        loc_preds =self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        #loc_pred (N * T * W * H * loc)
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        #MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0],-1,motion_cat)
            result['state'] = motion_cls_preds
        return result


class policy_net4(nn.Module):
    def __init__(self, in_channels=13, input_feat_sz=32):
        super(policy_net4, self).__init__()
        feat_map_sz = input_feat_sz // 4
        self.n_feat = int(256 * feat_map_sz * feat_map_sz)
        self.lidar_encoder = lidar_encoder(height_feat_size=in_channels)

        # Encoder
        # down 1 
        self.conv1 = conv2DBatchNormRelu(512, 512, k_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(512, 256, k_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(256, 256, k_size=3, stride=2, padding=1)

        # down 2
        self.conv4 = conv2DBatchNormRelu(256, 256, k_size=3, stride=1, padding=1)
        self.conv5 = conv2DBatchNormRelu(256, 256, k_size=3, stride=2, padding=1)

    def forward(self, features_map):
        _, _, _, _, outputs1 = self.lidar_encoder(features_map)
        outputs = self.conv1(outputs1)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        return outputs


class MIMOGeneralDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, query_size, key_size, warp_flag, attn_dropout=0.1):
        super().__init__()
        self.sparsemax = Sparsemax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(query_size, key_size)
        self.warp_flag = warp_flag
        print('Msg size: ',query_size,'  Key size: ', key_size)

    def forward(self, qu, k, v, sparse=True):
        # qu (batch,5,32)
        # k (batch,5,1024)
        # v (batch,5,channel,size,size)
        query = self.linear(qu)  # (batch,5,key_size)

        # normalization
        # query_norm = query.norm(p=2,dim=2).unsqueeze(2).expand_as(query)
        # query = query.div(query_norm + 1e-9)

        # k_norm = k.norm(p=2,dim=2).unsqueeze(2).expand_as(k)
        # k = k.div(k_norm + 1e-9)
        # generate the
        attn_orig = torch.bmm(k, query.transpose(2, 1))  # (batch,5,5)  column: differnt keys and the same query

        # scaling [not sure]
        # scaling = torch.sqrt(torch.tensor(k.shape[2],dtype=torch.float32)).cuda()
        # attn_orig = attn_orig/ scaling # (batch,5,5)  column: differnt keys and the same query

        attn_orig_softmax = self.softmax(attn_orig)  # (batch,5,5)

        attn_shape = attn_orig_softmax.shape
        bats, key_num, query_num = attn_shape[0], attn_shape[1], attn_shape[2]
        attn_orig_softmax_exp = attn_orig_softmax.view(bats, key_num, query_num, 1, 1, 1)

        if self.warp_flag==1:
            v_exp = v
        else:
            v_exp = torch.unsqueeze(v, 2)
            v_exp = v_exp.expand(-1, -1, query_num, -1, -1, -1)

        output = attn_orig_softmax_exp * v_exp  # (batch,5,channel,size,size)
        output_sum = output.sum(1)  # (batch,1,channel,size,size)

        return output_sum, attn_orig_softmax


class km_generator(nn.Module):
    def __init__(self, out_size=128, input_feat_sz=32):
        super(km_generator, self).__init__()
        feat_map_sz = input_feat_sz // 4
        self.n_feat = int(256 * feat_map_sz * feat_map_sz)
        self.fc = nn.Sequential(
            nn.Linear(self.n_feat, 256), #            
            nn.ReLU(inplace=True),
            nn.Linear(256, 128), #             
            nn.ReLU(inplace=True),
            nn.Linear(128, out_size)) #            

    def forward(self, features_map):
        outputs = self.fc(features_map.view(-1, self.n_feat))
        return outputs