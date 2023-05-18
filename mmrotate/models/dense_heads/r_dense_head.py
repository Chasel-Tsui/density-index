# Copyright (c) OpenMMLab. All rights reserved.
from cv2 import minAreaRect
from mmdet.core.bbox.iou_calculators.builder import build_iou_calculator
import torch.nn as nn
import torch
import numpy as np
from mmcv.ops import box_iou_rotated, DeformConvG, DeformConv2d
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmdet.core import anchor, images_to_levels, multi_apply, unmap, reduce_mean 
from mmrotate.core import obb2hbb,rotated_anchor_inside_flags
from mmrotate.ops.minarearect import minaerarect
from mmrotate.core.bbox import poly2obb, obb2poly

from ..builder import ROTATED_HEADS, build_loss
from .rotated_anchor_head import RotatedAnchorHead


@ROTATED_HEADS.register_module()
class RDenseHead(RotatedAnchorHead):
    r"""An anchor-based head used in `RotatedRetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int, optional): Number of stacked convolutions.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        anchor_generator (dict): Config dict for anchor generator
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 dilation_rate = 4,
                 iou_calculator = dict(type='RBboxMetrics2D'),
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 loss_density = dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.count = 0
        self.dilation_rate = dilation_rate
        super(RDenseHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_density = build_loss(loss_density)
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        if self.dilation_rate == False:
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                self.cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
                self.reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
        else:
            for i in range(self.stacked_convs-1):
                chn = self.in_channels if i == 0 else self.feat_channels
                self.cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
                self.reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            for i in range(1):
                self.cls_convs.append(
                        DeformConv2d(
                            self.feat_channels,
                            self.feat_channels,
                            3,
                            stride=1,
                            padding=1,
                            groups=1,
                            bias=False))           
                self.reg_convs.append(
                        DeformConv2d(
                            self.feat_channels,
                            self.feat_channels,
                            3,
                            stride=1,
                            padding=1,
                            groups=1,
                            bias=False))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 5, 3, padding=1)
        self.density = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (torch.Tensor): Features of a single scale level.

        Returns:
            tuple (torch.Tensor):

                - cls_score (torch.Tensor): Cls scores for a single scale \
                    level the channels number is num_anchors * num_classes.
                - bbox_pred (torch.Tensor): Box energies / deltas for a \
                    single scale level, the channels number is num_anchors * 4.
        """
        cls_feat = x # batch * channels * feature_w * feature_h
        reg_feat = x
        if self.dilation_rate == 0:
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)
        else:
            for reg_conv in self.reg_convs[:-1]:
                reg_feat = reg_conv(reg_feat)

            density = self.density(reg_feat) # get the predicted density batch_size * 1 * feature_w * feature_h

            item_init = torch.ones_like(density)
            item = item_init+((density<=0.2)&(density>0.1))*item_init+(density<=0.1)*item_init*2
            item = torch.clamp(item, min=0, max=self.dilation_rate) 
            zeros = torch.zeros_like(item)
            sampling_loc = torch.cat((-item,-item,-item,zeros,-item,item,zeros,-item, zeros, zeros, zeros,item,item,-item,item,zeros,item,item), dim=1)
            reg_feat = self.reg_convs[self.stacked_convs - 1](reg_feat, sampling_loc)

            for cls_conv in self.cls_convs[:-1]:
                cls_feat = cls_conv(cls_feat)
            cls_feat = self.cls_convs[self.stacked_convs - 1](cls_feat, sampling_loc)

        # offset batch_size * 18 * feature_w * feature_h (128,64,32,16,8), offset [y0, x0, y1, x1, y2, x2, ..., y8, x8]
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred, density


    def loss_single(self, cls_score, bbox_pred, density, anchors, labels, label_weights, bbox_targets, bbox_weights, density_targets, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (torch.Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (torch.Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W).
            anchors (torch.Tensor): Box reference for each scale level with
                shape (N, num_total_anchors, 5).
            labels (torch.Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (torch.Tensor): Label weights of each anchor with
                shape (N, num_total_anchors)
            bbox_targets (torch.Tensor): BBox regression targets of each anchor
            weight shape (N, num_total_anchors, 5).
            bbox_weights (torch.Tensor): BBox regression loss weights of each
                anchor with shape (N, num_total_anchors, 5).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            tuple (torch.Tensor):

                - loss_cls (torch.Tensor): cls. loss for each scale level.
                - loss_bbox (torch.Tensor): reg. loss for each scale level.
        """
        # classification loss
        labels = labels.reshape(-1)
        density_targets = density_targets.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        density = density.permute(0, 2, 3, 1).reshape(-1)
        
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 5)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        
        if self.use_qwl == True:
            # targets need to be decoded
            bg_class_ind = self.num_classes
            pos_inds = torch.where(
            ((labels >= 0) & (labels < bg_class_ind)) > 0)[0]
            num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_pred[0].device)
            num_pos = reduce_mean(num_pos)
            neg_inds = (labels >= bg_class_ind).nonzero().reshape(-1)
            num_neg = torch.tensor(
            len(neg_inds), dtype=torch.float, device=bbox_pred[0].device)
            num_neg = reduce_mean(num_neg)
            if num_pos > 0:
                pos_labels = labels[pos_inds]
                
                anchors = anchors.reshape(-1, 5)
                de_bbox_pred = self.bbox_coder.decode(anchors[pos_inds], bbox_pred[pos_inds])
                de_bbox_targets = self.bbox_coder.decode(anchors[pos_inds], bbox_targets[pos_inds])
                
                pred_pos_ious = box_iou_rotated(de_bbox_pred, de_bbox_targets.detach())
                pred_pos_ious=pred_pos_ious.diag().squeeze()
                #ave_iou = torch.mean(pred_pos_ious)
                #print(ave_iou)

                self.count += 1
                if self.count <= 2000:
                    pos_ious = 1
                else:
                    pos_ious = pred_pos_ious.clone().detach()
                cls_iou_targets = torch.zeros_like(cls_score)

                cls_iou_targets[pos_inds, pos_labels] = pos_ious


                loss_cls = self.loss_cls(
                cls_score, cls_iou_targets, weight=None, avg_factor=num_pos)

                loss_bbox = self.loss_bbox(
                bbox_pred,
                bbox_targets.detach(),
                bbox_weights,
                avg_factor=num_pos)
            else:
                cls_iou_targets = torch.zeros_like(cls_score)
                loss_cls = self.loss_cls(
                cls_score, cls_iou_targets, weight=None, avg_factor=num_neg)

                loss_bbox = self.loss_bbox(
                bbox_pred,
                bbox_targets.detach(),
                bbox_weights,
                avg_factor=num_neg)
        else:
            loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

            loss_density = self.loss_density(density, density_targets, label_weights, avg_factor=(num_total_samples/10))
        
            loss_bbox = self.loss_bbox(
                bbox_pred,
                bbox_targets,
                bbox_weights,
                avg_factor=num_total_samples)

            #loss_offsets = self.loss_offsets(pos_offsets, induced_offsets, offset_weights, avg_factor=num_pos)

        return loss_cls, loss_bbox, loss_density

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'density'))
    def loss(self,
             cls_scores,
             bbox_preds,
             density,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            cls_scores,
            bbox_preds,
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, density_targets_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i, _ in enumerate(anchor_list):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        # bbox_preds, a list of length 5 (batch * width height * 5)
        # labels, a list of length 5 (batch * width height)


        losses_cls, losses_bbox, losses_density = multi_apply(
        ##losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            density,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            density_targets_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_density=losses_density)
        ##return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def _get_targets_single(self,
                            flat_cls_scores,
                            flat_bbox_preds,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=False):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (torch.Tensor): Multi-level anchors of the image,
                which are concatenated into a single tensor of shape
                (num_anchors, 5)
            valid_flags (torch.Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (torch.Tensor): Ground truth bboxes of the image,
                shape (num_gts, 5).
            img_meta (dict): Meta info of the image.
            gt_bboxes_ignore (torch.Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 5).
            img_meta (dict): Meta info of the image.
            gt_labels (torch.Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple (list[Tensor]):

                - labels_list (list[Tensor]): Labels of each level
                - label_weights_list (list[Tensor]): Label weights of each \
                  level
                - bbox_targets_list (list[Tensor]): BBox targets of each level
                - bbox_weights_list (list[Tensor]): BBox weights of each level
                - num_total_pos (int): Number of positive samples in all images
                - num_total_neg (int): Number of negative samples in all images
        """
        num_gts = gt_bboxes.size(0)
        inside_flags = rotated_anchor_inside_flags(
            flat_anchors, valid_flags, img_meta['img_shape'][:2],
            self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        anchors = flat_anchors 
        num_valid_anchors = anchors.shape[0]

        
        if self.assign_by_circumhbbox is not None:
            gt_bboxes_assign = obb2hbb(gt_bboxes, self.assign_by_circumhbbox)
            assign_result = self.assigner.assign(
                flat_cls_scores, flat_bbox_preds, flat_anchors, gt_bboxes_assign, gt_bboxes_ignore,
                None if self.sampling else gt_labels)
        else:
            assign_result = self.assigner.assign(
                flat_cls_scores, flat_bbox_preds, flat_anchors, gt_bboxes, gt_bboxes_ignore,
                None if self.sampling else gt_labels)

        
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        density_targets = torch.zeros(num_valid_anchors, dtype=torch.long, device=labels.device)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        if len(pos_inds) > 0:

            ###### get the density target of a cetrain positive sample
            gt_density = self.iou_calculator(gt_bboxes, gt_bboxes, mode='gjsd_10')
            density_k = 1
            if num_gts == 1:
                ave_density = torch.zeros_like(gt_density).squeeze(-1)
            elif (num_gts > 1) & (num_gts <= density_k):
                ave_density  = (torch.sum(gt_density, 1)-1)/(num_gts-1)
            else:
                density, d_index = gt_density.sort(1, descending=True)
                density_k1 = density_k + 1
                density = density[...,:density_k1]
                ave_density = (torch.sum(density, 1)-1)/density_k
                ave_density = torch.clamp(ave_density, min=0, max=1)

            extended_density = ave_density.repeat(num_valid_anchors, 1) # (num_anchors * num_gts)
            density_targets[pos_inds] = extended_density[pos_inds, sampling_result.pos_assigned_gt_inds].long()


            ######

            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    anchors[pos_inds, :], sampling_result.pos_gt_bboxes) # WARNING: when encode, anchors changed 
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            density_targets = unmap(density_targets, num_total_anchors, inside_flags, fill = 0)


        return (labels, label_weights, bbox_targets, bbox_weights, density_targets, pos_inds, neg_inds, sampling_result)
    
    def get_targets(self,
                    cls_scores_list,
                    bbox_pred_list,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 5).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            offsets (list[list[Tensor]]): Offsets of DCN.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each \
                    level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels, [128^2, 64^2, 32^2, 16^2, 8^2]
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i])) # a list whose len is batch size, each element is a tensor
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        lvl_scores = []
        lvl_bboxes = []
        concat_cls_scores_list = []
        concat_bbox_pred_list = []
        for i in range(len(cls_scores_list)):
            reshaped_scores = cls_scores_list[i].detach().reshape(num_imgs,self.num_classes,-1)
            reshaped_bboxes = bbox_pred_list[i].detach().reshape(num_imgs,5,-1)
            lvl_scores.append(reshaped_scores)
            lvl_bboxes.append(reshaped_bboxes)
        cat_lvl_scores = torch.cat(lvl_scores, dim=-1)
        cat_lvl_bboxes = torch.cat(lvl_bboxes, dim=-1)
    
        for j in range(num_imgs):
            concat_cls_scores_list.append(cat_lvl_scores[j,...])
            concat_bbox_pred_list.append(cat_lvl_bboxes[j,...])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            concat_cls_scores_list,
            concat_bbox_pred_list,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, all_density_targets,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:8]
        rest_results = list(results[8:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        density_targets_list = images_to_levels(all_density_targets,
                                             num_level_anchors)

                                            
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, density_targets_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)



    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def filter_bboxes(self, cls_scores, bbox_preds):
        """Filter predicted bounding boxes at each position of the feature
        maps. Only one bounding boxes with highest score will be left at each
        position. This filter will be used in R3Det prior to the first feature
        refinement stage.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)

        Returns:
            list[list[Tensor]]: best or refined rbboxes of each level \
                of each image.
        """
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)

        num_imgs = cls_scores[0].size(0)

        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device=device)

        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            cls_score = cls_scores[lvl]
            bbox_pred = bbox_preds[lvl]

            anchors = mlvl_anchors[lvl]

            cls_score = cls_score.permute(0, 2, 3, 1)
            cls_score = cls_score.reshape(num_imgs, -1, self.num_anchors,
                                          self.cls_out_channels)

            cls_score, _ = cls_score.max(dim=-1, keepdim=True)
            best_ind = cls_score.argmax(dim=-2, keepdim=True)
            best_ind = best_ind.expand(-1, -1, -1, 5)

            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, self.num_anchors, 5)
            best_pred = bbox_pred.gather(
                dim=-2, index=best_ind).squeeze(dim=-2)

            anchors = anchors.reshape(-1, self.num_anchors, 5)

            for img_id in range(num_imgs):
                best_ind_i = best_ind[img_id]
                best_pred_i = best_pred[img_id]
                best_anchor_i = anchors.gather(
                    dim=-2, index=best_ind_i).squeeze(dim=-2)
                best_bbox_i = self.bbox_coder.decode(best_anchor_i,
                                                     best_pred_i)
                bboxes_list[img_id].append(best_bbox_i.detach())

        return bboxes_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def refine_bboxes(self, cls_scores, bbox_preds):
        """This function will be used in S2ANet, whose num_anchors=1.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, 5, H, W)

        Returns:
            list[list[Tensor]]: refined rbboxes of each level of each image.
        """
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)
        num_imgs = cls_scores[0].size(0)
        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device=device)

        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            bbox_pred = bbox_preds[lvl]
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, 5)
            anchors = mlvl_anchors[lvl]

            for img_id in range(num_imgs):
                bbox_pred_i = bbox_pred[img_id]
                decode_bbox_i = self.bbox_coder.decode(anchors, bbox_pred_i)
                bboxes_list[img_id].append(decode_bbox_i.detach())

        return bboxes_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   density,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 6) tensor, where the first 5 columns
                are bounding box positions (cx, cy, w, h, a) and the
                6-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.

        Example:
            >>> import mmcv
            >>> self = AnchorHead(
            >>>     num_classes=9,
            >>>     in_channels=1,
            >>>     anchor_generator=dict(
            >>>         type='AnchorGenerator',
            >>>         scales=[8],
            >>>         ratios=[0.5, 1.0, 2.0],
            >>>         strides=[4,]))
            >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
            >>> cfg = mmcv.Config(dict(
            >>>     score_thr=0.00,
            >>>     nms=dict(type='nms', iou_thr=1.0),
            >>>     max_per_img=10))
            >>> feat = torch.rand(1, 1, 3, 3)
            >>> cls_score, bbox_pred = self.forward_single(feat)
            >>> # note the input lists are over different levels, not images
            >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
            >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
            >>>                               img_metas, cfg)
            >>> det_bboxes, det_labels = result_list[0]
            >>> assert len(result_list) == 1
            >>> assert det_bboxes.shape[1] == 5
            >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device=device)
        
        result_list = []
        for img_id, _ in enumerate(img_metas):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if with_nms:
                # some heads don't support with_nms argument
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale)
            else:
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale,
                                                    with_nms)
            result_list.append(proposals)
        return result_list

