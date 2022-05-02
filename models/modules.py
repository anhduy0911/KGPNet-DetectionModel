import torch
import torch.nn as nn
from detectron2.layers import Conv2d, ShapeSpec, get_norm
import fvcore.nn.weight_init as weight_init
from detectron2.config import configurable
from typing import List
import numpy as np
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads, FastRCNNOutputLayers
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.layers import ShapeSpec

class Attention(nn.Module):
    def __init__(self, hidden_size, method="dot"):
        '''
        Module return the alignment scores
        '''
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        
        # Defining the layers/weights required depending on alignment scoring method
        if method == "general":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == "concat":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
            self.weight = nn.Linear(hidden_size, 1, bias=False)
  
    def forward(self, decoder_hidden, encoder_outputs):
        if self.method == "dot":
            # For the dot scoring method, no weights or linear layers are involved
            return encoder_outputs.bmm(decoder_hidden.view(1,-1,1)).squeeze(-1)
        elif self.method == "general":
            # For general scoring, decoder hidden state is passed through linear layers to introduce a weight matrix
            out = self.fc(decoder_hidden)
            return encoder_outputs.bmm(out.view(1,-1,1)).squeeze(-1)
        elif self.method == "concat":
            # For concat scoring, decoder hidden state and encoder outputs are concatenated first
            out = decoder_hidden + encoder_outputs.unsqueeze(1)
            out = torch.tanh(self.fc(out))
            
            return self.weight(out).squeeze(-1)

class KGPNetOutputLayers(FastRCNNOutputLayers):
    """
    Layers for predicting Fast R-CNN outputs:

    1. proposal-to-detection box regression deltas
    2. pseudo classification scores
    3. projection module
    4. attention module
    5. classification scores
    """
    @configurable
    def __init__(self, **kwargs):
        for arg in kwargs:
            print(arg)
        self.hidden_size = hidden_size = kwargs["hidden_size"]
        self.num_classes = num_classes = kwargs["num_classes"]
        roi_features = kwargs['input_shape']
        self.graph_embedding = torch.load(kwargs['graph_ebd_path'])

        kwargs.pop("hidden_size")
        kwargs.pop("graph_ebd_path")
        super().__init__(**kwargs)

        self.pseudo_detector = FastRCNNOutputLayers(**kwargs)
        self.projection = nn.Sequential(
            nn.Linear(roi_features, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
        ) 

        self.attention = Attention(hidden_size, method='concat')
        self.attention_dense = nn.Linear(hidden_size + roi_features, hidden_size)
        
        self.cls_score = nn.Linear(hidden_size, num_classes + 1)
        
        num_bbox_reg_classes = 1 if kwargs['cls_agnostic_bbox_reg'] else num_classes
        box_dim = len( kwargs['box2box_transform'].weights)
        self.bbox_pred = nn.Linear(hidden_size + roi_features, num_bbox_reg_classes * box_dim)
        self.softmax = nn.Softmax(dim=-1)

    @classmethod
    def from_config(cls, cfg):
        return {
            "input_shape": cfg.MODEL.ROI_HEADS.PREDICTOR_INPUT_SHAPE,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "graph_ebd_path": cfg.MODEL.ROI_HEADS.GRAPH_EBDS_PATH,
            "hidden_size": cfg.MODEL.ROI_HEADS.PREDICTOR_HIDDEN_SIZE,
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type"     : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight"           : {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
            # fmt: on
        }

    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        
        pseudo_scores, _ = self.pseudo_detector(x)
        mapped_visual_embedding = self.projection(x)
        # print(mapped_visual_embedding.shape)
        pseudo_scores = self.softmax(pseudo_scores)
        # g_embedding = torch.rand((self.num_classes + 1, self.hidden_size), device=x.device)
        self.graph_embedding = self.graph_embedding.to(x.device)
        condensed_graph_embedding = torch.mm(pseudo_scores, self.graph_embedding)
        # print(condensed_graph_embedding.shape)
        # context attention module
        # scores = torch.mm(mapped_visual_embedding, condensed_graph_embedding.t())
        scores = self.attention(mapped_visual_embedding, condensed_graph_embedding)
        # print(scores.shape)
        distribution = self.softmax(scores)
        # print(distribution.shape)
        context_val = torch.mm(distribution.t(), mapped_visual_embedding)
        # print(context_val.shape)
        context_and_visual_vec = torch.cat([context_val, x], dim=-1)
        # print(context_and_visual_vec.shape)
        attention_vec = nn.Tanh()(self.attention_dense(context_and_visual_vec))
        # print(attention_vec.shape)
        # enhanced_vec = torch.cat([attention_vec, x], dim=-1)

        proposal_deltas = self.bbox_pred(attention_vec)
        scores = self.cls_score(attention_vec)
        
        return scores, proposal_deltas

@ROI_HEADS_REGISTRY.register()
class KGPStandardROIHeads(StandardROIHeads):
  def __init__(self, cfg, input_shape):
    super().__init__(cfg, input_shape,
                    box_predictor=KGPNetOutputLayers(cfg))
    # self.box_predictor=KGPNetOutputLayers(cfg, input_shape)

