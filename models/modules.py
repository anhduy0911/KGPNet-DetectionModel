import torch
import torch.nn as nn
from torch.nn import functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import pandas as pd
import fvcore.nn.weight_init as weight_init
from detectron2.config import configurable
from typing import List, Tuple
import numpy as np
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads, FastRCNNOutputLayers
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.layers import ( 
    ShapeSpec,
    cross_entropy,
    cat,
    batched_nms
)
from detectron2.structures import Instances, Boxes
from detectron2.utils.events import get_event_storage
from utils.losses import JS_loss_fast_compute, KL_loss_fast_compute, graph_embedding_loss
import config as CFG
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

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

class GCN(nn.Module):
    def __init__(self, n_class=CFG.n_classes) -> None:
        super(GCN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(CFG.n_classes, 32)
        self.tanh = nn.Tanh()
        # self.conv2 = pyg_nn.GCNConv(32, 32)
        # self.conv3 = pyg_nn.GCNConv(32, 32)
        # self.conv4 = pyg_nn.GCNConv(32, 32)
        self.conv5 = pyg_nn.GCNConv(32, 128)
    
    def forward(self, data):
        x, edge_idx, edge_w = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_idx, edge_w)
        x = self.tanh(x)
        # x = self.conv2(x, edge_idx, edge_w)
        # x = self.relu(x)
        # x = self.conv3(x, edge_idx, edge_w)
        # x = self.relu(x)
        # x = self.conv4(x, edge_idx, edge_w)
        # x = self.relu(x)
        x = self.conv5(x, edge_idx, edge_w)

        return x

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
        # for arg in kwargs:
        #     print(arg)

        self.hidden_size = hidden_size = kwargs["hidden_size"]
        self.num_classes = num_classes = kwargs["num_classes"]
        roi_features = kwargs['input_shape']
        self.arg = kwargs.copy()
        # self.graph_embedding = torch.load(kwargs['graph_ebd_path'])
        self.device = kwargs["device"]

        kwargs.pop("device")
        kwargs.pop("hidden_size")
        kwargs.pop("graph_ebd_path")
        kwargs.pop("train_gcn")
        super().__init__(**kwargs)

        self.pseudo_detector = FastRCNNOutputLayers(**kwargs)
        # self.warmstart_pseudo_output_heads()

        self.g_embedding, self.dense_adj_matrix = self.build_graph_data()
        self.g_embedding = self.g_embedding.to(self.device)
        self.dense_adj_matrix = self.dense_adj_matrix.to(self.device)
        
        # self.graph_module = self.warmstart_gcn()
        self.gcn = GCN(num_classes)
        self.projection = nn.Sequential(
            nn.Linear(roi_features, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
        ) 
        self.attention = Attention(hidden_size, method='concat')
        self.attention_dense = nn.Linear(hidden_size * 2, hidden_size)
        self.cls_score = nn.Linear(hidden_size + roi_features, num_classes + 1)
        
        num_bbox_reg_classes = 1 if kwargs['cls_agnostic_bbox_reg'] else num_classes
        box_dim = len( kwargs['box2box_transform'].weights)
        self.bbox_pred = nn.Linear(hidden_size + roi_features, num_bbox_reg_classes * box_dim)
        self.softmax = nn.Softmax(dim=-1)

    @classmethod
    def from_config(cls, cfg):
        return {
            "device": cfg.MODEL.DEVICE,
            "train_gcn": cfg.MODEL.TRAIN_GCN,
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
            "loss_weight"           : {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT, "loss_linking": cfg.MODEL.ROI_HEADS.LINKING_LOSS_WEIGHT},
            # fmt: on
        }

    def build_graph_data(self):
        mapped_pill_idx = pickle.load(open(CFG.pill_root + 'name2id.pkl', "rb"))
        edge_index = []
        edge_weight = []
        
        pill_edge = pd.read_csv(CFG.graph_root + 'pill_pill_graph.csv', header=0)
        for x, y, w in pill_edge.values:
            assert(w > 0)
            if x in mapped_pill_idx and y in mapped_pill_idx:
                edge_index.append([mapped_pill_idx[x], mapped_pill_idx[y]])
                edge_weight.append(w)
                edge_index.append([mapped_pill_idx[y], mapped_pill_idx[x]])
                edge_weight.append(w)
        
        data = Data(x=torch.eye(self.arg['num_classes'], dtype=torch.float32), edge_index=torch.tensor(edge_index).t().contiguous(), edge_attr=torch.tensor(edge_weight).unsqueeze(1))
        # print(data)
        return data, to_dense_adj(data.edge_index, edge_attr=data.edge_attr).squeeze()

    def warmstart_gcn(self):
        import os.path as path
        # print('Warmstarting GCN...')
        # heads = ([8] * (3-1)) + [1]
        # print(heads)
        gcn = GCN(self.arg['num_classes']).to(self.device)
        
        if path.isfile(CFG.g_warmstart_path) and not self.arg['train_gcn']:
            gcn.load_state_dict(torch.load(CFG.g_warmstart_path))
            return gcn
        else:
            optimizer = torch.optim.AdamW(gcn.parameters())
            criteria = graph_embedding_loss
            gcn.train()
            for i in tqdm(range(5000)):
                optimizer.zero_grad()
                out_feats = gcn(self.g_embedding)
                loss = criteria(out_feats, self.dense_adj_matrix)
                print('Ep {}: Loss {}'.format(i, loss.item()))

                loss.backward()
                optimizer.step()
            
            torch.save(gcn.state_dict(), CFG.g_warmstart_path)
            self.visualize_g_embedding(gcn, 'g_embedding_after_warmstart')
            print('FINISH PLOTTING>>>')
            return gcn

    def visualize_g_embedding(self, gcn, name):
        out_features = gcn(self.g_embedding)
        dot_inp = torch.matmul(out_features, out_features.t())
        norm_inp = torch.norm(out_features, dim=1) + 1e-6
        norm_mtx_inp = torch.matmul(norm_inp.unsqueeze(1), norm_inp.unsqueeze(0))
        cosine_inp = dot_inp / norm_mtx_inp
        cosine_inp = 1/2 * (cosine_inp + 1)
        cosine_inp = cosine_inp - torch.eye(CFG.n_classes).to(cosine_inp.device)
        cosine_inp = cosine_inp / torch.max(cosine_inp, dim=1)[0]
        # cosine_inp_max = torch.max(cosine_inp, dim=1, keepdim=True)[0]
        # cosine_inp_min = torch.min(cosine_inp, dim=1, keepdim=True)[0]
        # cosine_inp_scaled = (cosine_inp - cosine_inp_min) / (cosine_inp_max - cosine_inp_min + 1e-6)
        cosine_inp_mask = cosine_inp > 0.8
        import seaborn as sns

        # cosine_inp_scaled = cosine_inp_scaled.detach().cpu().numpy()
        cosine_inp_mask = cosine_inp_mask.detach().cpu().numpy()
        cosine_inp = cosine_inp.detach().cpu().numpy()
        ax = sns.heatmap(cosine_inp * cosine_inp_mask, cmap='viridis', vmin=0, vmax=1)
        plt.savefig(CFG.log_dir_data + name + '.png', dpi=100)

    def warmstart_pseudo_output_heads(self):
        baseline_model = torch.load(CFG.warmstart_path + 'model_final.pth')
        # print(baseline_model['model'].keys())

        self.pseudo_detector.bbox_pred.weight.data = baseline_model['model']['roi_heads.box_predictor.bbox_pred.weight']
        self.pseudo_detector.bbox_pred.bias.data = baseline_model['model']['roi_heads.box_predictor.bbox_pred.bias']
        self.pseudo_detector.cls_score.weight.data = baseline_model['model']['roi_heads.box_predictor.cls_score.weight']
        self.pseudo_detector.cls_score.bias.data = baseline_model['model']['roi_heads.box_predictor.cls_score.bias']

        for param in self.pseudo_detector.parameters():
            param.requires_grad = False
        
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
        self.g_embedding = self.g_embedding.to(x.device)
        g_embedding = self.gcn(self.g_embedding)
        # print(g_embedding.shape)
        g_embedding = F.pad(g_embedding, (0, 0, 0, 1), mode='constant', value=0)
        # print(g_embedding.shape)
        condensed_graph_embedding = torch.mm(pseudo_scores, g_embedding)
        # print(condensed_graph_embedding.shape)
        # context attention module
        # scores = torch.mm(mapped_visual_embedding, condensed_graph_embedding.t())
        scores = self.attention(mapped_visual_embedding, condensed_graph_embedding)
        # print(scores.shape)
        distribution = self.softmax(scores)
        # print(distribution.shape)
        context_val = torch.mm(distribution.t(), mapped_visual_embedding)
        # print(context_val.shape)
        context_and_visual_vec = torch.cat([context_val, mapped_visual_embedding], dim=-1)
        # print(context_and_visual_vec.shape)
        attention_vec = nn.Tanh()(self.attention_dense(context_and_visual_vec))
        # print(attention_vec.shape)
        enhanced_vec = torch.cat([attention_vec, x], dim=-1)

        proposal_deltas = self.bbox_pred(enhanced_vec)
        scores = self.cls_score(enhanced_vec)
        
        return scores, proposal_deltas, mapped_visual_embedding, g_embedding

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas, mapped_visual, g_embedding = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        graph_gtruths = [g_embedding[p.gt_classes] for p in proposals]
        graph_gtruths = torch.stack(graph_gtruths, dim=0).view(-1, self.hidden_size).contiguous()
        
        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        losses = {
            "loss_cls": cross_entropy(scores, gt_classes, reduction="mean"),
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
            "loss_linking": JS_loss_fast_compute(graph_gtruths, mapped_visual)
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_classes`` are expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas, _ = predictions
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas, _ = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)

@ROI_HEADS_REGISTRY.register()
class KGPStandardROIHeads(StandardROIHeads):
  def __init__(self, cfg, input_shape):
    super().__init__(cfg, input_shape,
                    box_predictor=KGPNetOutputLayers(cfg))
    # self.box_predictor=KGPNetOutputLayers(cfg, input_shape)

def _log_classification_stats(pred_logits, gt_classes, prefix="fast_rcnn"):
    """
    Log the classification metrics to EventStorage.

    Args:
        pred_logits: Rx(K+1) logits. The last column is for background class.
        gt_classes: R labels
    """
    num_instances = gt_classes.numel()
    if num_instances == 0:
        return
    pred_classes = pred_logits.argmax(dim=1)
    bg_class_ind = pred_logits.shape[1] - 1

    fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_ind)
    num_fg = fg_inds.nonzero().numel()
    fg_gt_classes = gt_classes[fg_inds]
    fg_pred_classes = pred_classes[fg_inds]

    num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
    num_accurate = (pred_classes == gt_classes).nonzero().numel()
    fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

    storage = get_event_storage()
    storage.put_scalar(f"{prefix}/cls_accuracy", num_accurate / num_instances)
    if num_fg > 0:
        storage.put_scalar(f"{prefix}/fg_cls_accuracy", fg_num_accurate / num_fg)
        storage.put_scalar(f"{prefix}/false_negative", num_false_negative / num_fg)

def fast_rcnn_inference(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]