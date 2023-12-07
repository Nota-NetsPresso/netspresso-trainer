# Task heads

*Only params and stage_params will be contained*

## FC

### Field list

Fully connected layer is ...

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "fc" to use FC head. |
| `params.hidden_size` | (int) |
| `params.num_layer` | (int) |

## All-MLP decoder

### Field list

All-MLP decoder is ...

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "all_mlp_decoder" to use AllMLPDecoder head. |
| `params.decoder_hidden_size` | (int) |
| `params.classifier_dropout_prob` | (float) |

## RetinaNet head

RetinaNet head is ...

### Field list

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "all_mlp_decoder" to use AllMLPDecoder head. |
| `params.anchor_sizes` | (list[list[int]]) |
| `params.aspect_ratios` | (list[float]) |
| `params.num_anchors` | (int) |
| `params.norm_layer` | (str) |
| `params.topk_candidates` | (int) |
| `params.score_thresh` | (float) |
| `params.nms_thresh` | (float) |
| `params.class_agnostic` | (bool) |

## YOLOX head

YOLOX head is ...

### Field list

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "yolox_head" to use YOLOX head. |
| `params.act_type` | (float) |
| `params.score_thresh` | (float) |
| `params.class_agnostic` | (bool) |