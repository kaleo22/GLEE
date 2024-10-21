import detectron2
import numpy as np
import cv2
import torch
from os import path
from detectron2.config import get_cfg
from GLEE.glee.models.glee_model import GLEE_Model
from GLEE.glee.config_deeplab import add_deeplab_config
from GLEE.glee.config import add_glee_config
import torch.nn.functional as F
import torchvision
import math
from scipy.optimize import linear_sum_assignment
import argparse
from PIL import Image
import os

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--version', type=str,  default='Lite', help='select model version from [Lite,Plus,Pro]')
    parser.add_argument('--input_image', type=str,  default='./Examples/000000001000.jpg', help='path to image')
    parser.add_argument('--output', type=str,  default='./outputs', help='path to save detection results')
    parser.add_argument('--task', type=str, default='detection', help='mode: detection/grounding')
    parser.add_argument('--text', type=str, default='person,bicycle,car,motorcycle,airplane', help='category list split by ,\  or a sentence')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--sim_thres', type=float, default=0.1, help='Similarity Threshold')
    return parser


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

 
def LSJ_box_postprocess( out_bbox,  padding_size, crop_size, img_h, img_w):
    # postprocess box height and width
    boxes = box_cxcywh_to_xyxy(out_bbox)
    lsj_sclae = torch.tensor([padding_size[1], padding_size[0], padding_size[1], padding_size[0]]).to(out_bbox)
    crop_scale = torch.tensor([crop_size[1], crop_size[0], crop_size[1], crop_size[0]]).to(out_bbox)
    boxes = boxes * lsj_sclae
    boxes = boxes / crop_scale
    boxes = torch.clamp(boxes,0,1)

    scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
    scale_fct = scale_fct.to(out_bbox)
    boxes = boxes * scale_fct
    return boxes

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
                [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933],
                [0.494, 0.000, 0.556], [0.494, 0.000, 0.000], [0.000, 0.745, 0.000],
                [0.700, 0.300, 0.600],[0.000, 0.447, 0.741], [0.850, 0.325, 0.098]]


def main(args):


    print(f"Is CUDA available: {torch.cuda.is_available()}")
    # True
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    # Tesla T4

    coco_class_name = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    YTBVISOVIS_class_name = ['lizard', 'cat', 'horse', 'eagle', 'frog', 'Horse', 'monkey', 'bear', 'parrot', 'giant_panda', 'truck', 'zebra', 'rabbit', 'skateboard', 'tiger', 'shark', 'Person', 'Poultry', 'Zebra', 'Airplane', 'elephant', 'Elephant', 'Turtle', 'snake', 'train', 'Dog', 'snowboard', 'airplane', 'Lizard', 'dog', 'Cat', 'earless_seal', 'boat', 'Tiger', 'motorbike', 'duck', 'fox', 'Monkey', 'Bird', 'Bear', 'tennis_racket', 'Rabbit', 'Giraffe', 'Motorcycle', 'fish', 'Boat', 'deer', 'ape', 'Bicycle', 'Parrot', 'Cow', 'turtle', 'mouse', 'owl', 'Fish', 'surfboard', 'Giant_panda', 'Sheep', 'hand', 'Vehical', 'sedan', 'leopard', 'person', 'giraffe', 'cow']
    class_agnostic_name = ['object']

    if torch.cuda.is_available():
        print('use cuda')
        device = 'cuda'
    else:
        print('use cpu')
        device='cpu'

    if 'Lite'  in args.version:
        cfg_r50 = get_cfg()
        add_deeplab_config(cfg_r50)
        add_glee_config(cfg_r50)
        conf_files_r50 = 'GLEE/configs/R50.yaml'
        checkpoints_r50 = torch.load('GLEE_DEMO_MODEL_ZOO/GLEE_R50_Scaleup10m.pth') 
        cfg_r50.merge_from_file(conf_files_r50)
        GLEEmodel = GLEE_Model(cfg_r50, None, device, None, True).to(device)
        GLEEmodel.load_state_dict(checkpoints_r50, strict=False)
        GLEEmodel.eval()
        inference_type = 'resize_shot'  # or LSJ 
    elif 'Plus' in args.version:
        cfg_swin = get_cfg()
        add_deeplab_config(cfg_swin)
        add_glee_config(cfg_swin)
        conf_files_swin = 'GLEE/configs/SwinL.yaml'
        checkpoints_swin = torch.load('GLEE_DEMO_MODEL_ZOO/GLEE_SwinL_Scaleup10m.pth') 
        cfg_swin.merge_from_file(conf_files_swin)
        GLEEmodel = GLEE_Model(cfg_swin, None, device, None, True).to(device)
        GLEEmodel.load_state_dict(checkpoints_swin, strict=False)
        GLEEmodel.eval()
        inference_type = 'resize_shot'  # or LSJ 
    elif 'Pro' in args.version:
        cfg_eva02 = get_cfg()
        add_deeplab_config(cfg_eva02)
        add_glee_config(cfg_eva02)
        conf_files_eva02 = 'GLEE/configs/EVA02.yaml'
        checkpoints_eva = torch.load('GLEE_DEMO_MODEL_ZOO/GLEE_EVA02_Scaleup10m.pth') 
        cfg_eva02.merge_from_file(conf_files_eva02)
        GLEEmodel = GLEE_Model(cfg_eva02, None, device, None, True).to(device)
        GLEEmodel.load_state_dict(checkpoints_eva, strict=False)
        GLEEmodel.eval()
        inference_type = 'LSJ'
    else:
        assert False, 'model version not defined!'

       


    pixel_mean = torch.Tensor( [123.675, 116.28, 103.53]).to(device).view(3, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).to(device).view(3, 1, 1)
    normalizer = lambda x: (x - pixel_mean) / pixel_std
    inference_size = 800
    
    size_divisibility = 32

    FONT_SCALE = 1.5e-3
    THICKNESS_SCALE = 1e-3
    TEXT_Y_OFFSET_SCALE = 1e-2 


    if inference_type != 'LSJ':
        resizer = torchvision.transforms.Resize(inference_size,antialias=True)
    else:
        resizer = torchvision.transforms.Resize(size = 1535, max_size=1536, antialias=True)


    inputimage = np.array(Image.open(args.input_image))
    


    ori_image = torch.as_tensor(np.ascontiguousarray( inputimage.transpose(2, 0, 1)))
    ori_image = normalizer(ori_image.to(device))[None,]
    _,_, ori_height, ori_width = ori_image.shape

    if inference_type == 'LSJ':
        resize_image = resizer(ori_image)
        image_size = torch.as_tensor((resize_image.shape[-2],resize_image.shape[-1]))
        re_size = resize_image.shape[-2:]
        infer_image = torch.zeros(1,3,1536,1536).to(ori_image)
        infer_image[:,:,:image_size[0],:image_size[1]] = resize_image
        padding_size = (1536,1536)
    else:
        resize_image = resizer(ori_image)
        image_size = torch.as_tensor((resize_image.shape[-2],resize_image.shape[-1]))
        re_size = resize_image.shape[-2:]
        if size_divisibility > 1:
            stride = size_divisibility
            # the last two dims are H,W, both subject to divisibility requirement
            padding_size = ((image_size + (stride - 1)).div(stride, rounding_mode="floor") * stride).tolist()
            infer_image = torch.zeros(1,3,padding_size[0],padding_size[1]).to(resize_image)
            infer_image[0,:,:image_size[0],:image_size[1]] = resize_image
            # reversed_image = infer_image*pixel_std +  pixel_mean
            # reversed_image = torch.clip(reversed_image,min=0,max=255)
            # reversed_image = reversed_image[0].permute(1,2,0)
            # reversed_image = reversed_image.int().cpu().numpy().copy()
            # cv2.imwrite('test.png',reversed_image[:,:,::-1])

    results_select=['box','name','score']  # or ['box','mask'] #选择要可视化的部分
    topK_instance = args.topk
    threshold_select = args.sim_thres

    if args.task == 'detection':
        batch_category_name = args.text.split(',')
        prompt_list = []
        task="coco"
    elif args.task == 'grounding':
        batch_category_name = []
        prompt_list = {'grounding':[args.text]}
        task="grounding"
    else:
        assert False, 'task not defined!'
        
    with torch.no_grad():
        (outputs,_) = GLEEmodel(infer_image, prompt_list, task=task, batch_name_list=batch_category_name, is_train=False)

        mask_pred = outputs['pred_masks'][0]
        mask_cls = outputs['pred_logits'][0]
        boxes_pred = outputs['pred_boxes'][0]

        scores = mask_cls.sigmoid().max(-1)[0]
        scores_per_image, topk_indices = scores.topk(topK_instance, sorted=True)
            
        valid = scores_per_image>threshold_select
        topk_indices = topk_indices[valid]
        scores_per_image = scores_per_image[valid]

        pred_class = mask_cls[topk_indices].max(-1)[1].tolist()
        pred_boxes = boxes_pred[topk_indices] 


        boxes = LSJ_box_postprocess(pred_boxes,padding_size,re_size, ori_height,ori_width)
        mask_pred = mask_pred[topk_indices]
        assert len(mask_pred)>0 ,'not enough object to visualize, turn thres bigger'
        pred_masks = F.interpolate( mask_pred[None,], size=(padding_size[0], padding_size[1]), mode="bilinear", align_corners=False  )
        pred_masks = pred_masks[:,:,:re_size[0],:re_size[1]]
        pred_masks = F.interpolate( pred_masks, size=(ori_height,ori_width), mode="bilinear", align_corners=False  )
        pred_masks = (pred_masks>0).detach().cpu().numpy()[0]
        
        if 'mask' in results_select:
            mask_image_mix_ration=0.5
            zero_mask = np.zeros_like(inputimage) 
            for nn, mask in enumerate(pred_masks):
                # mask = mask.numpy()
                mask = mask.reshape(mask.shape[0], mask.shape[1], 1)

                lar = np.concatenate((mask*COLORS[nn%12][2], mask*COLORS[nn%12][1], mask*COLORS[nn%12][0]), axis = 2)
                zero_mask = zero_mask+ lar


            lar_valid = zero_mask>0
            masked_image = lar_valid*inputimage
            img_n = masked_image*mask_image_mix_ration + np.clip(zero_mask,0,1)*255*(1-mask_image_mix_ration)
            max_p = img_n.max()
            img_n = 255*img_n/max_p
            ret = (~lar_valid*inputimage)*mask_image_mix_ration + img_n
            ret = ret.astype('uint8') 
        else:
            ret = inputimage

        if 'box' in results_select:

            line_width = max(ret.shape) /200
    
            for nn,(classid, box) in enumerate(zip(pred_class,boxes)):
                x1,y1,x2,y2 = box.long().tolist()
                RGB = (COLORS[nn%12][2]*255,COLORS[nn%12][1]*255,COLORS[nn%12][0]*255)
                cv2.rectangle(ret, (x1,y1), (x2,y2), RGB,  math.ceil(line_width) )
                if args.task == 'detection'  :
                    label = ''
                    if 'name' in results_select:
                        label +=  batch_category_name[classid]  
                    if 'score' in results_select:
                        label +=  str(scores_per_image[nn].item())[:3] 
                   
                    if len(label)==0:
                        continue
                    height, width, _ = ret.shape
                    FONT = cv2.FONT_HERSHEY_COMPLEX
                    label_width, label_height = cv2.getTextSize(label, FONT, min(width, height) * FONT_SCALE, math.ceil(min(width, height) * THICKNESS_SCALE))[0]

                    cv2.rectangle(ret, (x1,y1), (x1+label_width,(y1 -label_height) - int(height * TEXT_Y_OFFSET_SCALE)), RGB, -1)

                    cv2.putText(
                            ret,
                            label,
                            (x1, y1 - int(height * TEXT_Y_OFFSET_SCALE)),
                            fontFace=FONT,
                            fontScale=min(width, height) * FONT_SCALE,
                            thickness=math.ceil(min(width, height) * THICKNESS_SCALE),
                            color=(255,255,255),
                        )

        ret = ret.astype('uint8')
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        Image.fromarray(ret).save(os.path.join(args.output, args.input_image.split('/')[-1]))
        # cv2.imwrite( os.path.join(args.output, args.input_image.split('/')[-1]),ret )


   

if __name__ == '__main__':
    parser = argparse.ArgumentParser('image path check script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)