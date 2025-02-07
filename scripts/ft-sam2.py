import os
import numpy as np
import torch
import rasterio
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Configuration
TRAIN_IMAGES_DIR = "/dss/dsstbyfs02/pn49cu/pn49cu-dss-0016/sam2_fasteo/GhanaMining3bands_final/train_imgs"
TRAIN_MASKS_DIR = "/dss/dsstbyfs02/pn49cu/pn49cu-dss-0016/sam2_fasteo/GhanaMining3bands_final/train_masks"
VAL_IMAGES_DIR = "/dss/dsstbyfs02/pn49cu/pn49cu-dss-0016/sam2_fasteo/GhanaMining3bands_final/val_imgs"
VAL_MASKS_DIR = "/dss/dsstbyfs02/pn49cu/pn49cu-dss-0016/sam2_fasteo/GhanaMining3bands_final/val_masks"

SAM2_CHECKPOINT = "/dss/dsstbyfs02/pn49cu/pn49cu-dss-0016/segment-anything-2/checkpoints/sam2_hiera_small.pt"
MODEL_CONFIG = "sam2_hiera_s.yaml"

NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 4e-5

def load_data(images_dir, masks_dir):
    """
    Load image and mask pairs from specified directories.
    
    Args:
        images_dir (str): Path to images directory
        masks_dir (str): Path to masks directory
    
    Returns:
        list: Data pairs of image and annotation paths
    """
    data = []
    image_files = sorted(f for f in os.listdir(images_dir) if f.endswith("_IMG.tif"))
    mask_files = sorted(f for f in os.listdir(masks_dir) if f.endswith("_MASK.tif"))

    for img_file in image_files:
        mask_file = img_file.replace("_IMG.tif", "_MASK.tif")
        
        if mask_file in mask_files:
            image_path = os.path.join(images_dir, img_file)
            mask_path = os.path.join(masks_dir, mask_file)
            data.append({"image": image_path, "annotation": mask_path})
        else:
            print(f"Warning: No corresponding mask for image {img_file}")
    
    return data

def read_batch(data, idx):
    """
    Read a batch of images and their corresponding masks and points.
    
    Args:
        data (list): List of image-mask pairs
        idx (int): Index of the data to read
    
    Returns:
        tuple: Image, masks, points, and labels
    """
    ent = data[idx]
    with rasterio.open(ent["image"]) as src:
        Img = np.dstack([src.read(1), src.read(2), src.read(3)])
    
    with rasterio.open(ent["annotation"]) as src:
        ann_map = src.read(1)
    
    inds = np.unique(ann_map)
    points = []
    masks = []
    
    for ind in inds:
        mask = (ann_map == ind).astype(np.uint8)
        masks.append(mask)
        coords = np.argwhere(mask > 0)
        yx = np.array(coords[np.random.randint(len(coords))])
        points.append([[yx[1], yx[0]]])
    
    return Img, np.array(masks), np.array(points), np.ones([len(masks), 1])

def read_batch_test(data):
    """
    Read a random batch from test data.
    
    Args:
        data (list): List of test image-mask pairs
    
    Returns:
        tuple: Image, masks, points, and labels
    """
    ent = data[np.random.randint(len(data))]
    
    with rasterio.open(ent["image"]) as src:
        Img = np.dstack([src.read(1), src.read(2), src.read(3)])
    
    with rasterio.open(ent["annotation"]) as src:
        ann_map = src.read(1)
    
    inds = np.unique(ann_map)
    points = []
    masks = []
    
    for ind in inds:
        mask = (ann_map == ind).astype(np.uint8)
        masks.append(mask)
        coords = np.argwhere(mask > 0)
        yx = np.array(coords[np.random.randint(len(coords))])
        points.append([[yx[1], yx[0]]])
    
    return Img, np.array(masks), np.array(points), np.ones([len(masks), 1])

def evaluate_model(predictor, test_data):
    """
    Evaluate the model on test data and calculate mean IoU.
    
    Args:
        predictor (SAM2ImagePredictor): Trained SAM2 model predictor
        test_data (list): List of test image-mask pairs
    
    Returns:
        float: Mean Intersection over Union (IoU)
    """
    total_iou = 0
    count = 0
    
    with torch.no_grad():
        for test_sample in test_data:
            image, mask, input_point, input_label = read_batch_test([test_sample])
            
            if mask.shape[0] == 0:
                continue
            
            predictor.set_image(image)
            
            mask_input, unnorm_coords, labels, _ = predictor._prep_prompts(
                input_point, input_label, box=None, mask_logits=None, normalize_coords=True
            )
            
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None
            )
            
            batched_mode = unnorm_coords.shape[0] > 1
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features
            )
            
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
            prd_mask = torch.sigmoid(prd_masks[:, 0])
            
            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            
            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            union = gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter
            iou = inter / (union + 1e-7)
            
            total_iou += iou.mean().item()
            count += 1
    
    return total_iou / count if count > 0 else 0

def main():
    # Load training and validation data
    train_data = load_data(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR)
    test_data = load_data(VAL_IMAGES_DIR, VAL_MASKS_DIR)
    
    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(test_data)}")
    
    # Initialize SAM2 model and predictor
    sam2_model = build_sam2(MODEL_CONFIG, SAM2_CHECKPOINT, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Set training mode for specific model components
    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)
    
    # Setup optimizer and gradient scaler
    optimizer = torch.optim.AdamW(
        params=predictor.model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    mean_iou = 0
    for itr in range(NUM_EPOCHS * len(train_data)):
        with torch.cuda.amp.autocast():
            image, mask, input_point, input_label = read_batch(train_data, (itr % len(train_data)))
            
            if mask.shape[0] == 0:
                continue
            
            predictor.set_image(image)
            
            mask_input, unnorm_coords, labels, _ = predictor._prep_prompts(
                input_point, input_label, box=None, mask_logits=None, normalize_coords=True
            )
            
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None
            )
            
            batched_mode = unnorm_coords.shape[0] > 1
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features
            )
            
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
            
            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])
            
            seg_loss = (-gt_mask * torch.log(prd_mask + 1e-5) - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-5)).mean()
            
            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.05
            
            predictor.model.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update mean IoU with exponential moving average
            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
            print(f"Step: {itr}, Training Accuracy (IoU): {mean_iou}")
    
    # Final model save and evaluation
    torch.save(predictor.model.state_dict(), "model_cocoa_10epochs.torch")
    print("Model saved at end of training")
    
    test_iou = evaluate_model(predictor, test_data)
    print(f"Test Accuracy (IoU) after 10 epochs: {test_iou}")

if __name__ == "__main__":
    main()