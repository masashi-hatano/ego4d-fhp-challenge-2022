import torch

def scaling(cfg, preds, size):
    crop_size = cfg.DATA.TRAIN_CROP_SIZE
    h, w = size
    h = h[:,None].cuda()
    w = w[:,None].cuda()
    odd = torch.zeros(preds.shape).cuda()
    even = torch.zeros(preds.shape).cuda()
    for i in range(20):
        if i%2:
            odd[:,i] = 1
        else:
            even[:,i] = 1
    preds = preds*(h*odd/crop_size+w*even/crop_size)
    
    return preds