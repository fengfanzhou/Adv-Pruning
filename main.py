import os.path

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from config import get_args
from utils.fr_model.fr_interface import get_fr_model
from utils.image_process import bgr2rgb, img_normalize


def cal_dis(embedding1, embedding2):
    dis = F.cosine_similarity(embedding1, embedding2)
    return dis

def preprocess(x):
    x = bgr2rgb(x)
    x = img_normalize(x)
    return x


def read_img(img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    img = torch.tensor(img, dtype=torch.float)
    img = img.permute((2, 0, 1))
    return img


def save_img(img_path, img):
    dir_name = os.path.dirname(img_path)
    os.makedirs(dir_name, exist_ok=True)
    img = img.permute((1, 2, 0)).detach().cpu().numpy()
    cv2.imwrite(img_path, img)


def main():
    args = get_args()
    attacker_img = read_img(args.attacker_img_path).unsqueeze(0).cuda()
    victim_img = read_img(args.victim_img_path).unsqueeze(0).cuda()
    adv_image = adv_pruning(args, attacker_img, victim_img)
    save_img(args.adv_img_path, adv_image[0])
    print(f"Saved the adversarial example in {args.adv_img_path}")


def float_mask_to_bool_mask(mask, kappa):
    original_mask_shape = mask.shape
    bs = original_mask_shape[0]
    t = mask.reshape((bs, -1))
    img_numel = t.shape[1]
    k = int(img_numel * kappa)
    values, indices = t.topk(k, dim=1, largest=False)
    x = torch.ones_like(t) > 0.0
    y = x.scatter(1, indices, False)
    z = y.reshape(original_mask_shape)
    return z


def adv_pruning(args, attacker_img, victim_img):
    attacker_model = get_fr_model(args)
    attacker_img = attacker_img.cuda()
    victim_img = victim_img.cuda()
    # Priming
    adv_image = attacker_img.detach().clone().requires_grad_(True)
    attacker_feat = attacker_model(preprocess(attacker_img)).detach()
    victim_feat = attacker_model(preprocess(victim_img)).detach()
    for i in range(args.round):
        adv_feat = attacker_model(preprocess(adv_image))
        loss_impersonation = cal_dis(victim_feat, adv_feat)
        loss_impersonation = 1.0 - loss_impersonation
        loss_impersonation = args.lam_priming * loss_impersonation
        loss_impersonation = torch.mean(loss_impersonation)
        loss_dodging = cal_dis(attacker_feat, adv_feat)
        loss_dodging = 1.0 + loss_dodging
        loss_dodging = torch.mean(loss_dodging)
        loss = loss_impersonation + loss_dodging
        loss.backward()
        grad = adv_image.grad
        adv_image = adv_image - grad.sign()
        adv_image = torch.clamp(adv_image, attacker_img - args.epsilon, attacker_img + args.epsilon)
        adv_image = torch.clamp(adv_image, 0, 255)
        adv_image = adv_image.detach().clone().requires_grad_(True)
    # Pruning
    adv_image_pretrained = adv_image.detach().clone()
    p = adv_image_pretrained - attacker_img
    p_abs = torch.abs(p)
    neg_bool_original_mask = float_mask_to_bool_mask(p_abs, args.kappa)
    bool_original_mask = ~neg_bool_original_mask
    p = p * neg_bool_original_mask
    adv_image_pretrained = attacker_img + p

    # Restoration
    adv_image = adv_image_pretrained.detach().clone().requires_grad_(True)
    for i in range(args.round):
        adv_feat = attacker_model(preprocess(adv_image))
        loss_impersonation = cal_dis(victim_feat, adv_feat)
        loss_impersonation = 1.0 - loss_impersonation
        loss_impersonation = args.lam_restoration * loss_impersonation
        loss_impersonation = torch.mean(loss_impersonation)
        loss_dodging = cal_dis(attacker_feat, adv_feat)
        loss_dodging = 1.0 + loss_dodging
        loss_dodging = torch.mean(loss_dodging)
        loss = loss_impersonation + loss_dodging
        loss.backward()
        grad = adv_image.grad
        adv_image = adv_image - grad.sign()
        p = adv_image - adv_image_pretrained
        p = bool_original_mask * p
        adv_image = adv_image_pretrained + p
        adv_image = torch.clamp(adv_image, attacker_img - args.epsilon, attacker_img + args.epsilon)
        adv_image = torch.clamp(adv_image, 0, 255)
        adv_image = adv_image.detach().clone().requires_grad_(True)
    return adv_image


if __name__ == '__main__':
    main()
