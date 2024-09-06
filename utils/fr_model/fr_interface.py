import torch

from utils.fr_model.irse import MobileFaceNet


def get_fr_model(args):
    fr_model = MobileFaceNet(512)
    fr_model.load_state_dict(torch.load(args.fr_model_ckpt_path))
    fr_model = fr_model.cuda()
    fr_model.eval()
    return fr_model