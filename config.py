import os
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fr_model_ckpt_path', type=str)
    parser.add_argument('--attacker_img_path', type=str)
    parser.add_argument('--victim_img_path', type=str)
    parser.add_argument('--adv_img_path', type=str)
    parser.add_argument('--round', type=int, default=200)
    parser.add_argument('--kappa', type=float, default=0.7)
    parser.add_argument('--lam_priming', type=float, default=1.0)
    parser.add_argument('--lam_restoration', type=float, default=0.5)
    parser.add_argument('--epsilon', type=int, default=10)
    args = parser.parse_args()
    return args
