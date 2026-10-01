#!/usr/bin/env python3
"""
预计算 CLIP 文本特征并保存为 .npz 文件。
用法:
    python tools/precompute_clip_text_features.py --text_dir text_descriptions_example --out ./text_features.npz --device cpu

输出格式（.npz）键名示例:
    bottle_2D -> (N, D) numpy array
    bottle_3D -> (M, D) numpy array
"""
import os
import argparse
import clip
import torch
import numpy as np


def main(text_dir, out_file, model_name='ViT-B/32', device='cpu'):
    device = torch.device(device)
    model, _ = clip.load(model_name, device=device)
    model.eval()

    keys = {}
    for fname in os.listdir(text_dir):
        if not fname.endswith('.txt'):
            continue
        cat = os.path.splitext(fname)[0]
        path = os.path.join(text_dir, fname)
        texts_2d = []
        texts_3d = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if s.startswith('2D:'):
                    t = s[3:].strip()
                    if t:
                        texts_2d.append(t)
                elif s.startswith('3D:'):
                    t = s[3:].strip()
                    if t:
                        texts_3d.append(t)

        # encode lists (possibly empty)
        if texts_2d:
            with torch.no_grad():
                tokens = clip.tokenize(texts_2d).to(device)
                feats = model.encode_text(tokens).cpu().numpy().astype(np.float32)
            keys[f"{cat}_2D"] = feats
        if texts_3d:
            with torch.no_grad():
                tokens = clip.tokenize(texts_3d).to(device)
                feats = model.encode_text(tokens).cpu().numpy().astype(np.float32)
            keys[f"{cat}_3D"] = feats

    # save
    np.savez_compressed(out_file, **keys)
    print(f'saved {out_file} with keys: {list(keys.keys())}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--model', default='ViT-B/32')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    main(args.text_dir, args.out, args.model, args.device)
