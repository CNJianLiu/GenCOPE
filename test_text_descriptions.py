#!/usr/bin/env python3
"""
快速测试脚本：验证文本描述加载功能
"""

import sys
import os

# 测试文本加载
def test_text_loading():
    """测试文本描述是否正确加载"""
    
    text_descriptions_dir = './text_descriptions_example'
    
    if not os.path.exists(text_descriptions_dir):
        print(f"ERROR: {text_descriptions_dir} not found!")
        return False
    
    cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
    text_descriptions = {}
    
    for cat_name in cat_names:
        txt_file = os.path.join(text_descriptions_dir, f'{cat_name}.txt')
        if not os.path.exists(txt_file):
            print(f"WARNING: {txt_file} not found")
            continue
        
        descriptions_2d = []
        descriptions_3d = []
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('2D:'):
                    desc = line[3:].strip()
                    if desc:
                        descriptions_2d.append(desc)
                elif line.startswith('3D:'):
                    desc = line[3:].strip()
                    if desc:
                        descriptions_3d.append(desc)
        
        text_descriptions[cat_name] = {
            "2D": descriptions_2d,
            "3D": descriptions_3d
        }
        print(f"✓ Loaded {len(descriptions_2d)} 2D and {len(descriptions_3d)} 3D descriptions for '{cat_name}'")
    
    # 测试随机选取
    print("\n=== 随机选取示例 ===")
    import numpy as np
    for cat_name in ['mug', 'bottle', 'bowl']:
        if cat_name in text_descriptions:
            texts_2d = text_descriptions[cat_name]['2D']
            texts_3d = text_descriptions[cat_name]['3D']
            if texts_2d and texts_3d:
                idx_2d = np.random.randint(len(texts_2d))
                idx_3d = np.random.randint(len(texts_3d))
                print(f"\n{cat_name.upper()}:")
                print(f"  2D: {texts_2d[idx_2d]}")
                print(f"  3D: {texts_3d[idx_3d]}")
    
    return True


def test_dataset_integration():
    """测试与 TrainingDataset 的集成"""
    print("\n=== 测试与 TrainingDataset 集成 ===")
    
    # 模拟检查 dataset.py 修改
    dataset_file = './provider/dataset.py'
    if not os.path.exists(dataset_file):
        print(f"WARNING: {dataset_file} not found, skipping integration test")
        return True
    
    with open(dataset_file, 'r') as f:
        content = f.read()
    
    checks = [
        ('text_descriptions_dir' in content, "参数 text_descriptions_dir 已添加"),
        ('_load_text_descriptions' in content, "方法 _load_text_descriptions() 已添加"),
        ('_get_text_descriptions' in content, "方法 _get_text_descriptions() 已添加"),
        ("ret_dict['text_texture']" in content, "返回值 text_texture 已添加"),
        ("ret_dict['text_shape']" in content, "返回值 text_shape 已添加"),
    ]
    
    all_ok = True
    for check, msg in checks:
        if check:
            print(f"✓ {msg}")
        else:
            print(f"✗ {msg}")
            all_ok = False
    
    return all_ok


if __name__ == '__main__':
    print("=" * 60)
    print("测试文本描述加载功能")
    print("=" * 60)
    
    test1 = test_text_loading()
    test2 = test_dataset_integration()
    
    print("\n" + "=" * 60)
    if test1 and test2:
        print("✓ 所有测试通过！")
        print("\n快速开始：")
        print("1. 将你的文本描述文件放在 text_descriptions/ 目录中")
        print("2. 初始化 TrainingDataset 时传入 text_descriptions_dir 参数")
        print("3. 数据集将自动加载文本描述并在 __getitem__ 中随机返回")
        sys.exit(0)
    else:
        print("✗ 部分测试失败，请检查修改是否完整")
        sys.exit(1)
