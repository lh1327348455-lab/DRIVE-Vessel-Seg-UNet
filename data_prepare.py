import os
from PIL import Image
from tqdm import tqdm

# ================= 配置区域 =================
# 请修改为你解压后的 DRIVE 数据集根目录
# 确保该目录下有 'training' 和 'test' 两个文件夹
DRIVE_ROOT_PATH = "E:\DRIVE" 

# 输出目录设置
TARGET_ROOT = "data"
# ===========================================

def process_drive_subset(subset_name, output_img_dir, output_mask_dir):
    """
    处理 DRIVE 数据集的子集 (training 或 test)
    subset_name: 'training' 或 'test'
    """
    print(f"正在处理 {subset_name} 集...")
    
    # 建立输出目录
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    # 定义源路径
    subset_path = os.path.join(DRIVE_ROOT_PATH, subset_name)
    images_path = os.path.join(subset_path, "images")
    
    # 注意：DRIVE的标注文件夹名通常是 '1st_manual'
    masks_path = os.path.join(subset_path, "1st_manual") 

    if not os.path.exists(images_path) or not os.path.exists(masks_path):
        print(f"错误: 找不到路径 {images_path} 或 {masks_path}")
        return

    files = [f for f in os.listdir(images_path) if f.endswith(".tif")]
    
    for file in tqdm(files):
        # 1. 提取文件ID (例如 '21_training.tif' -> '21')
        file_id = file.split("_")[0] 
        
        # 2. 处理原图
        img = Image.open(os.path.join(images_path, file))
        # 统一 Resize 到 512x512 (科研建议：后续改为 Padding 或 RandomCrop)
        img = img.resize((512, 512))
        # 保存为 png
        img.save(os.path.join(output_img_dir, f"{file_id}.png"))

        # 3. 处理 Mask (找对应的 manual1 文件)
        # 训练集通常叫 '21_manual1.gif'，测试集同理
        mask_filename = f"{file_id}_manual1.gif"
        mask_src_path = os.path.join(masks_path, mask_filename)
        
        if os.path.exists(mask_src_path):
            mask = Image.open(mask_src_path)
            mask = mask.resize((512, 512), resample=Image.Resampling.NEAREST) # 关键：最近邻插值
            # Pytorch-UNet 默认寻找 _mask 后缀
            mask.save(os.path.join(output_mask_dir, f"{file_id}_mask.png"))
        else:
            print(f"警告: 找不到对应的 Mask 文件 {mask_filename}")

# --- 主执行逻辑 ---
if __name__ == "__main__":
    # 1. 处理训练集 -> 放入 data/imgs 和 data/masks (供 train.py 使用)
    process_drive_subset(
        subset_name="training",
        output_img_dir=os.path.join(TARGET_ROOT, "imgs"),
        output_mask_dir=os.path.join(TARGET_ROOT, "masks")
    )

    # 2. 处理测试集 -> 放入 data/test_imgs 和 data/test_masks (供 predict.py 或 评估脚本使用)
    process_drive_subset(
        subset_name="test",
        output_img_dir=os.path.join(TARGET_ROOT, "test_imgs"),
        output_mask_dir=os.path.join(TARGET_ROOT, "test_masks")
    )
    
    print("\n全部处理完成！目录结构如下：")
    print("data/")
    print("  ├── imgs/         (训练原图)")
    print("  ├── masks/        (训练标签)")
    print("  ├── test_imgs/    (测试原图)")
    print("  └── test_masks/   (测试标签 - 用于计算 Dice)")