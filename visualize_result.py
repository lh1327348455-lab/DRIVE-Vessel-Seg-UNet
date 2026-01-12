import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

# 配置路径 (选一张效果最好的图，比如 01_test.tif)
img_path = 'data/test_imgs/01.png'  # 注意后缀可能是 .tif 或 .png
mask_path = 'data/test_masks/01_mask.png' # 注意后缀
pred_path = 'result_01_3.png' # 你刚才生成的最好结果

def load_and_prep(path, is_mask=False):
    if not os.path.exists(path): return np.zeros((512,512))
    img = Image.open(path).resize((512, 512))
    return np.array(img)

# 1. 读取数据
original = load_and_prep(img_path)
ground_truth = load_and_prep(mask_path, is_mask=True)
prediction = load_and_prep(pred_path, is_mask=True)

# 2. 制作 False Positive / False Negative 可视化
# 绿色 = 正确预测 (TP)
# 红色 = 漏报 (FN - 应该是血管但没预测出来)
# 蓝色 = 误报 (FP - 背景被预测成了血管)
h, w = 512, 512
diff_map = np.zeros((h, w, 3), dtype=np.uint8)

# 归一化到 0-1
gt = (ground_truth > 0).astype(int)
pred = (prediction > 0).astype(int)

diff_map[..., 1] = (gt & pred) * 255  # Green: Intersection
diff_map[..., 0] = (gt & ~pred) * 255 # Red: Missed
diff_map[..., 2] = (~gt & pred) * 255 # Blue: Noise

# 3. 绘图
fig, ax = plt.subplots(1, 4, figsize=(20, 5))

ax[0].imshow(original)
ax[0].set_title("Original Image (CLAHE)")
ax[0].axis('off')

ax[1].imshow(ground_truth, cmap='gray')
ax[1].set_title("Ground Truth")
ax[1].axis('off')

ax[2].imshow(prediction, cmap='gray')
ax[2].set_title("My Prediction (Dice: 0.71)")
ax[2].axis('off')

ax[3].imshow(diff_map)
ax[3].set_title("Error Map (G:Correct, R:Miss, B:Noise)")
ax[3].axis('off')

plt.tight_layout()
plt.savefig('final_report_figure.png', dpi=300)
print("✅ 最终对比图已生成：final_report_figure.png")