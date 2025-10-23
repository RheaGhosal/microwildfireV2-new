import pandas as pd, cv2, numpy as np, json, os
df = pd.read_csv("data/train.csv")
train = df[df.split=="train"].reset_index(drop=True)

# Channel means/stds over a sample (or all if small)
K = min(5000, len(train))
idx = np.random.RandomState(42).choice(len(train), K, replace=False)

sum_c = np.zeros(3); sumsq_c = np.zeros(3); n_pix = 0
pos = 0; neg = 0
for i in idx:
    img = cv2.imread(train.loc[i,'path'])[:,:,::-1] / 255.0  # BGR->RGB
    h,w,_ = img.shape; n_pix += h*w
    sum_c += img.reshape(-1,3).sum(axis=0)
    sumsq_c += (img.reshape(-1,3)**2).sum(axis=0)
pos = int(train['label'].sum()); neg = int((1-train['label']).sum())

mean = (sum_c / n_pix).tolist()
var = (sumsq_c / n_pix) - np.array(mean)**2
std = np.sqrt(np.maximum(var, 1e-12)).tolist()
pos_weight = float(neg / max(pos,1))

os.makedirs("out", exist_ok=True)
with open("out/norm_and_class.json","w") as f:
    json.dump({"mean":mean,"std":std,"pos_weight":pos_weight,"n_train":len(train), "pos":pos, "neg":neg}, f, indent=2)
print("Saved -> out/norm_and_class.json")

