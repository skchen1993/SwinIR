import matplotlib.pyplot as plt
import pickle
import os

iter = []
train_l1_y = []
set5valid_x = []
set5valid_y = []

record_path = "./set5test_results/chart/"
iter_record = os.path.join(record_path, "iter.txt")
train_l1_y_record = os.path.join(record_path, "train_l1_y.txt")
set5valid_x_record = os.path.join(record_path, "set5valid_x.txt")
set5valid_y_record = os.path.join(record_path, "set5valid_y.txt")


with open(iter_record, "rb") as fp:
    iter = pickle.load(fp)
with open(train_l1_y_record, "rb") as fp:
    train_l1_y = pickle.load(fp)
with open(set5valid_x_record, "rb") as fp:
    set5valid_x = pickle.load(fp)
with open(set5valid_y_record, "rb") as fp:
    set5valid_y = pickle.load(fp)

# chart plot
dir_chart = record_path
print("-----plot chart-----")

plt.figure(figsize=(10, 7))
plt.plot(iter, train_l1_y, color='orange', label='l1_loss')
plt.xlabel('current_step')
plt.ylabel('l1_loss')
plt.legend()
dir2 = os.path.join(dir_chart, "SwinIR_l1loss.png")
plt.savefig(dir2)

plt.figure(figsize=(10, 7))
plt.plot(set5valid_x, set5valid_y, color='orange', label='psnr_y')
plt.xlabel('current_step')
plt.ylabel('psnr')
plt.legend()
dir3 = os.path.join(dir_chart, "SwinIR_psnr.png")
plt.savefig(dir3)