import sys
import os
sys.path.append('../../')
from path_model import *

# 读数据
class3_path = './data/class3_no_neg.inter'
class4_path = './data/class4_no_neg.inter'
BATCH_SIZE = 256
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
saved_3_path = './savedir/class3_seq.pth'
saved_4_path = './savedir/class4_seq.pth'

train(class3_path, device, BATCH_SIZE, saved_3_path)
train(class4_path, device, BATCH_SIZE, saved_4_path)
