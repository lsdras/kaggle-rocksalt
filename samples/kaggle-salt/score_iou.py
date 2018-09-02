import numpy as np
import csv
import os

def val_IoU(train_rle, result_rle):    
    train = str_to_list(train_rle).reshape(-1,2)
    result = str_to_list(result_rle).reshape(-1,2)
    train_list = np.zeros(101*101, dtype=np.uint8)
    result_list = np.zeros(101*101, dtype=np.uint8)
    for dot,iter in train:
        dot = dot-1
        train_list[dot:dot+iter] = np.ones(iter)
    for dot,iter in result:
        dot = dot-1
        result_list[dot:dot+iter] = np.ones(iter)
    Union = np.zeros(101*101, dtype=np.uint8)
    Inter = np.zeros(101*101, dtype=np.uint8)
    for i in range(101*101):
        if train_list[i]==1 or result_list[i]==1: Union[i]=1
        if train_list[i]==1 and result_list[i]==1: Inter[i]=1
    U = sum(Union)
    I = sum(Inter)
    if U==0: return 1
    IoU_score = I/U
    return IoU_score

def str_to_list(rleString):
    if rleString =='' or rleString ==None: return np.array([])
    rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
    return np.array(rleNumbers)

chart = open(os.path.join('../../dataset/salt/train.csv'), 'r', encoding='utf-8')
lines = csv.reader(chart)
chart2 = open(os.path.join('../../results/salt/1im1dep1zeros_train/submit.csv'), 'r', encoding='utf-8')
lines2 = csv.reader(chart2)
result = []
for line in lines2:
    result.append(line)
scores = []
for line in lines:
    train_rle = line[1]
    result_rle = None
    for image_id, rle in result:        
        if image_id==line[0]:
            result_rle = rle            
            score = val_IoU(train_rle,result_rle)
            print("{}: {}\n".format(image_id,score))
            scores.append(score)

print("average score: {}".format(sum(scores)/len(scores)))