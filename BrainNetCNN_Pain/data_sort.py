import os
import pandas as pd
import shutil

path = 'D:\Pytorch\BrainNetCNN_pain//functional_connectivity_SFNproject'

# make data dirs
categories = ['healthy', 'pain'] #pain green
categories_dir = 'matrix108'
for i in range(len(categories)):
    dirname =  os.path.join(os.path.dirname(path),categories_dir,categories[i])
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    else:
        continue

# read names in excel file
df = pd.read_excel('D:\Pytorch\Pain_NoPain_AgeSelect.xlsx')
name=[]
split = 82
split_index = split-2
column_name = 'MRI檔名'
for i in df[column_name]:
    name.append(i)

# copy and move txt files
missing = []
for i in range(len(name)) :
    src = os.path.join(path, name[i], 'zFChoa.txt')
    dst_dir = os.path.join(os.path.dirname(path), categories_dir)
    if os.path.isfile(src):
        if (i < split_index):
            dst = os.path.join(dst_dir, categories[0], name[i]+'.txt')
            shutil.copyfile(src, dst)
        elif(i>=split_index):
            dst = os.path.join(dst_dir, categories[1], name[i] + '.txt')
            shutil.copyfile(src, dst)
        else:
            continue
    else:
        missing.append(name[i])

print('缺少的數據')
print(missing)
