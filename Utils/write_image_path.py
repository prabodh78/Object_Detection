import glob
import os

dataset_folder_name = 'person_voco_client_v1'
train_images_path = 'data/{}/train/*.jpg'.format(dataset_folder_name)
test_images_path = 'data/{}/test/*.jpg'.format(dataset_folder_name)

imgs = glob.glob(train_images_path)
file = open("train.txt", "w")
name = 'data/{}/train'.format(dataset_folder_name)
for i in imgs:
    data = os.path.join(name, os.path.basename(i))
    file.write("{}\n".format(data))

imgs = glob.glob(test_images_path)
file = open("test.txt", "w")
name = 'data/{}/test'.format(dataset_folder_name)
for i in imgs:
    data = os.path.join(name, os.path.basename(i))
    file.write("{}\n".format(data))
