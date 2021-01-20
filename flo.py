import torch
import numpy as np
import os
from PIL import Image, TarIO
import pickle
import tarfile
import scipy.io

class flower(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        super(flower, self).__init__()

        self.root = root
        self.train = train
        self.transform = transform


        if self._check_processed():
            print('Train file has been extracted' if self.train else 'Test file has been extracted')
        else:
            self._extract()

        if self.train:
            self.train_data, self.train_label, self.train_name = pickle.load(
                open(os.path.join(self.root, 'processed/train.pkl'), 'rb')
            )
        else:
            self.test_data, self.test_label, self.test_name = pickle.load(
                open(os.path.join(self.root, 'processed/test.pkl'), 'rb')
            )

    def __len__(self):
        return len(self.train_data) if self.train else len(self.test_data)

    def __getitem__(self, idx):
        if self.train:
            img, label, name = self.train_data[idx], self.train_label[idx], self.train_name[idx]
        else:
            img, label, name = self.test_data[idx], self.test_label[idx], self.test_name[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, name

    def _check_processed(self):
        assert os.path.isdir(self.root) == True
        #assert os.path.isfile(os.path.join(self.root, 'CUB_200_2011.tgz')) == True
        return (os.path.isfile(os.path.join(self.root, 'processed/train.pkl')) and
                os.path.isfile(os.path.join(self.root, 'processed/test.pkl')))

    def _extract(self):
        processed_data_path = os.path.join(self.root, 'processed')
        if not os.path.isdir(processed_data_path):
            os.mkdir(processed_data_path)

        imagelabel_path = os.path.join(self.root, "imagelabels.mat")
        train_test_split_path = os.path.join(self.root, "setid.mat")
        imgdir_path = os.path.join(self.root, "images")

        # tar = tarfile.open(cub_tgz_path, 'r:gz')
        imagelabel = scipy.io.loadmat(imagelabel_path)
        imagelabel = imagelabel['labels'].reshape([-1])
        train_test_split = scipy.io.loadmat(train_test_split_path)
        train_test_split['trnid'] = train_test_split['trnid'].reshape([-1])
        train_test_split['valid'] = train_test_split['valid'].reshape([-1])
        train_test_split['tstid'] = train_test_split['tstid'].reshape([-1])
        id2train = {}
        for id in train_test_split['trnid']:
            id2train[id] = "train"
        for id in train_test_split['valid']:
            id2train[id] = "train"
        for id in train_test_split['tstid']:
            id2train[id] = "test"
        

        train_data = []
        train_labels = []
        train_name = []
        test_data = []
        test_labels = []
        test_name = []
        print('Start extract images..')
        cnt = 0
        train_cnt = 0
        test_cnt = 0

        for id in range(1, 8190):
            cnt += 1
            name = "image_" + format(id, '05d') + ".jpg"
            image_path = os.path.join(imgdir_path, name)
            image = Image.open(image_path)
            if image.getbands()[0] == 'L':
                image = image.convert('RGB')

            image_np = np.array(image)
            image.close()

            label = imagelabel[id - 1] - 1
            # print(image_np)
            # print(label)
            # print(name)
            # exit()

            if id2train[id] == "train":
                train_cnt += 1
                train_data.append(image_np)
                train_labels.append(label)
                train_name.append(name)
            else:
                test_cnt += 1
                test_data.append(image_np)
                test_labels.append(label)
                test_name.append(name)
            if cnt%1000 == 0:
                print('{} images have been extracted'.format(cnt))


        print('Total images: {}, training images: {}. testing images: {}'.format(cnt, train_cnt, test_cnt))
        pickle.dump((train_data, train_labels, train_name),
                    open(os.path.join(self.root, 'processed/train.pkl'), 'wb'))
        pickle.dump((test_data, test_labels, test_name),
                    open(os.path.join(self.root, 'processed/test.pkl'), 'wb'))