import yaml
import torch
import glob
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from faceDataset import FaceDataset
from focalLoss import FocalLoss
from faceNet import FaceNetRes50, FaceNetEb2, FaceNetIBN50
from dataTransform import get_train_transform, get_test_transform

def train(train_loader, model, fl, kl, optimizer):
    for m in model:
        m.train()
    losses = [[],[],[]]
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()

        output = []
        for m in model:
            output.append(m(input))

        loss = 0.0
        for n in range(3):
            focal_loss = fl(output[n], target)
            kl_loss = 0.0
            for m in range(3):
                if n == m:
                    continue
                kl_loss += kl(F.log_softmax(output[n], dim=1), F.softmax(output[m], dim=1)).item()
            train_loss = focal_loss + kl_loss / 2
            losses[n].append(train_loss.item())
            loss += train_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if i % 100 == 0:
            print('Train loss-b1:{0}'.format(np.mean(losses[0])))
            print('Train loss-b2:{0}'.format(np.mean(losses[1])))
            print('Train loss-res:{0}'.format(np.mean(losses[2])))

def validate(val_loader, model, ce, kl):
    for m in model:
        m.eval()
    accs = [[],[],[]]
    losses = [[],[],[]]
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            output = []
            for m in model:
                output.append(m(input))

            for n in range(len(model)):
                logpt = F.log_softmax(output, dim=1)
                py = F.softmax(target, dim=1)
                acc = 1 - ce[n](logpt, py)

                kl_loss = 0
                for m in range(len(model)):
                    if n == m:
                        continue
                    kl_loss += kl[n](F.log_softmax(output[n], dim=1),
                                  F.softmax(output[m], dim=1))
                val_loss = kl_loss / (len(model) - 1)

                accs[n].append(acc)
                losses[n].append(val_loss)

        print(' * Val Acc@1 {0}'.format(np.mean(accs)))
        print(' * Val loss@1 {0}'.format(np.mean(losses)))
        return np.mean(accs)

def train_main():
    train_df = pd.read_csv('train.csv')
    train_png = train_df['name'].values
    train_labels = train_df['label'].values

    with open('basic_conf.yml') as f:
        args = yaml.load(f)

    train_preprocess = get_train_transform(args)
    val_preprocess = get_test_transform(args)
    alphas = torch.Tensor([1.063, 4.468, 1.021, 0.441, 0.787, 0.815, 1.406]).cuda()

    skf = KFold(n_splits=5, random_state=233, shuffle=True)
    i = 0
    for flod_idx, (train_idx, val_idx) in enumerate(skf.split(train_png, train_png)):
        train_data = FaceDataset(train_png[train_idx][:], train_labels[train_idx][:], transform=train_preprocess)
        val_data = FaceDataset(train_png[val_idx][:], train_labels[val_idx][:], transform=val_preprocess)
        data_loader = DataLoader(dataset=train_data, batch_size=2,
                                 shuffle=True, num_workers=5, pin_memory=True)
        val_loader = DataLoader(dataset=val_data, batch_size=10,
                                shuffle=False, num_workers=5, pin_memory=True)


        model = [FaceNetRes50().cuda(), FaceNetEb2().cuda(), FaceNetIBN50().cuda()]
        optimizer = torch.optim.SGD([{'params': model[0].parameters()},
                                      {'params': model[1].parameters()},
                                      {'params': model[2].parameters()}], lr=args['lr'],
                                     momentum=args['momentum'], weight_decay=args['weight_decay'])

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args['epoch_decay'], gamma=args['gamma'])
        focalloss = FocalLoss(alpha=alphas).cuda()
        klloss = nn.KLDivLoss(reduction='batchmean')
        celoss = nn.KLDivLoss()

        best_acc = 0.0
        i = 0

        for epoch in range(args['epoch']):

            print('\nEpoch: ', epoch)
            train(data_loader, model, focalloss, klloss, optimizer)
            val_acc = validate(val_loader, model, celoss, klloss)

            scheduler.step()

            if val_acc > best_acc:
                best_acc = val_acc
                i = flod_idx
                torch.save(model[0].state_dict(), 'res50_fold{0}.pt'.format(flod_idx))
                torch.save(model[1].state_dict(), 'efficientnet_b2_fold{0}.pt'.format(flod_idx))
                torch.save(model[2].state_dict(), 'ibn50_fold{0}.pt'.format(flod_idx))

def predict(test_loader, model, tta=10):
    model.eval()

    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            for i, (input, _) in enumerate(test_loader):
                input = input.cuda()

                output = model(input)
                output = output.data.cpu().numpy()

                test_pred.append(output)
        test_pred = np.vstack(test_pred)

        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta

def test_main():
    test_png = glob.glob('test/*')
    test_png = np.array(test_png)
    test_png.sort()

    with open('basic_conf.yml') as f:
        args = yaml.load(f)

    test_preprocess = get_test_transform(args)
    test_data = FaceDataset(test_png, [], transform=test_preprocess)
    test_loader = DataLoader(dataset=test_data, batch_size=10,
                             shuffle=False, num_workers=5, pin_memory=True)

    #model = [FaceNetRes50().cuda(), FaceNetEb2().cuda(), FaceNetRes18().cuda()]

    #model[0].load_state_dict(torch.load('resnet50_fold0.pt'))
    #model[1].load_state_dict(torch.load('efficientnet_b2_fold0.pt'))
    #model[2].load_state_dict(torch.load('resnet18_fold0.pt'))

    model = FaceNetRes50().cuda()
    model.load_state_dict(torch.load('resnet50_fold0.pt'))
    test_pred = predict(test_loader, model, 5)

    cls_name = np.array(['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised'])
    submit_df = pd.DataFrame({'name': test_png, 'label': cls_name[test_pred.argmax(1)]})
    submit_df['name'] = submit_df['name'].apply(lambda x: x.split('\\')[-1])
    submit_df = submit_df.sort_values(by='name')
    submit_df.to_csv('submit8-11.csv', index=None)


def MixupCutmix():
    import os, glob
    import pandas as pd
    import cv2

    lbl_list = ['angry','disgusted','fearful','happy','neutral','sad','surprised']


    pngs_list = []
    label_list = []
    N = 7
    for i in range(N):
        path = 'train/' + lbl_list[i] + '/*'
        pngs = glob.glob(path)
        np.random.shuffle(pngs)
        pngs_list.append(pngs)
        label = [['0.0']*N] * len(pngs)
        for x in label:
            x[i] = '1.0'
        label_list.append(label)

    if not os.path.exists('train/mixup/'):
        os.makedirs('train/mixup/')
    if not os.path.exists('train/cutmix/'):
        os.makedirs('train/cutmix/')

    mixup_path = []
    cutmix_path = []
    mixup_label = []
    cutmix_label = []
    index = 0
    for _ in range(70):
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                randint1, randint2 = np.random.randint(len(pngs_list[i]), size=(2))
                randint3, randint4 = np.random.randint(len(pngs_list[j]), size=(2))

                radio = np.round(np.random.rand(), 1)

                img1 = cv2.imread(pngs_list[i][randint1])
                img2 = cv2.imread(pngs_list[i][randint2])

                img3 = cv2.imread(pngs_list[j][randint3])
                img4 = cv2.imread(pngs_list[j][randint4])

                mixup = cv2.addWeighted(img1, radio, img2, 1-radio, 0)

                cut = np.int(np.round(img3.shape[1]*radio))
                cutmix = np.concatenate((img3[:,0:cut,:],img4[:,cut:-1,:]),axis=1)
                l = ['0.0'] * N
                l[i], l[j] = str(radio), str(np.round(1-radio, 1))


                cv2.imwrite('train/mixup/'+str(index)+'.png', mixup)
                cv2.imwrite('train/cutmix/'+str(index)+'.png', cutmix)

                mixup_path.append('train/mixup/'+str(index)+'.png')
                mixup_label.append(l)
                cutmix_path.append('train/cutmix/'+str(index)+'.png')
                cutmix_label.append(l)
                index += 1

    pngs_list.append(mixup_path)
    pngs_list.append(cutmix_path)
    label_list.append(mixup_label)
    label_list.append(cutmix_label)

    e1 = [png for pngs in pngs_list for png in pngs]
    e2 = [label for lables in label_list for label in lables]


    train_csv = {}
    train_csv['name'] = e1
    train_csv['label'] = e2

    test_submit = pd.DataFrame(train_csv, columns=['name', 'label'])
    test_submit['label'] = test_submit['label'].apply(lambda x: ' '.join(x))
    test_submit.to_csv('train.csv', index=None)

if __name__ == '__main__':
    # 算力不够，跑不了5个fold，跑了一个，选了个验证集分数最高的模型来跑结果
    #MixupCutmix()
    train_main()
    #test_main()

