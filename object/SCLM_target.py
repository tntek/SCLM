import argparse
import os, sys
import os.path as osp
from torch.nn.modules.activation import Threshold
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from numpy import linalg as LA
from loss import CrossEntropyLabelSmooth

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def train_target(args):
    dset_loaders = data_load(args)
    # set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'    
    netC.load_state_dict(torch.load(modelpath))

    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    ent_old_val = 0

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            netC.eval()
            mem_label, ent_new_val, feas_SNTg_dic, feas_SNTl_dic = obtain_label(dset_loaders['test'], netF, netB, netC, args, ent_old_val)
            ent_old_val = ent_new_val
            mem_label = torch.from_numpy(mem_label).cuda()
            ada_dic_num = feas_SNTg_dic.size(0)
            print("feas_SNTg_dic_num:{}".format(ada_dic_num))
            netF.train()
            netB.train()
            netC.train()

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        fea_f_ori = netF(inputs_test)

        fea_f_snt_g = SNT_global_detect(fea_f_ori, feas_SNTg_dic)
        fea_f_snt_g = torch.from_numpy(fea_f_snt_g).cuda()

        fea_f_snt_l = SNT_local_detect(fea_f_ori, feas_SNTl_dic, ada_dic_num)
        fea_f_snt_l = fea_f_snt_l.cuda()

        outputs_test_ori = netC(netB(fea_f_ori))
        outputs_test_snt_g = netC(netB(fea_f_snt_g))
        outputs_test_snt_l = netC(netB(fea_f_snt_l))
        
        softmax_out_ori = nn.Softmax(dim=1)(outputs_test_ori)
        softmax_out_snt_g = nn.Softmax(dim=1)(outputs_test_snt_g)
        softmax_out_snt_l = nn.Softmax(dim=1)(outputs_test_snt_l)

        output_ori_re = softmax_out_ori.unsqueeze(1)
        output_snt_g_re = softmax_out_snt_g.unsqueeze(1)
        output_snt_l_re = softmax_out_snt_l.unsqueeze(1)

        output_snt_g_re = output_snt_g_re.permute(0,2,1)
        output_snt_l_re = output_snt_l_re.permute(0,2,1)

        classifier_loss_snt_g = torch.log(torch.bmm(output_ori_re,output_snt_g_re)).sum(-1)
        classifier_loss_snt_l = torch.log(torch.bmm(output_ori_re,output_snt_l_re)).sum(-1)

        loss_const_snt_g = -torch.mean(classifier_loss_snt_g)
        loss_const_snt_l = -torch.mean(classifier_loss_snt_l)

        if args.cls_par > 0:
            pred = mem_label[tar_idx]
            ss_ori_loss = nn.CrossEntropyLoss()(outputs_test_ori, pred)
            ss_ori_loss *= args.cls_par

            ss_snt_loss = loss_const_snt_g + loss_const_snt_l
            ss_snt_loss *= args.cls_snt

            classifier_loss = ss_ori_loss + ss_snt_loss

            if iter_num < interval_iter and args.dset == "VISDA-C":
                classifier_loss *= 0
        
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            softmax_out = softmax_out_ori + softmax_out_snt_g + softmax_out_snt_l
            entropy_loss = torch.mean(loss.Entropy(softmax_out))

            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss

            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss
        
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            netF.train()
            netB.train()
            netC.train()

    if args.issave:   
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
        
    return netF, netB, netC

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def obtain_label(loader, netF, netB, netC, args, ent_old_val_):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas_f = netF(inputs)
            feas = netB(feas_f)
            outputs = netC(feas)
            if start_test:
                all_fea_f = feas_f.float().cpu()
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea_f = torch.cat((all_fea_f, feas_f.float().cpu()), 0)
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    # entropy with momenum
    #===============================================================================
    SNTl_dic_ = copy.deepcopy(all_fea_f)
    ent_cur = ent.cpu().numpy()
    ent_cur_val = ent_cur
    ent_var_tmp = ent_old_val_ - ent_cur_val
    ent_var_ = np.maximum(ent_var_tmp, -ent_var_tmp)
    ent_with_mom = args.new_ent_par * ent_cur_val + (1.0 - args.new_ent_par) * ent_var_
    #===============================================================================

    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()


    # EntMomClustering
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    net_pred_label = predict.cpu().numpy()
    confi_net_pred_lst = []
    confi_all_fea_lst = []
    confi_aff_lst = []
    for i in range(K):
        idx_i = np.where(net_pred_label == i)[0]
        ent_fin_slt_cls = ent_with_mom[idx_i]
        all_fea_cls = all_fea[idx_i, :]
        aff_cls = aff[idx_i, :]
        pred_cls = net_pred_label[idx_i]
        if idx_i.shape[0] > 0:
            confi_all_fea_i, confi_aff_i, pred_i = get_confi_fea_and_op(ent_fin_slt_cls, all_fea_cls, aff_cls, pred_cls, args)
            confi_all_fea_lst.append(confi_all_fea_i)
            confi_aff_lst.append(confi_aff_i)
            confi_net_pred_lst.append(pred_i)

    confi_all_fea = np.vstack(tuple(confi_all_fea_lst))
    confi_aff = np.vstack(tuple(confi_aff_lst))
    confi_pred_slt = np.hstack(tuple(confi_net_pred_lst))

    initc_confi = confi_aff.transpose().dot(confi_all_fea) 
    initc_confi = initc_confi / (1e-8 + confi_aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[confi_pred_slt].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]

    initc_ori = aff.transpose().dot(all_fea)
    initc_ori = initc_ori / (1e-8 + aff.sum(axis=0)[:,None])

    initc = args.initc_par * initc_confi + (1.0 - args.initc_par) * initc_ori
    
    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    # construct SNTg dictionary
    #-------------------------------------------------------------------------------
    K_pred = K
    confi_cls_lst = []
    all_fea_f = all_fea_f.cpu().numpy()
    for i in range(K_pred):
        idx_i = np.where(pred_label == i)[0]
        ent_fin_slt_cls = ent_with_mom[idx_i]
        all_fea_f_cls = all_fea_f[idx_i, :]
        if idx_i.shape[0] > 0:
            confi_cls_i = SNTg_dic_cls(ent_fin_slt_cls, all_fea_f_cls, K_pred, args)
            confi_cls_lst.append(confi_cls_i)

    feas_SNTg_dic_ = np.vstack(tuple(confi_cls_lst))
    feas_SNTg_dic_ = torch.from_numpy(feas_SNTg_dic_).cpu()
    #-------------------------------------------------------------------------------

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return pred_label.astype('int'), ent_cur, feas_SNTg_dic_, SNTl_dic_

def get_confi_fea_and_op(ent_cls, fea_cls, aff_cls, pred_cls, args):
    
    len_confi = int(ent_cls.shape[0] * args.confi_par) + 1
    ent_cls_tensor = torch.from_numpy(ent_cls)
    idx_confi = ent_cls_tensor.topk(len_confi, largest = False)[-1]
    fea_confi = fea_cls[idx_confi, :]
    aff_confi = aff_cls[idx_confi, :]
    pred_slt = pred_cls[idx_confi]

    return fea_confi, aff_confi, pred_slt

def SNTg_dic_cls(ent_cls, fea_cls, K, args):
    
    balance_num = int(2 * fea_cls.shape[1] / K)
    len_confi = int(ent_cls.shape[0] * args.confi_par)
    ent_fin = torch.from_numpy(ent_cls)

    if len_confi > balance_num:
        len_confi = balance_num
        print("== cls is not balance ==")

    idx_confi = ent_fin.topk(len_confi, largest = False)[-1]
    fea_confi = fea_cls[idx_confi, :]

    return fea_confi

def SNT_global_detect(data_t_batch, data_s_confi):
    data_t = data_t_batch.detach()
    data_s = data_s_confi.detach()
    data_t_ = data_t.cpu().numpy()
    data_s_ = data_s.cpu().numpy()

    X = np.transpose(data_s_)
    Y = np.transpose(data_t_)
    beta = 20.0

    Xt = np.transpose(X)
    I = np.identity(Xt.shape[0])

    par_1 = np.matmul(Xt, X)
    par_2 = np.multiply(beta, I)

    B = par_1 + par_2
    Binv = np.linalg.inv(B)
    C = np.matmul(Binv, Xt)
    recon_fea = np.matmul(C, Y)

    idx_recon = np.argmax(recon_fea, axis = 0)
    recon_from_confi = data_s_[idx_recon, :]

    return recon_from_confi

def SNT_local_detect(data_q, data_all, ada_num_):
    data_q_ = data_q.detach()
    data_all_ = data_all.detach()
    data_q_ = data_q_.cpu().numpy()
    data_all_ = data_all_.cpu().numpy()

    sim_slt, fea_nh_nval = get_sim_in_batch(data_q_, data_all_, ada_num_)
    mask_fea_in_batch = get_mask_in_batch(sim_slt, fea_nh_nval)
    re_tmp = get_similar_fea_in_batch(mask_fea_in_batch, data_all_)

    re = torch.from_numpy(re_tmp)
    return re

def get_sim_in_batch(Q, X, basis_num_):
    Xt = np.transpose(X)
    Simo = np.dot(Q, Xt)               
    nq = np.expand_dims(LA.norm(Q, axis=1), axis=1)
    nx = np.expand_dims(LA.norm(X, axis=1), axis=0)
    Nor = np.dot(nq, nx)
    Sim_f = 1 - (Simo / Nor) 

    indices_min = np.argmin(Sim_f, axis=1)
    indices_row = np.arange(0, Q.shape[0], 1)
    Sim_f[indices_row, indices_min] = 999
    Sim_f_sorted = np.sort(Sim_f, axis = 1)

    threshold_num = X.shape[0]//basis_num_
    get_nh_nval = Sim_f_sorted[:, threshold_num]

    return Sim_f, get_nh_nval

def get_mask_in_batch(Sim_f, fea_nh_nval):

    fea_nh_nval_f = np.expand_dims(fea_nh_nval, axis = 1)

    fea_nh_nval_zerof = np.zeros_like(Sim_f)
    fea_nh_nval_ff = fea_nh_nval_f + fea_nh_nval_zerof
    
    fea_nh_nval_slt = Sim_f - fea_nh_nval_ff

    all_1 = np.ones_like(Sim_f)
    fea_nh_nval_slt = torch.from_numpy(fea_nh_nval_slt)
    all_1 = torch.from_numpy(all_1)
    fea_nh_nval_zerof = torch.from_numpy(fea_nh_nval_zerof)

    mask_fea = torch.where(fea_nh_nval_slt <= 0.0, all_1, fea_nh_nval_zerof)
    mask_fea = mask_fea.cpu().numpy()

    return mask_fea

def get_similar_fea_in_batch(mask_fea_f, fea_all_f):
    ln = mask_fea_f.shape[0]
    ext_fea_list = []
    
    for k in range(ln):
        idx_hunter_feas = np.where(mask_fea_f[k] == 1.0)[0]
        fea_hunter_k = fea_all_f[idx_hunter_feas]

        if fea_hunter_k.shape[0] > 1:
            fea_hunter = np.mean(fea_hunter_k, axis=0)
        else:
            fea_hunter = fea_hunter_k
        ext_fea_list.append(fea_hunter) 

    ext_fea_arr = np.vstack(tuple(ext_fea_list))
    return ext_fea_arr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SCLM')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
 
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--cls_snt', type=float, default=0.1)

    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--new_ent_par', type=float, default=0.3)
    parser.add_argument('--initc_par', type=float, default=0.3)
    parser.add_argument('--confi_par', type=float, default=0.3)
    
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='Office31/r0/sclm')
    parser.add_argument('--output_src', type=str, default='Office31/r0/src')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    print("============   SCLM   ============")
    folder = './data/'
    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

    if args.dset == 'office-home':
        if args.da == 'pda':
            args.class_num = 65
            args.src_classes = [i for i in range(65)]
            args.tar_classes = [i for i in range(25)]

    args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
    args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
    args.name = names[args.s][0].upper()+names[args.t][0].upper()

    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.savename = 'par_' + str(args.cls_par)
    if args.da == 'pda':
        args.gent = ''
        args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
    args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    train_target(args)