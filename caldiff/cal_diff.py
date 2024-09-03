import os
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm import tqdm
import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model_cm import MaskedNLLLoss, LSTMModel, GRUModel, Model, MaskedMSELoss, FocalLoss, Model_Inter, Model_Intra
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
import pandas as pd
import pickle as pk
import datetime
import ipdb
from clloss import CKSCL_context, CKSCL_tav
from tqdm import tqdm
CKSCL_m = CKSCL_tav()
CKSCL_c = CKSCL_context()



def create_class_weight_SCL(label):
    unique = list(set(label.cpu().detach().numpy().tolist()))

    # one = sum(label)
    labels_dict = {l:(label==l).sum().item() for l in unique}
    # labels_dict = {0 : len(label) - one, 1: one}
    total = sum(list(labels_dict.values()))
    weights = []
    for i in range(max(unique)+1):
        if i not in unique:
            weights.append(0)
        else:
            weights.append(total/labels_dict[i])
    return weights

# seed = 1475 # We use seed = 1475 on IEMOCAP and seed = 67137 on MELD
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _init_fn(worker_id):
        np.random.seed(int(args.seed)+worker_id)

def get_train_valid_sampler(trainset, valid=0.1, dataset='IEMOCAP'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False, epoch_ratio=-1):
    trainset = MELDDataset('data/MELD_features_raw1.pkl', epoch_ratio=epoch_ratio)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset('data/MELD_features_raw1.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False, epoch_ratio=-1):
    trainset = IEMOCAPDataset(epoch_ratio=epoch_ratio)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory, worker_init_fn=_init_fn)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory, worker_init_fn=_init_fn)

    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    """
    
    """
    losses, preds, labels, masks = [], [], [], []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    max_sequence_len = []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything(args.seed)
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]        

        max_sequence_len.append(textf.size(0))
        
        log_prob, alpha, alpha_f, alpha_b, _ = model(textf, qmask, umask)
        lp_ = log_prob.transpose(0,1).contiguous().view(-1, log_prob.size()[2])
        labels_ = label.view(-1)
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_,1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)
    
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [alphas, alphas_f, alphas_b, vids]


def train_or_eval_graph_model(model, model_mm, model_intra, model_inter, loss_function, dataloader, epoch, cuda, modals, optimizer=None, train=False, dataset='IEMOCAP'):

    Difficulties = {}

    losses, preds, labels = [], [], []
    scores, vids = [], []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train or optimizer!=None

    model.eval()
    model_mm.load_state_dict(model.state_dict())
    model_mm.eval()
    model_intra.load_state_dict(model.state_dict())
    model_intra.eval()
    model_inter.load_state_dict(model.state_dict())
    model_inter.eval()

    seed_everything(args.seed)
    for data in tqdm(dataloader):
        if train:
            optimizer.zero_grad()
        
        textf1,textf2,textf3,textf4, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        if args.multi_modal:
            if args.mm_fusion_mthd=='concat':
                if modals == 'avl':
                    textf = torch.cat([acouf, visuf, textf1,textf2,textf3,textf4],dim=-1)
                elif modals == 'av':
                    textf = torch.cat([acouf, visuf],dim=-1)
                elif modals == 'vl':
                    textf = torch.cat([visuf, textf1,textf2,textf3,textf4],dim=-1)
                elif modals == 'al':
                    textf = torch.cat([acouf, textf1,textf2,textf3,textf4],dim=-1)
                else:
                    raise NotImplementedError
            elif args.mm_fusion_mthd=='gated':
                textf = textf
        else:
            if modals == 'a':
                textf = acouf
            elif modals == 'v':
                textf = visuf
            elif modals == 'l':
                textf = textf
            else:
                raise NotImplementedError

        lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]
        if args.multi_modal and args.mm_fusion_mthd=='gated':
            log_prob, e_i, e_n, e_t, e_l, hidden = model(textf, qmask, umask, lengths, acouf, visuf)
        elif args.multi_modal and args.mm_fusion_mthd=='concat_subsequently':   
            log_prob, e_i, e_n, e_t, e_l, hidden = model([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)
        elif args.multi_modal and args.mm_fusion_mthd=='concat_DHT':   
            log_prob, e_i, e_n, e_t, e_l, hidden = model([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)
        else:
            log_prob, e_i, e_n, e_t, e_l, hidden = model(textf, qmask, umask, lengths)
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        with torch.no_grad():
            ft_scl = hidden
            mask_textf1 = torch.mean(textf1, dim=0).repeat(textf1.shape[0], 1, 1)
            mask_textf2 = torch.mean(textf2, dim=0).repeat(textf2.shape[0], 1, 1)
            mask_textf3 = torch.mean(textf3, dim=0).repeat(textf3.shape[0], 1, 1)
            mask_textf4 = torch.mean(textf4, dim=0).repeat(textf4.shape[0], 1, 1)

            mask_acouf = torch.mean(acouf, dim=0).repeat(acouf.shape[0], 1, 1)
            mask_visuf = torch.mean(visuf, dim=0).repeat(visuf.shape[0], 1, 1)
            
            # mask multimodal data
            log_prob2t, e_i, e_n, e_t, e_l, hiddent = model_mm([mask_textf1, mask_textf2, mask_textf3, mask_textf4], qmask, umask, lengths, acouf, visuf, epoch)
            log_prob2a, e_i, e_n, e_t, e_l, hiddena = model_mm([textf1,textf2,textf3,textf4], qmask, umask, lengths, mask_acouf, visuf, epoch)
            log_prob2v, e_i, e_n, e_t, e_l, hiddenv = model_mm([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, mask_visuf, epoch)

            # mask inter context and intra contex
            log_prob2intra, e_i, e_n, e_t, e_l, hidden_intra = model_intra([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)
            log_prob2inter, e_i, e_n, e_t, e_l, hidden_inter = model_inter([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)


            labels_ = label

            pred2t_ = torch.argmax(log_prob2t, 1) # batch*seq_len
            pred2a_ = torch.argmax(log_prob2a, 1) # batch*seq_len
            pred2v_ = torch.argmax(log_prob2v, 1) # batch*seq_len
            pred2c_intra = torch.argmax(log_prob2intra, 1)
            pred2c_inter = torch.argmax(log_prob2inter, 1)

        lp_ = log_prob 
        pred_ = torch.argmax(lp_,1) # batch*seq_len

        rank_lossm = 0
        for i in range(len(log_prob)):
            num = labels_[i]
            if lp_[i][num] <= log_prob2t[i][num]:
                rank_lossm += (torch.sub(log_prob2t[i][num], lp_[i][num]))
            if lp_[i][num] <= log_prob2a[i][num]:
                rank_lossm += (torch.sub(log_prob2a[i][num], lp_[i][num]))
            if lp_[i][num] <= log_prob2v[i][num]:
                rank_lossm += (torch.sub(log_prob2v[i][num], lp_[i][num]))

        rank_lossc = 0 
        for i in range(len(log_prob)):
            num = labels_[i]
            if lp_[i][num] <= log_prob2intra[i][num]:
                rank_lossc += (torch.sub(log_prob2intra[i][num], lp_[i][num]))
            if lp_[i][num] <= log_prob2inter[i][num]:
                rank_lossc += (torch.sub(log_prob2inter[i][num], lp_[i][num]))

        rank_loss = rank_lossm + rank_lossc

        Mscl_pred_t = torch.zeros_like(pred_).cuda()
        Mscl_pred_v = torch.zeros_like(pred_).cuda()
        Mscl_pred_a = torch.zeros_like(pred_).cuda()
        for index, x in enumerate(Mscl_pred_t):
            if pred_[index] != labels_[index] and pred2t_[index] == labels_[index]:
                Mscl_pred_t[index] = 0
            if pred_[index] == labels_[index] and pred2t_[index] != labels_[index]:
                Mscl_pred_t[index] = 1
            if pred_[index] == labels_[index] and pred2t_[index] == labels_[index]:
                if lp_[index][labels_[index]] <= log_prob2t[index][labels_[index]]:
                    Mscl_pred_t[index] = 0
                if lp_[index][labels_[index]] > log_prob2t[index][labels_[index]]:
                    Mscl_pred_t[index] = 1

            if pred_[index] != labels_[index] and pred2a_[index] == labels_[index]:
                Mscl_pred_a[index] = 0
            if pred_[index] == labels_[index] and pred2a_[index] != labels_[index]:
                Mscl_pred_a[index] = 1
            if pred_[index] == labels_[index] and pred2a_[index] == labels_[index]:
                if lp_[index][labels_[index]] <= log_prob2a[index][labels_[index]]:
                    Mscl_pred_a[index] = 0
                if lp_[index][labels_[index]] > log_prob2a[index][labels_[index]]:
                    Mscl_pred_a[index] = 1

            if pred_[index] != labels_[index] and pred2v_[index] == labels_[index]:
                Mscl_pred_v[index] = 0
            if pred_[index] == labels_[index] and pred2v_[index] != labels_[index]:
                Mscl_pred_v[index] = 1
            if pred_[index] == labels_[index] and pred2v_[index] == labels_[index]:
                if lp_[index][labels_[index]] <= log_prob2v[index][labels_[index]]:
                    Mscl_pred_v[index] = 0
                if lp_[index][labels_[index]] > log_prob2v[index][labels_[index]]:
                    Mscl_pred_v[index] = 1

        Cscl_pred_intra = torch.zeros_like(pred_).cuda()
        Cscl_pred_inter = torch.zeros_like(pred_).cuda()
        for index, x in enumerate(Cscl_pred_intra):
            if pred_[index] != labels_[index] and pred2c_intra[index] == labels_[index]:
                Cscl_pred_intra[index] = 0
            if pred_[index] == labels_[index] and pred2c_intra[index] != labels_[index]:
                Cscl_pred_intra[index] = 1
            if pred_[index] == labels_[index] and pred2c_intra[index] == labels_[index]:
                if lp_[index][labels_[index]] <= log_prob2intra[index][labels_[index]]:
                    Cscl_pred_intra[index] = 0
                if lp_[index][labels_[index]] > log_prob2intra[index][labels_[index]]:
                    Cscl_pred_intra[index] = 1

            if pred_[index] != labels_[index] and pred2c_inter[index] == labels_[index]:
                Cscl_pred_inter[index] = 0
            if pred_[index] == labels_[index] and pred2c_inter[index] != labels_[index]:
                Cscl_pred_inter[index] = 1
            if pred_[index] == labels_[index] and pred2c_inter[index] == labels_[index]:
                if lp_[index][labels_[index]] <= log_prob2inter[index][labels_[index]]:
                    Cscl_pred_inter[index] = 0
                if lp_[index][labels_[index]] > log_prob2inter[index][labels_[index]]:
                    Cscl_pred_inter[index] = 1
            

        Mscl_tav = CKSCL_m(ft_scl, labels1=Mscl_pred_t, labels2=Mscl_pred_a, labels3=Mscl_pred_v, weight1=create_class_weight_SCL(Mscl_pred_t), weight2=create_class_weight_SCL(Mscl_pred_a), weight3=create_class_weight_SCL(Mscl_pred_v))
        Cscl_c = CKSCL_c(ft_scl, labels1=Cscl_pred_intra, labels2=Cscl_pred_inter, weight1=create_class_weight_SCL(Cscl_pred_intra), weight2=create_class_weight_SCL(Cscl_pred_inter))


        loss = loss_function(log_prob, label)

        print('loss: ', float(loss))
        print('Mscl_tav: ', float(Mscl_tav))
        print('Cscl_c: ', float(Cscl_c))
        print('rank_loss: ', float(rank_loss))
        diff_loss = float(rank_lossm) + float(rank_lossc)
        Difficulties[data[-1][0]] = float((diff_loss + torch.max(torch.nonzero(qmask[:,0,:])[:,1]) + 1) / (torch.max(torch.nonzero(qmask[:,0,:])[:,1]) + 1 + qmask.shape[0]))
    pickle.dump(Difficulties, open('caldiff/{}_diff.pkl'.format(args.Dataset), 'wb'))









if __name__ == '__main__':
    path = './saved/IEMOCAP/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--base-model', default='LSTM', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')

    parser.add_argument('--graph-model', action='store_true', default=True, help='whether to use graph model after recurrent encoding')

    parser.add_argument('--nodal-attention', action='store_true', default=True, help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')

    parser.add_argument('--windowp', type=int, default=10, help='context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowf', type=int, default=10, help='context window size for constructing edges in graph model for future utterances')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    
    parser.add_argument('--l2', type=float, default=0.00003, metavar='L2', help='L2 regularization weight')
    
    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')
    
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
    
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    
    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')
    
    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')
    
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    parser.add_argument('--graph_type', default='relation', help='relation/GCN3/DeepGCN/MMGCN/MMGCN2')

    parser.add_argument('--use_topic', action='store_true', default=False, help='whether to use topic information')

    parser.add_argument('--alpha', type=float, default=0.2, help='alpha')

    parser.add_argument('--multiheads', type=int, default=6, help='multiheads')

    parser.add_argument('--graph_construct', default='full', help='single/window/fc for MMGCN2; direct/full for others')

    parser.add_argument('--use_gcn', action='store_true', default=False, help='whether to combine spectral and none-spectral methods or not')

    parser.add_argument('--use_residue', action='store_true', default=False, help='whether to use residue information or not')

    parser.add_argument('--multi_modal', action='store_true', default=False, help='whether to use multimodal information')

    parser.add_argument('--mm_fusion_mthd', default='concat', help='method to use multimodal information: concat, gated, concat_subsequently')

    parser.add_argument('--modals', default='avl', help='modals to fusion')

    parser.add_argument('--av_using_lstm', action='store_true', default=False, help='whether to use lstm in acoustic and visual modality')

    parser.add_argument('--Deep_GCN_nlayers', type=int, default=4, help='Deep_GCN_nlayers')

    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')

    parser.add_argument('--use_speaker', action='store_true', default=True, help='whether to use speaker embedding')

    parser.add_argument('--use_modal', action='store_true', default=False, help='whether to use modal embedding')

    parser.add_argument('--norm', default='LN2', help='NORM type')

    parser.add_argument('--testing', action='store_true', default=False, help='testing')

    parser.add_argument('--num_L', type=int, default=3, help='num_hyperconvs')

    parser.add_argument('--num_K', type=int, default=4, help='num_convs')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--ckpt_path', type=str)

    args = parser.parse_args()
    today = datetime.datetime.now()
    print(args)
    if args.av_using_lstm:
        name_ = args.mm_fusion_mthd+'_'+args.modals+'_'+args.graph_type+'_'+args.graph_construct+'using_lstm_'+args.Dataset
    else:
        name_ = args.mm_fusion_mthd+'_'+args.modals+'_'+args.graph_type+'_'+args.graph_construct+str(args.Deep_GCN_nlayers)+'_'+args.Dataset

    if args.use_speaker:
        name_ = name_+'_speaker'
    if args.use_modal:
        name_ = name_+'_modal'

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size
    modals = args.modals
    feat2dim = {'IS10':1582,'3DCNN':512,'textCNN':100,'bert':768,'denseface':342,'MELD_text':600,'MELD_audio':300}
    D_audio = feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024 #feat2dim['textCNN'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_text']

    if args.multi_modal:
        if args.mm_fusion_mthd=='concat':
            if modals == 'avl':
                D_m = D_audio+D_visual+D_text
            elif modals == 'av':
                D_m = D_audio+D_visual
            elif modals == 'al':
                D_m = D_audio+D_text
            elif modals == 'vl':
                D_m = D_visual+D_text
            else:
                raise NotImplementedError
        else:
            D_m = 1024
    else:
        if modals == 'a':
            D_m = D_audio
        elif modals == 'v':
            D_m = D_visual
        elif modals == 'l':
            D_m = D_text
        else:
            raise NotImplementedError
    D_g = 512 if args.Dataset=='IEMOCAP' else 1024
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 512
    n_speakers = 9 if args.Dataset=='MELD' else 2
    n_classes  = 7 if args.Dataset=='MELD' else 6 if args.Dataset=='IEMOCAP' else 1

    if args.graph_model:
        seed_everything(args.seed)

        model = Model(args.base_model,
                                 D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                 n_speakers=n_speakers,
                                 max_seq_len=200,
                                 window_past=args.windowp,
                                 window_future=args.windowf,
                                 n_classes=n_classes,
                                 listener_state=args.active_listener,
                                 context_attention=args.attention,
                                 dropout=args.dropout,
                                 nodal_attention=args.nodal_attention,
                                 no_cuda=args.no_cuda,
                                 graph_type=args.graph_type,
                                 use_topic=args.use_topic,
                                 alpha=args.alpha,
                                 multiheads=args.multiheads,
                                 graph_construct=args.graph_construct,
                                 use_GCN=args.use_gcn,
                                 use_residue=args.use_residue,
                                 D_m_v = D_visual,
                                 D_m_a = D_audio,
                                 modals=args.modals,
                                 att_type=args.mm_fusion_mthd,
                                 av_using_lstm=args.av_using_lstm,
                                 Deep_GCN_nlayers=args.Deep_GCN_nlayers,
                                 dataset=args.Dataset,
                                 use_speaker=args.use_speaker,
                                 use_modal=args.use_modal,
                                 norm = args.norm,
                                 num_L = args.num_L,
                                 num_K = args.num_K)
        model_mm = Model(args.base_model,
                                 D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                 n_speakers=n_speakers,
                                 max_seq_len=200,
                                 window_past=args.windowp,
                                 window_future=args.windowf,
                                 n_classes=n_classes,
                                 listener_state=args.active_listener,
                                 context_attention=args.attention,
                                 dropout=args.dropout,
                                 nodal_attention=args.nodal_attention,
                                 no_cuda=args.no_cuda,
                                 graph_type=args.graph_type,
                                 use_topic=args.use_topic,
                                 alpha=args.alpha,
                                 multiheads=args.multiheads,
                                 graph_construct=args.graph_construct,
                                 use_GCN=args.use_gcn,
                                 use_residue=args.use_residue,
                                 D_m_v = D_visual,
                                 D_m_a = D_audio,
                                 modals=args.modals,
                                 att_type=args.mm_fusion_mthd,
                                 av_using_lstm=args.av_using_lstm,
                                 Deep_GCN_nlayers=args.Deep_GCN_nlayers,
                                 dataset=args.Dataset,
                                 use_speaker=args.use_speaker,
                                 use_modal=args.use_modal,
                                 norm = args.norm,
                                 num_L = args.num_L,
                                 num_K = args.num_K)
        model_intra = Model_Intra(args.base_model,
                                 D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                 n_speakers=n_speakers,
                                 max_seq_len=200,
                                 window_past=args.windowp,
                                 window_future=args.windowf,
                                 n_classes=n_classes,
                                 listener_state=args.active_listener,
                                 context_attention=args.attention,
                                 dropout=args.dropout,
                                 nodal_attention=args.nodal_attention,
                                 no_cuda=args.no_cuda,
                                 graph_type=args.graph_type,
                                 use_topic=args.use_topic,
                                 alpha=args.alpha,
                                 multiheads=args.multiheads,
                                 graph_construct=args.graph_construct,
                                 use_GCN=args.use_gcn,
                                 use_residue=args.use_residue,
                                 D_m_v = D_visual,
                                 D_m_a = D_audio,
                                 modals=args.modals,
                                 att_type=args.mm_fusion_mthd,
                                 av_using_lstm=args.av_using_lstm,
                                 Deep_GCN_nlayers=args.Deep_GCN_nlayers,
                                 dataset=args.Dataset,
                                 use_speaker=args.use_speaker,
                                 use_modal=args.use_modal,
                                 norm = args.norm,
                                 num_L = args.num_L,
                                 num_K = args.num_K)
        model_inter = Model_Inter(args.base_model,
                                 D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                 n_speakers=n_speakers,
                                 max_seq_len=200,
                                 window_past=args.windowp,
                                 window_future=args.windowf,
                                 n_classes=n_classes,
                                 listener_state=args.active_listener,
                                 context_attention=args.attention,
                                 dropout=args.dropout,
                                 nodal_attention=args.nodal_attention,
                                 no_cuda=args.no_cuda,
                                 graph_type=args.graph_type,
                                 use_topic=args.use_topic,
                                 alpha=args.alpha,
                                 multiheads=args.multiheads,
                                 graph_construct=args.graph_construct,
                                 use_GCN=args.use_gcn,
                                 use_residue=args.use_residue,
                                 D_m_v = D_visual,
                                 D_m_a = D_audio,
                                 modals=args.modals,
                                 att_type=args.mm_fusion_mthd,
                                 av_using_lstm=args.av_using_lstm,
                                 Deep_GCN_nlayers=args.Deep_GCN_nlayers,
                                 dataset=args.Dataset,
                                 use_speaker=args.use_speaker,
                                 use_modal=args.use_modal,
                                 norm = args.norm,
                                 num_L = args.num_L,
                                 num_K = args.num_K)
        print ('Graph NN with', args.base_model, 'as base model.')
        name = 'Graph'

    else:
        if args.base_model == 'GRU':
            model = GRUModel(D_m, D_e, D_h, 
                              n_classes=n_classes, 
                              dropout=args.dropout)

            print ('Basic GRU Model.')


        elif args.base_model == 'LSTM':
            model = LSTMModel(D_m, D_e, D_h, 
                              n_classes=n_classes, 
                              dropout=args.dropout)

            print ('Basic LSTM Model.')

        else:
            print ('Base model must be one of DialogRNN/LSTM/GRU/Transformer')
            raise NotImplementedError

        name = 'Base'

    if cuda:
        model.cuda()
        model_mm.cuda()
        model_intra.cuda()
        model_inter.cuda()

    model.load_state_dict(torch.load(args.ckpt_path))
    if args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668])

    if args.Dataset == 'MELD':
        loss_function = FocalLoss()
    else:
        if args.class_weight:
            if args.graph_model:
                #loss_function = FocalLoss()
                loss_function  = nn.NLLLoss(loss_weights.cuda() if cuda else loss_weights)
            else:
                loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        else:
            if args.graph_model:
                loss_function = nn.NLLLoss()
            else:
                loss_function = MaskedNLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    lr = args.lr
    


    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []



    if args.Dataset == 'MELD':
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                    batch_size=batch_size,
                                                                    num_workers=2,
                                                                    epoch_ratio=-1)
    elif args.Dataset == 'IEMOCAP':
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                    batch_size=batch_size,
                                                                    num_workers=2,
                                                                    epoch_ratio=-1)
    else:
        print("There is no such dataset")
    
    train_or_eval_graph_model(model, model_mm, model_intra, model_inter, loss_function, train_loader, 0, cuda, args.modals, optimizer, False, dataset=args.Dataset)
            
            
    
    