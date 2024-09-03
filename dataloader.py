import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import pickle, pandas as pd
import numpy
class IEMOCAPDataset(Dataset):

    def __init__(self, train=True, epoch_ratio=-1):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open('data/IEMOCAP_features.pkl', 'rb'), encoding='latin1')

        _, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4,\
        _, _, _, _ = pickle.load(open('data/iemocap_features_roberta.pkl', 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

        if epoch_ratio != -1:
            assert epoch_ratio > 0 and epoch_ratio <= 1
            self.diff = pickle.load(open('caldiff/IEMOCAP_diff.pkl', 'rb'))
            sorted_diff = [item[0] for item in sorted(self.diff.items(), key=lambda x: x[1])]
            # sorted_diff.reverse()
            sorted_diff = sorted_diff[:int(epoch_ratio*len(sorted_diff))]

            #### align to origin order            
            keysdict = {item:item_index for item_index, item in enumerate(self.keys)}
            sorted_diff = {item:keysdict[item]  for item in sorted_diff}
            sorted_diff = [item[0] for item in sorted(sorted_diff.items(), key=lambda x: x[1])]

            self.keys = sorted_diff
            self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(numpy.array(self.roberta1[vid])),\
               torch.FloatTensor(numpy.array(self.roberta2[vid])),\
               torch.FloatTensor(numpy.array(self.roberta3[vid])),\
               torch.FloatTensor(numpy.array(self.roberta4[vid])),\
               torch.FloatTensor(numpy.array(self.videoVisual[vid])),\
               torch.FloatTensor(numpy.array(self.videoAudio[vid])),\
               torch.FloatTensor(numpy.array([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]])),\
               torch.FloatTensor(numpy.array([1]*len(self.videoLabels[vid]))),\
               torch.LongTensor(numpy.array(self.videoLabels[vid])),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<7 else pad_sequence(dat[i], True) if i<9 else dat[i].tolist() for i in dat]

class MELDDataset(Dataset):

    def __init__(self, path, train=True, epoch_ratio=-1):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid, _ = pickle.load(open(path, 'rb'))

        _, _, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
            _, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open("data/meld_features_roberta.pkl", 'rb'), encoding='latin1')

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)
        if epoch_ratio != -1:
            assert epoch_ratio > 0 and epoch_ratio <= 1
            self.diff = pickle.load(open('caldiff/MELD_diff.pkl', 'rb'))
            sorted_diff = [item[0] for item in sorted(self.diff.items(), key=lambda x: x[1])]
            # sorted_diff.reverse()
            sorted_diff = sorted_diff[:int(epoch_ratio*len(sorted_diff))]

            #### align to origin order            
            keysdict = {item:item_index for item_index, item in enumerate(self.keys)}
            sorted_diff = {item:keysdict[item]  for item in sorted_diff}
            sorted_diff = [item[0] for item in sorted(sorted_diff.items(), key=lambda x: x[1])]

            self.keys = sorted_diff
            self.len = len(self.keys)


    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(numpy.array(self.roberta1[vid])),\
               torch.FloatTensor(numpy.array(self.roberta2[vid])),\
               torch.FloatTensor(numpy.array(self.roberta3[vid])),\
               torch.FloatTensor(numpy.array(self.roberta4[vid])),\
               torch.FloatTensor(numpy.array(self.videoVisual[vid])),\
               torch.FloatTensor(numpy.array(self.videoAudio[vid])),\
               torch.FloatTensor(numpy.array(self.videoSpeakers[vid])),\
               torch.FloatTensor(numpy.array([1]*len(self.videoLabels[vid]))),\
               torch.LongTensor(numpy.array(self.videoLabels[vid])),\
               vid  

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label+=self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<7 else pad_sequence(dat[i], True) if i<9 else dat[i].tolist() for i in dat]
