import yaml
import numpy as np
import torch
from model import Siamese
import torch.nn as nn
from torch.autograd import Variable
from torch import Tensor
from data import myDS, mytestDS
from dataExtractor import DataExtractor
FLAGS = None
import os
from utils import load_embed, save_embed, get_embedding
from torch.utils.data import DataLoader
from datetime import datetime
import pandas as pd
from tqdm import tqdm



with open('config.yaml', 'r') as f:
    config = yaml.load(f)
dataExtractor = DataExtractor(config)
train_q1, train_q2, train_label =  dataExtractor.get_training_list()
val_q1, val_q2, val_label = dataExtractor.get_val_list()
test_q1, test_q2 = dataExtractor.get_testing_list()

all_sentences = train_q1 + train_q2 + val_q1 + val_q2 + test_q1 + test_q2
train_data = myDS(train_q1, train_q2, train_label, all_sentences)
val_data = myDS(val_q1, val_q2, val_label, all_sentences)
test_data = mytestDS(test_q1, test_q2, all_sentences)

'''Embedding'''
full_embed_path = config['embedding']['full_embedding_path']
cur_embed_path = config['embedding']['cur_embedding_path']

if os.path.exists(cur_embed_path) and not config['make_dict']:
    embed_dict = load_embed(cur_embed_path)
    print('Loaded existing embedding.')
else:
    print('Making embedding...')
    embed_dict = get_embedding(train_data.vocab._id2word, full_embed_path) #  not using <unk> token for OOV words
    save_embed(embed_dict,cur_embed_path)
    print('Saved generated embedding.')

vocab_size = len(embed_dict)
embedding = nn.Embedding(vocab_size, config['model']['embed_dim'])
embed_list = []
for word in train_data.vocab._id2word:
    embed_list.append(embed_dict[word])

weight_matrix = np.array(embed_list)
# pass weights to nn embedding
embedding.weight = nn.Parameter(torch.from_numpy(weight_matrix).type(torch.FloatTensor), requires_grad = False)


""" Model Preparation """

# embedding
config['embedding_matrix'] = embedding
config['vocab_size'] = len(embed_dict)

# model
siamese = Siamese(config)
print(siamese)

# loss func
loss_weights = Variable(torch.FloatTensor([1, 1])) # why loss weight 1:3
if torch.cuda.is_available():
    loss_weights = loss_weights.cuda()
criterion = torch.nn.CrossEntropyLoss(loss_weights)

# optimizer
learning_rate = config['training']['learning_rate']
if config['training']['optimizer'] == 'sgd':
    optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, siamese.parameters()), lr=learning_rate) # why filter
elif config['training']['optimizer'] == 'adam':
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, siamese.parameters()), lr=learning_rate)
elif config['training']['optimizer'] == 'adadelta':
    optimizer = torch.optim.Adadelta(filter(lambda x: x.requires_grad, siamese.parameters()), lr=learning_rate)
elif config['training']['optimizer'] == 'rmsprop':
    optimizer = torch.optim.RMSprop(filter(lambda x: x.requires_grad, siamese.parameters()), lr=learning_rate)
print('Optimizer:', config['training']['optimizer'])
print('Learning rate:', config['training']['learning_rate'])

# log info
train_log_string = '%s :: Epoch %i :: Iter %i / %i :: train loss: %0.4f'
valid_log_string = '%s :: Epoch %i :: valid loss: %0.4f\n'

# Restore saved model (if one exists).
ckpt_path = os.path.join(config['ckpt_dir'], config['experiment_name']+'.pt')

if os.path.exists(ckpt_path):
    print('Loading checkpoint: %s' % ckpt_path)
    ckpt = torch.load(ckpt_path)
    epoch = ckpt['epoch']
    siamese.load_state_dict(ckpt['siamese'])
    optimizer.load_state_dict(ckpt['optimizer'])
else:
    epoch = 1
    print('Fresh start!\n')

if torch.cuda.is_available():
    criterion = criterion.cuda()
    siamese = siamese.cuda()
    embedding = embedding.cuda()
    config['embedding_matrix'] = embedding

if config['task'] == 'train':

    # save every epoch for visualization
    train_loss_record = []
    valid_loss_record = []
    best_record = 10.0

    # training
    print('Experiment: {}\n'.format(config['experiment_name']))

    #while epoch < config['training']['num_epochs']:
    #for epoch in tqdm(range(epoch, config['training']['num_epochs']+1)):
    while epoch < config['training']['num_epochs']:
        print('Start Epoch {} Training...'.format(epoch))

        # los
        train_loss = []
        train_loss_sum = []
        # dataloader

        train_dataloader = DataLoader(dataset=train_data, shuffle=True, num_workers=16, batch_size=config['model']['batch_size'])

        #for idx, data in enumerate(train_dataloader, 0):
        for idx, data in tqdm(enumerate(train_dataloader, 0)):
            # get data
            s1, s2, label = data


            # clear gradients
            optimizer.zero_grad()

            # input
            output = siamese(s1, s2)
            output = output.squeeze(0)

            # label cuda
            label = Variable(label)
            if torch.cuda.is_available():
                label = label.cuda()

            # loss backward
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.data.cpu())
            train_loss_sum.append(loss.data.cpu())

            # Every once and a while check on the loss
            if ((idx + 1) % 5000) == 0:
                print(train_log_string % (datetime.now(), epoch, idx + 1, len(train_q1), np.mean(train_loss)))
                train_loss = []

        # Record at every epoch
        print('Train Loss at epoch {}: {}\n'.format(epoch, np.mean(train_loss_sum)))
        train_loss_record.append(np.mean(train_loss_sum))

        # Valid
        print('Epoch {} Validating...'.format(epoch))

        # loss
        valid_loss = []
        # dataloader
        valid_dataloader = DataLoader(dataset=val_data, shuffle=True, num_workers=2, batch_size=config['model']['batch_size'])

        for idx, data in tqdm(enumerate(valid_dataloader, 0)):
            # get data
            s1, s2, label = data

            # input
            output = siamese(s1, s2)
            output = output.squeeze(0)

            # label cuda
            label = Variable(label)
            if torch.cuda.is_available():
                label = label.cuda()

            # loss
            loss = criterion(output, label)
            valid_loss.append(loss.data.cpu())

        print(valid_log_string % (datetime.now(), epoch, np.mean(valid_loss)))
        # Record
        valid_loss_record.append(np.mean(valid_loss))
        epoch += 1

        if np.mean(valid_loss)-np.mean(train_loss_sum) > 0.02:
             print("Early Stopping!")
             break

        # Keep track of best record
        if np.mean(valid_loss) < best_record:
            best_record = np.mean(valid_loss)
            # save the best model
            state_dict = {
                'epoch': epoch,
                'siamese': siamese.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state_dict, ckpt_path)
            print('Model saved!\n')

""" Inference """
#for ds in [ds1, ds2, ds3]
if config['task'] == 'inference':
    testDS = test_data
    # Do not shuffle here
    test_dataloader = DataLoader(dataset=testDS, num_workers=2, batch_size=config['model']['batch_size'])

    result = []
    for idx, data in tqdm(enumerate(test_dataloader, 0)):

        # get data
        s1, s2 = data

        # input
        output = siamese(s1,s2)
        output = output.squeeze(0)

        # feed output into softmax to get prob prediction
        sm = nn.Softmax(dim=1)
        res = sm(output.data)[:,1]
        result += res.data.tolist()

    result = pd.DataFrame(result)
    print(result.shape)
    print('Inference Done.')
    res_path = os.path.join(config['result']['filepath'], config['result']['filename'])
    result.to_csv(res_path, header=False, index=False)
    print('Result has written to', res_path, ', Good Luck!')


