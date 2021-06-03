import time
import datetime
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torchtext import data
from torchtext import datasets
from logger import setup_logger
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


n_hid = 128
batch_size = 64
epochs = 100
embedding = 100
lr = 6e-4
delta = 5.4e-2
gamma = 4.9
epsilon = 4.8


def get_data(bs,embedding_size):
    text = data.Field(tokenize='spacy', include_lengths=True)
    label = data.LabelField(dtype=torch.float)
    train_data, test_data = datasets.IMDB.splits(text, label)
    train_data, valid_data = train_data.split()

    max_vocab_size = 25_000
    text.build_vocab(train_data,
                     max_size=max_vocab_size,
                     vectors="glove.6B."+str(embedding_size)+"d",
                     unk_init=torch.Tensor.normal_)
    label.build_vocab(train_data)
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                                               batch_size=bs, sort=False)

    return train_iterator, valid_iterator, test_iterator, text

def zero_words_in_embedding(model, embedding_size, text, pad_idx):
    pretrained_embeddings = text.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    UNK_IDX = text.vocab.stoi[text.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(embedding_size)
    model.embedding.weight.data[pad_idx] = torch.zeros(embedding_size)
    

class coRNN(nn.Module):
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon):
        super(coRNN, self).__init__()
        self.n_hid = n_hid
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.i2h = nn.Linear(n_inp, n_hid)
        self.h2h = nn.Linear(n_hid+n_hid, n_hid, bias=False)

    def forward(self, x):
        hy = Variable(torch.zeros(x.size(1),self.n_hid)).to(device)
        hz = Variable(torch.zeros(x.size(1),self.n_hid)).to(device)
        inputs = self.i2h(x)
        for t in range(x.size(0)):
            hz = hz + self.dt * (torch.tanh(self.h2h(torch.cat((hz,hy),dim=1)) + inputs[t])
                                          - self.gamma * hy - self.epsilon * hz)
            hy = hy + self.dt * hz

        return hy

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx, dt, gamma, epsilon):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.rnn = coRNN(embedding_dim, hidden_dim, dt, gamma, epsilon).to(device)
        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        hidden = self.rnn(embedded)
        return self.readout(hidden)
    
    
## set up data iterators and dictonary:
train_iterator, valid_iterator, test_iterator, text_field = get_data(batch_size, embedding)


n_inp = len(text_field.vocab)
n_out = 1
pad_idx = text_field.vocab.stoi[text_field.pad_token]

model = RNNModel(n_inp, embedding, n_hid, n_out, pad_idx, delta, gamma, epsilon).to(device)

## zero embedding for <unk_token> and <padding_token>:
zero_words_in_embedding(model, embedding, text_field, pad_idx)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

# accuracy fn
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

# eval fn
def evaluate(data_iterator):

    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_iterator):
            
            text, text_lengths = batch.text
            text = text.to(device)
            text_lengths = text_lengths.to(device)
            label = batch.label.to(device)
            
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, label)
            acc = binary_accuracy(predictions, label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
    return epoch_loss / len(data_iterator), epoch_acc / len(data_iterator)

# train fn
def train(epoch):

    epoch_loss = 0
    epoch_acc = 0
    model.train()
    with tqdm(total=len(train_iterator.dataset)) as progress_bar:
        for i, batch in enumerate(train_iterator):
            
            optimizer.zero_grad()
            text, text_lengths = batch.text
            text = text.to(device)
            text_lengths = text_lengths.to(device)
            label = batch.label.to(device)
            
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, label)
            acc = binary_accuracy(predictions, label)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
            progress_bar.set_postfix(loss=(epoch_loss/(i+1)))
            progress_bar.update(text_lengths.size(0))
    
    logger.info("Epoch : {:d} | Lr: {:.6f} | Acc: {:.4f} | Loss: {:.4f} | Cost Time: {}".format(
                epoch, optimizer.param_groups[0]['lr'], (epoch_acc / len(train_iterator)), 
                (epoch_loss / len(train_iterator)), str(datetime.timedelta(seconds=int(time.time() - start_time)))))
    
    return epoch_loss / len(train_iterator), epoch_acc / len(train_iterator)



start_time = time.time()
    
# set logger
logger = setup_logger("sentiment_analysis", '/home/JinK/IMDb/runs/logs',
                      filename='{}_train_log.txt'.format(model.__class__.__name__), mode='a+')

for epoch in range(epochs):
    train_loss, train_acc = train(epoch)
    eval_loss, eval_acc = evaluate(valid_iterator)
    # test_loss, test_acc = evaluate(test_iterator)
    print('Train set: Loss: {:.4f}, Accuracy: {:.3f}\n'.format(train_loss, train_acc))
    print('Valid set: Loss: {:.4f}, Accuracy: {:.3f}\n'.format(eval_loss, eval_acc))
    # print('Test set: Loss: {:.4f}, Accuracy: {:.2f}%\n'.format(test_loss, test_acc))

    # save model after specified epochs
    if (epoch+1) % 25 == 0:
        print('saving model...')
        torch.save(model.state_dict(), f'models/{model.__class__.__name__}-e{epoch+1}.pth')
