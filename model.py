import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Classifier(nn.Module):
    def __init__(self, embedding, args):
        super(Classifier, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding)
        self.args = args

        self.embedding_dim = embedding.shape[1]
        self.rnn = RNN_Module(self.embedding_dim, args)
        
        self.news_dim = args.rnn_hidden_size*2 
        if args.model_type == 0:
            self.news_dim += args.doc_size
        self.feature_projection = nn.Sequential(nn.Dropout(args.dropout1),
                                               nn.Linear(self.news_dim, args.feature_dim),
                                               nn.ReLU())
        self.output_projection = nn.Sequential(nn.Dropout(args.dropout2),
                                               nn.Linear(2*args.feature_dim, 2),
                                               nn.LogSoftmax(dim=-1))

    def forward(self, inputs):
        title1 = self.embedding(inputs['title1'])
        title2 = self.embedding(inputs['title2'])
        title1 = self.rnn(title1, inputs['length1'])
        title2 = self.rnn(title2, inputs['length2'])
        
        mask1 = (inputs['title1'] == 0).unsqueeze(2)
        mask2 = (inputs['title2'] == 0).unsqueeze(2)
        title1 = title1.masked_fill(mask1, 0)
        title2 = title2.masked_fill(mask2, 0)
        title1 = torch.max(title1, dim=1)[0]
        title2 = torch.max(title2, dim=1)[0]
        
        if self.args.model_type == 0:
            doc1 = inputs['content1']   
            doc2 = inputs['content2']
            vec1 = torch.cat((title1, doc1), dim=1)
            vec2 = torch.cat((title2, doc2), dim=1)
        else:
            vec1 = title1
            vec2 = title2

        feature1 = self.feature_projection(vec1)
        feature2 = self.feature_projection(vec2)

        output = self.output_projection(torch.cat((feature1, feature2), dim=1))
        return output


class RNN_Module(nn.Module):
    def __init__(self, embedding_dim, args):
        super(RNN_Module, self).__init__()
        self.args = args
        self.news_dim = args.rnn_hidden_size + args.doc_size
        self.embedding_dim = embedding_dim
        self.lstm = nn.LSTM(num_layers=1, input_size=self.embedding_dim, hidden_size=args.rnn_hidden_size,
                            batch_first=True, bidirectional=True)

    def forward(self, input_ids, lengths):
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        max_length = seq_lengths[0]
        iperm_idx = torch.LongTensor(perm_idx.shape).fill_(0).to(self.args.device)
        for i, v in enumerate(perm_idx):
            iperm_idx[v.data] = i
        inRep = input_ids[perm_idx]

        inRep = pack_padded_sequence(inRep, seq_lengths.data.cpu().numpy(), batch_first=True)

        outRep, _ = self.lstm(inRep)

        outRep, _ = pad_packed_sequence(outRep, batch_first=True, total_length=max_length)
        outRep = outRep[iperm_idx]
        return outRep