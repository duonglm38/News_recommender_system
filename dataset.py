from torch.utils.data import Dataset
import torch


class MyDataset(Dataset):
    def __init__(self, args, data, idx2processedData, doc2vec):
        super(MyDataset, self).__init__()
        self.args = args
        self.idx2processedData = idx2processedData
        self.pairs = [p for p in data]
        self.doc2vec = {int(k):v for k, v in doc2vec.items()}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1 = self.idx2processedData[self.pairs[idx][0]]
        p2 = self.idx2processedData[self.pairs[idx][1]]
        content_vec1 = self.doc2vec[self.pairs[idx][0]]
        content_vec2 = self.doc2vec[self.pairs[idx][1]]
        label = self.pairs[idx][2]
        return p1, p2, content_vec1, content_vec2, label

    @staticmethod
    def collate_fn(batch):
        batch_title1 = []
        batch_title2 = []
        batch_content1 = []
        batch_content2 = []
        batch_labels = []

        length1 = [len(p[0]['title']) for p in batch]
        max_length1 = max(length1)
        length2 = [len(p[1]['title']) for p in batch]
        max_length2 = max(length2)

        for p1, p2, content_vec1, content_vec2, label in batch:
            amount_to_pad_title1 = max_length1 - len(p1['title'])
            amount_to_pad_title2 = max_length2 - len(p2['title'])
            title1 = p1['title'] + [0]*amount_to_pad_title1
            title2 = p2['title'] + [0]*amount_to_pad_title2
            batch_content1.append(content_vec1) 
            batch_content2.append(content_vec2) 
            batch_labels.append(label)
            batch_title1.append(title1)
            batch_title2.append(title2)
        return torch.LongTensor(batch_title1), torch.FloatTensor(batch_content1), torch.LongTensor(length1), \
               torch.LongTensor(batch_title2), torch.FloatTensor(batch_content2), torch.LongTensor(length2), \
               torch.LongTensor(batch_labels)

    @staticmethod
    def nodoc2vec_collate_fn(batch):
        batch_title1 = []
        batch_title2 = []
        batch_content1 = []
        batch_content2 = []
        batch_labels = []

        length1 = [(len(p[0]['title']) + len(p[0]['content'])) for p in batch]
        max_length1 = max(length1)
        length2 = [(len(p[1]['title']) + len(p[1]['content'])) for p in batch]
        max_length2 = max(length2)

        for p1, p2, content_vec1, content_vec2, label in batch:
            amount_to_pad_title1 = max_length1 - len(p1['title']) - len(p1['content'])
            amount_to_pad_title2 = max_length2 - len(p2['title']) - len(p2['content'])
            title1 = p1['title'] + p1['content'] + [0]*amount_to_pad_title1
            title2 = p2['title'] + p2['content'] + [0]*amount_to_pad_title2
            batch_labels.append(label)
            batch_title1.append(title1)
            batch_title2.append(title2)
        return torch.LongTensor(batch_title1), torch.FloatTensor(batch_content1), torch.LongTensor(length1), \
               torch.LongTensor(batch_title2), torch.FloatTensor(batch_content2), torch.LongTensor(length2), \
               torch.LongTensor(batch_labels)

    @staticmethod
    def justtitle_collate_fn(batch):
        batch_title1 = []
        batch_title2 = []
        batch_content1 = []
        batch_content2 = []
        batch_labels = []

        length1 = [len(p[0]['title']) for p in batch]
        max_length1 = max(length1)
        length2 = [len(p[1]['title']) for p in batch]
        max_length2 = max(length2)

        for p1, p2, content_vec1, content_vec2, label in batch:
            amount_to_pad_title1 = max_length1 - len(p1['title'])
            amount_to_pad_title2 = max_length2 - len(p2['title'])
            title1 = p1['title'] + [0]*amount_to_pad_title1
            title2 = p2['title'] + [0]*amount_to_pad_title2
            batch_labels.append(label)
            batch_title1.append(title1)
            batch_title2.append(title2)
        return torch.LongTensor(batch_title1), torch.FloatTensor(batch_content1), torch.LongTensor(length1), \
               torch.LongTensor(batch_title2), torch.FloatTensor(batch_content2), torch.LongTensor(length2), \
               torch.LongTensor(batch_labels)

    @staticmethod
    def justcontent_collate_fn(batch):
        batch_title1 = []
        batch_title2 = []
        batch_content1 = []
        batch_content2 = []
        batch_labels = []

        length1 = [len(p[0]['content']) for p in batch]
        max_length1 = max(length1)
        length2 = [len(p[1]['content']) for p in batch]
        max_length2 = max(length2)

        for p1, p2, content_vec1, content_vec2, label in batch:
            amount_to_pad_title1 = max_length1 - len(p1['content'])
            amount_to_pad_title2 = max_length2 - len(p2['content'])
            title1 = p1['content'] + [0]*amount_to_pad_title1
            title2 = p2['content'] + [0]*amount_to_pad_title2
            batch_labels.append(label)
            batch_title1.append(title1)
            batch_title2.append(title2)
        return torch.LongTensor(batch_title1), torch.FloatTensor(batch_content1), torch.LongTensor(length1), \
               torch.LongTensor(batch_title2), torch.FloatTensor(batch_content2), torch.LongTensor(length2), \
               torch.LongTensor(batch_labels)
