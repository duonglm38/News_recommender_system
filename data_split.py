import random
import _pickle as cPickle
random.seed(2019)


def main():
    with open('links.txt') as f:
        links = f.read().splitlines()[:517653]
    positive_pairs = []
    negative_pairs = []
    for link in links:
        token = [int(x) for x in link.split('-')]
        if token[2] == 1:
            positive_pairs.append((token[0], token[1], 1))
        elif random.random() < 0.21:
            negative_pairs.append((token[0], token[1], 0))
    print('num positive: {}, num negative: {}'.format(len(positive_pairs), len(negative_pairs)))

    random.shuffle(positive_pairs)
    random.shuffle(negative_pairs)

    val_start_idx1 = len(positive_pairs)*6//10
    val_start_idx2 = len(negative_pairs)*6//10
    test_start_idx1 = len(positive_pairs)*8//10
    test_start_idx2 = len(negative_pairs)*8//10
    train_data = positive_pairs[:val_start_idx1] + negative_pairs[:val_start_idx2]
    val_data = positive_pairs[val_start_idx1:test_start_idx1] + negative_pairs[val_start_idx2:test_start_idx2]
    test_data = positive_pairs[test_start_idx1:] + negative_pairs[test_start_idx2:]
    print('train, valid, test: {}, {}, {}'.format(len(train_data), len(val_data), len(test_data)))
    train_val_test = dict()
    train_val_test['train'] = train_data
    train_val_test['valid'] = val_data
    train_val_test['test'] = test_data

    with open('train_val_test.pkl', 'wb') as f:
        cPickle.dump(train_val_test, f)


if __name__ == '__main__':
    main()