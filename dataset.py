from collections import Counter

import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

import pre_processing
import utils


class CharacterTable:
    MASK_TOKEN = ''

    def __init__(self, chars):
        # make sure to be consistent with JS
        self.chars = [CharacterTable.MASK_TOKEN] + chars
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def __len__(self):
        return len(self.chars)

    def to_ids(self, css):
        return [
            [self.char_indices[c] for c in cs] for cs in css
        ]

    def to_niqud(self, cs):
        return [self.indices_char[cs[i].item()] for i in range(cs.size(dim=0))]

    def __repr__(self):
        return repr(self.chars)


letters_table = CharacterTable(pre_processing.SPECIAL_TOKENS + pre_processing.VALID_LETTERS)
dagesh_table = CharacterTable(pre_processing.DAGESH)
sin_table = CharacterTable(pre_processing.NIQQUD_SIN)
niqqud_table = CharacterTable(pre_processing.NIQQUD)

LETTERS_SIZE = len(letters_table)
NIQQUD_SIZE = len(niqqud_table)
DAGESH_SIZE = len(dagesh_table)
SIN_SIZE = len(sin_table)
PAD_INDEX = 20  # TODO check if to move to main or to config file
# A dictionary that converts a triplet (niqqud, dagesh, sin) into a unique ID for the triplet
niqqud_to_id_dict = {}

# A dictionary that an ID to a triplet (niqqud, dagesh, sin). The reverse of niqqud_to_id_dict
id_to_niqqud_dict = {}

Y2_SIZE = 0
for niqqud in range(NIQQUD_SIZE):
    for sin in range(SIN_SIZE):
        for dag in range(DAGESH_SIZE):
            niqqud_to_id_dict[(niqqud, dag, sin)] = Y2_SIZE
            id_to_niqqud_dict[Y2_SIZE] = (niqqud, dag, sin)
            Y2_SIZE += 1


def print_tables():
    print('const ALL_TOKENS =', letters_table.chars, end=';\n')
    print('const niqqud_array =', niqqud_table.chars, end=';\n')
    print('const dagesh_array =', dagesh_table.chars, end=';\n')
    print('const sin_array =', sin_table.chars, end=';\n')


def from_categorical(t):
    return np.argmax(t, axis=-1)


def merge(texts, tnss, nss, dss, sss):
    res = []
    for ts, tns, ns, ds, ss in zip(texts, tnss, nss, dss, sss):
        sentence = []
        for t, tn, n, d, s in zip(ts, tns, ns, ds, ss):
            if tn == 0:
                break
            sentence.append(t)
            if pre_processing.can_dagesh(t):
                sentence.append(dagesh_table.indices_char[d].replace(pre_processing.RAFE, ''))
            if pre_processing.can_sin(t):
                sentence.append(sin_table.indices_char[s].replace(pre_processing.RAFE, ''))
            if pre_processing.can_niqqud(t):
                sentence.append(niqqud_table.indices_char[n].replace(pre_processing.RAFE, ''))
        res.append(''.join(sentence))
    return res


class textDataset(Dataset):
    def __init__(self, base_paths, maxlen, minlen, tokenizer, split_sentence):

        def pad(ords, dtype='int32', value=PAD_INDEX):
            return utils.pad_sequences(ords, maxlen=maxlen, dtype=dtype, value=value)

        self.labels = []
        self.text = []
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.minlen = minlen
        self.split_sentence = split_sentence
        self.counter = {'N': Counter([i for i in range(NIQQUD_SIZE)]),
                        'D': Counter([i for i in range(DAGESH_SIZE)]),
                        'S': Counter([i for i in range(SIN_SIZE)])}

        corpora = read_corpora(base_paths)
        for (filename, heb_items) in corpora:
            if self.split_sentence:
                text, normalized, dagesh, sin, niqqud = zip(
                    *(zip(*line) for line in pre_processing.split_by_length(heb_items, maxlen)))
            else:
                text, normalized, dagesh, sin, niqqud = zip(
                    *(zip(*line) for line in pre_processing.split_by_sentence(heb_items, maxlen, minlen)))

            niqqud = pad(niqqud_table.to_ids(niqqud))
            dagesh = pad(dagesh_table.to_ids(dagesh))
            sin = pad(sin_table.to_ids(sin))

            for i in range(len(text)):
                self.counter['N'] += Counter(niqqud[i])
                self.counter['D'] += Counter(dagesh[i])
                self.counter['S'] += Counter(sin[i])

                self.labels.append({'N': niqqud[i], 'D': dagesh[i], 'S': sin[i]})
                self.text.append("".join(normalized[i]))

        self.counter['N'].pop(PAD_INDEX)
        self.counter['D'].pop(PAD_INDEX)
        self.counter['S'].pop(PAD_INDEX)
        # y3 = [[[0 for i in range(NIQQUD_SIZE + DAGESH_SIZE + SIN_SIZE)] for j in range(len(niqqud[q]))] for q in range(len(niqqud))]
        # for i in range(len(niqqud)):
        #     for j in range(len(niqqud[i])):
        #         y3[i][j][niqqud[i][j]] = 1/3
        #         y3[i][j][dagesh[i][j]] = 1/3
        #         y3[i][j][sin[i][j]] = 1/3
        #
        #     item = {
        #         "text": "".join(normalized[i]),
        #         "label": {'N': niqqud[i], 'D': dagesh[i], 'S': sin[i]},
        #         #"y2": [niqqud_to_id_dict[(niqqud[i][j], dagesh[i][j], sin[i][j])] for j in range(len(niqqud[i]))],
        #         #"y3": y3
        #     }
        #     self.data.append(item)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.text[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # todo remove? good for later QA?
            max_length=self.maxlen,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            input_ids=encoding['input_ids'].flatten(),
            attention_mask=encoding['attention_mask'].flatten(),
            label=label,
            text=text
        )


def read_corpora(base_paths):
    return tuple(
        [(filename, list(pre_processing.iterate_file(filename))) for filename in utils.iterate_files(base_paths)])


def calc_weights(train_data):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    a function by ido to calculate the recommended weights for the model
    Args:
        train_data:

    Returns:

    """
    tmp = train_data.counter['N']
    n_weights = torch.tensor([tmp[i] for i in range(len(tmp.keys()))])
    n_weights = (n_weights / n_weights.sum()).to(device)

    tmp = train_data.counter['D']
    d_weights = torch.tensor([tmp[i] for i in range(len(tmp.keys()))])
    d_weights = (d_weights / d_weights.sum()).to(device)

    tmp = train_data.counter['S']
    s_weights = torch.tensor([tmp[i] for i in range(len(tmp.keys()))])
    s_weights = (s_weights / s_weights.sum()).to(device)

    weights = {'N': n_weights, 'S': s_weights, 'D': d_weights}
    return weights


if __name__ == '__main__':
    # data = Data.concatenate([Data.from_text(x, maxlen=64) for x in read_corpora(['train1.txt'])])
    # data.print_stats()
    # print(np.concatenate([data.normalized[:1], data.sin[:1]]))
    # res = merge(data.text[:1], data.normalized[:1], data.niqqud[:1], data.dagesh[:1], data.sin[:1])
    # print(res)
    # train_dict = {}
    # train_dict["test"] = get_xy(load_data(tuple(['train1.txt', 'train2.txt']), maxlen=16).shuffle())
    test_data = textDataset(tuple([r"hebrew_diacritized/data/train/early_modern"]), 100, 10,
                            AutoTokenizer.from_pretrained('tavbert', use_fast=True), False)
    import json

    res = calc_weights(test_data)
    print(10 * "-" + "\n\n\nweights_early_modern\n\n\n" + 10 * "-")
    print(res)
    # with open("weights_early_modern.json", "w") as f:
    #     print(10 * "-" + "\n\n\nweights_early_modern\n\n\n" + 10 * "-")
    #     res_s = json.dumps(res, indent=4)
    #     print(res_s)
    #     f.write(res_s)

    test_data = textDataset(tuple([r"hebrew_diacritized/data/train/modern"]), 100, 10,
                            AutoTokenizer.from_pretrained('tavbert', use_fast=True), False)
    import json

    res = calc_weights(test_data)
    print(10 * "-" + "\n\n\nweights_modern\n\n\n" + 10 * "-")
    print(res)
    # with open("weights_modern.json", "w") as f:
    #     res_s = json.dumps(res, indent=4)
    #     print(10 * "-" + "\n\n\nweights_modern\n\n\n" + 10 * "-")
    #     print(res_s)
    #     f.write(res_s)

    # loader = DataLoader(test_data, shuffle=True)
    #
    # sample_batch = next(iter(loader))

    # text = metrics.format_output_y1(sample_batch['input_ids'].squeeze(),sample_batch['label']['N'].squeeze(),sample_batch['label']['D'].squeeze(),sample_batch['label']['S'].squeeze())
    # input = sample_batch["input_ids"]
    # mask = sample_batch["attention_mask"]
    # # _, predictions = model(input, mask)
    # x = 5

    # print_tables()
    # print(letters_table.to_ids(["שלום"]))
    # for i in range(1):
    #     sample = testData[i]
    #     print(sample)
    #     print(f"{i}: text type is {type(sample['text'])}. the text is:\n {sample['text']}\n\n\n")
    #     print(
    #         f"{i}: y1 type is {type(sample['y1'])}. y1 is:\n {sample['y1']}\n\n\n")
    #     x=5

# load_data.clear_cache()
