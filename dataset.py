from typing import Tuple, List
import random
import numpy as np

from cachier import cachier
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

tokenize = AutoTokenizer.from_pretrained("tau/tavbert-he")


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




class Data:
    text: np.ndarray = None
    normalized: np.ndarray = None
    dagesh: np.ndarray = None
    sin: np.ndarray = None
    niqqud: np.ndarray = None

    filenames: Tuple[str, ...] = ()

    @staticmethod
    def concatenate(others):
        """
        Used as a constructor
        :param others:
        :return:
        """
        self = Data()
        self.text = np.concatenate([x.text for x in others])
        self.normalized = np.concatenate([x.normalized for x in others],dtype=object)
        self.dagesh = np.concatenate([x.dagesh for x in others])
        self.sin = np.concatenate([x.sin for x in others])
        self.niqqud = np.concatenate([x.niqqud for x in others])
        return self

    def shapes(self):
        return self.text.shape, self.normalized.shape, self.dagesh.shape, self.sin.shape, self.niqqud.shape  # , self.kind.shape

    def shuffle(self):
        utils.shuffle_in_unison(
            self.text,
            self.normalized,
            self.dagesh,
            self.niqqud,
            self.sin
        )
        return self

    @staticmethod
    def from_text(heb_items, maxlen: int) -> 'Data':
        assert heb_items
        self = Data()
        text, normalized, dagesh, sin, niqqud = zip(*(zip(*line) for line in pre_processing.split_by_length(heb_items, maxlen)))

        def pad(ords, dtype='int32', value=0):
            return utils.pad_sequences(ords, maxlen=maxlen, dtype=dtype, value=value)

        self.normalized = tuple("".join(x) for x in normalized)
        self.dagesh = dagesh_table.to_ids(dagesh)
        self.sin = sin_table.to_ids(sin)
        self.niqqud = niqqud_table.to_ids(niqqud)
        self.text = text
        return self

    def __len__(self):
        return self.normalized.shape[0]

    def print_stats(self):
        print(self.shapes())


def read_corpora(base_paths):
    return tuple([(filename, list(pre_processing.iterate_file(filename))) for filename in utils.iterate_files(base_paths)])


@cachier()
def load_data(base_paths, maxlen: int) -> Data:
    corpora = read_corpora(base_paths)
    corpus = [(filename, Data.from_text(heb_items, maxlen)) for (filename, heb_items) in corpora]
    cs = [c for (_, c) in corpus]
    random.shuffle(cs)
    return Data.concatenate(cs)

def get_xy(d):
    x = d.normalized
    y = {'N': d.niqqud, 'D': d.dagesh, 'S': d.sin}
    return (x, y)

if __name__ == '__main__':
    # data = Data.concatenate([Data.from_text(x, maxlen=64) for x in read_corpora(['test1.txt'])])
    # data.print_stats()
    # print(np.concatenate([data.normalized[:1], data.sin[:1]]))
    # res = merge(data.text[:1], data.normalized[:1], data.niqqud[:1], data.dagesh[:1], data.sin[:1])
    # print(res)
    train_dict = {}
    train_dict["test"] = get_xy(load_data(tuple(['test1.txt']), maxlen=16).shuffle())
    print_tables()
    print(letters_table.to_ids(["שלום"]))

load_data.clear_cache()
