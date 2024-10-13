import typing
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import List, Dict

import dataset
import pre_processing


@dataclass(frozen=True)
class Document:
    name: str
    system: str
    text: str

    def sentences(self):
        return [sent + '.' for sent in self.text.split('. ') if len(pre_processing.remove_niqqud(sent)) > 15]

    def hebrew_items(self) -> List[pre_processing.HebrewItem]:
        return list(pre_processing.iterate_dotted_text(self.text))

    def tokens(self) -> List[pre_processing.Token]:
        return pre_processing.tokenize(self.hebrew_items())

    def vocalized_tokens(self) -> List[pre_processing.Token]:
        return [x.vocalize() for x in self.tokens()]


@dataclass(frozen=True)
class DocumentPack:
    source: str
    name: str
    docs: Dict[str, Document]

    def __getitem__(self, item):
        return self.docs[item]

    @property
    def expected(self):
        return self.docs['expected']

    @property
    def actual(self):
        assert len(self.docs) == 2
        assert 'expected' in self.docs
        return self.docs[(set(self.docs.keys()) - {'expected'}).pop()]


def read_document(system: str, path: Path) -> Document:
    return Document(path.name, system, ' '.join(path.read_text(encoding='utf8').strip().split()))


def read_document_pack(path_to_expected: Path, *systems: str) -> DocumentPack:
    return DocumentPack(path_to_expected.parent.name, path_to_expected.name,
                        {system: read_document(system, system_path_from_expected(path_to_expected, system))
                         for system in systems})


def iter_documents(*systems) -> typing.Iterator[DocumentPack]:
    for folder in basepath.iterdir():
        for path_to_expected in folder.iterdir():
            yield read_document_pack(path_to_expected, *systems)


def iter_documents_by_folder(*systems) -> typing.Iterator[List[DocumentPack]]:
    for folder in basepath.iterdir():
        yield [read_document_pack(path_to_expected, *systems) for path_to_expected in folder.iterdir()]


def system_path_from_expected(path: Path, system: str) -> Path:
    return Path(str(path).replace('expected', system))


def collect_failed_tokens(doc_pack, context):
    tokens_of = {system: doc_pack[system].tokens() for system in doc_pack.docs}
    for i in range(len(tokens_of['expected'])):
        res = {system: str(tokens_of[system][i]) for system in doc_pack.docs}
        if len(set(res.values())) > 1:
            pre_nonhebrew, _, post_nonhebrew = tokens_of['expected'][i].split_on_hebrew()
            pre = ' '.join(token_to_text(x) for x in tokens_of['expected'][i - context:i]) + ' ' + pre_nonhebrew
            post = post_nonhebrew + " " + ' '.join(
                token_to_text(x) for x in tokens_of['expected'][i + 1:i + context + 1])
            res = {system: token_to_text(tokens_of[system][i].split_on_hebrew()[1]) for system in doc_pack.docs}
            yield (pre, res, post)


def metric_cha(doc_pack: DocumentPack) -> float:
    """
    Calculate character-level agreement between actual and expected.
    """
    return mean_equal((x, y) for x, y in zip(doc_pack.actual.hebrew_items(), doc_pack.expected.hebrew_items())
                      if pre_processing.can_any(x.letter))


def metric_dec(doc_pack: DocumentPack) -> float:
    """
    Calculate nontrivial-decision agreement between actual and expected.
    """
    actual_hebrew = doc_pack.actual.hebrew_items()
    expected_hebrew = doc_pack.expected.hebrew_items()

    return mean_equal(
        ((x.niqqud, y.niqqud) for x, y in zip(actual_hebrew, expected_hebrew)
         if pre_processing.can_niqqud(x.letter)),

        ((x.dagesh, y.dagesh) for x, y in zip(actual_hebrew, expected_hebrew)
         if pre_processing.can_dagesh(x.letter)),

        ((x.sin, y.sin) for x, y in zip(actual_hebrew, expected_hebrew)
         if pre_processing.can_sin(x.letter)),
    )


def is_hebrew(token: pre_processing.Token) -> bool:
    return len([c for c in token.items if c.letter in pre_processing.HEBREW_LETTERS]) > 1


def metric_wor(doc_pack: DocumentPack) -> float:
    """
    Calculate token-level agreement between actual and expected,
    for tokens containing at least 2 Hebrew letters.
    """
    return mean_equal((x, y) for x, y in zip(doc_pack.actual.tokens(), doc_pack.expected.tokens())
                      if is_hebrew(x))


def metric_voc(doc_pack: DocumentPack) -> float:
    """
    Calculate token-level agreement over vocalization, between actual and expected,
    for tokens containing at least 2 Hebrew letters.
    """
    return mean_equal((x, y) for x, y in zip(doc_pack.actual.vocalized_tokens(), doc_pack.expected.vocalized_tokens())
                      if is_hebrew(x))


def token_to_text(token: pre_processing.Token) -> str:
    return str(token).replace(pre_processing.RAFE, '')


def mean_equal(*pair_iterables):
    total = 0
    acc = 0
    for pair_iterable in pair_iterables:
        pair_iterable = list(pair_iterable)
        total += len(pair_iterable)
        acc += sum(x == y for x, y in pair_iterable)
    return acc / total


def all_diffs_for_files(doc_pack: DocumentPack, system1: str, system2: str) -> None:
    triples = [(e, a1, a2) for (e, a1, a2) in zip(doc_pack.expected.sentences(),
                                                  doc_pack[system1].sentences(),
                                                  doc_pack[system2].sentences())
               if metric_wor(a1, e) < 0.90 or metric_wor(a2, e) < 0.90]
    triples.sort(key=lambda e_a1_a2: metric_cha(e_a1_a2[2], e_a1_a2[0]))
    for (e, a1, a2) in triples[:20]:
        print(f"{system1}: {metric_wor(a1, e):.2%}; {system2}: {metric_wor(a2, e):.2%}")
        print('סבבה:', a1)
        print('מקור:', e)
        print('גרוע:', a2)
        print()


def all_metrics(doc_pack: DocumentPack):
    return {
        'dec': metric_dec(doc_pack),
        'cha': metric_cha(doc_pack),
        'wor': metric_wor(doc_pack),
        'voc': metric_voc(doc_pack)
    }


def metricwise_mean(iterable):
    items = list(iterable)
    keys = items[0].keys()
    return {
        key: np.mean([item[key] for item in items])
        for key in keys
    }


def macro_average(system):
    return metricwise_mean(
        metricwise_mean(all_metrics(doc_pack) for doc_pack in folder_packs)
        for folder_packs in iter_documents_by_folder('expected', system)
    )


def micro_average(system):
    return metricwise_mean(all_metrics(doc_pack) for doc_pack in iter_documents('expected', system))


def format_latex(system, results):
    print(r'{system} & {dec:.2%} & {cha:.2%} & {wor:.2%} & {voc:.2%} \\'.format(system=system, **results)
          .replace('%', ''))


def all_stats(*output_paths):
    for output_path in output_paths:
        results = macro_average(output_path)
        format_latex(output_path, results)
        results = micro_average(output_path)
        format_latex(output_path, results)
        ew = 1 - results['wor']
        ev = 1 - results['voc']
        print(f'{(ew - ev) / ew:.2%}')
        print()
        return results


def all_failed():
    for doc_pack in iter_documents('expected', 'Dicta', 'Nakdimon'):
        for pre, ngrams, post in collect_failed_tokens(doc_pack, context=3):
            res = "|".join(ngrams.values())
            print(f'{doc_pack.source}|{doc_pack.name}| {pre}|{res}|{post} |')


from dataset import niqqud_table, dagesh_table, sin_table

def format_output_y1(pad_text, pad_niqqud, pad_dagesh, pad_sin, tokenizer):
    pad_text = pad_text.squeeze()
    pad_niqqud = pad_niqqud.squeeze()
    pad_dagesh = pad_dagesh.squeeze()
    pad_sin = pad_sin.squeeze()
    # find the index where we start padding
    pad_index = 0
    for i in range(pad_text.size(dim=0)):
        pad_index = i
        if pad_text[i] == 2:
            break

    # decode unpadded text
    coded_text = pad_text[1:pad_index]
    text = tokenizer.decode(coded_text,clean_up_tokenization_spaces=False)

    # remove padding
    niqqud = niqqud_table.to_niqud(pad_niqqud[1:pad_index])
    dagesh = dagesh_table.to_niqud(pad_dagesh[1:pad_index])
    sin = sin_table.to_niqud(pad_sin[1:pad_index])

    output = ''
    for i in range(len(niqqud)):
        diacritization = ''.join((text[2*i], dagesh[i], sin[i], niqqud[i]))
        output = ''.join((output, diacritization))
    return output.replace(pre_processing.RAFE, '')


def format_output_y2(text, y2) -> str:
    output = ''
    for letter, id in zip(text, y2):
        diacritization = ''.join(dataset.id_to_niqqud_dict[id])
        output = ''.join((output, diacritization))
    return output

basepath = Path('expected')

if __name__ == '__main__':
    basepath = Path('expected')
    all_stats(
        'predicted',
        # 'Dicta',
        # 'Nakdimon0',
        # 'Nakdimon',
    )

