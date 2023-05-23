import gc
import re

from importlib import import_module
from prettytable import PrettyTable


def huggingface_login():
    import huggingface_hub

    if not huggingface_hub.HfFolder.get_token():
        huggingface_hub.login()


def word_tokenizer(input: str):
    input = str.lower(input) \
        .replace('`', "'") \
        .replace("can't", "cannot") \
        .replace("n't", " not") \
        .replace("i'm", "i am")
    return re.findall(r"\w+|[^\w\s]+", input)


def subtitle(contents: str) -> str:
    return f'\n>>> {contents}:'


def title(contents: str) -> str:
    return '\n' + '/' * 20 + subtitle(contents)


def create_prettytable(results):
    if isinstance(results, dict):
        results = [results]
    table = PrettyTable()
    for row in results:
        table.add_row(row.values())
    table.field_names = results[0].keys()
    return table


def get_graph_for_print(graph, x_map):
    res = str(graph)
    for x, name in sorted(x_map.items(), key=lambda x: int(x[0].replace('X', '')), reverse=True):
        res = res.replace(x, name)
    return res



class TQDMPatch():
    def __init__(self, *k, **kw):
        from tqdm.auto import tqdm
        self.patch_fns = dir(tqdm)
        self.iterator = None
        for _k in list(k) + list(kw.values()):
            if hasattr(_k, '__iter__'):
                self.iterator = _k
                break

    def __iter__(self):
        return self.iterator.__iter__()

    def __getattr__(self, attr):
        if attr in self.patch_fns:
            return lambda *k, **kw: None        
        return attr

    def __call__(self, i, *k, **kw):
        return i


class silence_inner_tqdm:
    """Silence all outputs of tqdm in `causallearn.utils.FAS`."""

    def __init__(self, name: str = 'causallearn.utils.FAS'):
        self.name = name
        self.orig_ref = None

    def __enter__(self):
        try:
            referents = gc.get_referents(import_module(self.name))[0]
            if 'tqdm' in referents:
                self.orig_ref = referents['tqdm']
                referents['tqdm'] = TQDMPatch
        except (AttributeError, ModuleNotFoundError, ValueError):
            pass

    def __exit__(self, *args):
        try:
            gc.get_referents(import_module(self.name))[0]['tqdm'] = self.orig_ref
        except (ModuleNotFoundError, ValueError):
            pass
