import os
import glob
import random
import xml.etree.ElementTree as ET
from .base import Provider


class ADABProvider(Provider):
    """ADAB Arabic online handwriting dataset provider.

    Dataset layout expected (any nesting depth):
        <dataset_dir>/
            .../inkml/<name>.inkml   — stroke traces
            .../upx/<name>.upx       — Arabic word label
    """
    name = 'adab'

    # Reproducible 80 / 10 / 10 split
    _TRAIN_FRAC = 0.80
    _VAL_FRAC   = 0.10
    # remaining 10 % → test

    def __init__(self, dataset_dir='dataset', seed=42):
        self._dataset_dir = dataset_dir
        self._seed = int(seed)
        self._splits = None  # loaded lazily

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_training_data(self):
        train, _, _ = self._get_splits()
        return self._iter_examples(train)

    def get_validation_data(self):
        _, val, _ = self._get_splits()
        return self._iter_examples(val)

    def get_test_data(self):
        _, _, test = self._get_splits()
        return self._iter_examples(test)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_splits(self):
        if self._splits is None:
            pairs = self._find_pairs()
            rng = random.Random(self._seed)
            rng.shuffle(pairs)
            n = len(pairs)
            n_train = int(n * self._TRAIN_FRAC)
            n_val   = int(n * self._VAL_FRAC)
            self._splits = (
                pairs[:n_train],
                pairs[n_train:n_train + n_val],
                pairs[n_train + n_val:],
            )
            print(f'ADAB: {n} files -> train {len(self._splits[0])} / '
                  f'val {len(self._splits[1])} / test {len(self._splits[2])}')
        return self._splits

    def _find_pairs(self):
        """Return list of (inkml_path, upx_path) for every matched pair."""
        inkml_files = glob.glob(
            os.path.join(self._dataset_dir, '**', '*.inkml'), recursive=True
        )
        pairs = []
        for inkml_path in sorted(inkml_files):
            basename = os.path.splitext(os.path.basename(inkml_path))[0]
            inkml_dir = os.path.dirname(inkml_path)
            parent_dir = os.path.dirname(inkml_dir)
            upx_path = os.path.join(parent_dir, 'upx', basename + '.upx')
            if os.path.exists(upx_path):
                pairs.append((inkml_path, upx_path))
        return pairs

    def _iter_examples(self, pairs):
        for inkml_path, upx_path in pairs:
            try:
                strokes = self._parse_inkml(inkml_path)
                text    = self._parse_upx(upx_path)
                if strokes and text.strip():
                    yield strokes, text
            except Exception:
                continue

    # ------------------------------------------------------------------
    # File parsers
    # ------------------------------------------------------------------

    def _parse_inkml(self, path):
        tree = ET.parse(path)
        root = tree.getroot()
        ns = 'http://www.w3.org/2003/InkML'
        strokes = []
        for trace in root.findall(f'{{{ns}}}trace'):
            raw = (trace.text or '').strip()
            if not raw:
                continue
            points = []
            for pt in raw.split(','):
                parts = pt.strip().split()
                if len(parts) >= 2:
                    points.append((float(parts[0]), float(parts[1])))
            if points:
                strokes.append(points)
        return strokes

    def _parse_upx(self, path):
        tree = ET.parse(path)
        root = tree.getroot()
        ns = 'http://unipen.nici.ru.nl/upx'
        for alt in root.iter(f'{{{ns}}}alternate'):
            val = alt.get('value', '').strip()
            if val:
                return val
        return ''
