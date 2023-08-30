from zipfile import ZipFile
from pathlib import Path

from mmengine import get_file_backend, list_from_file

from mmpretrain.registry import DATASETS

import os.path as osp

from .base_dataset import BaseDataset


def skip_edge_elements(lst: list, skip_edge_items: int):
    if skip_edge_items == 0:
        return lst
    elif skip_edge_items < 0:
        raise ValueError("skip_edge_items must be non-negative.")
    else:
        return lst[skip_edge_items:-skip_edge_items]


@DATASETS.register_module()
class IVOCTDataset(BaseDataset):

    def __init__(self,
                 data_root: str,
                 skip_edge_slices: int = 0,
                 split: str = None,
                 **kwargs) -> None:

        self.backend = get_file_backend(data_root, enable_singleton=True)
        self.split = None if split is None else self.backend.join_path(data_root, split)
        self.skip_edge_slices = skip_edge_slices

        super().__init__(
            ann_file='',
            data_root=data_root,
            **kwargs)

    def load_data_list(self) -> list[dict]:
        if self.split is None:
            zip_ids = [
                p.stem
                for p in Path(self.data_root).glob('*.zip')
            ]
        else:
            zip_ids = list_from_file(self.split)

        data_list = []

        for zip_id in zip_ids:
            zip_path = self.backend.join_path(self.data_root, f'{zip_id}.zip')
            with ZipFile(zip_path) as zf:
                files = [f for f in zf.namelist() if f.endswith('.png')]

            files = skip_edge_elements(files, self.skip_edge_slices)
            for file in files:
                data_info = {
                    'img_path': file,
                    'zip_path': str(zip_path),
                }
                data_list.append(data_info)

        return data_list
