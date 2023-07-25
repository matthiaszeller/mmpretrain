from zipfile import ZipFile
from pathlib import Path

from mmengine import get_file_backend, list_from_file

from mmpretrain.registry import DATASETS

import os.path as osp

from .base_dataset import BaseDataset


@DATASETS.register_module()
class IVOCTDataset(BaseDataset):

    def __init__(self,
                 data_root: str,
                 split: str = None,
                 **kwargs) -> None:

        self.backend = get_file_backend(data_root, enable_singleton=True)
        self.split = None if split is None else self.backend.join_path(data_root, split)

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
                for file in zf.namelist():
                    if file.endswith('.png'):
                        data_info = {
                            'img_path': file,
                            'zip_path': str(zip_path),
                        }
                        data_list.append(data_info)

        return data_list
