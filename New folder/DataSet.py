import os.path as osp
from mmengine.fileio import list_from_file
from mmengine.dataset import BaseDataset
from mmaction.registry import DATASETS


@DATASETS.register_module()
class DatasetDogAction(BaseDataset):
    def __init__(self, ann_file, pipeline, data_root, data_prefix=dict(video=''),
                 test_mode=False, modality='RGB', **kwargs):
        self.modality = modality
        super(DatasetDogAction, self).__init__(ann_file=ann_file, pipeline=pipeline, data_root=data_root,
                                           data_prefix=data_prefix, test_mode=test_mode,
                                           **kwargs)

    def load_data_list(self):
        data_list = []
        fin = list_from_file(self.ann_file)
        for line in fin:
            line_split = line.strip().split()
            filename = line_split[0]
            labels = line_split[1].split(',')  # 分隔不同的标签
            labels = [int(label) for label in labels]  # 将标签转换为整数
            filename = osp.join(self.data_prefix['video'], filename)
            data_list.append(dict(filename=filename, labels=labels))  # 使用列表存储多个标签
        return data_list

    def get_data_info(self, idx: int) -> dict:
        data_info = super().get_data_info(idx)
        data_info['modality'] = self.modality
        data_info['labels'] = data_info['labels']  # 确保返回多标签
        return data_info