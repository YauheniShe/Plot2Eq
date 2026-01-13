import torch
from torch.utils.data import Dataset
import pickle
import gzip
from pathlib import Path

class SymbolicDataset(Dataset):
    def __init__(self, data_dir: str | Path) -> None:
        self.data = []
        data_path = Path(data_dir)
        if not data_path.exists():
            raise RuntimeError(f'Папка {data_path} не найдена')
        
        files = sorted(list(data_path.glob("chunk_*.pkl.gz")))

        if not files:
            raise RuntimeError(f'Файлы не найдены в {data_path}')
        
        print(f"Загрузка данных из {len(files)} файлов...")
        for f_path in files:
            with gzip.open(f_path, 'rb') as f:
                chunk = pickle.load(f)
                self.data.extend(chunk)
        print (f'Загружено {len(self.data)} примеров')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        # points shape: (Steps, 3) -> (x, y, mask)
        points = item['points']
        # tokens shape: (Seq_Len,)
        tokens = item['tokens']
        return {
            'src': torch.from_numpy(points).float(),
            'tgt': torch.from_numpy(tokens).long()
        }
    
def collate_fn(batch):
    src_list = [item['src'] for item in batch]
    tgt_list = [item['tgt'] for item in batch]

    src_batch = torch.stack(src_list)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=0)

    return src_batch, tgt_batch
