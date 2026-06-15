import torch, pathlib
from pathlib import Path
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset
from dataclasses import dataclass
from random import Random


class FastMRIBrainData(Dataset):

    def __init__(
        self,
        first_num_slices_only: int = 12,
        zero_mean: bool = True,
        denoise: bool = False,
        split = [90,5,5],
        split_seed: float = 1234,
        finetune: bool = False,
        use_saved_data: bool = False,
        data_used: int = None,
    ):
        super().__init__()
        self.zero_mean = zero_mean
        self.transform = T.Resize((256,256))
        self.slices = None
        self.split = split
        self.split_seed = split_seed
        self.denoise = denoise
        
        file_dir = pathlib.Path(__file__).parent.resolve()

        if use_saved_data is False:
            # extract test data
            data_list = np.load(file_dir / 'fastmri_brain_info.npy', allow_pickle=True).item()['info']

            Random(self.split_seed).shuffle(data_list)
            num_train = round(len(data_list) * self.split[0] / np.sum(self.split))
            num_val = round(len(data_list) * self.split[1] // np.sum(self.split))  
            
            # test volumes
            test_volumes = data_list[(num_train+num_val):]
            
            self.slices = []
            for volume_info in test_volumes:
                volume_path, num_slices = volume_info
                self.slices += [
                    (volume_path, i) 
                    for i in range(8, min(num_slices, first_num_slices_only))
                ]

            # shuffle the slices
            Random(self.split_seed).shuffle(self.slices)
            if finetune:
                # use last ten slices for finetuning
                self.slices = self.slices[-10:]

            structured_arr = np.array(self.slices, dtype=object)
            np.save(file_dir / 'test_data.npy', structured_arr, allow_pickle=True)

            if data_used is not None:
                self.slices = self.slices[:data_used]
                self._check_files()
                print("FastMRIBrainData [dataset]: All files exist locally!")

        else:   
            print("Saved data is being used...")
            slices = np.load(file_dir / 'test_data.npy', allow_pickle=True)
            self.slices = list(map(tuple, slices))

    def _check_files(self):
        ''' This function checks if all the files in self.slices exist '''
        for slice_path, slice_idx in self.slices:
            slice_path = pathlib.Path(str(slice_path).replace('gautschi', 'gilbreth'))
            full_slice_path = slice_path / f'slice_{slice_idx}.npy'
            if not full_slice_path.exists():
                raise FileNotFoundError(f'Following file path does not exist: {str(full_slice_path)}!')

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, i: int):
        # load image
        volume_path, slice_idx = self.slices[i]
        volume_path = Path(str(volume_path).replace('gautschi', 'gilbreth'))
        image = np.load(volume_path / f'slice_{slice_idx}.npy', allow_pickle=True)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        # resize to [256,256]
        image = self.transform(image)
        # remove the background noise
        if self.denoise:
            imax = image.max()
            image[image<0.07*imax] = 0
        # normalize to [0,1]
        image /= image.max()
        # zero mean
        image = 2*image - 1 if self.zero_mean else image
        return image