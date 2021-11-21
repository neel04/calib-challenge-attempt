from calib_dataset import CalibrationImageDataset
from torch.utils.data import DataLoader

exec(open("/content/Comma_AI/hvec.py").read())

# Constructing the Image dataset from HEVC files
#for i in tqdm(range(0,5)):
#  hevc_to_frames(i, f'./data_{i}')

#PyTorch Dataset creation
test_ds = CalibrationImageDataset('./')
train_dataloader = DataLoader(test_ds)

a_1, a_2 = next(iter(train_dataloader))
print(np.float64(a_2[0]))