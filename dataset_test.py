from torch.utils.data import Dataset, DataLoader, RandomSampler
from trajectory_dataset import TrajectoryDataset


class TestDataset:
    def __init__(self, blah):
        self.blah = blah

    def __len__(self):
        return len(self.blah)

    def __getitem__(self, idx):
        return self.blah[idx]

    def add(self, item):
        self.blah.append(item)

def main():
    data = TestDataset([1,2,3])
    dataloaded = DataLoader(data)
    dataloaded = RandomSampler(data)
    print(len(dataloaded))
    for blah in dataloaded:
        print(blah)

    data.add(4)
    print(len(dataloaded))

    for blah in dataloaded:
        print(blah)




if __name__ == "__main__":
    main()


