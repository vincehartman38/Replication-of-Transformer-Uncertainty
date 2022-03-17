from torch.utils.data import Dataset


class XSumDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.dataset = list(self._align_data().values())
        self.data_by_id = {x["id"]: x for x in self.dataset}

    def query_by_bbc_id(self, bbc_id):
        if str(bbc_id) in self.data_by_id:
            return self.data_by_id[str(bbc_id)]
        raise ValueError(f"no article for bbc_id: {bbc_id}")

    def _align_data(self):
        """
        Aligns data in xsum:

        Returns:
            dataset:  Dict(id, {
                'id': id of bbc article,
                'document': original document,
                'summary': true summary,

        """

        dataset = {}

        for data in self.data:
            dataset[str(data["id"])] = {
                "id": data["id"],
                "document": data["document"],
                "summary": data["summary"],
            }
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class CNNDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.dataset = list(self._align_data().values())
        self.data_by_id = {x["id"]: x for x in self.dataset}

    def query_by_bbc_id(self, bbc_id):
        if str(bbc_id) in self.data_by_id:
            return self.data_by_id[str(bbc_id)]
        raise ValueError(f"no article for bbc_id: {bbc_id}")

    def _align_data(self):
        """
        Aligns data in xsum:

        Returns:
            dataset:  Dict(id, {
                'id': id of bbc article,
                'document': original article,
                'summary': highlights,

        """

        dataset = {}

        for data in self.data:
            dataset[str(data["id"])] = {
                "id": data["id"],
                "document": data["article"],
                "summary": data["highlights"],
            }
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
