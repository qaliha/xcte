from os.path import join

from loader import DatasetFromFolder


def get_training_set(root_dir, scale_n_crop=True):
    train_dir = join(root_dir, "train")

    return DatasetFromFolder(train_dir, scale_n_crop)


def get_test_set(root_dir, scale_n_crop=True):
    test_dir = join(root_dir, "test")

    return DatasetFromFolder(test_dir, scale_n_crop)
