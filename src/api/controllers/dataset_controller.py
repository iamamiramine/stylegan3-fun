from fastapi import APIRouter

from src.domain.models.dataset.dataset_models import DatasetToolParameters

from src.application.dataset.services import dataset_service

router = APIRouter()


@router.post("/convert_dataset")
def convert_dataset(parameters: DatasetToolParameters) -> dict:
    """
    Description:
    ------------
        Convert an image dataset into a dataset archive usable with StyleGAN2 ADA PyTorch.

        The output dataset format can be either an image folder or an uncompressed zip archive.
        Zip archives makes it easier to move datasets around file servers and clusters, and may
        offer better training performance on network file systems.

        Images within the dataset archive will be stored as uncompressed PNG.
        Uncompresed PNGs can be efficiently decoded in the training loop.

        Class labels are stored in a file called 'dataset.json' that is stored at the
        dataset root folder.  This file has the following structure:

        \b
        {
            "labels": [
                ["00000/img00000000.png",6],
                ["00000/img00000001.png",9],
                ... repeated for every image in the dataset
                ["00049/img00049999.png",1]
            ]
        }

        If the 'dataset.json' file cannot be found, the dataset is interpreted as
        not containing class labels.

        Image scale/crop and resolution requirements:

        Output images must be square-shaped and they must all have the same power-of-two
        dimensions.

        To scale arbitrary input image size to a specific width and height, use the
        --resolution option.  Output resolution will be either the original
        input resolution (if resolution was not specified) or the one specified with
        --resolution option.

        Use the --transform=center-crop or --transform=center-crop-wide options to apply a
        center crop transform on the input image.  These options should be used with the
        --resolution option.  For example:

        \b
        python dataset_service.py --source LSUN/raw/cat_lmdb --dest /tmp/lsun_cat \\
            --transform=center-crop-wide --resolution=512x384

    Parameters:
    -----------
        parameters: DatasetToolParameters
            \b
            --source *_lmdb/                    Load LSUN dataset
            --source cifar-10-python.tar.gz     Load CIFAR-10 dataset
            --source train-images-idx3-ubyte.gz Load MNIST dataset
            --source path/                      Recursively load all images from path/
            --source dataset.zip                Recursively load all images from dataset.zip

            Specifying the output format and path:

            \b
            --dest /path/to/dir                 Save output files under /path/to/dir
            --dest /path/to/dataset.zip         Save output files into /path/to/dataset.zip

    Returns:
    --------
    dict
        A dictionary with the names and modalities of the datasets

    """
    return dataset_service.convert_dataset(parameters)
