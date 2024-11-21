import os
from typing import Dict

from dataclasses import dataclass, field

import hydra
from hydra.core.config_store import ConfigStore
from hydra.conf import HydraConf, SweepDir
import nibabel as nib
from nilearn.datasets import fetch_atlas_difumo
from nilearn.maskers import NiftiMapsMasker
from nilearn.image import clean_img, high_variance_confounds, smooth_img

import numpy as np

from mdshvb.utils import get_hydra_output_dir


@dataclass
class HCPPreprocessingConfig:
    subject: int = 100_206
    session: str = (
        "REST1_LR"  # choices: ["REST1_LR", "REST2_LR", "REST1_RL", "REST2_RL"]
    )
    difumo_dimension: int = 64
    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(sweep=SweepDir(subdir=""))
    )
    remove_confounds: bool = False
    clean_img_kwargs: Dict = field(
        default_factory=lambda: dict(
            detrend=True,
            low_pass=0.1,
            high_pass=0.01,
            t_r=0.74,
        )
    )
    smooth_img_kwargs: Dict = field(default_factory=lambda: dict(fwhm=4))


cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=HCPPreprocessingConfig)


@hydra.main(version_base=None, config_path="./conf")
def my_app(preprocessing_config: HCPPreprocessingConfig) -> None:

    HCP_DATA_ROOT = "/data/parietal/store/data/HCP900"

    output_dir = get_hydra_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    cfg = preprocessing_config.preprocess

    subject_images = nib.load(
        f"{HCP_DATA_ROOT}/{cfg.subject}/MNINonLinear/"
        f"Results/rfMRI_{cfg.session}/"
        f"rfMRI_{cfg.session}_hp2000_clean.nii.gz"
    )

    if cfg.remove_confounds:
        confounds = high_variance_confounds(subject_images)
    else:
        confounds = None

    cleaned_images = clean_img(
        subject_images, confounds=confounds, **cfg.clean_img_kwargs
    )

    smoothed_images = smooth_img(cleaned_images, **cfg.smooth_img_kwargs)

    difumo_atlas = fetch_atlas_difumo(dimension=cfg.difumo_dimension)
    difumo_maps = nib.load(difumo_atlas["maps"])

    dictionnary_dtseries = NiftiMapsMasker(
        maps_img=difumo_maps, standardize=True
    ).fit_transform(smoothed_images)

    np.save(
        f"{output_dir}/Y.npy",
        dictionnary_dtseries.T,
    )
    np.save(
        f"{output_dir}/c.npy",
        np.zeros(dictionnary_dtseries.shape[0]),
    )


if __name__ == "__main__":
    my_app()
