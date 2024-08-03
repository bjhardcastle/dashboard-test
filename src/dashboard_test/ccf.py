import functools
import logging
import tempfile
from typing import Iterable, Literal
import nrrd 
import numpy as np
import numpy.typing as npt
import upath 
import polars as pl
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

@functools.cache
def get_ccf_volume() -> npt.NDArray:
    """
    From ibl Atlas
    >>> volume = get_ccf_volume()
    
    """

    isilon_path = upath.UPath("//allen/programs/mindscope/workgroups/np-behavior/annotation_25.nrrd")
    cloud_path = upath.UPath("https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_25.nrrd")
    
    path = isilon_path if isilon_path.exists() else cloud_path
    if path.suffix not in (supported := ('.nrrd', '.npz')):
        raise ValueError(
            f'{path.suffix} files not supported, must be one of {supported}')
    if path.protocol: # cloud path - download it
        tempdir = tempfile.mkdtemp()
        temp_path = upath.UPath(tempdir) / path.name
        logger.info(f"Downloading CCF volume to temporary file {temp_path.as_posix()}")
        temp_path.write_bytes(path.read_bytes())
        path = temp_path
    logger.info(f"Using CCF volume from {path.as_posix()}")
    
    logger.info(f'Loading CCF volume from {path.as_posix()}')
    if path.suffix == '.nrrd':
        volume, _ = nrrd.read(path, index_order='C')  # ml, dv, ap
        # we want the coronal slice to be the most contiguous
        volume = np.transpose(volume, (2, 0, 1))  # image[iap, iml, idv]
    elif path.suffix == '.npz':
        volume = np.load(path)['arr_0']
    return volume

@functools.cache
def get_ccf_structure_tree_df() -> pl.DataFrame:
    isilon_path = upath.UPath('//allen/programs/mindscope/workgroups/np-behavior/ccf_structure_tree_2017.csv')
    github_path = upath.UPath('https://raw.githubusercontent.com/cortex-lab/allenCCF/master/structure_tree_safe_2017.csv')
    path = isilon_path if isilon_path.exists() else github_path
    return (
        pl.scan_csv(path.as_posix())
        .with_columns(
            color_hex_int=pl.col('color_hex_triplet').str.to_integer(base=16),
            color_hex_str=pl.lit('0x') + pl.col('color_hex_triplet'),
        )
        .with_columns(
            r=pl.col('color_hex_triplet').str.slice(0, 2).str.to_integer(base=16).mul(1/255),
            g=pl.col('color_hex_triplet').str.slice(2, 2).str.to_integer(base=16).mul(1/255),
            b=pl.col('color_hex_triplet').str.slice(4, 2).str.to_integer(base=16).mul(1/255),
        )
        .with_columns(
            color_rgb=pl.concat_list('r', 'g', 'b'),
        )
        .drop('r', 'g', 'b')
    ).collect()
    
def get_ccf_structure_info(acronym: str) -> dict:
    """
    >>> get_ccf_structure_info('MOs')
    {'id': 993, 'atlas_id': 831, 'name': 'Secondary motor area', 'acronym': 'MOs', 'st_level': None, 'ontology_id': 1, 'hemisphere_id': 3, 'weight': 8690, 'parent_structure_id': 500, 'depth': 7, 'graph_id': 1, 'graph_order': 24, 'structure_id_path': '/997/8/567/688/695/315/500/993/', 'color_hex_triplet': '1F9D5A', 'neuro_name_structure_id': None, 'neuro_name_structure_id_path': None, 'failed': 'f', 'sphinx_id': 25, 'structure_name_facet': 1043755260, 'failed_facet': 734881840, 'safe_name': 'Secondary motor area', 'color_hex_int': 2071898, 'color_hex_str': '0x1F9D5A', 'color_rgb': [0.12156862745098039, 0.615686274509804, 0.3529411764705882]}
    """
    results = get_ccf_structure_tree_df().filter(pl.col('acronym') == acronym)
    if len(results) == 0:
        raise ValueError(f'No area found with acronym {acronym}')
    if len(results) > 1:
        logger.warning(f"Multiple areas found: {results['acronym'].to_list()}. Using the first one")
    return results[0].limit(1).to_dicts()[0]

def get_children_in_volume(acronym: str, known_children: tuple[str, ...] | None = None) -> list[str]:
    """
    >>> get_children_in_volume('MOs')
    
    """
    children = list(known_children) or []
    children.append(
        get_ccf_structure_tree_df()
        .filter(pl.col('parent_structure_id') == get_ccf_structure_info(acronym)['id'])
        .get_column('acronym')
        .to_list()
    )
    if all(child in get_areas_in_volume() for child in children):
        return children
    for child in children:
        if child not in get_areas_in_volume():
            children.append(get_children_in_volume(child, children))
    
    
def get_ccf_volume_binary_mask(ccf_acronym: str | None = None) -> npt.NDArray:
    """
    # >>> volume = get_ccf_volume_binary_mask('MOs')
    # >>> assert volume.any()
    # >>> volume = get_ccf_volume_binary_mask()
    # >>> assert volume.any()
    """
    if not ccf_acronym:
        logger.warning('No acronym provided, returning mask for the whole volume')
        return get_ccf_volume() > 0
    ccf_id: int = get_ccf_structure_info(ccf_acronym)['id']
    if ccf_acronym in get_areas_in_volume():
        return get_ccf_volume() == ccf_id
    # call recursively on children and sum them up
    return np.sum([get_ccf_volume_binary_mask(child) for child in get_children_in_volume(ccf_acronym)], axis=0)



@functools.cache
def get_areas_in_volume() -> list[str]:
    """Reverse lookup on integers in ccf volume to get their corresponding acronyms
    
    >>> areas = get_areas_in_volume()
    >>> areas[0]
    'root'
    """
    return get_ccf_structure_tree_df().filter(pl.col('id').is_in(np.unique(get_ccf_volume())))['acronym'].to_list()


def get_ccf_projection(
    acronym: str | None = None,
    volume: npt.NDArray | None = None, 
    axis: Literal['sagittal', 'coronal', 'horizontal'] = 'horizontal', 
    slice_center_index: int | None = None, 
    thickness_indices: int = 500,
    apply_color: bool = True,
    alpha: float = 1,
) -> npt.NDArray:
    """
    >>> slice = get_ccf_projection(axis='coronal') # no acronym returns all non-zero areas
    >>> assert slice.any()
    """
    if volume is None:
        volume = get_ccf_volume_binary_mask(acronym)

    axis_dim = {'sagittal': 0, 'coronal': 1, 'horizontal': 2}[axis]
    if slice_center_index is None:
        slice_center_index = volume.shape[axis_dim] // 2
    
    s = slice(
        max(round(slice_center_index - 0.5 * thickness_indices), 0),
        min(round(slice_center_index + 0.5 * thickness_indices) + 1, volume.shape[axis_dim]),
        1,
    )
    if axis == 'coronal':
        slice_image = volume[s, :, :].any(axis=0)
    elif axis == 'sagittal':
        slice_image = volume[:, s, :].any(axis=1)
    elif axis == 'horizontal':
        slice_image = volume[:, :, s].any(axis=2)
    if not apply_color:
        return slice_image
    if acronym:
        rgb = get_ccf_structure_info(acronym)['color_rgb']
    else:
        rgb = [.5] * 3
    rgba: np.ndarray[functools.Any, np.dtype[functools.Any]] = np.array(rgb + [1])
    return np.stack([slice_image] * 4, axis=-1) * rgba


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # ax = plt.imshow(
    #     get_ccf_projection('CTX', axis='horizontal')
    # )
    # plt.show()
    import doctest
    doctest.testmod()