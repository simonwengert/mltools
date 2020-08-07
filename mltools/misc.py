import os
import numpy as np
import itertools as ito

import ase.io
# import mytools.checks


def get_info(p_xyz_file, keys):
    """
    Parse xyz-file for non-array-like data.

    Parameters
    ----------
    p_xyz_file : str
        Path the an (ase-readable) xyz-file.
    keys : list or str
        Desired keys of the ase-object.info dictionary.

    Returns
    -------
    info : dict
        Stores extracted information. Keys are `keys`,
        while corresponding values are lists with extracted
        information (ordering as in xyz-file).
    """
    if isinstance(keys, str):
        keys = list(keys)
    elif not isinstance(keys, list):
        raise TypeError("'keys' must be of type 'list' or 'str'")

    o_geos = ase.io.read(p_xyz_file+'@:')
    info = {key : [] for key in keys}
    for o_geo in o_geos:
        for key in keys:
            info[key].append(o_geo.info[key])
    return info


def get_arrays(p_xyz_file, keys, geo_resolved=False):
    """
    Parse xyz-file for array-like data.

    Parameters
    ----------
    p_xyz_file : str
        Path the an (ase-readable) xyz-file.
    keys : list or str
        Desired keys of the ase-object.arrays dictionary.
    geo_resolved : bool
        If set to `True`, the values of `arrays` will be lists
        with individual arrays per geometry. If set to `False`,
        the values will be concatenated arrays with data
        of all geometries.

    Returns
    -------
    arrays : dict
        Stores extracted information. Keys are `keys`,
        while corresponding values are arrays with extracted
        information (ordering as in xyz-file).
    """
    if isinstance(keys, str):
        keys = list(keys)
    elif not isinstance(keys, list):
        raise TypeError("'keys' must be of type 'list' or 'str'")

    o_geos = ase.io.read(p_xyz_file+'@:')

    arrays = {key: [] for key in keys}
    for o_geo in o_geos:
        for key in keys:
            arrays[key].append(o_geo.arrays[key])

    if not geo_resolved:
        for key in keys:
            arrays[key] = np.concatenate((arrays[key]))
    return arrays



# def plot_correlation(data, label_ref, xlabel='', ylabel='', buffr=0.1, p_save='', legend=True, latex=True):
#
#     helpers.set_defaults(rcParams)
#     # Set the font
#     # rcParams['text.latex.preamble'] = '\usepackage{libertine}, \
#     #                                    \usepackage[libertine]{newtxmath}, \
#     #                                    \usepackage{sfmath}, \
#     #                                    \usepackage[T1]{fontenc}, \
#     #                                    \usepackage{amsmath}'
#
#     if latex is True:
#         helpers.set_latex(rcParams, font = 'libertine')
#     else:
#         helpers.set_mathtext(rcParams, family = 'sans-serif')
#
#     width = helpers.width
#     height = width / (0.70*helpers.golden_ratio)
#
#     rcParams['figure.figsize'] = (width, height)
#     rcParams['figure.subplot.left'] = 0.19
#     rcParams['figure.subplot.right'] = 0.97
#     rcParams['figure.subplot.bottom'] = 0.17
#     rcParams['figure.subplot.top'] = 0.95
#     rcParams['figure.subplot.hspace'] = 0.4
#     rcParams['figure.subplot.wspace'] = 0.4
#     rcParams['font.size'] = 18
#
#     # # read data
#     # ref_model = 'aims_mbd_reference'
#     # p_ref_model = os.path.join(sc.P_DATA, ref_model)
#
#     # n_geos = os.listdir(p_ref_model)
#     # models = []
#     # if args.harris_mbd or args.empty:  # need data range also for empty plot
#     #     models.append('aims_mbd_harris')
#     # if args.dftb_mbd:
#     #     models.append('dftb_mbd')
#     # if args.dftb_ts:
#     #     models.append('dftb_ts')
#
#     # ref_n_geos = [value for n_geo in n_geos for value in mytools.data_handler.read_data(os.path.join(p_ref_model, n_geo, 'lattice_energy.txt'))[0]]
#     # ref_coh    = [value for n_geo in n_geos for value in mytools.data_handler.read_data(os.path.join(p_ref_model, n_geo, 'lattice_energy.txt'))[1]]
#
#     # # collect data
#     # coh_all = []
#     # n_geos_all = []
#
#     # for model in models:
#     #     p_model = os.path.join(sc.P_DATA, model)
#     #     tmp_n_geos = [value for n_geo in n_geos for value in mytools.data_handler.read_data(os.path.join(p_model, n_geo, 'lattice_energy.txt'))[0]]
#     #     tmp_coh    = [value for n_geo in n_geos for value in mytools.data_handler.read_data(os.path.join(p_model, n_geo, 'lattice_energy.txt'))[1]]
#
#     #     # checks
#     #     mytools.checks.list_ordering(ref_n_geos, tmp_n_geos)
#
#     #     n_geos_all.append(tmp_n_geos)
#     #     coh_all.append(tmp_coh)
#
#
#     # prep. plot
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#
#     # add line of perfect correlation
#     val_all = np.array([val for value in data.values() for val in value])
#     val_range = np.abs(np.max(val_all) - np.min(val_all))
#     val_min = np.min(val_all) - buffr*val_range
#     val_max = np.max(val_all) + buffr*val_range
#
#     ax.plot([val_min, val_max], [val_min, val_max], c='k')
#
#     ax.set_xlim(val_min, val_max)
#     ax.set_ylim(val_min, val_max)
#
#     for label, values in data.items():
#         if label != label_ref:
#             ax.scatter(data[label_ref], values, label=label, alpha=0.2)
#
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#
#     if legend:
#         plt.legend()
#
#     # if args.zoom:
#     #     ax.axis([val_min, 0, val_min, 0])
#     # else:
#     #     ax.axis([val_min, val_max, val_min, val_max])
#
#     # saving
#     if p_save:
#         if os.path.exists(p_save):
#
#             is_asking = True
#             while is_asking:
#                 overwrite = input('Overwrite {}?(y/n)'.format(p_save))
#                 if overwrite in ['y', 'yes']:
#                     helpers.write(p_save)
#                     is_asking = False
#                 elif overwrite in ['n', 'no']:
#                     is_asking = False
#                 else:
#                     print('Options are y/n. Choose one of them!')
#         else:
#             helpers.write(p_save)
#
#     return plt

