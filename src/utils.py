# internal python imports
import os
import csv
import functools
import itertools

# third party imports
import numpy as np
import scipy
import torch
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

######################################################################
# Auxiliary functions (from https://github.com/voxelmorph/voxelmorph)
######################################################################

def default_encoder_features():
    nb_features = [
        [16, 32, 32, 16],             # encoder
    ]
    return nb_features

def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features

def read_file_list(filename, prefix=None, suffix=None):
    '''
    Reads a list of files from a line-seperated text file.

    Parameters:
        filename: Filename to load.
        prefix: File prefix. Default is None.
        suffix: File suffix. Default is None.
    '''
    with open(filename, 'r') as file:
        content = file.readlines()
    filelist = [x.strip() for x in content if x.strip()]
    if prefix is not None:
        filelist = [prefix + f for f in filelist]
    if suffix is not None:
        filelist = [f + suffix for f in filelist]
    return filelist


def read_pair_list(filename, delim=None, prefix=None, suffix=None):
    '''
    Reads a list of registration file pairs from a line-seperated text file.

    Parameters:
        filename: Filename to load.
        delim: File pair delimiter. Default is a whitespace seperator (None).
        prefix: File prefix. Default is None.
        suffix: File suffix. Default is None.
    '''
    pairlist = [f.split(delim) for f in read_file_list(filename)]
    if prefix is not None:
        pairlist = [[prefix + f for f in pair] for pair in pairlist]
    if suffix is not None:
        pairlist = [[f + suffix for f in pair] for pair in pairlist]
    return pairlist


def load_volfile(
    filename,
    np_var='vol',
    add_batch_axis=False,
    add_feat_axis=False,
    pad_shape=None,
    resize_factor=1,
    ret_affine=False
):
    """
    Loads a file in nii, nii.gz, mgz, npz, or npy format. If input file is not a string,
    returns it directly (allows files preloaded in memory to be passed to a generator)

    Parameters:
        filename: Filename to load, or preloaded volume to be returned.
        np_var: If the file is a npz (compressed numpy) with multiple variables,
            the desired variable can be specified with np_var. Default is 'vol'.
        add_batch_axis: Adds an axis to the beginning of the array. Default is False.
        add_feat_axis: Adds an axis to the end of the array. Default is False.
        pad_shape: Zero-pad the array to a target shape. Default is None.
        resize: Volume resize factor. Default is 1
        ret_affine: Additionally returns the affine transform (or None if it doesn't exist).
    """
    if isinstance(filename, str) and not os.path.isfile(filename):
        raise ValueError("'%s' is not a file." % filename)

    if not os.path.isfile(filename):
        if ret_affine:
            (vol, affine) = filename
        else:
            vol = filename
    elif filename.endswith(('.nii', '.nii.gz', '.mgz')):
        import nibabel as nib
        img = nib.load(filename)
        vol = img.get_data().squeeze()
        affine = img.affine
    elif filename.endswith('.npy'):
        vol = np.load(filename)
        affine = None
    elif filename.endswith('.npz'):
        npz = np.load(filename)
        vol = next(iter(npz.values())) if len(npz.keys()) == 1 else npz[np_var]
        affine = None
    else:
        raise ValueError('unknown filetype for %s' % filename)

    if pad_shape:
        vol, _ = pad(vol, pad_shape)

    if add_feat_axis:
        vol = vol[..., np.newaxis]

    if resize_factor != 1:
        vol = resize(vol, resize_factor)

    if add_batch_axis:
        vol = vol[np.newaxis, ...]

    return (vol, affine) if ret_affine else vol


def save_volfile(array, filename, affine=None):
    """
    Saves an array to nii, nii.gz, or npz format.

    Parameters:
        array: The array to save.
        filename: Filename to save to.
        affine: Affine vox-to-ras matrix. Saves LIA matrix if None (default).
    """
    if filename.endswith(('.nii', '.nii.gz')):
        import nibabel as nib
        if affine is None and array.ndim >= 3:
            # use LIA transform as default affine
            affine = np.array([[-1, 0, 0, 0],  # nopep8
                               [0, 0, 1, 0],  # nopep8
                               [0, -1, 0, 0],  # nopep8
                               [0, 0, 0, 1]], dtype=float)  # nopep8
            pcrs = np.append(np.array(array.shape[:3]) / 2, 1)
            affine[:3, 3] = -np.matmul(affine, pcrs)[:3]
        nib.save(nib.Nifti1Image(array, affine), filename)
    elif filename.endswith('.npz'):
        np.savez_compressed(filename, vol=array)
    else:
        raise ValueError('unknown filetype for %s' % filename)


def load_labels(arg):
    """
    Load label maps and return a list of unique labels as well as all maps.

    Parameters:
        arg: Path to folder containing label maps, string for globbing, or a list of these.

    Returns:
        np.array: List of unique labels.
        list: List of label maps, each as a np.array.
    """
    if not isinstance(arg, (tuple, list)):
        arg = [arg]

    # List files.
    import glob
    ext = ('.nii.gz', '.nii', '.mgz', '.npy', '.npz')
    files = [os.path.join(f, '*') if os.path.isdir(f) else f for f in arg]
    files = sum((glob.glob(f) for f in files), [])
    files = [f for f in files if f.endswith(ext)]

    # Load labels.
    if len(files) == 0:
        raise ValueError(f'no labels found for argument "{files}"')
    label_maps = []
    shape = None
    for f in files:
        x = np.squeeze(load_volfile(f))
        if shape is None:
            shape = np.shape(x)
        if not np.issubdtype(x.dtype, np.integer):
            raise ValueError(f'file "{f}" has non-integral data type')
        if not np.all(x.shape == shape):
            raise ValueError(f'shape {x.shape} of file "{f}" is not {shape}')
        label_maps.append(x)

    return np.unique(label_maps), label_maps


def load_pheno_csv(filename, training_files=None):
    """
    Loads an attribute csv file into a dictionary. Each line in the csv should represent
    attributes for a single training file and should be formatted as:

    filename,attr1,attr2,attr2...

    Where filename is the file basename and each attr is a floating point number. If
    a list of training_files is specified, the dictionary file keys will be updated
    to match the paths specified in the list. Any training files not found in the
    loaded dictionary are pruned.
    """

    # load csv into dictionary
    pheno = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        header = next(csv_reader)
        for row in csv_reader:
            pheno[row[0]] = np.array([float(f) for f in row[1:]])

    # make list of valid training files
    if training_files is None:
        training_files = list(training_files.keys())
    else:
        training_files = [f for f in training_files if os.path.basename(f) in pheno.keys()]
        # make sure pheno dictionary includes the correct path to training data
        for f in training_files:
            pheno[f] = pheno[os.path.basename(f)]

    return pheno, training_files


def pad(array, shape):
    """
    Zero-pads an array to a given shape. Returns the padded array and crop slices.
    """
    if array.shape == tuple(shape):
        return array, ...

    padded = np.zeros(shape, dtype=array.dtype)
    offsets = [int((p - v) / 2) for p, v in zip(shape, array.shape)]
    slices = tuple([slice(offset, l + offset) for offset, l in zip(offsets, array.shape)])
    padded[slices] = array

    return padded, slices


def resize(array, factor, batch_axis=False):
    """
    Resizes an array by a given factor. This expects the input array to include a feature dimension.
    Use batch_axis=True to avoid resizing the first (batch) dimension.
    """
    if factor == 1:
        return array
    else:
        if not batch_axis:
            dim_factors = [factor for _ in array.shape[:-1]] + [1]
        else:
            dim_factors = [1] + [factor for _ in array.shape[1:-1]] + [1]
        return scipy.ndimage.interpolation.zoom(array, dim_factors, order=0)


def dice(array1, array2, labels=None, include_zero=False):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.

    Parameters:
        array1: Input array 1.
        array2: Input array 2.
        labels: List of labels to compute dice on. If None, all labels will be used.
        include_zero: Include label 0 in label list. Default is False.
    """
    if labels is None:
        labels = np.concatenate([np.unique(a) for a in [array1, array2]])
        labels = np.sort(np.unique(labels))
    if not include_zero:
        labels = np.delete(labels, np.argwhere(labels == 0)) 

    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem


def affine_shift_to_matrix(trf, resize=None, unshift_shape=None):
    """
    Converts an affine shift to a matrix (over the identity).
    To convert back from center-shifted transform, provide image shape
    to unshift_shape.

    TODO: make ND compatible - currently just 3D
    """
    matrix = np.concatenate([trf.reshape((3, 4)), np.zeros((1, 4))], 0) + np.eye(4)
    if resize is not None:
        matrix[:3, -1] *= resize
    if unshift_shape is not None:
        T = np.zeros((4, 4))
        T[:3, 3] = (np.array(unshift_shape) - 1) / 2
        matrix = (np.eye(4) + T) @ matrix @ (np.eye(4) - T)
    return matrix


def extract_largest_vol(bw, connectivity=1):
    """
    Extracts the binary (boolean) image with just the largest component.
    TODO: This might be less than efficiently implemented.
    """
    lab = measure.label(bw.astype('int'), connectivity=connectivity)
    regions = measure.regionprops(lab, cache=False)
    areas = [f.area for f in regions]
    ai = np.argsort(areas)[::-1]
    bw = lab == ai[0] + 1
    return bw


def clean_seg(x, std=1):
    """
    Cleans a segmentation image.
    """

    # take out islands, fill in holes, and gaussian blur
    bw = extract_largest_vol(x)
    bw = 1 - extract_largest_vol(1 - bw)
    gadt = scipy.ndimage.gaussian_filter(bw.astype('float'), std)

    # figure out the proper threshold to maintain the total volume
    sgadt = np.sort(gadt.flatten())[::-1]
    thr = sgadt[np.ceil(bw.sum()).astype(int)]
    clean_bw = gadt > thr

    assert np.isclose(bw.sum(), clean_bw.sum(), atol=5), 'cleaning segmentation failed'
    return clean_bw.astype(float)


def clean_seg_batch(X_label, std=1):
    """
    Cleans batches of segmentation images.
    """
    if not X_label.dtype == 'float':
        X_label = X_label.astype('float')

    data = np.zeros(X_label.shape)
    for xi, x in enumerate(X_label):
        data[xi, ..., 0] = clean_seg(x[..., 0], std)

    return data


def filter_labels(atlas_vol, labels):
    """
    Filters given volumes to only include given labels, all other voxels are set to 0.
    """
    mask = np.zeros(atlas_vol.shape, 'bool')
    for label in labels:
        mask = np.logical_or(mask, atlas_vol == label)
    return atlas_vol * mask


def dist_trf(bwvol):
    """
    Computes positive distance transform from positive entries in a logical image.
    """
    revbwvol = np.logical_not(bwvol)
    return scipy.ndimage.morphology.distance_transform_edt(revbwvol)


def signed_dist_trf(bwvol):
    """
    Computes the signed distance transform from the surface between the binary
    elements of an image
    NOTE: The distance transform on either side of the surface will be +/- 1,
    so there are no voxels for which the distance should be 0.
    NOTE: Currently the function uses bwdist twice. If there is a quick way to
    compute the surface, bwdist could be used only once.
    """

    # get the positive transform (outside the positive island)
    posdst = dist_trf(bwvol)

    # get the negative transform (distance inside the island)
    notbwvol = np.logical_not(bwvol)
    negdst = dist_trf(notbwvol)

    # combine the positive and negative map
    return posdst * notbwvol - negdst * bwvol


def vol_to_sdt(X_label, sdt=True, sdt_vol_resize=1):
    """
    Computes the signed distance transform from a volume.
    """

    X_dt = signed_dist_trf(X_label)

    if not (sdt_vol_resize == 1):
        if not isinstance(sdt_vol_resize, (list, tuple)):
            sdt_vol_resize = [sdt_vol_resize] * X_dt.ndim
        if any([f != 1 for f in sdt_vol_resize]):
            X_dt = scipy.ndimage.interpolation.zoom(X_dt, sdt_vol_resize, order=1, mode='reflect')

    if not sdt:
        X_dt = np.abs(X_dt)

    return X_dt


def vol_to_sdt_batch(X_label, sdt=True, sdt_vol_resize=1):
    """
    Computes the signed distance transforms from volume batches.
    """

    # assume X_label is [batch_size, *vol_shape, 1]
    assert X_label.shape[-1] == 1, 'implemented assuming size is [batch_size, *vol_shape, 1]'
    X_lst = [f[..., 0] for f in X_label]  # get rows
    X_dt_lst = [vol_to_sdt(f, sdt=sdt, sdt_vol_resize=sdt_vol_resize)
                for f in X_lst]  # distance transform
    X_dt = np.stack(X_dt_lst, 0)[..., np.newaxis]
    return X_dt


def get_surface_pts_per_label(total_nb_surface_pts, layer_edge_ratios):
    """
    Gets the number of surface points per label, given the total number of surface points.
    """
    nb_surface_pts_sel = np.round(np.array(layer_edge_ratios) * total_nb_surface_pts).astype('int')
    nb_surface_pts_sel[-1] = total_nb_surface_pts - int(np.sum(nb_surface_pts_sel[:-1]))
    return nb_surface_pts_sel


def edge_to_surface_pts(X_edges, nb_surface_pts=None):
    """
    Converts edges to surface points.
    """

    # assumes X_edges is NOT in keras form
    surface_pts = np.stack(np.where(X_edges), 0).transpose()

    # random with replacements
    if nb_surface_pts is not None:
        chi = np.random.choice(range(surface_pts.shape[0]), size=nb_surface_pts)
        surface_pts = surface_pts[chi, :]

    return surface_pts


def sdt_to_surface_pts(X_sdt, nb_surface_pts,
                       surface_pts_upsample_factor=2, thr=0.50001, resize_fn=None):
    """
    Converts a signed distance transform to surface points.
    """
    us = [surface_pts_upsample_factor] * X_sdt.ndim

    if resize_fn is None:
        resized_vol = scipy.ndimage.interpolation.zoom(X_sdt, us, order=1, mode='reflect')
    else:
        resized_vol = resize_fn(X_sdt)
        pred_shape = np.array(X_sdt.shape) * surface_pts_upsample_factor
        assert np.array_equal(pred_shape, resized_vol.shape), 'resizing failed'

    X_edges = np.abs(resized_vol) < thr
    sf_pts = edge_to_surface_pts(X_edges, nb_surface_pts=nb_surface_pts)

    # can't just correct by surface_pts_upsample_factor because of how interpolation works...
    pt = [sf_pts[..., f] * (X_sdt.shape[f] - 1) / (X_edges.shape[f] - 1) for f in range(X_sdt.ndim)]
    return np.stack(pt, -1)


def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.

    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims], 
              where vol_shape is of len nb_dims

    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

def ndgrid(*args, **kwargs):
    """
    Disclaimer: This code is taken directly from the scitools package [1]
    Since at the time of writing scitools predominantly requires python 2.7 while we work with 3.5+
    To avoid issues, we copy the quick code here.
    Same as calling ``meshgrid`` with *indexing* = ``'ij'`` (see
    ``meshgrid`` for documentation).
    """
    kwargs['indexing'] = 'ij'
    return np.meshgrid(*args, **kwargs)


def volsize2ndgrid(volsize):
    """
    return the dense nd-grid for the volume with size volsize
    essentially return the ndgrid fpr
    """
    ranges = [np.arange(e) for e in volsize]
    return ndgrid(*ranges)

def sub2ind2d(siz, subs, **kwargs):
    """
    assumes column-order major
    """
    # subs is a list
    assert len(siz) == len(subs), \
        'found inconsistent siz and subs: %d %d' % (len(siz), len(subs))

    k = np.cumprod(siz[::-1])

    ndx = subs[-1]
    for i, v in enumerate(subs[:-1][::-1]):
        ndx = ndx + v * k[i]

    return ndx

def prod_n(lst):
    """
    Alternative to tf.stacking and prod, since tf.stacking can be slow
    """
    prod = lst[0]
    for p in lst[1:]:
        prod *= p
    return prod

def point_spatial_transformer(x, single=False, sdt_vol_resize=1):
    """
    Transforms surface points with a given deformation.
    Note that the displacement field that moves image A to image B will be "in the space of B".
    That is, `trf(p)` tells you "how to move data from A to get to location `p` in B". 
    Therefore, that same displacement field will warp *landmarks* in B to A easily 
    (that is, for any landmark `L(p)`, it can easily find the appropriate `trf(L(p))`
    via interpolation.
    TODO: needs documentation
    """

    # surface_points is a N x D or a N x (D+1) Tensor
    # trf is a *volshape x D Tensor
    surface_points, trf = x
    trf = trf * sdt_vol_resize
    surface_pts_D = surface_points.get_shape().as_list()[-1]
    trf_D = trf.get_shape().as_list()[-1]
    assert surface_pts_D in [trf_D, trf_D + 1]

    if surface_pts_D == trf_D + 1:
        li_surface_pts = K.expand_dims(surface_points[..., -1], -1)
        surface_points = surface_points[..., :-1]

    # just need to interpolate.
    # at each location determined by surface point, figure out the trf...
    # note: if surface_points are on the grid, gather_nd should work as well
    fn = lambda x: interpn(x[0], x[1])
    diff = tf.map_fn(fn, [trf, surface_points], fn_output_signature=tf.float32)
    ret = surface_points + diff

    if surface_pts_D == trf_D + 1:
        ret = tf.concat((ret, li_surface_pts), -1)

    return ret

######################################################################
# Neurite plotting functions (from https://github.com/adalca/neurite)
######################################################################

def slices(slices_in,           # the 2D slices
           titles=None,         # list of titles
           cmaps=None,          # list of colormaps
           norms=None,          # list of normalizations
           do_colorbars=False,  # option to show colorbars on each slice
           grid=False,          # option to plot the images in a grid or a single row
           width=15,            # width in in
           show=True,           # option to actually show the plot (plt.show())
           axes_off=True,
           plot_block=True,     # option to plt.show()
           facecolor=None,
           imshow_args=None):
    '''
    plot a grid of slices (2d images)
    '''

    # input processing
    if type(slices_in) == np.ndarray:
        slices_in = [slices_in]
    nb_plots = len(slices_in)
    slices_in = list(map(np.squeeze, slices_in))
    for si, slice_in in enumerate(slices_in):
        if len(slice_in.shape) != 2:
            assert len(slice_in.shape) == 3 and slice_in.shape[-1] == 3, \
                'each slice has to be 2d or RGB (3 channels)'

    def input_check(inputs, nb_plots, name, default=None):
        ''' change input from None/single-link '''
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [default]
        if len(inputs) == 1:
            inputs = [inputs[0] for i in range(nb_plots)]
        return inputs

    titles = input_check(titles, nb_plots, 'titles')
    cmaps = input_check(cmaps, nb_plots, 'cmaps', default='gray')
    norms = input_check(norms, nb_plots, 'norms')
    imshow_args = input_check(imshow_args, nb_plots, 'imshow_args')
    for idx, ia in enumerate(imshow_args):
        imshow_args[idx] = {} if ia is None else ia

    # figure out the number of rows and columns
    if grid:
        if isinstance(grid, bool):
            rows = np.floor(np.sqrt(nb_plots)).astype(int)
            cols = np.ceil(nb_plots / rows).astype(int)
        else:
            assert isinstance(grid, (list, tuple)), \
                "grid should either be bool or [rows,cols]"
            rows, cols = grid
    else:
        rows = 1
        cols = nb_plots

    # prepare the subplot
    fig, axs = plt.subplots(rows, cols)
    if rows == 1 and cols == 1:
        axs = [axs]

    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        # turn off axis
        ax.axis('off')

        # add titles
        if titles is not None and titles[i] is not None:
            ax.title.set_text(titles[i])

        # show figure
        im_ax = ax.imshow(slices_in[i], cmap=cmaps[i],
                          interpolation="nearest", norm=norms[i], **imshow_args[i])

        # colorbars
        # http://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        if do_colorbars:  # and cmaps[i] is not None
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im_ax, cax=cax)

    # clear axes that are unnecessary
    for i in range(nb_plots, col * row):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        if axes_off:
            ax.axis('off')

    # show the plots
    fig.set_size_inches(width, rows / cols * width)

    if facecolor is not None:
        fig.set_facecolor(facecolor)

    if show:
        plt.tight_layout()
        plt.show(block=plot_block)

    return (fig, axs)


def volume3D(vols, slice_nos=None, data_squeeze=True, **kwargs):
    """
    plot slices of a 3D volume by taking a middle slice of each axis

    Parameters:
        vols: a 3d volume or list of 3d volumes
        slice_nos (optional): a list of 3 elements of the slice numbers for each axis, 
            or list of lists of 3 elements. if None, the middle slices will be used.
        data_squeeze: remove singleton dimensions before plotting
    """
    if not isinstance(vols, (tuple, list)):
        vols = [vols]
    nb_vols = len(vols)
    vols = list(map(np.squeeze if data_squeeze else np.asarray, vols))
    assert all(v.ndim == 3 for v in vols), 'only 3d volumes allowed in volume3D'

    slics = []
    for vi, vol in enumerate(vols):

        these_slice_nos = slice_nos
        if slice_nos is None:
            these_slice_nos = [f // 2 for f in vol.shape]
        elif isinstance(slice_nos[0], (list, tuple)):
            these_slice_nos = slice_nos[vi]
        else:
            these_slice_nos = slice_nos

        slics = slics + [np.take(vol, these_slice_nos[d], d) for d in range(3)]

    if 'titles' not in kwargs.keys():
        kwargs['titles'] = ['axis %d' % d for d in range(3)] * nb_vols

    if 'grid' not in kwargs.keys():
        kwargs['grid'] = [nb_vols, 3]

    slices(slics, **kwargs)


def flow_legend(plot_block=True):
    """
    show quiver plot to indicate how arrows are colored in the flow() method.
    https://stackoverflow.com/questions/40026718/different-colours-for-arrows-in-quiver-plot
    """
    ph = np.linspace(0, 2 * np.pi, 13)
    x = np.cos(ph)
    y = np.sin(ph)
    u = np.cos(ph)
    v = np.sin(ph)
    colors = np.arctan2(u, v)

    norm = Normalize()
    norm.autoscale(colors)
    # we need to normalize our colors array to match it colormap domain
    # which is [0, 1]

    colormap = cm.winter

    plt.figure(figsize=(6, 6))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.quiver(x, y, u, v, color=colormap(norm(colors)), angles='xy', scale_units='xy', scale=1)
    plt.show(block=plot_block)


def flow(slices_in,           # the 2D slices
         titles=None,         # list of titles
         cmaps=None,          # list of colormaps
         width=15,            # width in in
         indexing='ij',       # plot vecs w/ matrix indexing 'ij' or cartesian indexing 'xy'
         img_indexing=True,   # whether to match the image view, i.e. flip y axis
         grid=False,          # option to plot the images in a grid or a single row
         show=True,           # option to actually show the plot (plt.show())
         quiver_width=None,
         plot_block=True,  # option to plt.show()
         scale=1):            # note quiver essentially draws quiver length = 1/scale
    '''
    plot a grid of flows (2d+2 images)
    '''

    # input processing
    nb_plots = len(slices_in)
    for slice_in in slices_in:
        assert len(slice_in.shape) == 3, 'each slice has to be 3d: 2d+2 channels'
        assert slice_in.shape[-1] == 2, 'each slice has to be 3d: 2d+2 channels'

    def input_check(inputs, nb_plots, name):
        ''' change input from None/single-link '''
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [None]
        if len(inputs) == 1:
            inputs = [inputs[0] for i in range(nb_plots)]
        return inputs

    assert indexing in ['ij', 'xy']
    slices_in = np.copy(slices_in)  # Since img_indexing, indexing may modify slices_in in memory

    if indexing == 'ij':
        for si, slc in enumerate(slices_in):
            # Make y values negative so y-axis will point down in plot
            slices_in[si][:, :, 1] = -slices_in[si][:, :, 1]

    if img_indexing:
        for si, slc in enumerate(slices_in):
            slices_in[si] = np.flipud(slc)  # Flip vertical order of y values

    titles = input_check(titles, nb_plots, 'titles')
    cmaps = input_check(cmaps, nb_plots, 'cmaps')
    scale = input_check(scale, nb_plots, 'scale')

    # figure out the number of rows and columns
    if grid:
        if isinstance(grid, bool):
            rows = np.floor(np.sqrt(nb_plots)).astype(int)
            cols = np.ceil(nb_plots / rows).astype(int)
        else:
            assert isinstance(grid, (list, tuple)), \
                "grid should either be bool or [rows,cols]"
            rows, cols = grid
    else:
        rows = 1
        cols = nb_plots

    # prepare the subplot
    fig, axs = plt.subplots(rows, cols)
    if rows == 1 and cols == 1:
        axs = [axs]

    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        # turn off axis
        ax.axis('off')

        # add titles
        if titles is not None and titles[i] is not None:
            ax.title.set_text(titles[i])

        u, v = slices_in[i][..., 0], slices_in[i][..., 1]
        colors = np.arctan2(u, v)
        colors[np.isnan(colors)] = 0
        norm = Normalize()
        norm.autoscale(colors)
        if cmaps[i] is None:
            colormap = cm.winter
        else:
            raise Exception("custom cmaps not currently implemented for plt.flow()")

        # show figure
        ax.quiver(u, v,
                  color=colormap(norm(colors).flatten()),
                  angles='xy',
                  units='xy',
                  width=quiver_width,
                  scale=scale[i])
        ax.axis('equal')

    # clear axes that are unnecessary
    for i in range(nb_plots, col * row):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        ax.axis('off')

    # show the plots
    fig.set_size_inches(width, rows / cols * width)
    plt.tight_layout()

    if show:
        plt.show(block=plot_block)

    return (fig, axs)

######################################################################
# Neurite auxiliary functions (from https://github.com/adalca/neurite)
######################################################################
 # import tensorflow as tf
 # import tensorflow.keras.backend as K

 # def interpn(vol, loc, interp_method='linear', fill_value=None):
 #    """
 #    N-D gridded interpolation in tensorflow
 #    vol can have more dimensions than loc[i], in which case loc[i] acts as a slice 
 #    for the first dimensions
 #    Parameters:
 #        vol: volume with size vol_shape or [*vol_shape, nb_features]
 #        loc: a N-long list of N-D Tensors (the interpolation locations) for the new grid
 #            each tensor has to have the same size (but not nec. same size as vol)
 #            or a tensor of size [*new_vol_shape, D]
 #        interp_method: interpolation type 'linear' (default) or 'nearest'
 #        fill_value: value to use for points outside the domain. If None, the nearest
 #            neighbors will be used (default).
 #    Returns:
 #        new interpolated volume of the same size as the entries in loc
 #    If you find this function useful, please cite the original paper this was written for:
 #        VoxelMorph: A Learning Framework for Deformable Medical Image Registration 
 #        G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
 #        IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 
 #        Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
 #        A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
 #        MedIA: Medical Image Analysis. (57). pp 226-236, 2019 
 #    TODO:
 #        enable optional orig_grid - the original grid points.
 #        check out tf.contrib.resampler, only seems to work for 2D data
 #    """

 #    if isinstance(loc, (list, tuple)):
 #        loc = tf.stack(loc, -1)
 #    nb_dims = loc.shape[-1]
 #    input_vol_shape = vol.shape

 #    if len(vol.shape) not in [nb_dims, nb_dims + 1]:
 #        raise Exception("Number of loc Tensors %d does not match volume dimension %d"
 #                        % (nb_dims, len(vol.shape[:-1])))

 #    if nb_dims > len(vol.shape):
 #        raise Exception("Loc dimension %d does not match volume dimension %d"
 #                        % (nb_dims, len(vol.shape)))

 #    if len(vol.shape) == nb_dims:
 #        vol = K.expand_dims(vol, -1)

 #    # flatten and float location Tensors
 #    if not loc.dtype.is_floating:
 #        target_loc_dtype = vol.dtype if vol.dtype.is_floating else 'float32'
 #        loc = tf.cast(loc, target_loc_dtype)
 #    elif vol.dtype.is_floating and vol.dtype != loc.dtype:
 #        loc = tf.cast(loc, vol.dtype)

 #    if isinstance(vol.shape, (tf.compat.v1.Dimension, tf.TensorShape)):
 #        volshape = vol.shape.as_list()
 #    else:
 #        volshape = vol.shape

 #    max_loc = [d - 1 for d in vol.get_shape().as_list()]

 #    # interpolate
 #    if interp_method == 'linear':
 #        # floor has to remain floating-point since we will use it in such operation
 #        loc0 = tf.floor(loc)

 #        # clip values
 #        clipped_loc = [tf.clip_by_value(loc[..., d], 0, max_loc[d]) for d in range(nb_dims)]
 #        loc0lst = [tf.clip_by_value(loc0[..., d], 0, max_loc[d]) for d in range(nb_dims)]

 #        # get other end of point cube
 #        loc1 = [tf.clip_by_value(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
 #        locs = [[tf.cast(f, 'int32') for f in loc0lst], [tf.cast(f, 'int32') for f in loc1]]

 #        # compute the difference between the upper value and the original value
 #        # differences are basically 1 - (pt - floor(pt))
 #        #   because: floor(pt) + 1 - pt = 1 + (floor(pt) - pt) = 1 - (pt - floor(pt))
 #        diff_loc1 = [loc1[d] - clipped_loc[d] for d in range(nb_dims)]
 #        diff_loc0 = [1 - d for d in diff_loc1]
 #        # note reverse ordering since weights are inverse of diff.
 #        weights_loc = [diff_loc1, diff_loc0]

 #        # go through all the cube corners, indexed by a ND binary vector
 #        # e.g. [0, 0] means this "first" corner in a 2-D "cube"
 #        cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
 #        interp_vol = 0

 #        for c in cube_pts:

 #            # get nd values
 #            # note re: indices above volumes via
 #            #   https://github.com/tensorflow/tensorflow/issues/15091
 #            #   It works on GPU because we do not perform index validation checking on GPU -- it's
 #            #   too expensive. Instead we fill the output with zero for the corresponding value.
 #            #   The CPU version caught the bad index and returned the appropriate error.
 #            subs = [locs[c[d]][d] for d in range(nb_dims)]

 #            # tf stacking is slow for large volumes, so we will use sub2ind and use single indexing.
 #            # indices = tf.stack(subs, axis=-1)
 #            # vol_val = tf.gather_nd(vol, indices)
 #            # faster way to gather than gather_nd, because gather_nd needs tf.stack which is slow :(
 #            idx = sub2ind2d(vol.shape[:-1], subs)
 #            vol_reshape = tf.reshape(vol, [-1, volshape[-1]])
 #            vol_val = tf.gather(vol_reshape, idx)

 #            # get the weight of this cube_pt based on the distance
 #            # if c[d] is 0 --> want weight = 1 - (pt - floor[pt]) = diff_loc1
 #            # if c[d] is 1 --> want weight = pt - floor[pt] = diff_loc0
 #            wts_lst = [weights_loc[c[d]][d] for d in range(nb_dims)]
 #            # tf stacking is slow, we will use prod_n()
 #            # wlm = tf.stack(wts_lst, axis=0)
 #            # wt = tf.reduce_prod(wlm, axis=0)
 #            wt = prod_n(wts_lst)
 #            wt = K.expand_dims(wt, -1)

 #            # compute final weighted value for each cube corner
 #            interp_vol += wt * vol_val

 #    else:
 #        assert interp_method == 'nearest', \
 #            'method should be linear or nearest, got: %s' % interp_method
 #        roundloc = tf.cast(tf.round(loc), 'int32')
 #        roundloc = [tf.clip_by_value(roundloc[..., d], 0, max_loc[d]) for d in range(nb_dims)]

 #        # get values
 #        # tf stacking is slow. replace with gather
 #        # roundloc = tf.stack(roundloc, axis=-1)
 #        # interp_vol = tf.gather_nd(vol, roundloc)
 #        idx = sub2ind2d(vol.shape[:-1], roundloc)
 #        interp_vol = tf.gather(tf.reshape(vol, [-1, vol.shape[-1]]), idx)

 #    if fill_value is not None:
 #        out_type = interp_vol.dtype
 #        fill_value = tf.constant(fill_value, dtype=out_type)
 #        below = [tf.less(loc[..., d], 0) for d in range(nb_dims)]
 #        above = [tf.greater(loc[..., d], max_loc[d]) for d in range(nb_dims)]
 #        out_of_bounds = tf.reduce_any(tf.stack(below + above, axis=-1), axis=-1, keepdims=True)
 #        interp_vol *= tf.cast(tf.logical_not(out_of_bounds), out_type)
 #        interp_vol += tf.cast(out_of_bounds, out_type) * fill_value

 #    # if only inputted volume without channels C, then return only that channel
 #    if len(input_vol_shape) == nb_dims:
 #        assert interp_vol.shape[-1] == 1, 'Something went wrong with interpn channels'
 #        interp_vol = interp_vol[..., 0]

 #    return interp_vol