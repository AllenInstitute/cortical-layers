import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import os
import json
from scipy.signal import convolve2d
from scipy import interpolate
import warnings
from hmmlearn import hmm
from caveclient import CAVEclient
from cortical_layers.LayerPredictor import BoundaryPredictor, LayerClassifier, LayerPrediction


class SubBoundaryPredictor:
    """
    finds the borders between sublayers within a given layer (currently just layer 6) for each column in a list of columns to be predicted
    """

    def __init__(self, bin_width=0.05,
                 step_size=0.005,
                 resolution=(4., 4., 40.),
                 features=("soma_volume",),
                 use_depth=False,
                 use_exc_soma_density=True,
                 use_soma_vol_std=True,
                 num_PCA=None,
                 save_figs=False,
                 verbose=False,
                 **kwargs):
        """
        :param bin_width (float in mm) How wide a bin should be used for calculating information by depth
        :param step_size (float in mm) How large of steps should be taken in the depth direction. Recommend <= bin_width
        :param resolution (tuple of floats) voxel resolution of the dataset
        :param features (tuple of Strings) are columns of the soma/nuc features table to use for calculation
               from {'nucleus_area', 'nucleus_area_to_volume',
                   'nucleus_center_mass_nm', 'nucleus_fold_area', 'nucleus_fract_fold',
                   'nucleus_id', 'nucleus_volume', 'cleft_segid', 'size', 'centroid_x',
                   'centroid_y', 'centroid_z', 'yr_um_translated', 'predict',
                   'is_watertight', 'soma_center_mass', 'soma_volume', 'soma_area',
                   'soma_area_to_volume', 'num_processes', 'avg_sdf', 'orients', 'vectors',
                   'n_soma_syn', 'soma_syn_density', 'nuc_id', 'seg_id', 'soma_y',
                   'soma_x', 'soma_z', 'xr', 'yr', 'nucleus_to_soma', 'sum_proc_sdf',
                   'cell_type_pred_num', 'cell_type_pred', 'umapx', 'umapy', 'visible',
                   'outline_color', 'outline_width', 'max_sdf'}
        :param use_depth determines whether depth should be a features
        :param use_soma_vol_std determines whether the standard deviation of soma volume at each depth should also
               be used as a feature (all the other features take the mean only)
        :param use_exc_soma_density determines whether the excitatory soma density at each depth should also
               be used as a feature (all the other features are means over cells)
        :param num_PCA (None or int)  indicates how many PCA modes should be used. Default: None, which indicates the
            raw features should be used without performing PCA
        :param save_figs=False dictates whether plots should be saved to disk
        :param verbose=False dictates whether printing and plotting occurs
        :param default_bounds (np.array of shape (7,)): the default boundaries to use to initialize the HMM.
               This value is updated every time predict_col is called
        :param cache_dir: default: ".", the directory where to store resulting plots and predictions
        :param name: default: "layers", the name of this layer predictor to use for file naming
        :param l1_2_thresh=120_000: density threshold in mm^-3 to use for determining layer 1-2 border
        :param l6_WM_thresh=50_000: density threshold in mm^-3 to use for determining layer 6-white matter border
        :param soma_table_path: The path of the pickled soma features table
        """
        self.bin_width = bin_width
        self.step_size = step_size
        self.resolution = np.asarray(resolution)
        self.features = list(features)
        self.use_depth = use_depth
        self.use_soma_vol_std = use_soma_vol_std
        self.use_exc_soma_density = use_exc_soma_density
        self.num_PCA = num_PCA
        self.save_figs = save_figs
        self.verbose = verbose

        # np.array([0.3, 0.400516, 0.555516, 0.700516, 0.830516, 1.010516, ]) from HMM trained on 2 PCA modes in column
        self.default_bounds = kwargs["default_bounds"] if "default_bounds" in kwargs else None
        self.name = kwargs["name"] if "name" in kwargs else "layers"
        self.cache_dir = kwargs["cache_dir"] if "cache_dir" in kwargs else "."
        self.l1_2_thresh = kwargs["l1_2_thresh"] if "l1_2_thresh" in kwargs else 120_000
        self.l6_WM_thresh = kwargs["l6_WM_thresh"] if "l6_WM_thresh" in kwargs else 50_000
        self.soma_table_path = kwargs[
            "soma_table_path"] if "soma_table_path" in kwargs else "Minnie_soma_nuc_feature_model_83_1.pkl"
        self.column_labels = None  # list of the column labels of varis

    def _init_data(self):
        # Gather data
        if self.verbose:
            print("connecting to server... ", end="")
        datastack_name = "minnie65_public_v117"  # TODO the only one supported because others don't have a soma table
        client = CAVEclient(datastack_name)
        if self.verbose:
            print("success.")

        if self.verbose:
            print("downloading all cells from datastack... ", end="")
        self.auto_cells = client.materialize.query_table("allen_soma_coarse_cell_class_model_v1")
        cells_by_root = self.auto_cells.copy()
        cells_by_root.index = cells_by_root.pt_root_id

        nuc_to_root = client.materialize.query_table("nucleus_neuron_svm")
        nuc_to_root.index = nuc_to_root.id
        if self.verbose:
            print("success.")

        if self.verbose:
            print("loading soma features... ", end="")
        self.soma_features = pd.read_pickle(
            self.soma_table_path)  # TODO this isn't available to everyone
        if self.verbose:
            print("success.")

        self.soma_features["seg_id"] = [nuc_to_root.loc[n].pt_root_id for n in self.soma_features.nuc_id]
        self.soma_features.index = self.soma_features.seg_id

        # size is equal to nucleus volume
        # soma area and nucleus area closely track their respective volumes
        # avg sdf is a list of the 'diameters' of processes (e.g. dendrites) that leave each cell body

    def predict(self, base_pred: LayerPrediction, layer: str, num_sublayers: int, min_depth=None, col_size=None):
        """
        Calculates the requested features at each depth and uses them to predict the boundaries between cortical layers
        for the provided region
        :param base_pred: a LayerPrediction object for which you would like to refine layer predictions within the given layer
        :param layer: the layer to subdivide
        :param num_sublayers: number of subdivisions
        :param min_depth = None the minimum depth (as a fraction of the way between the minimum depth within `layer` and
          the maximum depth within `layer`) at which a layer boundary should be placed
        :param col_size = None optionally override the column width from the base_pred to include a large/smaller number
          of cells in determining the sublayer borders

        :return np.array of shape (len(bboxs), 5): the respective layer boundaries [L1/L23, L23/L4, L4/L5, L5/L6, L6/WM] for each bbox in bboxs
        """
        LAYER_NAMES = ("L1", "L23", "L4", "L5", "L6", "WM")
        assert layer in LAYER_NAMES, "layer must be one of (\"L1\", \"L23\", \"L4\", \"L5\", \"L6\", \"WM\")"
        assert num_sublayers < 6, "too many sublayers"
        assert base_pred.bounds.shape[1] == 5, "given prediction doesn't have 5 layers"
        overall_bbox = base_pred.overall_bbox
        col_size = ((base_pred.cols_nm[0, 1, 0] - base_pred.cols_nm[0, 0, 0]) / 1000,  # microns
                    (base_pred.cols_nm[0, 1, 2] - base_pred.cols_nm[0, 0, 2]) / 1000) if col_size is None else col_size
        ngridpts = base_pred.ngridpts
        self._init_data()
        bboxs, col_center_xs, col_center_zs = BoundaryPredictor.get_snaking_cols(overall_bbox, col_size=col_size,
                                                                                 ngridpts=ngridpts)

        # initialize new layer boundaries and layer labels
        layer_labels = []
        results = np.empty((len(bboxs), base_pred.bounds.shape[1] + num_sublayers - 1))
        i = 0
        base_i = 0
        for l in LAYER_NAMES:
            # set the layer boundary to the corresponding layer bound in base_pred
            if i > 0:
                results[:, i - 1] = base_pred.bounds[:, base_i - 1]

            if l != layer:
                layer_labels.append(l)
                i += 1
            else:
                for j in range(num_sublayers):
                    layer_labels.append(l + "abcde"[j])
                    i += 1
            base_i += 1

        # sub_idxs = [subdivision1_idx, subdivision2_idx]  does not include divisions already present in base_pred
        sub_idxs = np.nonzero([layer_labels[i].startswith(layer) for i in range(len(layer_labels))])[0][:-1]

        pred = LayerPrediction(bboxs, ngridpts, results, col_center_xs, col_center_zs, overall_bbox,
                               cache_dir=self.cache_dir, name=self.name + "_prediction",
                               layer_labels=layer_labels)

        # initialize default bounds
        if self.default_bounds is None:
            self.default_bounds = np.zeros(pred.bounds.shape[1] + 2)
            self.default_bounds[1:-1] = pred.bounds[0]
            self.default_bounds[0] = 0.3  # pia border
            self.default_bounds[-1] = 1.1  # end of sample
            slope = (self.default_bounds[sub_idxs[-1] + 2] - self.default_bounds[sub_idxs[0]]) / (len(sub_idxs) + 1)
            self.default_bounds[sub_idxs + 1] = (sub_idxs - (sub_idxs[0] - 1)) * slope + self.default_bounds[
                sub_idxs[0]]

        for i, b in enumerate(bboxs):
            bbox = b / self.resolution
            if self.verbose:
                print("\nWORKING ON", i, bbox)
            start_depth = pred.bounds[i, sub_idxs[0] - 1] if sub_idxs[0] - 1 >= 0 else -1e10
            end_depth = pred.bounds[i, sub_idxs[-1] + 1] if sub_idxs[-1] + 1 < pred.bounds.shape[1] else 1e10
            pred.bounds[i, sub_idxs] = self._predict_col(bbox, start_depth, end_depth, num_sublayers, sub_idxs[0],
                                                         idx=i, min_depth=min_depth)
            if i % 10 == 9:
                pred.save()
                plt.close("all")  # free up RAM
        pred.save()
        plt.close("all")  # free up RAM
        plt.show()
        return pred

    def _predict_col(self, bbox, start_depth, end_depth, num_sublayers, start_sub_idx, idx=None, min_depth=None):
        """
        makes layer boundary predictions for the particular column provided by bbox
        :param bbox: bounding box of column
        :param idx: the index of bbox within the bboxs parameter passed into predict()
        :param start_sub_idx: the number of layer boundaries before the first subdivision
        :return: bounds: the predicted layer bounds
        """
        soma_features_root_ids = set(self.soma_features.seg_id)

        auto_col_cells = self.auto_cells[
            self.auto_cells.pt_position.apply(BoundaryPredictor.in_bbox, args=[bbox])].copy()
        auto_col_cells["mm_depth"] = [auto_col_cells.pt_position.iloc[i][1] * self.resolution[1] / 1_000_000 for i in
                                      range(len(auto_col_cells))]
        # add soma features columns to auto_col_cells
        ftrs = self.features.copy()
        # weird case where someone only wants the std but not the mean of cell size
        if self.use_soma_vol_std and "soma_volume" not in self.features:
            ftrs.append("soma_volume")
        for feature in ftrs:
            auto_col_cells[feature] = [
                (self.soma_features.loc[r][feature] if r in soma_features_root_ids and
                                                       type(self.soma_features.loc[r][
                                                                feature]) is not pd.Series else np.nan)
                for r in auto_col_cells.pt_root_id]

        if self.verbose:
            print("calculating features by depth... ", end="")
        bin_centers, varis = self._calculate_features(bbox, start_depth, end_depth, auto_col_cells, min_depth=min_depth)
        if self.verbose:
            print("success.")

        model = self._hmm_fit(bin_centers, varis, num_sublayers, start_sub_idx)

        bounds, hmm_layers, posteriors = self._hmm_predict(model, bin_centers, varis, start_sub_idx)

        # Plot model results!
        if self.verbose or self.save_figs:
            model_means = np.array([model.means_[l] for l in hmm_layers])
            model_stds = np.array([np.diagonal(np.sqrt(model.covars_[l])) for l in hmm_layers])

            colors = get_cmap("tab20").colors[::2] + get_cmap("tab20b").colors[::2]

            fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=130)

            for i in range(varis.shape[1]):
                c = colors[i % len(colors)]
                ax.plot(bin_centers, varis[:, i], color=c, label=self.column_labels[i])
                ax.plot(bin_centers, model_means[:, i], linestyle="-.", color=c)
                ax.fill_between(bin_centers, model_means[:, i] - model_stds[:, i],
                                model_means[:, i] + model_stds[:, i],
                                edgecolor="none", facecolor=c, alpha=0.2)

            ax.axvline(bounds[0], linestyle="--", color="blue", label="automatic bounds")
            for bound in bounds[1:]:
                ax.axvline(bound, linestyle="--", color="blue")
            # ax.plot(bin_centers, posteriors)
            ax.legend()
            ax.set_xlim(0.29, 1.11)

            if self.verbose:
                plt.draw()
            if self.save_figs:
                fig.savefig(os.path.join(self.cache_dir, f"{self.name}_{idx}.png"))

        return bounds

    def _calculate_features(self, bbox, start_depth, end_depth, auto_col_cells, min_depth=None):
        """
        calculates the features used for the HMM
        :param bbox: bbox of column
        :param auto_col_cells: df of (excitatory) cells in column with features attached
        :return bin_centers: the 1D array of mm depths at which the features were calculated
                varis: the 2D array of features to be used for the HMM, normalized and cleaned of nans
        """
        auto_exc_cells = auto_col_cells.query("classification_system == 'aibs_coarse_excitatory'")

        # cross sectional area to be layered, in mm^2
        xarea = self.resolution[0] * self.resolution[2] * (bbox[1][0] - bbox[0][0]) * (
                bbox[1][2] - bbox[0][2]) / 1_000_000. ** 2

        # min is pia border (with L1) and max is white matter border (with L6)
        min_y = max(np.min(auto_col_cells.mm_depth.values), start_depth)
        max_y = min(np.max(auto_col_cells.mm_depth.values), end_depth)
        if min_depth is not None:
            min_y = min_y + min_depth * (max_y - min_y)

        auto_exc_cells = auto_exc_cells.sort_values(axis="index", by="mm_depth")

        exc_soma_densities = []
        exc_soma_features_by_depth = dict((f, []) for f in self.features)
        exc_soma_vol_std_by_depth = []
        exc_soma_depths = []

        depths = np.arange(min_y, max_y, self.step_size)
        bin_centers = depths + self.step_size // 2
        prev_cutoff_idx = 0
        for curr_y in depths:
            # first index where pt_position[1] is greater than curr_y + bin_width
            cutoff_idx = BoundaryPredictor.get_cutoff_idx(auto_exc_cells, curr_y + self.bin_width, prev_cutoff_idx)
            lower_cutoff_idx = BoundaryPredictor.get_cutoff_idx(auto_exc_cells, curr_y, prev_cutoff_idx)

            current_exc_cells = auto_exc_cells.iloc[lower_cutoff_idx:cutoff_idx]

            for f in self.features:
                exc_soma_features_by_depth[f].append(current_exc_cells[f].mean())
            if self.use_soma_vol_std:
                exc_soma_vol_std_by_depth.append(current_exc_cells["soma_volume"].std())
            exc_soma_depths.append(current_exc_cells["mm_depth"].mean())

            exc_soma_densities.append(len(current_exc_cells))

            prev_cutoff_idx = lower_cutoff_idx

        exc_soma_densities = np.array(exc_soma_densities, dtype=float) / (self.bin_width * xarea)  # per mm^3

        exc_features_df = pd.DataFrame(exc_soma_features_by_depth)
        if self.use_exc_soma_density:
            exc_features_df["soma_density"] = exc_soma_densities
        if self.use_soma_vol_std:
            exc_features_df["soma_vol_std"] = exc_soma_vol_std_by_depth
        for col in exc_features_df.columns:
            exc_features_df[col] = BoundaryPredictor.clean_nans(exc_features_df[col], normalize=True)

        ### PCA on the features to remove correlations
        if self.num_PCA is not None:
            X = exc_features_df.values.T
            Xc = X - np.mean(X, axis=1, keepdims=True)  # mean subtract data first
            cov = Xc @ Xc.T / (X.shape[0] - 1)  # covariance

            variance, V = np.linalg.eig(cov)
            idxs = np.argsort(-variance)
            V = V[:, idxs]
            variance = variance[idxs]
            explained_variance = [sum(variance[:i + 1]) / sum(variance) for i in range(len(variance))]

            if self.verbose:
                plt.imshow(cov)
                plt.colorbar()
                plt.title("covariance matrix")
                plt.ylabel(" ".join(list(exc_features_df.columns)))
                plt.draw()

                plt.plot(range(1, len(explained_variance) + 1), explained_variance)
                plt.xlabel("num principle components")
                plt.ylabel("explained variance")
                plt.ylim([0, 1])
                plt.draw()

            # Yc is the projection of Xc onto the principal components
            Yc = V[:, :self.num_PCA].T @ Xc

        varis = exc_features_df.values if self.num_PCA is None else Yc.T
        self.column_labels = list(exc_features_df.columns) if self.num_PCA is None else [f"PCA {i}" for i in
                                                                                         range(self.num_PCA)]
        if self.use_depth:
            # this is here because depth shouldn't go into PCA
            varis = np.hstack([varis, BoundaryPredictor.clean_nans(exc_soma_depths, normalize=True).reshape(-1, 1)])
            self.column_labels.append("depth")

        return bin_centers, varis

    def _hmm_fit(self, bin_centers, varis, num_sublayers, start_sub_idx):
        """
        fits a hidden markov model to the variables given, which are measured at depths bin_centers
        :param bin_centers: depths of varis in mm
        :param varis: np.array of shape (len(bin_centers), num_features), normalized and free of nans
        :param start_sub_idx: the number of layer boundaries before the first subdivision
        :return: model: trained hidden markov model
        """
        model = hmm.GaussianHMM(n_components=num_sublayers, covariance_type="diag", init_params="", params="mc",
                                n_iter=1)
        # the model starts in the first state, and there is 0 probability of starting elsewhere
        model.startprob_ = np.zeros(model.n_components)
        model.startprob_[0] = 1
        p = 0.04
        transition_probs = np.full(model.n_components - 1, p)
        stay_probs = np.full(model.n_components, 1 - p)
        stay_probs[-1] = 1  # nothing deeper than white matter
        model.transmat_ = np.diag(stay_probs) + np.diag(transition_probs, k=1)

        # initialize means and variances
        nf = varis.shape[1]
        model.means_ = np.zeros((model.n_components, nf))
        covars = np.ones((model.n_components, nf))
        for i in range(model.n_components):
            # idxs = np.arange(i * len(varis) // num_sublayers, (i + 1) * len(varis) // num_sublayers)  # defaults to a uniform split
            idxs = (self.default_bounds[start_sub_idx + i] <= bin_centers) \
                   & (bin_centers < self.default_bounds[start_sub_idx + i + 1])
            if any(idxs):
                model.means_[i, :] = varis[idxs, :].mean(axis=0)
                covars[i, :] = varis[idxs, :].var(axis=0) + 1e-10
            else:
                model.means_[i, :] = 0
                covars[i, :] = 0.1  # arbitrary positive number
        model.covars_ = covars

        depth_centers = model.means_[:,
                        -1] if self.use_depth else None  # to help reduce major errors, these will be the means for the depth emissions

        # the package has it's own convergence monitor, but I want fine control over it so I'm doing it manually
        prev_score = 1e20  # big number
        score = model.score(varis)
        max_iters = 100
        tol = 0.001
        i = 0
        while abs(score - prev_score) > tol and i < max_iters:
            model.fit(varis)
            prev_score = score
            score = model.score(varis)
            i += 1
            if self.use_depth:
                # each iteration re-fix the "target" mean depth to the center of each default layer
                # so the model can't be terribly wrong
                model.means_[:, -1] = depth_centers
            if self.verbose:
                print("score:", model.score(varis))

        if abs(score - prev_score) > tol:
            raise ValueError("Did not converge")

        return model

    def _hmm_predict(self, model, bin_centers, varis, start_sub_idx):
        """
        computes the layer boundaries as predicted by model on observation varis, which were observed at depths bin_centers
        :param bin_centers: depths of varis in mm
        :param varis: np.array of shape (len(bin_centers), num_features), normalized and free of nans
        :param start_sub_idx: the number of layer boundaries before the first subdivision
        :return: bounds: predicted layer boundaries
                 hmm_layers: hmm prediction of which layer each index participates in (its hidden state)
                 posteriors: how confident the hmm is that each depth belongs to each state
        """
        hmm_layers = model.predict(varis).tolist()

        bounds = []
        for i in range(1, model.n_components):
            if i == hmm_layers[0]:
                bound = bin_centers[0]
            elif i in hmm_layers:
                idx = hmm_layers.index(i)
                bound = (bin_centers[idx] + bin_centers[idx - 1]) / 2
            else:  # i > hmm_layers[-1]:
                bound = bin_centers[-1] + (bin_centers[1] - bin_centers[0]) / 2
            bounds.append(bound)

        self.default_bounds[1 + start_sub_idx: 1 + start_sub_idx + model.n_components - 1] = bounds

        posteriors = model.predict_proba(varis)
        if self.verbose:
            print("bounds:", bounds)

        return bounds, hmm_layers, posteriors

    @staticmethod
    def in_bbox(p, bbox):
        """ returns whether point is between the two points given by bbox"""
        lower, upper = bbox
        return lower[0] <= p[0] < upper[0] and lower[1] <= p[1] < upper[1] and lower[2] <= p[2] < upper[2]

    @staticmethod
    def clean_nans(x, normalize=False):
        """linearly interpolates to fill in nans
        normalizing subtracts mean and divides by std
        """
        cleaned = np.array(x)
        mask = np.isfinite(cleaned)
        cleaned = np.interp(np.arange(len(cleaned)), np.arange(len(cleaned))[mask], cleaned[mask])
        if normalize:
            cleaned -= cleaned.mean()
            cleaned /= cleaned.std()
        return cleaned

    @staticmethod
    def get_cutoff_idx(df, y, prev_cutoff_idx, position_col="mm_depth"):
        """efficiently find the first index after prev_cutoff_idx where position[1] is greater than y"""
        i = prev_cutoff_idx
        if i >= len(df):
            return i
        while df.iloc[i][position_col] < y:
            i += 1
            if i >= len(df):
                return i
        return i  # new_cutoff_idx

    @staticmethod
    def get_snaking_cols(overall_bbox, col_size=(100., 100.), ngridpts=(32, 16)):
        """
        Calculates the bounding boxes of columns along a snaking path through the provided bounding box, such that
        adjacent bboxs in the resulting list are spatially adjacent
        :param overall_bbox: (np.array of shape (2, 3)) representing the minimum and maximum corner, in microns, of the
            desired region of the dataset
        :param col_size: (tuple of float) size, in microns, of the x and z dimensions of the columns. 100x100 columns
            provide sufficient information to make reasonably reliable HMMs
        :param ngridpts: (tuple of int) the x and z dimensions of the grid of cortical columns.
                ngridpts[0] * ngridpts[1] columns will be used
        :return: (list of 2x3 np.arrays of floats) the nm coordinates of the minimum corner and maximum corner of
                the col_size[0] x full depth x col_size[1] columns within the provided bounding box. They are listed starting
                from the minimum corner and snaking back and forth across the data set, ending at maximum x.
        """
        seg_low_um, seg_up_um = overall_bbox[0], overall_bbox[1]

        col_size = np.array(col_size)  # x and z dimensions of column
        col_center_xs = np.linspace(seg_low_um[0] + col_size[0] // 2, seg_up_um[0] - col_size[0] // 2, ngridpts[0])
        col_center_zs = np.linspace(seg_low_um[2] + col_size[1] // 2, seg_up_um[2] - col_size[1] // 2, ngridpts[1])

        cols_nm = []
        offx = col_size[0] // 2
        offz = col_size[1] // 2
        for i, x in enumerate(col_center_xs):
            for z in col_center_zs[::(-1) ** i]:
                col_low = [x - offx, seg_low_um[1], z - offz]
                col_up = [x + offx, seg_up_um[1], z + offz]
                cols_nm.append(np.array([col_low, col_up]) * 1_000)
        return cols_nm, col_center_xs, col_center_zs

    @staticmethod
    def smooth_bounds(pred, smoothness=1):
        """
        in-place smooths the prediction by convolving with 3x3 gaussian kernel
        :param pred: LayerPrediction object
        :param smoothness: radius of the gaussian
        """
        spatial_array = LayerPrediction.to_spatial_array(pred.bounds, pred.ngridpts)
        kernel_gridx, kernel_gridz = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
        kernel = np.exp(-(kernel_gridx ** 2 + kernel_gridz ** 2) / smoothness)
        kernel /= np.sum(kernel)
        smoothed_spatial_bounds = np.empty(spatial_array.shape)
        for i in range(spatial_array.shape[2]):
            smoothed_spatial_bounds[:, :, i] = convolve2d(spatial_array[:, :, i], kernel, boundary="symm", mode="same")
        pred.bounds = LayerPrediction.to_snaking_array(smoothed_spatial_bounds)


if __name__ == "__main__":
    resolution = np.array([4., 4., 40.])
    # conservative bbox only containing well-segmented areas
    seg_low_um = np.array([130_000, 50_000, 15_000]) * resolution / 1_000
    seg_up_um = np.array([355_000, 323_500, 27_500]) * resolution / 1_000
    name = "sub6_soma_vol_and_density_second_half_only_200micron"
    p = SubBoundaryPredictor(features=("soma_volume",),
                             num_PCA=None, use_depth=False, use_exc_soma_density=True, use_soma_vol_std=False,
                             resolution=resolution, save_figs=True, verbose=True, cache_dir="..\\" + name,
                             name=name)
    c = LayerClassifier(data=f"minnie65_phase3")
    pred = p.predict(c.pred, layer="L6", num_sublayers=2, min_depth=0.5, col_size=[200., 200.])
    inpt = pred.cols_nm[0:2, 0].copy()
    c.predict(inpt)
