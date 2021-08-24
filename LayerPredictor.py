import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import os
import pickle
from scipy.signal import convolve2d
from scipy import interpolate
import warnings
from hmmlearn import hmm
from caveclient import CAVEclient


class LayerPrediction:
    """
    stores the prediction of layer boundaries, including the columns for which predictions were made,
    the values of the prediction, and the shape of the grid over which the predictions were made
    """

    def __init__(self, columns_nm, num_gridpts, boundaries, col_center_xs, col_center_zs,
                 overall_bbox, name="layer_prediction", cache_dir="."):
        self.cols_nm = columns_nm
        self.ngridpts = num_gridpts
        self.bounds = boundaries
        self.col_center_xs, self.col_center_zs = col_center_xs, col_center_zs
        self.overall_bbox = overall_bbox
        self.name = name
        self.cache_dir = cache_dir

    def save_pickle(self):
        with open(os.path.join(self.cache_dir, f"{self.name}.pkl"), "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_pickle(dir):
        with open(dir, "rb") as f:
            o = pickle.load(f)
        return o

    @staticmethod
    def to_spatial_array(snaking_arr, ngridpts):
        """
        Convert a snaking array to a spatial array, given the size of the grid through which it snakes. Useful for
        taking the results of get_snaking_cols() or predict() to a spatially arranged array.
        :param snaking_arr: (np.array of shape (ngridpts[0] * ngridpts[1], ...)), in which the first dimension
                            should be reshaped into a 2D spatial array
        :param ngridpts: (list-like of length 2) representing the number of grid points in the x and z directions
        :return: np.array of shape (ngridpts[0], ngridpts[1], *snaking_arr.shape[1:])
        """
        spatial_array = snaking_arr.reshape((ngridpts[0], ngridpts[1], *snaking_arr.shape[1:])).copy()
        for i in range(1, spatial_array.shape[0], 2):  # reverse every other x slice
            spatial_array[i, :, ...] = spatial_array[i, ::-1, ...]
        return spatial_array

    @staticmethod
    def to_snaking_array(spatial_arr):
        """
        convert spatial array to snaking array
        :param spatial_arr:
        :return: snaking array (np.array of shape (spatial_arr.shape[0] * spatial_arr.shape[1], ...)),
                in which the first dimension has been reshaped from the given a 2D spatial array
        """
        smoothed_bounds = []
        for i in range(spatial_arr.shape[0]):
            if i % 2 == 0:
                smoothed_bounds.extend(spatial_arr[i, :, ...].tolist())
            else:
                smoothed_bounds.extend(spatial_arr[i, ::-1, ...].tolist())
        return np.array(smoothed_bounds)


class BoundaryPredictor:
    """
    finds the borders between layers for each column in a list of columns to be predicted
    """

    def __init__(self, bin_width=0.05,
                 step_size=0.005,
                 resolution=(4., 4., 40.),
                 features=("soma_volume",),
                 use_depth=False,
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
        :param num_PCA (None or int)  indicates how many PCA modes should be used. Default: None, which indicates the
            raw features should be used without performing PCA
        :param save_figs=False dictates whether plots should be saved to disk
        :param verbose=False dictates whether printing and plotting occurs
        :param default_bounds (np.array of shape (7,)): the default boundaries to use to initialize the HMM.
               This value is updated every time predict_col is called
        :param: cache_dir: default: ".", the directory where to store resulting plots and predictions
        :param: name: default: "layers", the name of this layer predictor to use for file naming
        """
        self.bin_width = bin_width
        self.step_size = step_size
        self.resolution = np.asarray(resolution)
        self.features = list(features)
        self.use_depth = use_depth
        self.use_soma_vol_std = use_soma_vol_std
        self.num_PCA = num_PCA
        self.save_figs = save_figs
        self.verbose = verbose
        self.default_bounds = kwargs["default_bounds"] if "default_bounds" in kwargs else np.array(
            [0.3, 0.400516, 0.555516, 0.700516, 0.830516, 1.010516, 1.1])  # from HMM trained on 2 PCA modes in column
        self.name = kwargs["name"] if "name" in kwargs else "layers"
        self.cache_dir = kwargs["cache_dir"] if "cache_dir" in kwargs else "."
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
            "Minnie_soma_nuc_feature_model_83_1.pkl")  # TODO this isn't available to everyone
        if self.verbose:
            print("success.")

        self.soma_features["seg_id"] = [nuc_to_root.loc[n].pt_root_id for n in self.soma_features.nuc_id]
        self.soma_features.index = self.soma_features.seg_id

        # size is equal to nucleus volume
        # soma area and nucleus area closely track their respective volumes
        # avg sdf is a list of the 'diameters' of processes (e.g. dendrites) that leave each cell body

    def predict_with_sensitivity(self, overall_bbox, col_size=(100., 100.), ngridpts=(32, 16), ntrials=6, noise_scale=0.02):
        """
        Calculates the requested features at each depth and uses them to predict the boundaries between cortical layers
        for the provided region, at a x,z resolution defined by col_size and step_size.
        It runs ntrials trials of each boundary with randomly perturbed initial conditions
        and returns the std of the resulting bounds
        :param overall_bbox (2x3 np.array of floats) the micron coordinates of the minimum corner and maximum corner of
                the region of interest, which will be broken into many smaller columns of size col_size and predicted
        :param col_size: (tuple of float) size, in microns, of the x and z dimensions of the columns. 100x100 columns
               provide sufficient information to make reasonably reliable HMMs
        :param ngrdipts: (tuple of int) the x and z dimensions of the grid of cortical columns.
                ngridpts[0] * ngridpts[1] columns will be used
        :param ntrials=10 (int) number of trials to use for sensitivity analysis
        :param noise_scale=0.02 (float) the std in mm of random normal noise to use for sensitivity analysis
        :return LayerPrediction object, with bounds=np.array of shape (len(bboxs), 5, ntrials): the respective layer
            boundaries [L1/L23, L23/L4, L4/L5, L5/L6, L6/WM] for each bbox in bboxs, and where the innermost list contains the results of the `ntrials` trials
        """
        self._init_data()
        bboxs, col_center_xs, col_center_zs = BoundaryPredictor.get_snaking_cols(overall_bbox, col_size=col_size, ngridpts=ngridpts)

        results = np.empty((len(bboxs), 5, ntrials))
        pred = LayerPrediction(bboxs, ngridpts, results, col_center_xs, col_center_zs, overall_bbox, cache_dir=self.cache_dir, name=self.name + "_sensitivity_prediction")

        for i, b in enumerate(bboxs):
            bbox = b / self.resolution
            if self.verbose:
                print("\nWORKING ON", i, bbox)

            default_bounds = self.default_bounds
            for tr in range(ntrials):
                if self.verbose:
                    print("\ntrial", tr)

                self.default_bounds[1:-1] = default_bounds[1:-1] + np.random.normal(loc=0, scale=noise_scale, size=5)
                results[i, :, tr] = self._predict_col(bbox, idx=i * ntrials + tr)
            self.default_bounds[1:-1] = results[i, :, :].mean(axis=1)
            if i % 10 == 9:
                pred.save_pickle()
                plt.close("all")  # free up RAM
        pred.save_pickle()
        plt.close("all")  # free up RAM
        return pred

    def predict(self, overall_bbox, col_size=(100., 100.), ngridpts=(32, 16)):
        """
        Calculates the requested features at each depth and uses them to predict the boundaries between cortical layers
        for the provided region, at a x,z resolution defined by col_size and step_size
        :param overall_bbox (2x3 np.array of floats) the micrometer coordinates of the minimum corner and maximum corner of
                the region of interest, which will be broken into many smaller columns of size col_size and predicted
        :param col_size: (tuple of float) size, in microns, of the x and z dimensions of the columns. 100x100 columns
            provide sufficient information to make reasonably reliable HMMs
        :param ngrdipts: (tuple of int) the x and z dimensions of the grid of cortical columns.
                ngridpts[0] * ngridpts[1] columns will be used
        :return np.array of shape (len(bboxs), 5): the respective layer boundaries [L1/L23, L23/L4, L4/L5, L5/L6, L6/WM] for each bbox in bboxs
        """
        self._init_data()
        bboxs, col_center_xs, col_center_zs = BoundaryPredictor.get_snaking_cols(overall_bbox, col_size=col_size, ngridpts=ngridpts)

        results = np.empty((len(bboxs), 5))
        pred = LayerPrediction(bboxs, ngridpts, results, col_center_xs, col_center_zs, overall_bbox, cache_dir=self.cache_dir, name=self.name + "_prediction")

        for i, b in enumerate(bboxs):
            bbox = b / self.resolution
            if self.verbose:
                print("\nWORKING ON", i, bbox)
            results[i, :] = self._predict_col(bbox, idx=i)
            if i % 10 == 9:
                pred.save_pickle()
                plt.close("all")  # free up RAM
        pred.save_pickle()
        plt.close("all")  # free up RAM
        plt.show()
        return pred

    def _predict_col(self, bbox, idx=None):
        """
        makes layer boundary predictions for the particular column provided by bbox
        :param bbox: bounding box of column
        :param idx: the index of bbox within the bboxs parameter passed into predict()
        :return: bounds: the predicted layer bounds
        """
        soma_features_root_ids = set(self.soma_features.seg_id)

        auto_col_cells = self.auto_cells[self.auto_cells.pt_position.apply(BoundaryPredictor.in_bbox, args=[bbox])].copy()
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
        bin_centers, varis, exc_soma_densities = self._calculate_features(bbox, auto_col_cells)
        if self.verbose:
            print("success.")

            model = self._hmm_fit(bin_centers, varis)

        bounds, hmm_layers, posteriors = self._hmm_predict(model, bin_centers, varis, exc_soma_densities)

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

    def _calculate_features(self, bbox, auto_col_cells):
        """
        calculates the features used for the HMM
        :param bbox: bbox of column
        :param auto_exc_cells: df of cells in column with features attached
        :return bin_centers: the 1D array of mm depths at which the features were calculated
                varis: the 2D array of features to be used for the HMM, normalized and cleaned of nans
                exc_soma_densities: density of excitatory somas in #/mm^3
        """
        auto_exc_cells = auto_col_cells.query("classification_system == 'aibs_coarse_excitatory'")

        # cross sectional area to be layered, in mm^2
        xarea = self.resolution[0] * self.resolution[2] * (bbox[1][0] - bbox[0][0]) * (
                    bbox[1][2] - bbox[0][2]) / 1_000_000. ** 2

        # min is pia border (with L1) and max is white matter border (with L6)
        min_y = np.min(auto_col_cells.mm_depth.values)
        max_y = np.max(auto_col_cells.mm_depth.values)

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
        self.column_labels = list(exc_features_df.columns) if self.num_PCA is None else [f"PCA {i}" for i in range(self.num_PCA)]
        if self.use_depth:
            # this is here because depth shouldn't go into PCA
            varis = np.hstack([varis, BoundaryPredictor.clean_nans(exc_soma_depths, normalize=True).reshape(-1, 1)])
            self.column_labels.append("depth")

        return bin_centers, varis, exc_soma_densities

    def _hmm_fit(self, bin_centers, varis):
        """
        fits a hidden markov model to the variables given, which are measured at depths bin_centers
        :param bin_centers: depths of varis in mm
        :param varis: np.array of shape (len(bin_centers), num_features), normalized and free of nans
        :return: model: trained hidden markov model
        """
        model = hmm.GaussianHMM(n_components=6, covariance_type="diag", init_params="", params="mc", n_iter=1)
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
            idxs = (self.default_bounds[i] <= bin_centers) & (bin_centers < self.default_bounds[i + 1])
            if any(idxs):
                model.means_[i, :] = varis[idxs, :].mean(axis=0)
                covars[i, :] = varis[idxs, :].var(axis=0) + 1e-10
            else:
                model.means_[i, :] = 0
                covars[i, :] = 0.1
        model.covars_ = covars

        depth_centers = model.means_[:,
                        -1] if self.use_depth else None  # to help reduce major errors, these will be the means for the depth emissions

        # the package has it's own convergence monitor, but I want fine control over it so I'm doing it manually
        prev_score = 100_000_000_000  # big number
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

    def _hmm_predict(self, model, bin_centers, varis, exc_soma_densities):
        """
        computes the layer boundaries as predicted by model on observation varis, which were observed at depths bin_centers
        :param bin_centers: depths of varis in mm
        :param varis: np.array of shape (len(bin_centers), num_features), normalized and free of nans
        :param exc_soma_densities: np.array of shape (len(bin_centers),) that contains the density of exc somas in mm^-3
        :return: bounds: predicted layer boundaries
                 hmm_layers: hmm prediction of which layer each index participates in (its hidden state)
                 posteriors: how confident the hmm is that each depth belongs to each state
        """
        hmm_layers = model.predict(varis).tolist()

        bounds = []
        for i in range(2, model.n_components - 1):
            idx = hmm_layers.index(i)
            bounds.append((bin_centers[idx] + bin_centers[idx - 1]) / 2)

        # linearly interpolate the outermost bounds based on exc soma density
        l1_2_thresh = 120_000
        l1_2_idx = np.nonzero(exc_soma_densities > l1_2_thresh)[0][0]
        l1_2_bound = bin_centers[l1_2_idx - 1] + (bin_centers[l1_2_idx] - bin_centers[l1_2_idx - 1]) \
                    * (l1_2_thresh - exc_soma_densities[l1_2_idx - 1]) / (
                                exc_soma_densities[l1_2_idx] - exc_soma_densities[l1_2_idx - 1])
        l6_wm_thresh = 50_000
        l6_wm_idx = np.nonzero(exc_soma_densities > l6_wm_thresh)[0][-1]
        l6_wm_bound = bin_centers[l6_wm_idx] + (bin_centers[l6_wm_idx + 1] - bin_centers[l6_wm_idx]) \
                    * (l6_wm_thresh - exc_soma_densities[l6_wm_idx]) / (
                                exc_soma_densities[l6_wm_idx + 1] - exc_soma_densities[l6_wm_idx])
        bounds = [l1_2_bound] + bounds + [l6_wm_bound]
        bounds = np.array(bounds)
        self.default_bounds[1:-1] = bounds

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
        :param ngrdipts: (tuple of int) the x and z dimensions of the grid of cortical columns.
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
        :param smoothness: radius of the gaussian
        """
        spatial_array = LayerPrediction.to_spatial_array(pred.bounds, pred.ngridpts)
        kernel_gridx, kernel_gridz = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
        kernel = np.exp(-(kernel_gridx ** 2 + kernel_gridz ** 2) / smoothness)
        kernel /= np.sum(kernel)
        smoothed_spatial_bounds = np.empty(spatial_array.shape)
        for i in range(5):
            smoothed_spatial_bounds[:, :, i] = convolve2d(spatial_array[:, :, i], kernel, boundary="symm", mode="same")
        pred.bounds = LayerPrediction.to_snaking_array(smoothed_spatial_bounds)


class LayerClassifier:
    """
    Takes the resulting layer borders from LayerPredictor, cleans them up,
    and classifies which layer a given point is in
    """
    LAYER_NAMES = np.array(["L1", "L23", "L4", "L5", "L6", "WM"], dtype=object)

    def __init__(self, data_path):
        """
        :param data_path: path to the pickled LayerPrediction object. Currently only simple predictions are supported
                        as opposed to predictions generated using LayerPredictor.predict_with_sensitivity()
        """
        self.data_path = data_path
        self.pred = LayerPrediction.load_pickle(data_path)
        if self.pred.bounds.ndim != 2:
            raise ValueError("""Currently only simple predictions are supported
                        as opposed to predictions generated using LayerPredictor.predict_with_sensitivity()""")
        self.spatial_array = LayerPrediction.to_spatial_array(self.pred.bounds, self.pred.ngridpts)
        self.layer_funcs = []
        for i in range(5):
            self.layer_funcs.append(
                interpolate.interp2d(self.pred.col_center_xs, self.pred.col_center_zs,
                                     self.spatial_array[:, :, i].T, kind="linear"))

    def predict(self, pts):
        """
        returns which layer each of the given points belongs to
        :param pts: (np.array of shape (n, 3)) nanometer points to classify the layer of
        :return:
        """
        if any(pts[:, 0] > self.pred.overall_bbox[1][0] * 1000 + 50_000) \
                or any(pts[:, 0] < self.pred.overall_bbox[0][0] * 1000 - 50_000) \
                or any(pts[:, 2] > self.pred.overall_bbox[1][2] * 1000 + 50_000) \
                or any(pts[:, 2] < self.pred.overall_bbox[0][2] * 1000 - 50_000):  # um
            warnings.warn(
                """Some points are more than 50 microns laterally from the region where the predictions were made.
                The will be mapped to the nearest known point.""")

        results = np.zeros((pts.shape[0],), dtype=object)
        bounds = np.zeros((pts.shape[0], len(self.layer_funcs) + 2), dtype=np.float64)
        bounds[:, 0] = -2 ** 30
        for i in range(len(self.layer_funcs)):
            for j in range(bounds.shape[0]):
                bounds[j, i + 1] = self.layer_funcs[i](pts[j, 0] / 1_000, pts[j, 2] / 1_000)[0] * 1_000_000
        bounds[:, -1] = 2 ** 30
        for i in range(bounds.shape[1] - 1):
            results[(bounds[:, i] < pts[:, 1]) & (pts[:, 1] <= bounds[:, i + 1])] = LayerClassifier.LAYER_NAMES[i]
        return results


if __name__ == "__main__":
    # minnie_col = np.array([[672444., 200000., 805320.], [772444., 1294000., 905320.]])  # nm
    resolution = np.array([4., 4., 40.])
    # conservative bbox only containing well-segmented areas
    seg_low_um = np.array([130_000, 50_000, 15_000]) * resolution / 1_000
    seg_up_um = np.array([355_000, 323_500, 27_500]) * resolution / 1_000
    name = "small_test"
    p = BoundaryPredictor(features=("soma_volume",), num_PCA=None, use_depth=False, use_soma_vol_std=True,
                          resolution=resolution, save_figs=True, verbose=True,
                          name=name)
    # bboxs = [minnie_col + i * np.array([25_000, 0, 0]) for i in range(27)]  # move along x
    pred = p.predict(np.array([seg_low_um, seg_up_um]), col_size=(100, 100), ngridpts=(2, 2))
    # bounds = p.predict_with_sensitivity(cols_nm, ntrials=8, noise_scale=0.02)
    print(pred)

    pred = LayerPrediction.load_pickle(f"{name}_prediction.pkl")
    c = LayerClassifier(data_path=f"{name}_prediction.pkl")
    inpt = np.array(pred.cols_nm)[0:2, 0].copy()
    c.predict(inpt)
