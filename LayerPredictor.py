import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from hmmlearn import hmm
from caveclient import CAVEclient


class LayerPredictor:

    def __init__(self, bin_width=0.05,
                 step_size=0.005,
                 resolution=(4., 4., 40.),
                 features=("soma_volume", "n_soma_syn", "nucleus_volume", "nucleus_fract_fold", "sum_proc_sdf"),
                 use_depth=True,
                 use_soma_vol_std=True,
                 num_PCA=None,
                 verbose=False):
        """
        :param bin_width (float in mm) How wide a bin should be used for calculating information by depth
        :param step_size (float in mm) How large of steps should be taken in the depth direction. Recommend <= bin_width
        :param resolution (tuple of floats) voxel resolution of the dataset
        :param features (tuple of Strings) are columns of the soma/nuc features table to use for calculation
        :param use_depth determines whether depth should be a features
        :param use_soma_vol_std determines whether the standard deviation of soma volume at each depth should additionally
               be used as a feature
        :param num_PCA (None or int)  indicates how many PCA modes should be used. Default: None, which indicates the raw features should
               be used without performing PCA
        :param verbose dictates whether printing and plotting occurs
        """
        self.bin_width = bin_width
        self.step_size = step_size
        self.resolution = resolution
        self.features = features
        self.use_depth = use_depth
        self.use_soma_vol_std = use_soma_vol_std
        self.num_PCA = num_PCA
        self.verbose = verbose

    def predict(self, bboxs):
        """
        Calculates the requested features at each depth and uses them to predict the boundaries between cortical layers
        for the provided regions
        :param bboxs (list of 2x3 np.arrays of floats) the nm coordinates of the minimum corner and maximum corner of
                the region of interest. Recommend a region 100_000 nm x full depth x 100_000 nm.
                A separate analysis will be performed on each bbox
        """
        # bounding box of the proofread column we're interested in
        self.resolution = np.array(self.resolution)

        # Gather data
        if self.verbose:
            print("connecting to server... ", end="")
        datastack_name = "minnie65_phase3_v1"
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
        self.soma_features = pd.read_pickle("Minnie_soma_nuc_feature_model_83_1.pkl")  # TODO this isn't available to everyone
        if self.verbose:
            print("success.")


        self.soma_features["seg_id"] = [nuc_to_root.loc[n].pt_root_id for n in self.soma_features.nuc_id]
        self.soma_features.index = self.soma_features.seg_id

        # size is equal to nucleus volume
        # soma area and nucleus area closely track their respective volumes
        # avg sdf is a list of the 'diameters' of processes (e.g. dendrites) that leave each cell body

        results = []
        for b in bboxs:
            bbox = b / self.resolution
            if self.verbose:
                print("\nWORKING ON", bbox)
            results.append(self._predict_col(bbox))
        return results

    def _predict_col(self, bbox):
        """
        makes layer boundary predictions for the particular column provided by bbox
        :param bbox: bounding box of column
        :return: bounds: the predicted layer bounds
        """
        soma_features_root_ids = set(self.soma_features.seg_id)

        auto_col_cells = self.auto_cells[self.auto_cells.pt_position.apply(LayerPredictor.in_bbox, args=[bbox])].copy()
        auto_col_cells["mm_depth"] = [auto_col_cells.pt_position.iloc[i][1] * self.resolution[1] / 1_000_000 for i in
                                      range(len(auto_col_cells))]
        # add soma features columns to auto_col_cells
        for feature in self.features:
            auto_col_cells[feature] = [
                (self.soma_features.loc[r][feature] if r in soma_features_root_ids and
                type(self.soma_features.loc[r][feature]) is not pd.Series else np.nan)
                for r in auto_col_cells.pt_root_id]

        # weird case where someone only wants the std but not the mean of cell size
        if self.use_soma_vol_std and "soma_volume" not in self.features:
            auto_col_cells["soma_volume"] = [
                (self.soma_features.loc[r]["soma_volume"] if r in soma_features_root_ids and
                type(self.soma_features.loc[r]["soma_volume"]) is not pd.Series else np.nan)
                for r in auto_col_cells.pt_root_id]

        auto_exc_cells = auto_col_cells.query("classification_system == 'aibs_coarse_excitatory'")

        bin_centers, varis = self._calculate_features(bbox, auto_exc_cells)

        model = self._hmm_fit(bin_centers, varis)

        bounds, layers, posteriors = self._hmm_predict(model, bin_centers, varis)

        model_means = np.array([model.means_[l] for l in layers])
        model_stds = np.array([np.diagonal(np.sqrt(model.covars_[l])) for l in layers])

        if self.verbose:
            # Plot model results!

            colors = get_cmap("tab20").colors[::2] + get_cmap("tab20b").colors[::2]

            fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=130)
            ax2 = ax.twinx()

            for i in range(varis.shape[1]):
                c = colors[i % len(colors)]
                ax2.plot(bin_centers, varis[:, i], color=c, label=f"variable {i}")
                ax2.plot(bin_centers, model_means[:, i], linestyle="-.", color=c)
                ax2.fill_between(bin_centers, model_means[:, i] - model_stds[:, i],
                                 model_means[:, i] + model_stds[:, i],
                                 edgecolor="none", facecolor=c, alpha=0.2)

            # for i, f in enumerate(features):
            #     c = colors[i % len(colors)]
            #     ax2.plot(bin_centers, exc_soma_features_by_depth[f] / np.nanmax(exc_soma_features_by_depth[f]), linestyle="-", color=c, label="exc " + f)
            #     ax2.plot(bin_centers, model_means[:, i + 2], linestyle="-.", color=c, label="exc "+ f + " model mean")
            #     ax2.fill_between(bin_centers, model_means[:, i+2] - model_stds[:, i+2], model_means[:, i+2] + model_stds[:, i+2], edgecolor="none", facecolor=c, alpha=0.2)
            #     ax2.plot(bin_centers, normalized_smooth_exc_features[f], linestyle=":", label="exc fit " + f)

            labels = np.array([107000, 147000, 184000, 224000, 265000]) * self.resolution[
                1] / 1_000_000  # manual labels minnie65 col
            ax.axvline(labels[0], linestyle="--", color="k", label="manual")
            for lab in labels[1:]:
                ax.axvline(lab, linestyle="--", color="k")
            ax.axvline(bounds[0], linestyle="--", color="blue", label="automatic")
            for bound in bounds[1:]:
                ax.axvline(bound, linestyle="--", color="blue")
            ax.plot(bin_centers, posteriors)
            ax.legend(bbox_to_anchor=[1.1, 1])
            ax2.legend(bbox_to_anchor=[1.1, 0.6])
            ax.set_xlabel("depth ($mm$)")
            ax.set_ylim([0, 3])
            # ax.set_ylabel("soma density (per $mm^{3}$)")
            # ax2.set_ylabel("synapse density (per $mm^3$)")
            plt.show()
            fig.savefig(f"{bbox[0, 0]}x.svg")

        return bounds

    def _calculate_features(self, bbox, auto_exc_cells):
        """
        calculates the features used for the HMM
        :param bbox: bbox of column
        :param auto_exc_cells: df of excitatory cells in column with features attached
        :return bin_centers: the 1D array of mm depths at which the features were calculated
                varis: the 2D array of features to be used for the HMM
        """
        # cross sectional area to be layered, in mm^2
        xarea = self.resolution[0] * self.resolution[2] * (bbox[1][0] - bbox[0][0]) * (bbox[1][2] - bbox[0][2]) / 1_000_000 ** 2

        # min is pia border (with L1) and max is white matter border (with L6)
        min_y = np.min(auto_exc_cells.mm_depth.values)
        max_y = np.max(auto_exc_cells.mm_depth.values)

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
            cutoff_idx = LayerPredictor.get_cutoff_idx(auto_exc_cells, curr_y + self.bin_width, prev_cutoff_idx)
            lower_cutoff_idx = LayerPredictor.get_cutoff_idx(auto_exc_cells, curr_y, prev_cutoff_idx)

            current_exc_cells = auto_exc_cells.iloc[lower_cutoff_idx:cutoff_idx]

            for f in self.features:
                exc_soma_features_by_depth[f].append(current_exc_cells[f].mean())
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
            exc_features_df[col] = LayerPredictor.clean_nans(exc_features_df[col], normalize=True)

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
                plt.show()

                plt.plot(range(1, len(explained_variance) + 1), explained_variance)
                plt.xlabel("num principle components")
                plt.ylabel("explained variance")
                plt.ylim([0, 1])
                plt.show()

            # Yc is the projection of Xc onto the principal components
            Yc = V[:, :self.num_PCA].T @ Xc

        varis = exc_features_df.values if self.num_PCA is None else Yc.T
        if self.use_depth:
            # this is here because depth shouldn't go into PCA
            varis = np.hstack([varis, LayerPredictor.clean_nans(exc_soma_depths, normalize=True).reshape(-1, 1)])

        return bin_centers, varis

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
        default_bounds = np.array(
            [0.3, 0.400516, 0.555516, 0.700516, 0.830516, 1.010516, 1.1])  # from HMM trained on 2 PCA modes in column
        model.means_ = np.zeros((model.n_components, nf))
        covars = np.ones((model.n_components, nf))
        for i in range(model.n_components):
            idxs = (default_bounds[i] <= bin_centers) & (bin_centers < default_bounds[i + 1])
            if any(idxs):
                model.means_[i, :] = varis[idxs, :].mean(axis=0)
                covars[i, :] = varis[idxs, :].var(axis=0) + 1e-10
            else:
                model.means_[i, :] = 0
                covars[i, :] = 0.1
        model.covars_ = covars

        depth_centers = model.means_[:, -1] if self.use_depth else None  # to help reduce major errors, these will be the means for the depth emissions

        # the package has it's own convergence monitor, but I want fine control over it so I'm doing it manually
        prev_score = -1
        score = model.score(varis)
        max_iters = 100
        tol = 0.001
        i = 0
        while abs(score - prev_score) > tol and i < max_iters:
            model.fit(varis)
            prev_score = score
            score = model.score(varis)
            print(score)
            i += 1
            # TODO: I should instead not use the HMM for this, just a threshold for the outer borders
            if self.use_depth:
                # each iteration re-fix the "target" mean depth to the center of each default layer
                # so the model can't be terribly wrong
                model.means_[:, -1] = depth_centers
            print("after correction:", model.score(varis))

        if abs(score - prev_score) > tol:
            raise ValueError("Did not converge")

        return model

    def _hmm_predict(self, model, bin_centers, varis):
        """
        computes the layer boundaries as predicted by model on observation varis, which were observed at depths bin_centers
        :return: bounds: predicted layer boundaries
                 layers: prediction of which layer each index participates in (its hidden state)
                 posteriors: how confident the hmm is that each depth belongs to each state
        """
        layers = model.predict(varis).tolist()

        bounds = []
        for i in range(1, model.n_components):
            idx = layers.index(i)
            bounds.append((bin_centers[idx] + bin_centers[idx - 1]) / 2)
        bounds = np.array(bounds)

        posteriors = model.predict_proba(varis)
        if self.verbose:
            print("bounds:", bounds)

        return bounds, layers, posteriors


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


if __name__ == "__main__":
    p = LayerPredictor(num_PCA=None, use_depth=False, verbose=True)
    minnie_col = np.array([[672444., 200000., 805320.], [772444., 1294000., 905320.]])
    bboxs = [minnie_col + i * np.array([50_000, 0, 0]) for i in range(12)]

    # quadrants = [minnie_col.copy() for i in range(4)]
    #
    # midx = (minnie_col[0, 0] + minnie_col[1, 0]) / 2
    # midz = (minnie_col[0, 2] + minnie_col[1, 2]) / 2
    #
    # quadrants[0][0, 0] = midx
    # quadrants[0][0, 2] = midz
    #
    # quadrants[1][1, 0] = midx
    # quadrants[1][0, 2] = midz
    #
    # quadrants[2][1, 0] = midx
    # quadrants[2][1, 2] = midz
    #
    # quadrants[3][0, 0] = midx
    # quadrants[3][1, 2] = midz

    print(p.predict(bboxs))
