import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from hmmlearn import hmm
from caveclient import CAVEclient

def in_bbox(p, bbox):
    """ returns whether point is between the two points given by bbox"""
    lower, upper = bbox
    return lower[0] <= p[0] < upper[0] and lower[1] <= p[1] < upper[1] and lower[2] <= p[2] < upper[2]


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


def predict(bbox,
    bin_width=0.05,
    step_size=0.005,
    resolution=(4., 4., 40.),
    features=("soma_volume", "n_soma_syn", "nucleus_volume", "nucleus_fract_fold", "sum_proc_sdf"),
    use_depth=True,
    use_soma_vol_std=True,
    num_PCA=None):
    """
    Calculates the requested features at each depth and uses them to predict the boundaries between cortical layers for
    the provided region
    :param bbox (2x3 np.array of floats) the nm coordinates of the minimum corner and maximum corner of the region of
           interest. Recommend a region 100_000 nm x full depth x 100_000 nm.
    :param bin_width (float in mm) How wide a bin should be used for calculating information by depth
    :param step_size (float in mm) How large of steps should be taken in the depth direction. Recommend <= bin_width
    :param resolution (tuple of floats) voxel resolution of the dataset
    :param features (tuple of Strings) are columns of the soma/nuc features table to use for calculation
    :param use_depth determines whether depth should be a features
    :param use_soma_vol_std determines whether the standard deviation of soma volume at each depth should additionally
           be used as a feature
    :param num_PCA (None or int)  indicates how many PCA modes should be used. Default: None, which indicates the raw features should
           be used without performing PCA
    """
    # bounding box of the proofread column we're interested in
    resolution = np.array(resolution)
    y_resolution = resolution[1]
    bbox /= resolution
    labels = np.array([107000, 147000, 184000, 224000, 265000]) * y_resolution / 1_000_000  # manual labels minnie65 col

    # Gather data

    datastack_name = "minnie65_phase3_v1"
    client = CAVEclient(datastack_name)

    all_cells = client.materialize.query_table("allen_soma_coarse_cell_class_model_v1")
    all_cells["num_soma"] = all_cells.groupby("pt_root_id").transform("count")["valid"]
    cells_by_root = all_cells.copy()
    cells_by_root.index = cells_by_root.pt_root_id

    valid_ids = set(all_cells.query("num_soma == 1").pt_root_id)

    soma_features = pd.read_pickle("Minnie_soma_nuc_feature_model_83_1.pkl")  # TODO this isn't available to everyone

    nuc_to_root = client.materialize.query_table("nucleus_neuron_svm")

    nuc_to_root.index = nuc_to_root.id

    soma_features["seg_id"] = [nuc_to_root.loc[n].pt_root_id for n in soma_features.nuc_id]
    soma_features.index = soma_features.seg_id

    # size is equal to nucleus volume
    # soma area and nucleus area closely track their respective volumes
    # avg sdf is a list of the 'diameters' of processes (e.g. dendrites) that leave each cell body

    soma_features_root_ids = set(soma_features.seg_id)

    auto_cells = client.materialize.query_table("allen_soma_coarse_cell_class_model_v1")

    auto_col_cells = auto_cells[auto_cells.pt_position.apply(in_bbox, args=[bbox])].copy()
    auto_col_cells["mm_depth"] = [auto_col_cells.pt_position.iloc[i][1] * 4 / 1_000_000 for i in
                                  range(len(auto_col_cells))]
    # add soma features columns to auto_col_cells
    for feature in features:
        auto_col_cells[feature] = [
            (soma_features.loc[r][feature] if r in soma_features_root_ids and
            type(soma_features.loc[r][feature]) is not pd.Series else np.nan)
            for r in auto_col_cells.pt_root_id]

    # weird case where someone only wants the std but not the mean of cell size
    if use_soma_vol_std and "soma_volume" not in features:
        auto_col_cells["soma_volume"] = [
            (soma_features.loc[r]["soma_volume"] if r in soma_features_root_ids and
            type(soma_features.loc[r]["soma_volume"]) is not pd.Series else np.nan)
            for r in auto_col_cells.pt_root_id]

    auto_exc_cells = auto_col_cells.query("classification_system == 'aibs_coarse_excitatory'")

    # ### CALCULATE FEATURES ###

    # cross sectional area to be layered, in mm^2
    xarea = resolution[0] * resolution[2] * (bbox[1][0] - bbox[0][0]) * (bbox[1][2] - bbox[0][2]) / 1_000_000 ** 2

    # min is pia border (with L1) and max is white matter border (with L6)
    min_y = np.min(auto_exc_cells.mm_depth.values)
    max_y = np.max(auto_exc_cells.mm_depth.values)

    auto_exc_cells = auto_exc_cells.sort_values(axis="index", by="mm_depth")

    exc_soma_densities = []
    exc_soma_features_by_depth = dict((f, []) for f in features)
    exc_soma_vol_std_by_depth = []
    exc_soma_depths = []

    depths = np.arange(min_y, max_y, step_size)
    bin_centers = depths + step_size // 2
    prev_cutoff_idx = 0
    for curr_y in depths:
        # first index where pt_position[1] is greater than curr_y + bin_width
        cutoff_idx = get_cutoff_idx(auto_exc_cells, curr_y + bin_width, prev_cutoff_idx)
        lower_cutoff_idx = get_cutoff_idx(auto_exc_cells, curr_y, prev_cutoff_idx)

        current_exc_cells = auto_exc_cells.iloc[lower_cutoff_idx:cutoff_idx]

        for f in features:
            exc_soma_features_by_depth[f].append(current_exc_cells[f].mean())
        exc_soma_vol_std_by_depth.append(current_exc_cells["soma_volume"].std())
        exc_soma_depths.append(current_exc_cells["mm_depth"].mean())

        exc_soma_densities.append(len(current_exc_cells))

        prev_cutoff_idx = lower_cutoff_idx

    exc_soma_densities = np.array(exc_soma_densities, dtype=float) / (bin_width * xarea)  # per mm^3

    exc_features_df = pd.DataFrame(exc_soma_features_by_depth)
    exc_features_df["soma_density"] = exc_soma_densities
    if use_soma_vol_std:
        exc_features_df["soma_vol_std"] = exc_soma_vol_std_by_depth
    for col in exc_features_df.columns:
        exc_features_df[col] = clean_nans(exc_features_df[col], normalize=True)

    ### PCA on the features to remove correlations
    if num_PCA is not None:
        X = exc_features_df.values.T
        Xc = X - np.mean(X, axis=1, keepdims=True)  # mean subtract data first
        cov = Xc @ Xc.T / (X.shape[0] - 1)  # covariance
        plt.imshow(cov)
        plt.colorbar()
        plt.title("covariance matrix")
        plt.ylabel(" ".join(list(exc_features_df.columns)))
        plt.show()

        variance, V = np.linalg.eig(cov)
        idxs = np.argsort(-variance)
        V = V[:, idxs]
        variance = variance[idxs]
        explained_variance = [sum(variance[:i + 1]) / sum(variance) for i in range(len(variance))]

        plt.plot(explained_variance)
        plt.xlabel("num principle components")
        plt.ylabel("explained variance")
        plt.show()

        # Yc is the projection of Xc onto the principal components
        Yc = V[:, :num_PCA].T @ Xc

    colors = get_cmap("tab20").colors[::2] + get_cmap("tab20b").colors[::2]

    model = hmm.GaussianHMM(n_components=6, covariance_type="diag", init_params="", params="mc", n_iter=1)
    # the model starts in the first state, and there is 0 probability of starting elsewhere
    model.startprob_ = np.zeros(model.n_components)
    model.startprob_[0] = 1
    p = 0.04
    transition_probs = np.full(model.n_components - 1, p)
    stay_probs = np.full(model.n_components, 1 - p)
    stay_probs[-1] = 1  # nothing deeper than white matter
    model.transmat_ = np.diag(stay_probs) + np.diag(transition_probs, k=1)

    varis = exc_features_df.values if num_PCA is None else Yc.T
    if use_depth:
        # this is here because depth shouldn't go into PCA
        varis = np.hstack([varis, clean_nans(exc_soma_depths, normalize=True).reshape(-1, 1)])

    # initialize means and variances
    nf = varis.shape[1]
    default_bounds = np.array(
        [0.3, 0.400516, 0.555516, 0.700516, 0.830516, 1.010516, 1.1])  # from HMM trained on 2 PCA modes in column
    model.means_ = np.zeros((model.n_components, nf))
    covars = np.ones((model.n_components, nf))
    for i in range(model.n_components):
        idxs = (default_bounds[i] <= bin_centers) & (bin_centers < default_bounds[i + 1])
        model.means_[i, :] = varis[idxs, :].mean(axis=0)
        covars[i, :] = varis[idxs, :].var(axis=0) + 1e-10
    model.covars_ = covars

    depth_centers = model.means_[:, -1] if use_depth else None  # to help reduce major errors, these will be the means for the depth emissions

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
        if use_depth:
            # each iteration re-fix the "target" mean depth to the center of each default layer
            # so the model can't be terribly wrong
            model.means_[:, -1] = depth_centers
        print("after correction:", model.score(varis))

    if abs(score - prev_score) > tol:
        raise ValueError("Did not converge")

    layers = model.predict(varis).tolist()

    bounds = []
    for i in range(1, model.n_components):
        idx = layers.index(i)
        bounds.append((bin_centers[idx] + bin_centers[idx - 1]) / 2)
    bounds = np.array(bounds)

    model_means = np.array([model.means_[l] for l in layers])
    model_stds = np.array([np.diagonal(np.sqrt(model.covars_[l])) for l in layers])

    posteriors = model.predict_proba(varis)
    print(bounds)

    # Plot model results!

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=130)
    ax2 = ax.twinx()

    for i in range(varis.shape[1]):
        c = colors[i % len(colors)]
        ax2.plot(bin_centers, varis[:, i], color=c, label=f"variable {i}")
        ax2.plot(bin_centers, model_means[:, i], linestyle="-.", color=c)
        ax2.fill_between(bin_centers, model_means[:, i] - model_stds[:, i], model_means[:, i] + model_stds[:, i],
                         edgecolor="none", facecolor=c, alpha=0.2)

    # for i, f in enumerate(features):
    #     c = colors[i % len(colors)]
    #     ax2.plot(bin_centers, exc_soma_features_by_depth[f] / np.nanmax(exc_soma_features_by_depth[f]), linestyle="-", color=c, label="exc " + f)
    #     ax2.plot(bin_centers, model_means[:, i + 2], linestyle="-.", color=c, label="exc "+ f + " model mean")
    #     ax2.fill_between(bin_centers, model_means[:, i+2] - model_stds[:, i+2], model_means[:, i+2] + model_stds[:, i+2], edgecolor="none", facecolor=c, alpha=0.2)
    #     ax2.plot(bin_centers, normalized_smooth_exc_features[f], linestyle=":", label="exc fit " + f)

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

if __name__ == "__main__":
    predict(np.array([[ 672444., 200000., 805320.], [ 772444., 1294000., 905320.]]) + np.array([000_000, 0, 0]),
            num_PCA=3)