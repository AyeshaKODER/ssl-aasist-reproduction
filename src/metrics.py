"""
EER and min t-DCF, Section 6.1:

  "We use two evaluation metrics: the Equal Error Rate (EER) [64] and the
  Minimum Tandem Detection Cost Function (min t-DCF) [65]."

"""
import numpy as np


def compute_eer(bonafide_scores, spoof_scores):
    """Equal Error Rate: threshold where false-accept rate == false-reject
    rate. bonafide/spoof scores: higher score => more likely bonafide."""
    scores = np.concatenate([bonafide_scores, spoof_scores])
    labels = np.concatenate([np.ones(len(bonafide_scores)),
                              np.zeros(len(spoof_scores))])
    thresholds = np.unique(scores)
    order = np.argsort(thresholds)
    thresholds = thresholds[order]

    frr = []  # false reject rate (bonafide rejected)
    far = []  # false accept rate (spoof accepted)
    for th in thresholds:
        pred_bonafide = scores >= th
        frr.append(np.mean(~pred_bonafide[labels == 1]) if (labels == 1).any() else 0.0)
        far.append(np.mean(pred_bonafide[labels == 0]) if (labels == 0).any() else 0.0)
    frr = np.array(frr)
    far = np.array(far)

    diff = frr - far
    idx = np.argmin(np.abs(diff))
    eer = (frr[idx] + far[idx]) / 2.0
    eer_threshold = thresholds[idx]
    return eer * 100.0, eer_threshold  # as a percentage, matching Table 2/3


def compute_min_tdcf(cm_bonafide, cm_spoof, asv_bonafide=None, asv_spoof_impostor=None,
                      p_target=0.05, c_miss=1.0, c_fa=1.0,
                      c_miss_asv=1.0, c_fa_asv=10.0, p_miss_asv=0.05, p_fa_asv=0.05):
    """Minimum tandem detection cost function, following the tandem-DCF
    formulation (Kinnunen et al. 2020, cited as [65]).

    If ASV scores aren't supplied, uses fixed nominal ASV operating-point
    error rates (p_miss_asv/p_fa_asv defaults) as a stand-in — pass real ASV
    scores from an ASV system evaluated on the same trials for an accurate
    number; the official scoring toolkit computes these from the ASV system's
    own score file.
    """
    cm_scores = np.concatenate([cm_bonafide, cm_spoof])
    cm_labels = np.concatenate([np.ones(len(cm_bonafide)), np.zeros(len(cm_spoof))])
    thresholds = np.unique(cm_scores)

    c0 = c_miss_asv * p_miss_asv * p_target + c_fa_asv * p_fa_asv * (1 - p_target)
    c1 = c_miss * p_target
    c2 = c_fa * (1 - p_target)

    best_tdcf = np.inf
    for th in thresholds:
        pred = cm_scores >= th
        p_miss_cm = np.mean(~pred[cm_labels == 1]) if (cm_labels == 1).any() else 0.0
        p_fa_cm = np.mean(pred[cm_labels == 0]) if (cm_labels == 0).any() else 0.0
        tdcf = c0 + c1 * p_miss_cm + c2 * p_fa_cm
        best_tdcf = min(best_tdcf, tdcf)

    # Normalise, per convention, by the cost of a naive "always accept"/
    # "always reject" system
    tdcf_default = min(c1, c2) + c0
    return best_tdcf / tdcf_default if tdcf_default > 0 else best_tdcf
