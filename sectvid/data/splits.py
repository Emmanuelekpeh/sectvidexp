"""Splits: held out by clip AND by identity (spec 6.1).

Splits are identity-first: a fraction of identities is held out entirely and
never appears in training. The remaining identities are split into train and
val at clip level. Behavior stratification is applied when behaviors are
labeled; until then clips are unlabeled and stratification is a no-op.
"""
import numpy as np


def curation_check(meta_by_clip):
    """Clips whose identity has NOT been human-reviewed. Splits are
    identity-first (spec 6.1): the default identity_id = clip_id is only safe
    when every clip shows a distinct person, so splits must not be generated
    while this list is non-empty. meta_by_clip: {clip_id: meta dict}."""
    return sorted(c for c, m in meta_by_clip.items() if not m.get("identity_reviewed"))


def split_by_identity(clip_to_identity, identity_holdout_frac, test_frac, seed):
    """Three-way identity split: unseen (never trained on), val, train.
    Identities do not overlap between any two splits."""
    rng = np.random.default_rng(seed)
    identities = sorted(set(clip_to_identity.values()))
    order = np.arange(len(identities))
    rng.shuffle(order)
    n_unseen = int(round(identity_holdout_frac * len(identities)))
    n_val = int(round(test_frac * max(len(identities) - n_unseen, 0)))
    unseen_ids = set(identities[i] for i in order[:n_unseen])
    val_ids = set(identities[i] for i in order[n_unseen:n_unseen + n_val])

    train, val, unseen = [], [], []
    for clip, ident in clip_to_identity.items():
        if ident in unseen_ids:
            unseen.append(clip)
        elif ident in val_ids:
            val.append(clip)
        else:
            train.append(clip)
    return train, val, unseen


def leakage_report(train, val, unseen, clip_to_identity):
    """Any clip or identity overlap between disjoint splits is a failure."""
    def ids(clips):
        return {clip_to_identity[c] for c in clips}

    train_s, val_s, unseen_s = set(train), set(val), set(unseen)
    reports = {
        "train_val_clip_overlap": sorted(train_s & val_s),
        "train_unseen_clip_overlap": sorted(train_s & unseen_s),
        "val_unseen_clip_overlap": sorted(val_s & unseen_s),
        "train_val_identity_overlap": sorted(ids(train_s) & ids(val_s)),
        "train_unseen_identity_overlap": sorted(ids(train_s) & ids(unseen_s)),
        "val_unseen_identity_overlap": sorted(ids(val_s) & ids(unseen_s)),
    }
    reports["clean"] = all(len(v) == 0 for v in reports.values())
    return reports
