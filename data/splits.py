"""City-level splits. Kept separate from preprocess so the SAME preprocessing
code runs for the dev 11/3 split, the 5-fold city-grouped CV, and the final
14-city fit. Splits never leak into transform fitting."""
import numpy as np

# 14 labelled OSCD train cities (the only ones with ground truth).
TRAIN_CITIES = ["aguasclaras","bercy","bordeaux","nantes","paris","rennes","saclay_e",
                "abudhabi","cupertino","pisa","beihai","hongkong","beirut","mumbai"]

# 10 OSCD test cities. Their labels are published separately by OSCD but were
# never downloaded or used in this project, so nothing here is trained, tuned or
# selected on them; final mode only predicts over them.
TEST_CITIES = ["brasilia","montpellier","norcia","rio","saclay_w","valencia",
               "dubai","lasvegas","milano","chongqing"]

# Fixed dev split for smoke tests / engineering (Europe + US + Asia in val).
_DEV_VAL = ["paris", "cupertino", "beihai"]

def get_dev_split():
    """(train_cities, val_cities) — fixed 11/3 split for pipeline smoke tests."""
    val = list(_DEV_VAL)
    train = [c for c in TRAIN_CITIES if c not in val]
    return train, val

def get_grouped_folds(n_splits=5, seed=0):
    """Deterministic n-fold *city-grouped* CV over the 14 train cities (NOT
    leave-one-region-out; true LORO would be 14 folds of 1 held-out city each).
    Returns [(train_cities, val_cities), ...]; every city is a val city once."""
    rng = np.random.RandomState(seed)
    order = list(TRAIN_CITIES)
    rng.shuffle(order)
    groups = [order[i::n_splits] for i in range(n_splits)]   # round-robin buckets
    folds = []
    for g in groups:
        val = list(g)
        train = [c for c in TRAIN_CITIES if c not in val]
        folds.append((train, val))
    return folds
