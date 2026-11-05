"""Feature engineering compatible with the SGCC LightGBM research model.

The feature calculations mirror SGCC_Final_Model_Preparation.ipynb. The
excluded date is fixed from the training dataset (rather than inferred from
an uploaded batch) so one-row or small uploads cannot change the feature schema.
"""
import warnings
import numpy as np
import pandas as pd

EXCLUDED_DATES = ["10/3/2014"]

def build_feature_matrix(df, raw_date_columns, feature_columns):
    """Return (features, identifiers, optional labels, warnings).

    df must contain all date columns from raw_date_columns. CONS_NO and FLAG
    are optional; missing CONS_NO is replaced with stable row IDs. FLAG, if
    present, is returned only for evaluation and never enters model features.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if df.empty:
        raise ValueError("The uploaded dataset contains no data rows.")
    missing_dates = [c for c in raw_date_columns if c not in df.columns]
    if missing_dates:
        raise ValueError(
            f"CSV is missing {len(missing_dates)} required SGCC daily columns. "
            f"Examples: {missing_dates[:8]}"
        )
    warnings_out = []
    if "CONS_NO" in df.columns:
        ids = df["CONS_NO"].fillna("").astype(str).to_numpy()
    else:
        ids = np.array([f"row_{i+1:06d}" for i in range(len(df))], dtype=object)
        warnings_out.append("CONS_NO was not present; generated row IDs are shown in the output.")

    labels = None
    if "FLAG" in df.columns:
        raw_labels = pd.to_numeric(df["FLAG"], errors="coerce")
        valid = raw_labels.notna() & raw_labels.isin([0, 1])
        if not bool(valid.all()):
            raise ValueError("FLAG must contain only 0/1 labels when supplied. Remove FLAG for prediction-only use.")
        labels = raw_labels.astype(int).to_numpy()

    use_dates = [c for c in raw_date_columns if c not in EXCLUDED_DATES]
    # Same order as training (chronological, as verified from the source CSV).
    dates = pd.to_datetime(use_dates, format="%m/%d/%Y", errors="coerce")
    if dates.isna().any():
        raise ValueError("One or more date headers could not be parsed using the training date format.")
    # Numeric conversion is defensive: unexpected strings become missing values.
    numeric = df[use_dates].apply(pd.to_numeric, errors="coerce")
    bad_numeric = int(numeric.isna().sum().sum())
    original_nonempty = int(df[use_dates].notna().sum().sum())
    if bad_numeric > 0 and original_nonempty > 0:
        warnings_out.append(
            f"{bad_numeric:,} cells are missing or non-numeric after parsing; the model pipeline handles missing values."
        )
    X = numeric.to_numpy(dtype=np.float32, copy=True)
    n, p = X.shape
    missing = np.isnan(X)
    obs = ~missing
    cnt = obs.sum(axis=1).astype(np.float32)
    zero = obs & (X == 0)
    positive = obs & (X > 0)
    feats = {}
    def add(k, v):
        feats[k] = np.asarray(v, dtype=np.float32)
    add('observed_days', cnt)
    add('missing_pct', missing.mean(axis=1) * 100)
    add('zero_pct_observed', np.divide(zero.sum(axis=1), cnt, out=np.zeros(n, dtype=np.float32), where=cnt>0)*100)
    add('zero_days', zero.sum(axis=1))
    add('nonzero_days', positive.sum(axis=1))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        add('mean_daily', np.nanmean(X, axis=1))
        add('std_daily', np.nanstd(X, axis=1))
        mx = np.full(n, np.nan, dtype=np.float32)
        valid_rows = cnt > 0
        mx[valid_rows] = np.nanmax(X[valid_rows], axis=1)
        add('max_daily_log', np.log1p(mx))
    qs = [10, 25, 50, 75, 90, 95, 99]
    quant = {q: np.full(n, np.nan, dtype=np.float32) for q in qs}
    for lo in range(0, n, 1500):
        hi = min(lo+1500, n)
        block = np.sort(X[lo:hi], axis=1)
        counts = cnt[lo:hi].astype(np.int64)
        rr = np.arange(hi-lo)
        for q in qs:
            pos = (counts-1)*(q/100.0)
            left = np.floor(np.maximum(pos, 0)).astype(int)
            right = np.ceil(np.maximum(pos, 0)).astype(int)
            frac = pos-left
            vals = block[rr, left] + (block[rr, right]-block[rr, left])*frac
            vals[counts == 0] = np.nan
            quant[q][lo:hi] = vals
    for q, v in quant.items():
        add('median_daily' if q == 50 else f'q{q}_daily', v)
    wins_mean = np.full(n, np.nan, dtype=np.float32)
    wins_std = np.full(n, np.nan, dtype=np.float32)
    for lo in range(0, n, 1500):
        hi = min(lo+1500, n)
        block = X[lo:hi]
        cap = quant[99][lo:hi, None]
        with np.errstate(invalid='ignore'):
            clipped = np.minimum(block, cap)
            wins_mean[lo:hi] = np.nanmean(clipped, axis=1)
            wins_std[lo:hi] = np.nanstd(clipped, axis=1)
    add('mean_winsor99', wins_mean)
    add('std_winsor99', wins_std)
    add('cv_daily', feats['std_daily']/(feats['mean_daily']+1e-3))
    longz = np.zeros(n, dtype=np.int16)
    curz = np.zeros(n, dtype=np.int16)
    longm = np.zeros(n, dtype=np.int16)
    curm = np.zeros(n, dtype=np.int16)
    for j in range(p):
        curz = np.where(zero[:, j], curz+1, 0)
        longz = np.maximum(longz, curz)
        curm = np.where(missing[:, j], curm+1, 0)
        longm = np.maximum(longm, curm)
    add('longest_zero_streak', longz)
    add('longest_missing_streak', longm)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        def nanmean(a):
            return np.nanmean(a, axis=1)
        windows = {
            'first_30d': (0, min(30, p)),
            'last_30d': (max(0, p-30), p),
            'last_90d': (max(0, p-90), p),
            'last_180d': (max(0, p-180), p),
            'first_half': (0, p//2),
            'second_half': (p//2, p),
            'prior_90d': (max(0, p-120), max(0, p-30)),
        }
        for name, (a, b) in windows.items():
            add(name+'_mean', nanmean(X[:, a:b]))
            denom = obs[:, a:b].sum(axis=1)
            add(name+'_zero_pct', np.divide(zero[:, a:b].sum(axis=1), denom, out=np.zeros(n, dtype=np.float32), where=denom>0)*100)
        add('half_mean_change', feats['second_half_mean']-feats['first_half_mean'])
        add('half_mean_change_ratio', (feats['second_half_mean']-feats['first_half_mean'])/(np.abs(feats['first_half_mean'])+1e-3))
        add('last30_vs_prior90_ratio', (feats['last_30d_mean']-feats['prior_90d_mean'])/(np.abs(feats['prior_90d_mean'])+1e-3))
        for k, a in enumerate(range(0, p, 90), 1):
            add(f'block{k:02d}_mean', nanmean(X[:, a:min(a+90, p)]))
        t = np.arange(p, dtype=np.float64)[None, :]
        valid = obs.astype(np.float64)
        vals = np.nan_to_num(X, nan=0).astype(np.float64)
        cntd = valid.sum(axis=1)
        sumt = (valid*t).sum(axis=1)
        sumx = vals.sum(axis=1)
        sumtx = (vals*t).sum(axis=1)
        sumt2 = (valid*t*t).sum(axis=1)
        den = cntd*sumt2-sumt*sumt
        slope = np.divide(cntd*sumtx-sumt*sumx, den, out=np.zeros(n), where=den != 0)
        add('linear_trend_slope', slope)
        add('linear_trend_slope_scaled', slope/(feats['mean_daily']+1e-3))
        weekdays = np.array([d.weekday() for d in dates])
        wkday = np.isin(weekdays, [0, 1, 2, 3, 4])
        wkend = ~wkday
        add('weekday_mean', nanmean(X[:, wkday]))
        add('weekend_mean', nanmean(X[:, wkend]))
        add('weekend_weekday_ratio', (feats['weekend_mean']-feats['weekday_mean'])/(np.abs(feats['weekday_mean'])+1e-3))
        months = np.array([d.month for d in dates])
        for m in range(1, 13):
            mask = months == m
            if mask.any():
                add(f'month_{m:02d}_mean', nanmean(X[:, mask]))
    F = pd.DataFrame(feats).replace([np.inf, -np.inf], np.nan)
    missing_features = [c for c in feature_columns if c not in F.columns]
    extra_features = [c for c in F.columns if c not in feature_columns]
    if missing_features or extra_features:
        raise ValueError(f"Feature schema mismatch. Missing={missing_features}; unexpected={extra_features}")
    F = F.loc[:, feature_columns]
    return F, ids, labels, warnings_out
