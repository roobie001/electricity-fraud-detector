import json, warnings
from pathlib import Path
import joblib
import pandas as pd
from feature_engineering import build_feature_matrix
ROOT=Path(__file__).resolve().parent
model=joblib.load(ROOT/'lightgbm_sgcc_research.joblib')
meta=json.loads((ROOT/'model_metadata.json').read_text())
dates=json.loads((ROOT/'raw_date_columns.json').read_text())
config=json.loads((ROOT/'app_config.json').read_text())
assert model.n_features_in_ == len(meta['feature_columns']) == 66
assert 'FLAG' not in meta['feature_columns'] and 'CONS_NO' not in meta['feature_columns']
assert 0 < config['nested_cv_median_threshold'] < 1
assert 0 < config['bundle_exploratory_oof_threshold'] < 1
assert list(model.feature_name_) == meta['feature_columns']
raw=pd.read_csv(ROOT/'sample_sgcc_input.csv', low_memory=False)
features, ids, labels, warnings_out=build_feature_matrix(raw, dates, meta['feature_columns'])
assert features.shape == (25,66), features.shape
scores=model.predict_proba(features)[:,1]
assert len(scores)==25 and scores.min()>=0 and scores.max()<=1
assert bool(pd.Series(scores).notna().all())
try:
    build_feature_matrix(pd.DataFrame(), dates, meta['feature_columns'])
except ValueError as exc:
    assert 'no data rows' in str(exc)
else:
    raise AssertionError('Empty input should be rejected clearly.')
print('MODEL_TYPE:',type(model).__name__)
print('FEATURES:',features.shape)
print('RAW_DATE_COLUMNS:',len(dates))
print('SAMPLE_ROWS:',len(raw))
print('SCORE_RANGE:',float(scores.min()),float(scores.max()))
print('FIRST_5_SCORES:',[round(float(x),6) for x in scores[:5]])
print('WARNINGS:',warnings_out)
print('SCHEMA_AND_EMPTY_INPUT_CHECKS: PASS')
