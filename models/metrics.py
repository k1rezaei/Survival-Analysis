import numpy as np
import prepare_data
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc, concordance_index_censored, \
    concordance_index_ipcw


def c_index_censored(model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                     y_test: np.ndarray):
    event_indicator_train, event_time_train, event_indicator_test, event_time_test = \
        prepare_data.get_event_indicator_time(y_train, y_test)
    risk_score_test = model.predict(X_test)
    risk_score_train = model.predict(X_train)

    # print(list(zip(risk_score_train[:20], event_time_train[:20], event_indicator_train[:20])))
    return concordance_index_censored(event_indicator_train, event_time_train, risk_score_train)[0], \
           concordance_index_censored(event_indicator_test, event_time_test, risk_score_test)[0]


def c_index_icpw(model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                 y_test: np.ndarray):
    risk_score_test = model.predict(X_test)
    risk_score_train = model.predict(X_train)
    return concordance_index_ipcw(y_train, y_train, risk_score_train)[0], \
           concordance_index_ipcw(y_train, y_test, risk_score_test)[0]


def i_brier_score(model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                  y_test: np.ndarray, times: np.ndarray):
    survs_test = model.predict_survival_function(X_test)
    preds_test = np.asarray([[fn(t) for t in times] for fn in survs_test])

    survs_train = model.predict_survival_function(X_train)
    preds_train = np.asarray([[fn(t) for t in times] for fn in survs_train])

    return integrated_brier_score(y_train, y_train, preds_train, times), \
           integrated_brier_score(y_train, y_test, preds_test, times)


def c_auc_score(model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                y_test: np.ndarray, times: np.ndarray):
    model_chf_funcs_test = model.predict_cumulative_hazard_function(X_test)
    model_chf_funcs_train = model.predict_cumulative_hazard_function(X_train)

    risk_scores_test = np.row_stack([chf(times) for chf in model_chf_funcs_test])
    risk_scores_train = np.row_stack([chf(times) for chf in model_chf_funcs_train])

    return cumulative_dynamic_auc(y_train, y_train, risk_scores_train, times), \
           cumulative_dynamic_auc(y_train, y_test, risk_scores_test, times)
    # score on each point for train, mean for train, score on each point for test, mean for test
