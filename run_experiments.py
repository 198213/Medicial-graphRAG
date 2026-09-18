# run_experiments.py
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, gamma, multivariate_t, norm, lognorm, wilcoxon
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import QuantileRegressor
from sklearn.covariance import graphical_lasso
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ================== 全局配置 ==================
P = 500
N_NORMAL = 1000
N_ANOM = 100
N_TRAIN = 800
M_LOCAL = 50
ALPHA_FIXED = 0.01
TAU_PERCENTILE = 90

REPEATS = 5                 # 主实验重复次数，调试时可改 5   100
PARAM_SENS_REPEATS = 3       # 参数敏感性重复次数，调试时可改 3   20
BASE_SEED = 20250000
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "tables"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)

DIST_LIST = ["lognormal", "gamma"]
DIST_NAMES = {"lognormal": "对数正态", "gamma": "伽马"}
ANOMALY_TYPES = ["mean_shift", "cor_break", "nonlinear"]
ANOMALY_NAMES = {
    "mean_shift": "均值漂移",
    "cor_break": "相关结构破坏",
    "nonlinear": "非线性异常",
}
BASELINE_NAMES = ["MRCD", "Spatial Rank", "ROBPCA", "Huber", "Graphical LASSO", "Quantile Lasso"]
METHOD_ORDER = ["本文方法"] + BASELINE_NAMES

# ================== 稳健标准化与空间中心 ==================
def robust_standardize(X, med=None, mad=None):
    if med is None:
        med = np.median(X, axis=0)
    if mad is None:
        mad = np.median(np.abs(X - med), axis=0)
        mad[mad < 1e-6] = 1e-6
    return (X - med) / mad, med, mad

def spatial_median(X):
    n, d = X.shape
    def loss(m):
        return np.sum(np.linalg.norm(X - m.reshape(1, d), axis=1))
    init = np.median(X, axis=0)
    res = minimize(loss, init, method="L-BFGS-B", tol=1e-6)
    return res.x

# ================== 评价指标 ==================
def compute_metrics(scores, y_true, lcl, ucl):
    pred = ((scores < lcl) | (scores > ucl)).astype(int)
    tp = np.sum((pred == 1) & (y_true == 1))
    tn = np.sum((pred == 0) & (y_true == 0))
    fp = np.sum((pred == 1) & (y_true == 0))
    fn = np.sum((pred == 0) & (y_true == 1))
    alpha = fp / (tn + fp) if (tn + fp) > 0 else 0
    beta = fn / (tp + fn) if (tp + fn) > 0 else 0
    arl0 = 1 / alpha if alpha > 0 else 9999
    arl1 = 1 / (1 - beta) if (1 - beta) > 0 else 9999
    acc = (tp + tn) / len(y_true)
    auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.5
    f1 = f1_score(y_true, pred)
    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    return {
        "ARL0": round(arl0, 2), "ARL1": round(arl1, 2),
        "AUC": round(auc, 4), "F1": round(f1, 4),
        "Precision": round(precision, 4), "Recall": round(recall, 4),
        "alpha": round(alpha, 4), "beta": round(beta, 4), "Acc": round(acc, 4),
    }

# ================== 稳健 MCD ==================
def robust_mcd(X):
    n, d = X.shape
    cen = np.median(X, axis=0)
    for _ in range(3):
        cov = np.cov((X - cen).T) + np.eye(d) * 1e-6
        mahal = np.sum((X - cen) @ np.linalg.inv(cov) * (X - cen), axis=1)
        cen = np.mean(X[mahal <= np.percentile(mahal, 95)], axis=0)
    cov = np.cov((X - cen).T) + np.eye(d) * 1e-6
    return cen, cov

# ================== 数据生成：高斯 Copula + 块相关 ==================
def _make_corr_matrix():
    cor = np.eye(P)
    cor[0:150, 0:150] = 0.95
    cor[150:300, 150:300] = 0.85
    cor[300:400, 300:400] = 0.65
    cor[400:500, 400:500] = 0.55
    cor += np.eye(P) * 1e-6
    return cor

def generate_fixed_cor_data(n_sample, dist_type, seed):
    rng = np.random.default_rng(seed)
    cor = _make_corr_matrix()
    Z = rng.multivariate_normal(np.zeros(P), cor, size=n_sample)
    U = norm.cdf(Z)
    U = np.clip(U, 1e-8, 1 - 1e-8)

    if dist_type == "lognormal":
        X = lognorm.ppf(U, s=0.25, scale=np.exp(1.0))
    elif dist_type == "gamma":
        X = gamma.ppf(U, a=2, scale=1)
    elif dist_type == "multivariate_t":
        X = multivariate_t.rvs(np.zeros(P), cor, df=3, size=n_sample, random_state=seed) + 3
    else:
        X = Z

    trend = np.linspace(0.2, 0.6, P)
    X = X + trend[None, :]
    return X

def generate_fixed_anomalies(n_sample, dist_type, anomaly_type="mean_shift", seed=123):
    rng = np.random.default_rng(seed)

    if anomaly_type == "mean_shift":
        n_pos = n_sample // 2
        n_neg = n_sample - n_pos
        X_pos = generate_fixed_cor_data(n_pos, dist_type, seed=seed + 1) + 0.9
        X_neg = generate_fixed_cor_data(n_neg, dist_type, seed=seed + 2) - 0.8
        return np.vstack([X_pos, X_neg])

    elif anomaly_type == "cor_break":
        X = generate_fixed_cor_data(n_sample, dist_type, seed=seed)
        high_idx = np.arange(0, 20)
        for j in high_idx:
            rng.shuffle(X[:, j])

        low_idx = np.arange(480, 500)
        target_corr = 0.9
        sub_X = X[:, low_idx].copy()
        n, d = sub_X.shape
        mean_vec = np.mean(sub_X, axis=0)
        cov_mat = np.cov(sub_X, rowvar=False) + np.eye(d) * 1e-6
        L_cov = np.linalg.cholesky(cov_mat)
        X_white = (sub_X - mean_vec) @ np.linalg.inv(L_cov.T)
        R_target = np.ones((d, d)) * target_corr
        np.fill_diagonal(R_target, 1.0)
        L_target = np.linalg.cholesky(R_target)
        X_new = X_white @ L_target.T + mean_vec
        X[:, low_idx] = X_new
        return X

    elif anomaly_type == "nonlinear":
        X_base = generate_fixed_cor_data(n_sample, dist_type, seed=seed)
        Z = rng.standard_normal((n_sample, 100))
        for i in range(0, 100, 2):
            X_base[:, i] = Z[:, i] + 2.5
            X_base[:, i + 1] = np.sin(Z[:, i]) + Z[:, i + 1] * 0.1
        X_anom = X_base.copy()
        Z_anom = rng.standard_normal((n_sample, 100))
        for i in range(0, 100, 2):
            X_anom[:, i] = Z_anom[:, i] + 2.5
            X_anom[:, i + 1] = Z_anom[:, i + 1]
        return X_anom

    else:
        raise ValueError("Unknown anomaly_type")

# ================== 病态矩阵安全的 graphical lasso ==================
def safe_graphical_lasso(emp_corr, alpha, max_iter=200):
    """高维强相关矩阵会使 sklearn 的 graphical_lasso 因 Cholesky 失败
    抛出 FloatingPointError。先对称化输入；失败时逐步加入岭收缩
    （向单位矩阵收缩）后重试，保证实验不被单次数值问题中断。"""
    S0 = np.asarray(emp_corr, dtype=float)
    S0 = (S0 + S0.T) / 2.0
    d = S0.shape[0]
    ridge = 0.0
    for attempt in range(7):
        try:
            return graphical_lasso(
                S0 + np.eye(d) * ridge, alpha=alpha, max_iter=max_iter
            )
        except FloatingPointError:
            ridge = 0.005 * (2 ** attempt)
    fallback = np.diag(1.0 / (np.diag(S0) + 1.0))
    return fallback, fallback

# ================== 全局骨架与权重提取 ==================
def get_global_model(X_std, alpha):
    _, cov_mcd = robust_mcd(X_std)
    d = cov_mcd.shape[0]
    R_global = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            R_global[i, j] = cov_mcd[i, j] / np.sqrt(cov_mcd[i, i] * cov_mcd[j, j])
    Theta, _ = safe_graphical_lasso(R_global, alpha=alpha, max_iter=200)
    A = np.abs(Theta) > 1e-6
    np.fill_diagonal(A, False)
    weights = np.zeros(d)
    for i in range(d):
        neighbors = np.where(A[i])[0]
        if len(neighbors) > 0:
            weights[i] = np.sum(np.abs(R_global[i, neighbors]))
        else:
            weights[i] = 0.0
    if np.sum(weights) < 1e-8:
        weights = np.ones(d) / d
    else:
        weights = weights / np.sum(weights)
    return A, R_global, weights

# ================== 本文方法：局部动态权重 ==================
def proposed_method(X_train, X_test, y_test,
                    m_local=M_LOCAL, alpha_fixed=ALPHA_FIXED, tau_percentile=TAU_PERCENTILE):
    X_train_std, med, mad = robust_standardize(X_train)
    X_test_std, _, _ = robust_standardize(X_test, med, mad)

    A, R_global, w_global = get_global_model(X_train_std, alpha=alpha_fixed)
    mu = spatial_median(X_train_std)

    def get_dir(X, mu, w):
        z = X - mu
        zw = z * w
        norm_ = np.linalg.norm(zw, axis=1, keepdims=True) + 1e-8
        return zw / norm_

    s_train_global = get_dir(X_train_std, mu, w_global)
    s_bar = np.median(s_train_global, axis=0)

    score_train_global = np.linalg.norm(s_train_global - s_bar, axis=1)
    med_score = np.median(score_train_global)
    mad_score = np.median(np.abs(score_train_global - med_score))
    if mad_score < 1e-6:
        mad_score = 1e-6
    z_train = (score_train_global - med_score) / mad_score

    lcl_cand = np.percentile(z_train, np.arange(1, 16))
    ucl_cand = np.percentile(z_train, np.arange(85, 100))
    best_lcl, best_ucl = lcl_cand[0], ucl_cand[-1]
    min_fp = len(z_train)
    for l in lcl_cand:
        for u in ucl_cand:
            if u <= l:
                continue
            fp = np.sum((z_train < l) | (z_train > u))
            if fp < min_fp:
                min_fp, best_lcl, best_ucl = fp, l, u

    triu_idx = np.triu_indices(P, k=1)
    global_abs_corr = np.abs(R_global[triu_idx])
    thresh_new_edge = np.percentile(global_abs_corr, tau_percentile)

    score_test_raw = np.zeros(len(X_test_std))
    for i in range(len(X_test_std)):
        x = X_test_std[i]
        dists = np.linalg.norm(X_train_std - x, axis=1)
        nn_idx = np.argpartition(dists, m_local - 1)[:m_local - 1]
        local_set = np.vstack([X_train_std[nn_idx], x])

        R_local, _ = spearmanr(local_set)

        skel_edges = A & np.triu(A, k=1)
        edge_weights = np.abs(R_local) * skel_edges
        local_weight_sum = np.sum(edge_weights, axis=1) + np.sum(edge_weights, axis=0)

        new_edges = (~A) & (np.abs(R_local) > thresh_new_edge)
        np.fill_diagonal(new_edges, False)
        new_edge_weights = np.abs(R_local) * new_edges
        local_weight_sum += np.sum(new_edge_weights, axis=1) + np.sum(new_edge_weights, axis=0)

        if np.sum(local_weight_sum) < 1e-8:
            w_local = np.ones(P) / P
        else:
            w_local = local_weight_sum / np.sum(local_weight_sum)

        dir_i = get_dir(x.reshape(1, -1), mu, w_local)
        score_test_raw[i] = np.linalg.norm(dir_i - s_bar)

    z_test = (score_test_raw - med_score) / mad_score
    return compute_metrics(z_test, y_test, best_lcl, best_ucl)

# ================== 基线方法：修正阈值 ==================
def baseline_methods(X_train, X_test, y_test):
    X_train_std, med, mad = robust_standardize(X_train)
    X_test_std, _, _ = robust_standardize(X_test, med, mad)
    res = {}

    # MRCD
    cen_mcd, cov_mcd = robust_mcd(X_train_std)
    inv_cov = np.linalg.inv(cov_mcd)
    s_train = np.sum((X_train_std - cen_mcd) @ inv_cov * (X_train_std - cen_mcd), axis=1)
    s_test = np.sum((X_test_std - cen_mcd) @ inv_cov * (X_test_std - cen_mcd), axis=1)
    res["MRCD"] = compute_metrics(s_test, y_test, -1e9, np.percentile(s_train, 99))

    # Spatial Rank
    cen_sp = spatial_median(X_train_std)
    s_train_sp = np.linalg.norm(X_train_std - cen_sp, axis=1)
    s_test_sp = np.linalg.norm(X_test_std - cen_sp, axis=1)
    res["Spatial Rank"] = compute_metrics(s_test_sp, y_test, -1e9, np.percentile(s_train_sp, 99))

    # ROBPCA
    pca = PCA(n_components=5, random_state=42)
    pca.fit(X_train_std - np.median(X_train_std, axis=0))
    def rp_score(X):
        xc = X - np.median(X_train_std, axis=0)
        return np.sum((xc - pca.inverse_transform(pca.transform(xc))) ** 2, axis=1)
    s_train_rp = rp_score(X_train_std)
    s_test_rp = rp_score(X_test_std)
    res["ROBPCA"] = compute_metrics(s_test_rp, y_test, -1e9, np.percentile(s_train_rp, 99))

    # Huber
    def huber_estimator(X, k=1.345, max_iter=100):
        n, d = X.shape
        mu = np.median(X, axis=0)
        for _ in range(max_iter):
            diff = X - mu
            dist = np.sqrt(np.sum(diff * diff, axis=1))
            w = np.where(dist <= k, 1.0, k / dist)
            mu_new = np.average(X, axis=0, weights=w)
            if np.linalg.norm(mu_new - mu) < 1e-6:
                break
            mu = mu_new
        diff = X - mu
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        w = np.where(dist <= k, 1.0, k / dist)
        cov = np.cov(diff.T, aweights=w) + np.eye(d) * 1e-6
        return mu, cov

    mu_huber, cov_huber = huber_estimator(X_train_std)
    inv_h = np.linalg.inv(cov_huber)
    s_train_h = np.sum((X_train_std - mu_huber) @ inv_h * (X_train_std - mu_huber), axis=1)
    s_test_h = np.sum((X_test_std - mu_huber) @ inv_h * (X_test_std - mu_huber), axis=1)
    res["Huber"] = compute_metrics(s_test_h, y_test, -1e9, np.percentile(s_train_h, 99))

    # Graphical LASSO
    rho, _ = spearmanr(X_train_std)
    precision, _ = safe_graphical_lasso(rho, alpha=0.1, max_iter=200)
    adjacency = np.abs(precision) > 1e-6
    degrees = np.sum(adjacency, axis=1)
    selected = np.where(degrees > 0)[0]
    if len(selected) < 5:
        selected = np.argsort(degrees)[-10:]
    cen_gl, cov_gl = robust_mcd(X_train_std[:, selected])
    inv_gl = np.linalg.inv(cov_gl)
    s_train_gl = np.sum((X_train_std[:, selected] - cen_gl) @ inv_gl * (X_train_std[:, selected] - cen_gl), axis=1)
    s_test_gl = np.sum((X_test_std[:, selected] - cen_gl) @ inv_gl * (X_test_std[:, selected] - cen_gl), axis=1)
    res["Graphical LASSO"] = compute_metrics(s_test_gl, y_test, -1e9, np.percentile(s_train_gl, 99))

    # Quantile Lasso
    selected_q = set()
    n_vars = min(20, P)
    for i in range(n_vars):
        y_i = X_train_std[:, i]
        X_others = np.delete(X_train_std, i, axis=1)
        qr = QuantileRegressor(quantile=0.5, alpha=0.1, solver="highs",
                               solver_options={"max_iter": 500})
        qr.fit(X_others, y_i)
        coef = qr.coef_
        non_zero_idx = np.where(np.abs(coef) > 1e-4)[0]
        adjusted_idx = [j if j < i else j + 1 for j in non_zero_idx]
        selected_q.update(adjusted_idx)
    selected_q = np.array(list(selected_q))
    if len(selected_q) < 5:
        selected_q = np.arange(P)[:10]
    cen_q, cov_q = robust_mcd(X_train_std[:, selected_q])
    inv_q = np.linalg.inv(cov_q)
    s_train_q = np.sum((X_train_std[:, selected_q] - cen_q) @ inv_q * (X_train_std[:, selected_q] - cen_q), axis=1)
    s_test_q = np.sum((X_test_std[:, selected_q] - cen_q) @ inv_q * (X_test_std[:, selected_q] - cen_q), axis=1)
    res["Quantile Lasso"] = compute_metrics(s_test_q, y_test, -1e9, np.percentile(s_train_q, 99))

    return res

# ================== 单次实验 ==================
def run_single_experiment(a_type, dist, rep, seed):
    X_normal = generate_fixed_cor_data(N_NORMAL, dist, seed)
    X_anom = generate_fixed_anomalies(N_ANOM, dist, anomaly_type=a_type, seed=seed + 100000)
    X_train = X_normal[:N_TRAIN]
    X_test = np.vstack([X_normal[N_TRAIN:], X_anom])
    y_test = np.hstack([np.zeros(N_NORMAL - N_TRAIN), np.ones(N_ANOM)])

    res_prop = proposed_method(X_train, X_test, y_test)
    res_base = baseline_methods(X_train, X_test, y_test)

    records = []
    for method, met in [("本文方法", res_prop)] + list(res_base.items()):
        rec = {"method": method, "anomaly_type": a_type, "dist": dist, "rep": rep, **met}
        records.append(rec)
    return records

# ================== 主实验 ==================
def run_main_experiment():
    tasks = []
    for a_type in ANOMALY_TYPES:
        for dist in DIST_LIST:
            for rep in range(REPEATS):
                seed = BASE_SEED + rep * 1000 + hash((a_type, dist)) % 1000
                tasks.append((a_type, dist, rep, seed))

    print(f"开始主实验：{len(tasks)} 个任务，并行执行...")
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_single_experiment)(*t) for t in tasks
    )
    all_records = [r for sublist in results for r in sublist]
    df = pd.DataFrame(all_records)
    df.to_csv(os.path.join(OUT_DIR, "all_runs.csv"), index=False)
    return df

# ================== 汇总表 ==================
def format_mean_std(mean, std):
    return f"{mean:.4f}±{std:.4f}"

def make_table(df, anomaly_type, table_name):
    sub = df[df["anomaly_type"] == anomaly_type].copy()
    metrics = ["ARL0", "ARL1", "AUC", "F1", "Precision", "Recall", "Acc"]
    rows = []
    for method in METHOD_ORDER:
        for dist in DIST_LIST:
            row = {"方法": method, "分布": DIST_NAMES[dist]}
            for m in metrics:
                vals = sub[(sub["method"] == method) & (sub["dist"] == dist)][m]
                if len(vals) == 0:
                    row[m] = "待填±待填"
                else:
                    row[m] = format_mean_std(vals.mean(), vals.std())
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "tables", f"{table_name}.csv"), index=False)
    with open(os.path.join(OUT_DIR, "tables", f"{table_name}.md"), "w", encoding="utf-8") as f:
        f.write(f"## {table_name}\n\n")
        f.write(out.to_markdown(index=False))
    return out

def make_overall_table(df):
    metrics = ["AUC", "F1", "ARL0", "ARL1"]
    rows = []
    for method in METHOD_ORDER:
        sub = df[df["method"] == method]
        row = {"方法": method}
        for m in metrics:
            row[m] = f"{sub[m].mean():.4f}"
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "tables", "table_4_9_overall.csv"), index=False)
    with open(os.path.join(OUT_DIR, "tables", "table_4_9_overall.md"), "w", encoding="utf-8") as f:
        f.write("## 表 4-9 各方法综合性能汇总\n\n")
        f.write(out.to_markdown(index=False))
    return out

def make_wilcoxon_table(df):
    rows = []
    for a_type in ANOMALY_TYPES:
        for dist in DIST_LIST:
            sub = df[(df["anomaly_type"] == a_type) & (df["dist"] == dist)]
            prop = sub[sub["method"] == "本文方法"]
            for base_name in BASELINE_NAMES:
                base = sub[sub["method"] == base_name]
                merged = prop.merge(base, on="rep", suffixes=("_prop", "_base"))
                for metric in ["AUC", "F1"]:
                    if len(merged) < 5:
                        p = np.nan
                    else:
                        _, p = wilcoxon(merged[f"{metric}_prop"], merged[f"{metric}_base"])
                    rows.append({
                        "异常类型": ANOMALY_NAMES[a_type],
                        "分布": DIST_NAMES[dist],
                        "对比方法": base_name,
                        "指标": metric,
                        "p值": round(p, 4) if not np.isnan(p) else "NA",
                        "是否显著": "是" if (not np.isnan(p) and p < 0.05) else "否",
                    })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "tables", "table_4_10_wilcoxon.csv"), index=False)
    with open(os.path.join(OUT_DIR, "tables", "table_4_10_wilcoxon.md"), "w", encoding="utf-8") as f:
        f.write("## 表 4-10 配对 Wilcoxon 符号秩检验\n\n")
        f.write(out.to_markdown(index=False))
    return out

# ================== 参数敏感性分析 ==================
def run_param_sensitivity():
    param_configs = {
        "m": [20, 30, 50, 80, 100],
        "alpha": [0.005, 0.01, 0.02, 0.05, 0.1],
        "tau": [80, 85, 90, 95],
    }
    all_rows = []
    for param_name, values in param_configs.items():
        print(f"参数敏感性：{param_name} ...")
        for val in values:
            for a_type in ANOMALY_TYPES:
                for dist in DIST_LIST:
                    arl0_list, arl1_list = [], []
                    for rep in range(PARAM_SENS_REPEATS):
                        seed = BASE_SEED + rep * 1000 + hash((a_type, dist, param_name, val)) % 1000
                        X_normal = generate_fixed_cor_data(N_NORMAL, dist, seed)
                        X_anom = generate_fixed_anomalies(N_ANOM, dist, anomaly_type=a_type, seed=seed + 100000)
                        X_train = X_normal[:N_TRAIN]
                        X_test = np.vstack([X_normal[N_TRAIN:], X_anom])
                        y_test = np.hstack([np.zeros(N_NORMAL - N_TRAIN), np.ones(N_ANOM)])

                        kwargs = {}
                        if param_name == "m":
                            kwargs["m_local"] = val
                        elif param_name == "alpha":
                            kwargs["alpha_fixed"] = val
                        elif param_name == "tau":
                            kwargs["tau_percentile"] = val

                        met = proposed_method(X_train, X_test, y_test, **kwargs)
                        arl0_list.append(met["ARL0"])
                        arl1_list.append(met["ARL1"])
                    all_rows.append({
                        "param": param_name,
                        "value": val,
                        "anomaly_type": a_type,
                        "dist": dist,
                        "ARL0_mean": np.mean(arl0_list),
                        "ARL1_mean": np.mean(arl1_list),
                    })
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT_DIR, "param_sensitivity.csv"), index=False)
    return df

def plot_param_heatmaps(df):
    for param_name in ["m", "alpha", "tau"]:
        sub = df[df["param"] == param_name]
        for metric in ["ARL0", "ARL1"]:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for ax, dist in zip(axes, DIST_LIST):
                pv = sub[sub["dist"] == dist].pivot(
                    index="anomaly_type", columns="value", values=f"{metric}_mean"
                )
                pv = pv.reindex(ANOMALY_TYPES)
                sns.heatmap(
                    pv, annot=True, fmt=".2f",
                    cmap="viridis" if metric == "ARL0" else "viridis_r",
                    ax=ax, cbar_kws={"label": metric}
                )
                ax.set_title(f"{DIST_NAMES[dist]} - {metric}")
                ax.set_xlabel(param_name)
                ax.set_ylabel("异常类型")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, "figures", f"fig_{param_name}_{metric}.png"), dpi=300)
            plt.savefig(os.path.join(OUT_DIR, "figures", f"fig_{param_name}_{metric}.pdf"))
            plt.close()

# ================== 生成报告 ==================
def make_report():
    lines = ["# 4.5 模拟结果与分析\n"]
    for fname in ["table_4_6_mean_shift.md", "table_4_7_cor_break.md", "table_4_8_nonlinear.md",
                  "table_4_9_overall.md", "table_4_10_wilcoxon.md"]:
        path = os.path.join(OUT_DIR, "tables", fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                lines.append(f.read())
            lines.append("\n")
    lines.append("## 参数敏感性图\n")
    for param in ["m", "alpha", "tau"]:
        for metric in ["ARL0", "ARL1"]:
            img = f"figures/fig_{param}_{metric}.png"
            lines.append(f"![{param}_{metric}]({img})\n")
    with open(os.path.join(OUT_DIR, "report_4_5.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ================== 主入口 ==================
if __name__ == "__main__":
    print("=" * 80)
    print("主实验开始")
    print("=" * 80)
    df_main = run_main_experiment()
    print("\n主实验完成，生成表格...")
    make_table(df_main, "mean_shift", "table_4_6_mean_shift")
    make_table(df_main, "cor_break", "table_4_7_cor_break")
    make_table(df_main, "nonlinear", "table_4_8_nonlinear")
    make_overall_table(df_main)
    make_wilcoxon_table(df_main)

    print("\n参数敏感性分析开始...")
    df_sens = run_param_sensitivity()
    plot_param_heatmaps(df_sens)

    make_report()
    print("\n全部完成，结果已保存到 results/ 目录。")
    print("终端仅显示进度，详细表见 results/tables/，图见 results/figures/。")