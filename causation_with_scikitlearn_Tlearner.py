# Causal method with scikit learn.
# https://medium.com/data-science-collective/causal-ai-in-action-drive-conversions-with-uplift-modeling-a43ccdb1e5aa


# Create dataset
def simulate_free_delivery_subscription_extended(n_samples=5000, seed=42):
    """
    Simulate data for a randomized trial measuring the effect of a discount on
    subscription to a free delivery program,
    with extended demographic features.
    Parameters
    ----------
    n_samples : int
        Number of observations (customers) to generate.
    seed : int
        Random seed for reproducibility.
    Returns
    -------
    df : pd.DataFrame
        Simulated dataset with user features (including extra demographics),
        a treatment indicator, and a subscription outcome.
    """
    np.random.seed(seed)

    age = np.random.randint(18, 70, size=n_samples)
    purchase_freq = np.random.poisson(lam=2, size=n_samples)
    avg_spend = np.round(np.random.gamma(shape=2.0, scale=50.0, size=n_samples), 2)

    gender = np.random.binomial(1, p=0.5, size=n_samples)
    regions = np.random.choice(['Urban', 'Suburban', 'Rural'],
                               size=n_samples,
                               p=[0.5, 0.3, 0.2])
    treatment = np.random.binomial(1, 0.5, size=n_samples)

    region_effect_map = {'Urban': 0.3, 'Suburban': 0.2, 'Rural': 0.0}
    region_effect = np.array([region_effect_map[r] for r in regions])

    log_odds_baseline = (
        -4.0
        - 0.02 * age
        + 0.5  * purchase_freq
        + 0.01 * avg_spend
        + 0.1  * gender     
        + region_effect     
    )

    treatment_effect = 1.0
    log_odds = log_odds_baseline + treatment * treatment_effect
    prob_subscription = expit(log_odds)

    subscription = np.random.binomial(1, prob_subscription)

    df = pd.DataFrame({
        'age': age,
        'purchase_freq': purchase_freq,
        'avg_spend': avg_spend,
        'gender': gender,
        'region': regions,
        'treatment': treatment,
        'subscription': subscription
    })

    return df

X = df.drop(columns=['subscription', 'treatment'])
T = df['treatment']
y = df['subscription']

X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(
    X, T, y, test_size=0.2, random_state=42
)

encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

X_train_encoded = encoder.fit_transform(X_train[['region']])
X_test_encoded = encoder.transform(X_test[['region']])

encoded_cols = encoder.get_feature_names_out(['region'])
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_cols, index=X_train.index)
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_cols, index=X_test.index)
s
X_train_final = pd.concat([X_train.drop(columns='region'), X_train_encoded_df], axis=1)
X_test_final = pd.concat([X_test.drop(columns='region'), X_test_encoded_df], axis=1)

# T-learner.  (create 2 models, one for treatment, another for control group)
t_learner = TwoModels(
    estimator_trmnt=GradientBoostingClassifier(n_estimators=100, random_state=42),
    estimator_ctrl=GradientBoostingClassifier(n_estimators=100, random_state=42)
)

t_learner.fit(X_train_final, y_train, T_train)

uplift_preds_t = t_learner.predict(X_test_final)


# Visualize performance. AUC curve.
import matplotlib.pyplot as plt
import numpy as np
from sklift.metrics import qini_curve, qini_auc_score

# Calculate Qini curve points
qini_x, qini_y = qini_curve(y_test, uplift_preds, T_test)

# Compute Qini AUC score
qini_auc = qini_auc_score(y_test, uplift_preds, T_test)

# Random targeting baseline
random_y = np.linspace(0, qini_y[-1], len(qini_x))

# Plot the Qini Curve
plt.figure(figsize=(8, 6))
plt.plot(qini_x, qini_y, label='S-Learner Qini Curve', linewidth=2)
plt.plot(qini_x, random_y, 'r--', label='Random Targeting')

# Display Qini AUC score at the bottom-right corner
plt.text(0.95, 0.05, 
         f'Qini AUC = {qini_auc:.4f}',
         fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.7),
         horizontalalignment='right', 
         verticalalignment='bottom', 
         transform=plt.gca().transAxes)

# Formatting
plt.xlabel("Proportion of Population (sorted by predicted uplift)")
plt.ylabel("Cumulative Incremental Outcome")
plt.title("Qini Curve for S-Learner")
plt.legend(loc='upper left')
plt.grid(True)

plt.tight_layout()
plt.show()

