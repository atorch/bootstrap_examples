import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV

np.random.seed(998877)


def resample(dataset):

    if isinstance(dataset, pd.DataFrame):
        return dataset.sample(frac=1.0, replace=True)

    return np.random.choice(dataset, size=dataset.size, replace=True)


def bootstrap(dataset, estimator, n_bootstrap_samples):

    values = []

    for _ in range(n_bootstrap_samples):

        # Note that we sample *with* replacement
        resampled_dataset = resample(dataset)
        values.append(estimator(resampled_dataset))

    return values


def example_1(n_obs=55, verbose=True):
    """Based in part on All of Statistics by Larry Wasserman, Chapter 8

    See example 8.1, bootstrap for the median
    """

    # Our data comes from a triangle distribution
    # See https://en.wikipedia.org/wiki/Triangular_distribution
    # And see https://stats.stackexchange.com/questions/41467/consider-the-sum-of-n-uniform-distributions-on-0-1-or-z-n-why-does-the
    dataset = np.random.uniform(size=n_obs) + np.random.uniform(size=n_obs)

    sample_median = np.median(dataset)

    # How variable is the sample median?  What is its standard error?
    sample_medians = bootstrap(dataset, np.median, n_bootstrap_samples=500)

    standard_error = np.std(sample_medians)

    # This should contain the true population median (which is 1.0) approximately 95% of the time
    percentile_confidence_interval = np.percentile(sample_medians, q=[2.5, 97.5])

    if verbose:
        print("*** Example 1 ***")
        print(f"sample median: {sample_median} (the true population value is 1.0)")
        print(f"std error: {standard_error}")
        print(f"confidence interval: {percentile_confidence_interval}")

    return percentile_confidence_interval


def example_2(n_replications=250):
    """How often do the bootstrap percentile CIs from example 1 actually contain the true median?
    """

    ci_contains_true_median = []
    true_median = 1.0

    for _ in range(n_replications):

        confidence_interval = example_1(verbose=False)
        ci_contains_true_median.append(
            confidence_interval[0] < true_median < confidence_interval[1]
        )

    coverage = np.mean(ci_contains_true_median)
    print("*** Example 2 ***")
    print(f"CI coverage: {coverage} (this should be close to 0.95)")


def simulate_dataframe(n_obs):

    df = pd.DataFrame({"x1": np.random.uniform(size=n_obs), "x2": np.random.uniform(size=n_obs)})

    df["x3"] = df["x2"] + np.random.uniform(size=n_obs)

    df["x4"] = np.random.normal(size=n_obs)
    df["x5"] = df["x4"] + np.random.normal(size=n_obs, scale=2.0)

    df["epsilon"] = np.random.normal(size=n_obs)

    df["y"] = 10 + 5 * df["x1"] - 3 * df["x2"] + 1 * df["x3"] - 4 * df["x5"] + df["epsilon"]

    return df


def get_model_coefficients(df):

    model = ElasticNetCV(n_alphas=10, cv=5, l1_ratio=0.9)

    predictors = ["x1", "x2", "x3", "x4", "x5"]

    model.fit(X=df[predictors], y=df["y"])

    return model.coef_


def example_3(n_obs=250):

    """Based in part on Statistical Learning with Sparsity, Chapter 6

    See 6.2 The Bootstrap, and especially Figure 6.4
    """

    df = simulate_dataframe(n_obs)

    coef = get_model_coefficients(df)

    coefs = np.array(bootstrap(df, get_model_coefficients, n_bootstrap_samples=3))

    # TODO Plot
    # TODO How often is each coef zeroed out?


def main():

    # example_1()
    # example_2()

    # TODO Another example where bootstrap doesn't work well (estimating the max, for example)

    example_3()


if __name__ == "__main__":
    main()
