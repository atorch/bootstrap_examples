from functools import partial

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV

np.random.seed(998877)


def simulate_confidence_interval_for_mean(n_obs=35):

    """Simple confidence interval for the mean (don't need the bootstrap)
    """

    dataset = np.random.uniform(size=n_obs)

    sample_mean = np.mean(dataset)

    # See https://numpy.org/doc/stable/reference/generated/numpy.var.html
    # In standard statistical practice, ddof=1 provides an unbiased estimator of the variance of a hypothetical infinite population.
    # Also see https://stats.stackexchange.com/questions/100041/how-exactly-did-statisticians-agree-to-using-n-1-as-the-unbiased-estimator-for
    std_error = np.sqrt(np.var(dataset, ddof=1) / n_obs)

    confidence_interval = [sample_mean - 1.96 * std_error, sample_mean + 1.96 * std_error]

    return confidence_interval


def example_0(n_simulations=1000):

    ci_contains_true_mean = []

    # True mean of a Uniform[0, 1]
    true_mean = 0.5

    for _ in range(n_simulations):

        confidence_interval = simulate_confidence_interval_for_mean()

        ci_contains_true_mean.append(confidence_interval[0] < true_mean < confidence_interval[1])

    coverage = np.mean(ci_contains_true_mean)

    print("*** Example 0 ***")
    print("Simple confidence interval for the mean")
    print(f"CI coverage: {coverage} (this should be close to 0.95)")


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


def example_1(n_obs=55, verbose=True, statistic=np.median, population_value=1.0):
    """Based in part on All of Statistics by Larry Wasserman, Chapter 8

    See example 8.1, bootstrap for the median
    """

    # Our data comes from a triangle distribution
    # See https://en.wikipedia.org/wiki/Triangular_distribution
    # And see https://stats.stackexchange.com/questions/41467/consider-the-sum-of-n-uniform-distributions-on-0-1-or-z-n-why-does-the
    dataset = np.random.uniform(size=n_obs) + np.random.uniform(size=n_obs)

    sample_statistic = statistic(dataset)

    # How variable is the sample median?  What is its standard error?
    sample_statistics = bootstrap(dataset, statistic, n_bootstrap_samples=500)

    standard_error = np.std(sample_statistics)

    # This should contain the true population median (which is 1.0) approximately 95% of the time
    percentile_confidence_interval = np.percentile(sample_statistics, q=[2.5, 97.5])

    if verbose:
        print("*** Example 1 ***")
        print(f"sample {statistic.__name__}: {sample_statistic} (the true population value is {population_value})")
        print(f"std error: {standard_error}")
        print(f"confidence interval: {percentile_confidence_interval}")

    return percentile_confidence_interval


def example_2(n_replications=500):
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
    print("Bootstrap percentile confidence interval for the median")
    print(f"CI coverage: {coverage} (this should be close to 0.95)")


def example_2_with_a_twist(n_replications=500):
    """How often do the bootstrap percentile CIs from example 1 actually contain the true 75th percentile?
    """

    ci_contains_true_75th_percentile = []
    true_75th_percentile = 2 - np.sqrt(1/2)

    calculate_75th_percentile = partial(np.percentile, q=75)

    for _ in range(n_replications):

        confidence_interval = example_1(verbose=False, statistic=calculate_75th_percentile)
        ci_contains_true_75th_percentile.append(
            confidence_interval[0] < true_75th_percentile < confidence_interval[1]
        )

    coverage = np.mean(ci_contains_true_75th_percentile)
    print("*** Example 2 ***")
    print("Bootstrap percentile confidence interval for the 75th percentile")
    print(f"CI coverage: {coverage} (this should be close to 0.95)")


def example_3(n_replications=500):
    """An example where the bootstrap does *not* work

    Try building a confidence interval for the _max_ instead of the median

    See https://stats.stackexchange.com/questions/9664/what-are-examples-where-a-naive-bootstrap-fails
    """

    ci_contains_true_max = []
    true_max = 2.0

    example_1(verbose=True, statistic=np.max, population_value=true_max)

    for _ in range(n_replications):

        confidence_interval = example_1(verbose=False, statistic=np.max, population_value=true_max)
        ci_contains_true_max.append(
            confidence_interval[0] < true_max < confidence_interval[1]
        )

    coverage = np.mean(ci_contains_true_max)
    print("*** Example 3 ***")
    print("Bootstrap percentile confidence interval for the maximum")
    print(f"CI coverage: {coverage} (isn't close to 0.95!)")


def simulate_dataframe(n_obs):

    df = pd.DataFrame({"x1": np.random.uniform(size=n_obs), "x2": np.random.uniform(size=n_obs)})

    df["x3"] = df["x2"] + np.random.uniform(size=n_obs)

    df["x4"] = np.random.normal(size=n_obs)
    df["x5"] = df["x4"] + np.random.normal(size=n_obs, scale=2.0)

    df["epsilon"] = np.random.normal(size=n_obs)

    df["y"] = 10 + 5 * df["x1"] - 3 * df["x2"] + 1 * df["x3"] - 4 * df["x5"] + df["epsilon"]

    return df


def get_model_coefficients(df):

    model = ElasticNetCV(n_alphas=10, cv=5, l1_ratio=0.95)

    predictors = ["x1", "x2", "x3", "x4", "x5"]

    model.fit(X=df[predictors], y=df["y"])

    return model.coef_


def example_4(n_obs=90, n_replications=100):

    """Based in part on Statistical Learning with Sparsity, Chapter 6

    See 6.2 The Bootstrap, and especially Figure 6.4
    """

    df = simulate_dataframe(n_obs)

    coef = get_model_coefficients(df)

    coefs = np.array(bootstrap(df, get_model_coefficients, n_bootstrap_samples=n_replications))

    # TODO Plot
    # TODO How often is each coef zeroed out?
    fraction_zeroed_out = np.mean(np.isclose(coefs, 0.0), axis=0)

def main():

    # An example where you don't need the bootstrap
    example_0()

    example_1()
    example_2()
    example_2_with_a_twist()

    # An example where the boostrap does not work
    example_3()

    # A more complicated example involving an elastic net
    example_4()


if __name__ == "__main__":
    main()
