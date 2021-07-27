import numpy as np

np.random.seed(998877)


def bootstrap(dataset, estimator, n_bootstrap_samples):

    values = []

    for _ in range(n_bootstrap_samples):

        # Note that we sample _with replacement_
        resampled_dataset = np.random.choice(dataset, size=dataset.size, replace=True)
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
    """How often do the boostrap percentile CIs from example 1 actually contain the true median?
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


def main():

    example_1()
    example_2()


if __name__ == "__main__":
    main()
