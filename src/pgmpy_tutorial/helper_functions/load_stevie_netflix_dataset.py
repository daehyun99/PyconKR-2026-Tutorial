"""
Reference: https://www.tylervigen.com/spurious/correlation/3268_popularity-of-the-first-name-stevie_correlates-with_netflixs-stock-price

"""
from scipy import stats
import numpy as np


def load_stevie_netflix_dataset():
    # These modules make it easier to perform the calculation

    # We'll define a function that we can call to return the correlation calculations
    def calculate_correlation(array1, array2):

        # Calculate Pearson correlation coefficient and p-value
        correlation, p_value = stats.pearsonr(array1, array2)

        # Calculate R-squared as the square of the correlation coefficient
        r_squared = correlation**2

        return correlation, r_squared, p_value

    # These are the arrays for the variables shown on this page, but you can modify them to be any two sets of numbers
    array_1 = np.array([232,215,211,252,229,217,240,210,227,254,260,318,312,379,444,473,629,801,1147,1217,])
    array_2 = np.array([0.85,4.11,1.8,3.87,3.71,3.79,4.22,7.93,25,10.04,13.6,52.4,49.15,109,124.96,196.1,259.28,326.1,539,605.61,])
    array_1_name = "Popularity of the first name Stevie"
    array_2_name = "Netflix's stock price (NFLX)"

    # Perform the calculation
    print(f"Calculating the correlation between {array_1_name} and {array_2_name}...")
    correlation, r_squared, p_value = calculate_correlation(array_1, array_2)

    # Print the results
    print("Correlation Coefficient:", correlation)
    print("R-squared:", r_squared)
    print("P-value:", p_value)
    return array_1, array_2
