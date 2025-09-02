import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 10)

def task1():
    poland_vacation = pd.read_csv("multiTimeline.csv", sep=",", skiprows=3, index_col="PL_Month",
                                  names=["PL_Month", "PL_Number"])
    US_vacation = pd.read_csv("multiTimeline (1).csv", sep=",", skiprows=3, index_col="US_Month",
                              names=["US_Month", "US_Number"])
    UK_vacation = pd.read_csv("multiTimeline (2).csv", sep=",", skiprows=3, index_col="UK_Month",
                              names=["UK_Month", "UK_Number"])
    combined_dataset = pd.concat([poland_vacation, US_vacation, UK_vacation], axis=1)
    print(combined_dataset.describe())
    print(combined_dataset.head())
    combined_dataset.plot()
    plt.xlabel("Dates [months]")
    plt.ylabel("Number of searching")
    plt.title("Plot of vacation search results")
    plt.show()
    combined_dataset.boxplot()
    plt.show()
    plt.hist(poland_vacation["PL_Number"], bins=20, label="PL_Vacation", alpha=0.7)
    plt.hist(US_vacation["US_Number"], bins=20, label="US_Vacation", alpha=0.7)
    plt.hist(UK_vacation["UK_Number"], bins=20, label="UK_Vacation", alpha=0.7)
    plt.title("Vacation search results")
    plt.xlabel("Number of search")
    plt.ylabel("Number of numbers of searches")
    plt.legend()
    plt.show()
    combined_dataset.plot.kde()
    plt.title("Kernel Density")
    plt.xlabel("Number of search")
    plt.xlim([0, 100])
    plt.show()

def create_table(city_temps):
    city_temps = city_temps.sort_values(by='Value', ascending=True)
    city_temps["Rank (out of 84)"] = np.arange(1, len(city_temps) + 1)
    mean = city_temps.mean()
    text = "Anomaly 1939 - 2022 Mean: " + str(np.round(mean['Value'])) + "F"
    city_temps.rename(columns={"Value": "Average Temperature", "Anomaly": text}, inplace=True)
    city_temps.sort_values(by='Date', inplace=True)
    display(city_temps)

def task2():
    city_temps = pd.read_csv("data.csv", sep=",", skiprows=5, index_col="Date",
                             names=["Date", "Value", "Anomaly"], date_format="%Y%m")
    city_temps.where(city_temps == -99, np.nan)
    city_temps.interpolate(method='linear', inplace=True)
    print(city_temps.head())
    city_temps.plot()
    plt.title("Average Temperature")
    plt.ylabel("Temperature [Fahrenheit]")
    plt.xlabel("Dates [months]")
    plt.show()

    plt.hist(city_temps['Value'], bins=20, label="Value")
    plt.hist(city_temps['Anomaly'], bins=20)
    plt.title("Histogram of Average Temperatures and Anomalies")
    plt.xlabel("Values")
    plt.ylabel("Number of values")
    plt.show()

    city_temps.plot.kde()
    plt.title("Kernel density of Average Temperatures and Anomalies")
    plt.xlabel("Values")
    plt.show()
    print(city_temps.describe())
    create_table(city_temps)


if __name__ == "__main__":
    task2()
