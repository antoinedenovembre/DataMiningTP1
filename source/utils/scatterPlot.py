import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def scatterPlot(x_DF, y_DF, algo_name):
    temp_DF = pd.DataFrame(data=x_DF.loc[:, 0:1], index=x_DF.index)
    temp_DF = pd.concat((temp_DF, y_DF), axis=1, join="inner")
    print(x_DF)
    print(y_DF)
    temp_DF.columns = ["First Vector", "Second Vector", "Label"]
    sns.lmplot(x="First Vector", y="Second Vector", hue="Label",
               data=temp_DF, fit_reg=False)
    ax = plt.gca()
    ax.set_title("Separation of Observations using " + algo_name)
    plt.show()
