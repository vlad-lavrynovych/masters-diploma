import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)

df = pd.read_csv("cleaned_syn.csv")
print(df.head())
print(df.describe()[1:][list(df.select_dtypes(include=[np.number]))])
df.describe(include='O').transpose()
df.isna().sum()
df['HeartDisease'].value_counts()

# gradient = df.describe()[1:][list(df.select_dtypes(include=[np.number]))].T.style.background_gradient(cmap='Blues')
# dfi.export(gradient, 'out/gradient.png')

# Visualization

from matplotlib import pyplot as plt
import seaborn as sns

plt.pie(x=df['HeartDisease'].value_counts(), labels=df['HeartDisease'].value_counts().index)
plt.title('Healthy - Sick ratio')
plt.show()
# sns.set_context(font_scale=2, rc={"font.size": 45, "axes.titlesize": 55, "axes.labelsize": 45})
sns.set_style("ticks")
s = sns.catplot(kind='count', data=df, x='AgeCategory', hue='HeartDisease',
                order=df['AgeCategory'].sort_values().unique())
s.tick_params(axis='x', rotation=45)
s.legend.set_title('Atherosclerosis')
plt.title('Variation of Age for each target class')
plt.show()

i = 1
plt.figure(figsize=(25, 15))
for feature in df.select_dtypes(include=[object]):
    plt.subplot(3, 5, i)
    sns.set(palette='Paired')
    sns.set_style("ticks")
    ax = sns.countplot(x=feature, data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    i += 1
plt.show()

label = LabelEncoder()
for col in df:
    df[col] = label.fit_transform(df[col])
print(df)
cmap = sns.diverging_palette(250, 10, as_cmap=True)
sns.heatmap(df.corr(), cmap=cmap, vmin=-1, vmax=1,
            square=True,
            linewidth=.5)
plt.show()
