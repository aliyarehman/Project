

```python
# importing necessary libraries for analysis 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm 
from scipy.stats import pearsonr
from numpy.random import seed
from numpy import mean
from numpy import var
from math import sqrt
from pandas_ml import ConfusionMatrix
from sklearn.metrics import confusion_matrix
import itertools

# Use ggplot style for visulizations
plt.style.use('seaborn-dark')
```


```python
# Import HIV data
data = pd.read_excel('HIV1.xlsx', index=False)

# Read the required columns
data = data[['sample_id', 'data_1', 'Cutoff ', 'data_2', 'Cutoff .1']]

# Set new column names
data.columns = [['id', 'evo_data', 'evo_resp', 'rmp_data', 'rmp_resp']]

# Change negative and positive values to 0 and 1 respectively 
data = data.replace('POSITIVE', 1)
data = data.replace('Negative', 0)


# Print columns and head
print(data.columns)
data.head()
```

    MultiIndex(levels=[['evo_data', 'evo_resp', 'id', 'rmp_data', 'rmp_resp']],
               labels=[[2, 0, 1, 3, 4]])





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>id</th>
      <th>evo_data</th>
      <th>evo_resp</th>
      <th>rmp_data</th>
      <th>rmp_resp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17V0161024</td>
      <td>0.18</td>
      <td>0</td>
      <td>0.09</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17V0161767</td>
      <td>4.05</td>
      <td>1</td>
      <td>3.90</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17V0161827</td>
      <td>0.12</td>
      <td>0</td>
      <td>0.06</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17V0161840</td>
      <td>0.12</td>
      <td>0</td>
      <td>0.05</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17V0161877</td>
      <td>0.12</td>
      <td>0</td>
      <td>0.06</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Describe the 5 point statistics of the dataset
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>evo_data</th>
      <th>evo_resp</th>
      <th>rmp_data</th>
      <th>rmp_resp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>324.000000</td>
      <td>324.000000</td>
      <td>324.000000</td>
      <td>324.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.251852</td>
      <td>0.040123</td>
      <td>0.159907</td>
      <td>0.043210</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.659762</td>
      <td>0.196552</td>
      <td>0.549919</td>
      <td>0.203644</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.100000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.110000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.130000</td>
      <td>0.000000</td>
      <td>0.060000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.140000</td>
      <td>0.000000</td>
      <td>0.060000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.120000</td>
      <td>1.000000</td>
      <td>3.900000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Scatter plot between evo and rmp data readings 

from pylab import rcParams
rcParams['figure.figsize'] = 15, 5
plt.scatter(data.evo_data, data.rmp_data)
plt.xlabel('EVO')
plt.ylabel('RMP')
plt.title('HIV')
plt.show()
```


![png](output_3_0.png)



```python
# Plot histograms to inspect the typical values

data['evo_data'].hist(bins=5)
data['rmp_data'].hist(bins=5)

```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x1a19a1fe10>]],
          dtype=object)




![png](output_4_1.png)



![png](output_4_2.png)



```python
# Plot distributions as overlapping kde plots

kde = data[['evo_data', 'rmp_data']].copy()
kde.plot.kde()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a17f98c50>




![png](output_5_1.png)



```python
# Convert data to numpy arrays and calculate pearson correlation coefficient

x = np.array(data.evo_data)
y = np.array(data.rmp_data)
```


```python
# Pearson Correlation

corr, _ = pearsonr(x,y)
print('Pearsons correlation: %.3f' % corr)
```

    Pearsons correlation: 0.974



```python
# Effect Size Measurement

# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = mean(d1), mean(d2)
    # calculate the effect size
    return (u1 - u2) / s


d = cohend(y,x)
print('Cohens d: %.3f' % d)
```

    Cohens d: -0.151



```python
# Run Independant and dependant t-tests 

from scipy import stats
stats.ttest_ind(y,x)
```




    Ttest_indResult(statistic=array([-1.92689814]), pvalue=array([0.05443061]))




```python
stats.ttest_rel(y,x)
```




    Ttest_relResult(statistic=array([-9.42224983]), pvalue=array([8.77853164e-19]))




```python
# Calculate confusion and matrix and diagnosticsa

EVO=np.hstack(np.array(data['evo_resp']))
RMP=np.hstack(np.array(data['rmp_resp']))

cm = ConfusionMatrix(RMP, EVO)
print(cm)
cm.print_stats()
```

    Predicted  False  True  __all__
    Actual                         
    False        307     3      310
    True           4    10       14
    __all__      311    13      324
    population: 324
    P: 14
    N: 310
    PositiveTest: 13
    NegativeTest: 311
    TP: 10
    TN: 307
    FP: 3
    FN: 4
    TPR: 0.7142857142857143
    TNR: 0.9903225806451613
    PPV: 0.7692307692307693
    NPV: 0.9871382636655949
    FPR: 0.00967741935483871
    FDR: 0.23076923076923078
    FNR: 0.2857142857142857
    ACC: 0.9783950617283951
    F1_score: 0.7407407407407407
    MCC: 0.7300300641806626
    informedness: 0.7046082949308756
    markedness: 0.7563690328963641
    prevalence: 0.043209876543209874
    LRP: 73.80952380952381
    LRN: 0.288506281991624
    DOR: 255.83333333333337
    FOR: 0.012861736334405145



```python
# Draw Confusion matrix for report
# Motivation for this visualization has been taken from stackoverflow discussion at:
# https://stackoverflow.com/questions/48817300/sklearn-plot-confusion-matrix-combined-across-trainingtest-sets

from pylab import rcParams
rcParams['figure.figsize'] = 10,10

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt ), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('RMP Results - Truth', fontsize=15)
    plt.xlabel('EVO Results', fontsize=15)


cm = confusion_matrix(RMP, EVO)

plt.figure()
plot_confusion_matrix(cm, classes = [0,1], title='Confusion matrix')
```

    Confusion matrix, without normalization
    [[307   3]
     [  4  10]]



![png](output_12_1.png)

