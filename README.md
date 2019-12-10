
# Final Project Submission

Please fill out:
* Student name: Darius Fuller
* Student pace: Part-time
* Scheduled project review date/time: 12/9/2019 
* Instructor name: James Irving
* Blog post URL:https://medium.com/@d_full22/my-first-data-science-project-3e9f93e3ceb8
* Video of 5-min Non-Technical Presentation: https://youtu.be/JBZgxmkv5Cg


# TABLE OF CONTENTS 

*Click to jump to matching Markdown Header.*<br><br>

<font size=4rem>
    
- [Introduction](#INTRODUCTION)<br>
- **[OBTAIN](#OBTAIN)**<br>
- **[SCRUB](#SCRUB)**<br>
- **[EXPLORE](#EXPLORE)**<br>
- **[MODEL](#MODEL)**<br>
- **[iNTERPRET](#iNTERPRET)**<br>
- [Conclusions/Recommendations](#CONCLUSIONS-&-RECOMMENDATIONS)<br>
</font>
___


# INTRODUCTION

My goal with this project is to determine what are the top three factors that have the most influence on the price of a given home in King County, 
Washington. 

I will be taking the position of a real estate professional looking to create a model that can help me gain a reference point for understanding 
how much to reasonably sell or buy a home.

Negotiations can be tough and prices are not always based within reason, so having a tool to help determine a realistic price point for the home will provide me with a discrete price region to target during the negotiation process.
    
> I will try to find (if any):
   * Top three positive/negative predictors for price
   * Whether or not the waterfront carries any weight on price
   * Differences between multi and single story homes

# OBTAIN

> #### Importing necessary packages


```python
%matplotlib inline
import warnings
warnings.simplefilter("ignore")
```


```python
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import math
from scipy import stats
!pip install -U fsds_100719
from fsds_100719.imports import *
```

    fsds_1007219  v0.4.45 loaded.  Read the docs: https://fsds.readthedocs.io/en/latest/ 
    


<style  type="text/css" >
</style><table id="T_2e8e2530_1ae5_11ea_963e_b831b59cb35c" ><caption>Loaded Packages and Handles</caption><thead>    <tr>        <th class="col_heading level0 col0" >Handle</th>        <th class="col_heading level0 col1" >Package</th>        <th class="col_heading level0 col2" >Description</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_2e8e2530_1ae5_11ea_963e_b831b59cb35crow0_col0" class="data row0 col0" >dp</td>
                        <td id="T_2e8e2530_1ae5_11ea_963e_b831b59cb35crow0_col1" class="data row0 col1" >IPython.display</td>
                        <td id="T_2e8e2530_1ae5_11ea_963e_b831b59cb35crow0_col2" class="data row0 col2" >Display modules with helpful display and clearing commands.</td>
            </tr>
            <tr>
                                <td id="T_2e8e2530_1ae5_11ea_963e_b831b59cb35crow1_col0" class="data row1 col0" >fs</td>
                        <td id="T_2e8e2530_1ae5_11ea_963e_b831b59cb35crow1_col1" class="data row1 col1" >fsds_100719</td>
                        <td id="T_2e8e2530_1ae5_11ea_963e_b831b59cb35crow1_col2" class="data row1 col2" >Custom data science bootcamp student package</td>
            </tr>
            <tr>
                                <td id="T_2e8e2530_1ae5_11ea_963e_b831b59cb35crow2_col0" class="data row2 col0" >mpl</td>
                        <td id="T_2e8e2530_1ae5_11ea_963e_b831b59cb35crow2_col1" class="data row2 col1" >matplotlib</td>
                        <td id="T_2e8e2530_1ae5_11ea_963e_b831b59cb35crow2_col2" class="data row2 col2" >Matplotlib's base OOP module with formatting artists</td>
            </tr>
            <tr>
                                <td id="T_2e8e2530_1ae5_11ea_963e_b831b59cb35crow3_col0" class="data row3 col0" >plt</td>
                        <td id="T_2e8e2530_1ae5_11ea_963e_b831b59cb35crow3_col1" class="data row3 col1" >matplotlib.pyplot</td>
                        <td id="T_2e8e2530_1ae5_11ea_963e_b831b59cb35crow3_col2" class="data row3 col2" >Matplotlib's matlab-like plotting module</td>
            </tr>
            <tr>
                                <td id="T_2e8e2530_1ae5_11ea_963e_b831b59cb35crow4_col0" class="data row4 col0" >np</td>
                        <td id="T_2e8e2530_1ae5_11ea_963e_b831b59cb35crow4_col1" class="data row4 col1" >numpy</td>
                        <td id="T_2e8e2530_1ae5_11ea_963e_b831b59cb35crow4_col2" class="data row4 col2" >scientific computing with Python</td>
            </tr>
            <tr>
                                <td id="T_2e8e2530_1ae5_11ea_963e_b831b59cb35crow5_col0" class="data row5 col0" >pd</td>
                        <td id="T_2e8e2530_1ae5_11ea_963e_b831b59cb35crow5_col1" class="data row5 col1" >pandas</td>
                        <td id="T_2e8e2530_1ae5_11ea_963e_b831b59cb35crow5_col2" class="data row5 col2" >High performance data structures and tools</td>
            </tr>
            <tr>
                                <td id="T_2e8e2530_1ae5_11ea_963e_b831b59cb35crow6_col0" class="data row6 col0" >sns</td>
                        <td id="T_2e8e2530_1ae5_11ea_963e_b831b59cb35crow6_col1" class="data row6 col1" >seaborn</td>
                        <td id="T_2e8e2530_1ae5_11ea_963e_b831b59cb35crow6_col2" class="data row6 col2" >High-level data visualization library based on matplotlib</td>
            </tr>
    </tbody></table>


> #### Grabbing the dataset


```python
pd.set_option('display.max_columns',0)
csv="https://raw.githubusercontent.com/learn-co-students/dsc-v2-mod1-final-project-online-ds-pt-100719/master/kc_house_data.csv"
kc_house_df = pd.read_csv(csv)
```

> #### Functions used throughout


```python
def minmaxscaler(dataseries):
    """Takes in a pandas.series and returns a min-max
        scaled verison of the series. Can also take in iterable
        objects like lists.
    
    Args:
        dataseries (obj.): contains the values to be 
            scaled.
            
    Return:
        dataseries: min-max scaled copy of dataseries arg
    """
    return (dataseries - min(dataseries)) / (max(dataseries) - min(dataseries))
```


```python
def nrmlizer(dataseries):
    """Takes in a pandas.series and returns a mean normalized
        verison of the series. Can also take in iterable
        objects like lists.
    
    Args:
        dataseries (obj.): contains the values to be 
            scaled.
            
    Return:
        dataseries: mean normalized copy of dataseries arg
    """
    return (dataseries - np.mean(dataseries)) / (max(dataseries) - min(dataseries))
```


```python
def dot_product(x, y):
    """Takes in two interable objects and returns a single
        int or float dot product of them. 
    
    Args:
        x (obj.): contains the values to be multiplied.
        y (obj.): contains the values to be multiplied.
            
    Return:
        total: dot product of x and y 
            ** = (x[0]*y[0]) + (x[1]*y[1])...(x[n]*y[n])
    """
    Xdummy = list(x)
    Ydummy = list(y)
    total = 0
    for idx in list(range(0, len(x))):
        total += (Xdummy[idx]*Ydummy[idx]) 
    return total
```


```python
def correlation(var1, var2):
    """Takes in two interable objects and returns a single
        value representing the Pearson correlation bewtween them.
        If the objects are unequal in length, the function stops.
    
    Args:
        var1 (obj.): contains the values to be correlated.
        var2 (obj.): contains the values to be correlated.
            
    Return:
        Pearson correlation value of var1/var2  
    """    
    if len(var1) != len(var2):
        return 'The lengths of both the lists should be equal.' 
    
    else: 
        mean_norm_var1 = nrmlizer(var1)
        mean_norm_var2 = nrmlizer(var2)
        
        var1_dot_var2 = dot_product(mean_norm_var1, mean_norm_var2)
        
        var1_squared = [i * i for i in mean_norm_var1]
        var2_squared = [i * i for i in mean_norm_var2]
        
        return round(var1_dot_var2 / math.sqrt(sum(var1_squared) * sum(var2_squared)), 2)
```


```python
def check_column(df, col_name, target):
    """Displays info on null values, datatype, correlation,
        value counts and displays .describe()
    
    Args:
        df (df): contains the columns
        col_name (str): name of the df column to show
        n_unique (int): Number of unique values top show.
        target (str): name of the df column to be y for df.plot
    
    Return:
        fig, ax (Matplotlib Figure and Axes)
        
        ax[0]: KDE plot of col_name
        ax[1]: scatterplot of col_name and target
    """
    print('DataType:')
    print('\t',df[col_name].dtypes)
    
    num_nulls = df[col_name].isna().sum()
    print(f'Null Values Present = {num_nulls}')
    
    display(df[col_name].describe().round(decimals=3))
    
    print('\nValue Counts:')
    display(df[col_name].value_counts().head())
    
    print('\nCorrelation:')
    display(correlation(df[col_name], df[target]))
    
    fig, ax = plt.subplots(ncols=2, figsize=(18,10));
    
    df[col_name].plot.hist(density=True, label=col_name + 'Histogram', ax=ax[0]);
    df.plot(kind='scatter', x=col_name, y=target, label=col_name, ax=ax[1]);
    
    return fig,ax;
```


```python
def grab_outliers(series):
    """Takes in an interable obj of z-score values 
        (designed for pandas.series) and outputs a list
        of boolean values using abs(value) < 3 as a threshold.
    
    Args:
        series (obj.): iterable object of z-score values
    
    Return:
        outlie (list): list of boolean values corresponding
            to 'series' input.
            * True = outlier
    """
    outlie = []
    for idx in list(range(0, len(series))):
        if abs(series[idx]) > 3:
            outlie.append(True)
        else:
            outlie.append(False)
    return outlie
```


```python
def outlie_drop_bycol(df, col_name):
    """Takes in a pd.DataFrame and one of its columns.
        Function will go to column and each row, dropping the
        row from the DataFrame inplace with outliers in that
        column with respect to z-scores within -3 and 3.
    
    Args:
        df (pd.DataFrame): DataFrame
        col_name (str): 'name' of targeted column 
    
    Return:
        .describe() of new column without outliers
    """
    #'column' of z-scores
    z_col = pd.Series(stats.zscore(df[col_name]))
    
    #'column' of T/F for [abs(z-score) > 3] 
    z_col_outlie = pd.Series(grab_outliers(z_col))
    
    #dropping rows with True values from df 'inplace'
    for idx, item in zip(list(range(0, len(z_col_outlie))), z_col_outlie):
        if z_col_outlie[idx] == True:
            df.drop(idx, axis=0, inplace=True)
        else:
            pass
        
    #check results
    return df[col_name].describe()
```


```python
def kde_hist_plot(data):
    """Takes in a pd.DataFrame. Function will go to each
        column, producing a subplot with a hist/kde overlay.
        
    Args:
        data (pd.DataFrame): DataFrame
            
    Return:
        plt.plot() figure for each column with histogram/kde combo 
    """
    for col in data:
        data[col].plot.hist(density=True, label=col + ' Historgram')
        #data[col].plot.kde(label=col + ' KDE')
        plt.legend()
        plt.show()
```

# SCRUB

## Varibable Definitions

>* **id** - unique identified for a house
>* **date** - house was sold
>* **price** -  is prediction target
>* **bedrooms** -  of Bedrooms/House
>* **bathrooms** -  of bathrooms/bedrooms
>* **sqft_living** -  footage of the home
>* **sqft_lot** -  footage of the lot
>* **floors** -  floors (levels) in house
>* **waterfront** - House which has a view to a waterfront
>* **view** - Has been viewed
>* **condition** - How good the condition is ( Overall )
>* **grade** - overall grade given to the housing unit, based on King County grading system
>* **sqft_above** - square footage of house apart from basement
>* **sqft_basement** - square footage of the basement
>* **yr_built** - Built Year
>* **yr_renovated** - Year when house was renovated
>* **zipcode** - zip
>* **lat** - Latitude coordinate
>* **long** - Longitude coordinate
>* **sqft_living15** - The square footage of interior housing living space for the nearest 15 neighbors
>* **sqft_lot15** - The square footage of the land lots of the nearest 15 neighbors

## Scatter matrix to check briefly

* When I first tried to run, it took a while, so I decided to split into smaller dataframes, making sure to include price in both


```python
pd.plotting.scatter_matrix(kc_house_df[['price', 'date', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition']], figsize=((20,20)));
```


![png](output_24_0.png)


* For the above scatter matrix, it is clear that a couple of the variables have positive correlations with each other, like the square footage and room counts. Otherwise, I have a lot of categorical variables. 
* View for example does not visually appear to have any effect on anything, and with there being a high skew, this may be dropped.
* Of the continuous variables, they seem to have a slight skew and will need to be transformed. The categorical variables are highly skewed


```python
pd.plotting.scatter_matrix(kc_house_df[['price', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']], figsize=((20,20)));
```


![png](output_26_0.png)


* Similar to the first matrix, the only variables to carry a visually identifiable correlation with price are square footage variables (excluding sqft_lot15). Grade is a categorical variable that for sure has a correlation. As for the locational data (lat, long, and zipcode) there appears to be some correlation, but its hard to tell whether or not the connection is strong enough to be significant.
* It is obvious that yr_renovated does not have much impact on price visually, nor any other variable, and since the bulk of data have a 0 for this category, it will be a strong candidate to drop. 

> After looking at both matrices, there are a couple features I already have in mind to drop. A few potential drops that I will need to look further at, but I will make sure to cut out some of the square footage variables, as they will guaranteed carry correlation and interfere with eachother since the individual home's square footage could be calculated within the 15 nearest neighbors. Additionally, things such as condition and grade describe the same thing about a house. I will need to determine how to deal with the zipcode/lat/long variables, as they will have captured, in essence, the same info about a home. 

## Working column by column

### 'Id'


```python
check_column(kc_house_df, 'id', 'price');
```

    DataType:
    	 int64
    Null Values Present = 0
    


    count    2.159700e+04
    mean     4.580474e+09
    std      2.876736e+09
    min      1.000102e+06
    25%      2.123049e+09
    50%      3.904930e+09
    75%      7.308900e+09
    max      9.900000e+09
    Name: id, dtype: float64


    
    Value Counts:
    


    795000620     3
    1825069031    2
    2019200220    2
    7129304540    2
    1781500435    2
    Name: id, dtype: int64


    
    Correlation:
    


    -0.02



![png](output_31_6.png)


> Now that I've determined there are multiples for the same house, I will drop these because my goal is to find out what features lead to a price and having the _exact_ same home with different selling prices introduces time as a variable, which _is_ relevant in the real world, but is not controllable by any individual. One can influence how many bathrooms or square footage a home has however.


```python
#creating a column of boolean values for duplicate IDs
dupe_idx = kc_house_df['id'].duplicated(keep='first')

#copying for safety
kc_house_nodupe = kc_house_df

#if corresponding index in kc_house_df matches True value, row is dropped
for idx in list(range(0, len(dupe_idx))):
   if dupe_idx[idx] == True:
      kc_house_nodupe.drop(idx, axis=0, inplace=True)
   else:
      pass
```


```python
kc_house_nodupe.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 21420 entries, 0 to 21596
    Data columns (total 21 columns):
    id               21420 non-null int64
    date             21420 non-null object
    price            21420 non-null float64
    bedrooms         21420 non-null int64
    bathrooms        21420 non-null float64
    sqft_living      21420 non-null int64
    sqft_lot         21420 non-null int64
    floors           21420 non-null float64
    waterfront       19067 non-null float64
    view             21357 non-null float64
    condition        21420 non-null int64
    grade            21420 non-null int64
    sqft_above       21420 non-null int64
    sqft_basement    21420 non-null object
    yr_built         21420 non-null int64
    yr_renovated     17616 non-null float64
    zipcode          21420 non-null int64
    lat              21420 non-null float64
    long             21420 non-null float64
    sqft_living15    21420 non-null int64
    sqft_lot15       21420 non-null int64
    dtypes: float64(8), int64(11), object(2)
    memory usage: 3.6+ MB
    

> As seen above, there is a descrepancy between my indices (0-21596) and datapoints (21420). I will reset the index and drop the duplicate column


```python
#reset index into new df for safety
kc_house_newidx = kc_house_nodupe.reset_index()

#drop 'index' column generated by .reset_index()
kc_house_newidx.drop('index', axis=1, inplace=True)
```


```python
kc_house_newidx.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21420 entries, 0 to 21419
    Data columns (total 21 columns):
    id               21420 non-null int64
    date             21420 non-null object
    price            21420 non-null float64
    bedrooms         21420 non-null int64
    bathrooms        21420 non-null float64
    sqft_living      21420 non-null int64
    sqft_lot         21420 non-null int64
    floors           21420 non-null float64
    waterfront       19067 non-null float64
    view             21357 non-null float64
    condition        21420 non-null int64
    grade            21420 non-null int64
    sqft_above       21420 non-null int64
    sqft_basement    21420 non-null object
    yr_built         21420 non-null int64
    yr_renovated     17616 non-null float64
    zipcode          21420 non-null int64
    lat              21420 non-null float64
    long             21420 non-null float64
    sqft_living15    21420 non-null int64
    sqft_lot15       21420 non-null int64
    dtypes: float64(8), int64(11), object(2)
    memory usage: 3.4+ MB
    

> #### Better!

### 'Date'


```python
kc_house_newidx.drop('date', axis=1, inplace=True)
```

> I have decided to drop the 'date' column because of similar reasons to removing the duplicate IDs. This type of information related to the timing of the home being sold is not what I am looking to predict around and speaks more to the economy the house is being sold in rather than the home itself

### 'Price'


```python
check_column(kc_house_newidx, 'price', 'price');
```

    DataType:
    	 float64
    Null Values Present = 0
    


    count      21420.000
    mean      540739.304
    std       367931.110
    min        78000.000
    25%       322500.000
    50%       450000.000
    75%       645000.000
    max      7700000.000
    Name: price, dtype: float64


    
    Value Counts:
    


    350000.0    172
    450000.0    171
    550000.0    156
    500000.0    151
    425000.0    150
    Name: price, dtype: int64


    
    Correlation:
    


    1.0



![png](output_43_6.png)


> Looking at this information, it is apparent there are some outliers that are skewing the data to the right. I will need to deal with these in order to ensure that my model has decent data to model off of. I will operate off of z-scores within -3 and 3.


```python
#outlie_drop_bycol(kc_house_newidx, 'price')
```


```python
#check_column(kc_house_newidx, 'price', 'price');
```

> Although not normal, this look a bit more managable to deal with.


```python
kc_house_newidx.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21420 entries, 0 to 21419
    Data columns (total 20 columns):
    id               21420 non-null int64
    price            21420 non-null float64
    bedrooms         21420 non-null int64
    bathrooms        21420 non-null float64
    sqft_living      21420 non-null int64
    sqft_lot         21420 non-null int64
    floors           21420 non-null float64
    waterfront       19067 non-null float64
    view             21357 non-null float64
    condition        21420 non-null int64
    grade            21420 non-null int64
    sqft_above       21420 non-null int64
    sqft_basement    21420 non-null object
    yr_built         21420 non-null int64
    yr_renovated     17616 non-null float64
    zipcode          21420 non-null int64
    lat              21420 non-null float64
    long             21420 non-null float64
    sqft_living15    21420 non-null int64
    sqft_lot15       21420 non-null int64
    dtypes: float64(8), int64(11), object(1)
    memory usage: 3.3+ MB
    

* Need to .reset_index() again


```python
#reset index into new df for safety
kc_house_newidx2 = kc_house_newidx.reset_index()

#drop 'index' column generated by .reset_index()
kc_house_newidx2.drop('index', axis=1, inplace=True)
```

### 'Bedrooms'


```python
check_column(kc_house_newidx2, 'bedrooms', 'price');
```

    DataType:
    	 int64
    Null Values Present = 0
    


    count    21420.000
    mean         3.374
    std          0.925
    min          1.000
    25%          3.000
    50%          3.000
    75%          4.000
    max         33.000
    Name: bedrooms, dtype: float64


    
    Value Counts:
    


    3    9731
    4    6849
    2    2736
    5    1586
    6     265
    Name: bedrooms, dtype: int64


    
    Correlation:
    


    0.31



![png](output_52_6.png)


> Need to drop the _ONE_ datapoint skewing this distribution so much. It looks to be the max value of 33. Otherwise the large majority of datapoints fall bewteen the IQR. 


```python
#finding the index of the 33 room home
kc_house_newidx2[kc_house_newidx2['bedrooms'] == 33]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15710</th>
      <td>2402100895</td>
      <td>640000.0</td>
      <td>33</td>
      <td>1.75</td>
      <td>1620</td>
      <td>6000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1040</td>
      <td>580.0</td>
      <td>1947</td>
      <td>0.0</td>
      <td>98103</td>
      <td>47.6878</td>
      <td>-122.331</td>
      <td>1330</td>
      <td>4700</td>
    </tr>
  </tbody>
</table>
</div>




```python
#drop it!
kc_house_newidx2.drop(15710, axis=0, inplace=True)
```


```python
#recheck
check_column(kc_house_newidx2, 'bedrooms', 'price');
```

    DataType:
    	 int64
    Null Values Present = 0
    


    count    21419.000
    mean         3.373
    std          0.903
    min          1.000
    25%          3.000
    50%          3.000
    75%          4.000
    max         11.000
    Name: bedrooms, dtype: float64


    
    Value Counts:
    


    3    9731
    4    6849
    2    2736
    5    1586
    6     265
    Name: bedrooms, dtype: int64


    
    Correlation:
    


    0.32



![png](output_56_6.png)


> #### Looks better!

### 'Bathrooms'


```python
check_column(kc_house_newidx2, 'bathrooms', 'price');
```

    DataType:
    	 float64
    Null Values Present = 0
    


    count    21419.000
    mean         2.118
    std          0.769
    min          0.500
    25%          1.750
    50%          2.250
    75%          2.500
    max          8.000
    Name: bathrooms, dtype: float64


    
    Value Counts:
    


    2.50    5352
    1.00    3794
    1.75    3019
    2.25    2031
    2.00    1913
    Name: bathrooms, dtype: int64


    
    Correlation:
    


    0.53



![png](output_59_6.png)


> It appears to have a few outliers, I will check this with a boxplot, and then address them appropriately


```python
sns.boxplot(data=kc_house_newidx2['bathrooms']);
```


![png](output_61_0.png)


> According to this, I should be looking to drop any rows with more than 4 bathrooms. I will do so to see what I will get out of it.


```python
test_df = kc_house_newidx2[kc_house_newidx2['bathrooms'] <= 4]
check_column(test_df, 'bathrooms', 'price');
```

    DataType:
    	 float64
    Null Values Present = 0
    


    count    21154.000
    mean         2.086
    std          0.713
    min          0.500
    25%          1.500
    50%          2.250
    75%          2.500
    max          4.000
    Name: bathrooms, dtype: float64


    
    Value Counts:
    


    2.50    5352
    1.00    3794
    1.75    3019
    2.25    2031
    2.00    1913
    Name: bathrooms, dtype: int64


    
    Correlation:
    


    0.48



![png](output_63_6.png)


> Although this distribution looks better, I do want to investigate other columns further before dropping rows from the dataframe. I can always program something in to drop all outliers in one go.

### 'Sqft_living'


```python
check_column(kc_house_newidx2, 'sqft_living', 'price');
```

    DataType:
    	 int64
    Null Values Present = 0
    


    count    21419.000
    mean      2083.154
    std        918.824
    min        370.000
    25%       1430.000
    50%       1920.000
    75%       2550.000
    max      13540.000
    Name: sqft_living, dtype: float64


    
    Value Counts:
    


    1300    136
    1440    133
    1400    132
    1660    128
    1800    128
    Name: sqft_living, dtype: int64


    
    Correlation:
    


    0.7



![png](output_66_6.png)


> There is a strong correlation here as-is! I will definitely look to remove some outliers, since there appears to be a bit of a right-sided tail. I will need to pick one or two of these 'sqft' variables to keep, since I am assumming there will be collinearity between them.

### 'Sqft_lot'


```python
check_column(kc_house_newidx2, 'sqft_lot', 'price');
```

    DataType:
    	 int64
    Null Values Present = 0
    


    count      21419.000
    mean       15128.464
    std        41531.720
    min          520.000
    25%         5040.000
    50%         7614.000
    75%        10692.000
    max      1651359.000
    Name: sqft_lot, dtype: float64


    
    Value Counts:
    


    5000    355
    6000    285
    4000    249
    7200    218
    7500    118
    Name: sqft_lot, dtype: int64


    
    Correlation:
    


    0.09



![png](output_69_6.png)


> Really low correlation and a large skew to the right. I will definitely need to remove outliers and/or normalize to 'unbunch' the data better. Will include this in the 'one shot' removal done later on.

> It also may not be very valuable as a predictor because near the zero mark on the scatter, the price variable goes to the top. It seems as the lot gets bigger, maybe indcating leaving city limits, the price isn't quite following this trend. 

### 'Floors'


```python
check_column(kc_house_newidx2, 'floors', 'price');
```

    DataType:
    	 float64
    Null Values Present = 0
    


    count    21419.000
    mean         1.496
    std          0.540
    min          1.000
    25%          1.000
    50%          1.500
    75%          2.000
    max          3.500
    Name: floors, dtype: float64


    
    Value Counts:
    


    1.0    10551
    2.0     8203
    1.5     1888
    3.0      609
    2.5      161
    Name: floors, dtype: int64


    
    Correlation:
    


    0.26



![png](output_72_6.png)


> Since this an obvious nominal category, I will need to create dummy variables for this column. I will, per my targeted goals, split this variable into a binary variable, with 1 indicating multi-leveled homes.

### 'Waterfront'


```python
check_column(kc_house_newidx2, 'waterfront', 'price');
```

    DataType:
    	 float64
    Null Values Present = 2353
    


    count    19066.000
    mean         0.008
    std          0.087
    min          0.000
    25%          0.000
    50%          0.000
    75%          0.000
    max          1.000
    Name: waterfront, dtype: float64


    
    Value Counts:
    


    0.0    18920
    1.0      146
    Name: waterfront, dtype: int64


    
    Correlation:
    


    nan



![png](output_75_6.png)


* There is a small amount of null values here, but it is evident (per the scatter plot) that there may be some influence on the price if this variable is set to one. My strategy will be to randomly sample and fill these null values because I do not want to drop roughly 10% of my data.


```python
#creating copy to manipulate
dummyColumn = kc_house_newidx2['waterfront'].copy()

#boolean column indicating if item is a NA or not
nullwater = dummyColumn.isna()

#setting the sampling of values to those True-valued indices in dummyColumn
dummyColumn.loc[nullwater] = dummyColumn.dropna().sample(nullwater.sum()).values

#finalizing it
kc_house_newidx2['waterfront'] = dummyColumn.copy()
```


```python
check_column(kc_house_newidx2, 'waterfront', 'price');
```

    DataType:
    	 float64
    Null Values Present = 0
    


    count    21419.000
    mean         0.008
    std          0.087
    min          0.000
    25%          0.000
    50%          0.000
    75%          0.000
    max          1.000
    Name: waterfront, dtype: float64


    
    Value Counts:
    


    0.0    21255
    1.0      164
    Name: waterfront, dtype: int64


    
    Correlation:
    


    0.25



![png](output_78_6.png)


> #### A little better!

### 'View'


```python
kc_house_newidx2.drop('view', axis=1, inplace=True)
```

> I have decided to drop the 'view' column for the same reasons I have dropped 'id' and 'date'. This variable only lets me know how many times the home was viewed before sold. Looking at the scatter plot on this data in the scatter matrix, there is no clear trend to base off of either.

### 'Condition'


```python
check_column(kc_house_newidx2, 'condition', 'price');
```

    DataType:
    	 int64
    Null Values Present = 0
    


    count    21419.000
    mean         3.411
    std          0.650
    min          1.000
    25%          3.000
    50%          3.000
    75%          4.000
    max          5.000
    Name: condition, dtype: float64


    
    Value Counts:
    


    3    13900
    4     5643
    5     1686
    2      162
    1       28
    Name: condition, dtype: int64


    
    Correlation:
    


    0.03



![png](output_84_6.png)


> This is an ordinal category and thus carries some influence on the price as the condition value gets higher. This is evidenced in the scatter plot. I will split this into dummy variables per each condition since there are not too many of them.

### 'Grade'


```python
check_column(kc_house_newidx2, 'grade', 'price');
```

    DataType:
    	 int64
    Null Values Present = 0
    


    count    21419.000
    mean         7.663
    std          1.172
    min          3.000
    25%          7.000
    50%          7.000
    75%          8.000
    max         13.000
    Name: grade, dtype: float64


    
    Value Counts:
    


    7     8888
    8     6041
    9     2606
    6     1995
    10    1130
    Name: grade, dtype: int64


    
    Correlation:
    


    0.67



![png](output_87_6.png)


> This is an almost-normal, ordinal category! I will split this into bins and further create dummy variables to help include this in the model since it carries such a strong correlation with price.

### 'Sqft_above'


```python
check_column(kc_house_newidx2, 'sqft_above', 'price');
```

    DataType:
    	 int64
    Null Values Present = 0
    


    count    21419.000
    mean      1791.205
    std        828.696
    min        370.000
    25%       1200.000
    50%       1560.000
    75%       2220.000
    max       9410.000
    Name: sqft_above, dtype: float64


    
    Value Counts:
    


    1300    210
    1010    204
    1200    203
    1220    186
    1140    183
    Name: sqft_above, dtype: int64


    
    Correlation:
    


    0.61



![png](output_90_6.png)


> This has less of a correlation than the sqft_living variable and describes almost the same thing. Also has a right skew and will require a log or dropping of outliers. May drop completely due to collinearity.

### 'Sqft_basement'

* After getting an error on check_column(), I found this column carries a placeholder of '?' and has a dtype of 'object'

* This needs to be changed to a number before manipulation. My plan is to split the values of those homes under the ? category into single and multi-leveled homes. Those multi-leveled homes will get the median value input while the remaining getting 0 input.


```python
kc_house_newidx2['sqft_basement'].value_counts().head()
```




    0.0      12717
    ?          452
    600.0      216
    500.0      206
    700.0      205
    Name: sqft_basement, dtype: int64




```python
#setting target columns to a copied df to manipulate
dummyDf = kc_house_newidx2[['sqft_basement', 'floors']].copy()

#just multi-leveled homes with a placeholder
basemcol = dummyDf[(dummyDf['sqft_basement'] == '?') & (dummyDf['floors'] > 1)].copy()

#dropping floors to access target only
basemcol = basemcol['sqft_basement']

#replacement of placeholder
basemcol.replace(to_replace='?', value='292', inplace=True)

#same as above but for single-leveled homes
basemcol2 = dummyDf[(dummyDf['sqft_basement'] == '?') & (dummyDf['floors'] <= 1)].copy()
basemcol2 = basemcol2['sqft_basement']
basemcol2.replace(to_replace='?', value='0', inplace=True)

#combining the two columns into one
basemcol_comb = pd.concat([basemcol, basemcol2])
```


```python
#updating 'sqft_basement' column with new values
kc_house_newidx2.update(basemcol_comb)
```


```python
#setting new column into df as a float dtype
kc_house_newidx2['sqft_basement'] = kc_house_newidx2['sqft_basement'].astype('float64', copy=False)
```


```python
check_column(kc_house_newidx2, 'sqft_basement', 'price');
```

    DataType:
    	 float64
    Null Values Present = 0
    


    count    21419.000
    mean       289.108
    std        438.978
    min          0.000
    25%          0.000
    50%          0.000
    75%        550.000
    max       4820.000
    Name: sqft_basement, dtype: float64


    
    Value Counts:
    


    0.0      12933
    292.0      236
    600.0      216
    500.0      206
    700.0      205
    Name: sqft_basement, dtype: int64


    
    Correlation:
    


    0.32



![png](output_98_6.png)


> There is a weak correlation that is visible via the scatter plot, but it is also very obvious that a strong skew is being caused due to over half of the data being a 0. May try to standardize and will be a potential candidate for dropping.

### 'Yr_built'


```python
check_column(kc_house_newidx2, 'yr_built', 'price');
```

    DataType:
    	 int64
    Null Values Present = 0
    


    count    21419.000
    mean      1971.094
    std         29.387
    min       1900.000
    25%       1952.000
    50%       1975.000
    75%       1997.000
    max       2015.000
    Name: yr_built, dtype: float64


    
    Value Counts:
    


    2014    559
    2006    453
    2005    450
    2004    429
    2003    420
    Name: yr_built, dtype: int64


    
    Correlation:
    


    0.05



![png](output_101_6.png)


> This is an interesting feature. There appears visually to be no real correlation between the year a home was built, and the price of a home. Due to this, I may choose to drop this one in order to trim down the number of features in my model. I will more than likely split the category in half (pre vs post 1975) just to see if this will have an influence, although I doubt it.

### 'Yr_renovated'


```python
check_column(kc_house_newidx2, 'yr_renovated', 'price');
```

    DataType:
    	 float64
    Null Values Present = 3804
    


    count    17615.000
    mean        83.852
    std        400.447
    min          0.000
    25%          0.000
    50%          0.000
    75%          0.000
    max       2015.000
    Name: yr_renovated, dtype: float64


    
    Value Counts:
    


    0.0       16875
    2014.0       73
    2003.0       31
    2013.0       31
    2007.0       30
    Name: yr_renovated, dtype: int64


    
    Correlation:
    


    nan



![png](output_104_6.png)


> By appearance, there is not much correlation between house price and date of the renovations done on the home. Additionally, approximately 17.7% of the data is null and of the remaining data 96% has the value of 0. There is not enough indication that this bears any weight on the price to keep it around. I will drop it. 


```python
kc_house_newidx2 = kc_house_newidx2.drop('yr_renovated', axis=1)
```

### 'Zipcode'


```python
check_column(kc_house_newidx2, 'zipcode', 'price');
```

    DataType:
    	 int64
    Null Values Present = 0
    


    count    21419.000
    mean     98077.873
    std         53.478
    min      98001.000
    25%      98033.000
    50%      98065.000
    75%      98117.000
    max      98199.000
    Name: zipcode, dtype: float64


    
    Value Counts:
    


    98103    599
    98038    586
    98115    576
    98052    571
    98117    548
    Name: zipcode, dtype: int64


    
    Correlation:
    


    -0.05



![png](output_108_6.png)


> Visually, it does not seem that zipcode alone provides a lot of influence on the price of a home. However, I can tell that some zipcodes do not have a high maximum value for price. I will split these into dummy columns, despite running into a large number of features because of it. King County includes more than Seattle, so I may be able to bin them into 'City' zipcode bins.

### 'Lat'


```python
check_column(kc_house_newidx2, 'lat', 'price');
```

    DataType:
    	 float64
    Null Values Present = 0
    


    count    21419.000
    mean        47.560
    std          0.139
    min         47.156
    25%         47.471
    50%         47.572
    75%         47.678
    max         47.778
    Name: lat, dtype: float64


    
    Value Counts:
    


    47.6624    17
    47.5322    17
    47.5491    17
    47.6846    17
    47.6711    16
    Name: lat, dtype: int64


    
    Correlation:
    


    0.31



![png](output_111_6.png)


> This has a similar setup to the zipcode data, but in a less categorical manner. I will have to choose between which to include because I will expect some collinearity. More than likely will drop this because some of my online research shows that it is not useful unless put into a category, which zipcode is already neatly done for me. Additionally, latitude and longitude need to be as precise as possible to work (no rounding).


```python
kc_house_newidx2 = kc_house_newidx2.drop('lat', axis=1)
```

### 'Long'


```python
check_column(kc_house_newidx2, 'long', 'price');
```

    DataType:
    	 float64
    Null Values Present = 0
    


    count    21419.000
    mean      -122.214
    std          0.141
    min       -122.519
    25%       -122.328
    50%       -122.230
    75%       -122.125
    max       -121.315
    Name: long, dtype: float64


    
    Value Counts:
    


    -122.290    113
    -122.300    110
    -122.362    102
    -122.291    100
    -122.363     99
    Name: long, dtype: int64


    
    Correlation:
    


    0.02



![png](output_115_6.png)


> Less evident that longitude affects price like the latitude does, but for similar reasons, I will drop this variable. Normally the two are used in conjunction as well to plot points on a map. This will not be very useful for my regression model.


```python
kc_house_newidx2 = kc_house_newidx2.drop('long', axis=1)
```

### 'Sqft_living15'


```python
check_column(kc_house_newidx2, 'sqft_living15', 'price');
```

    DataType:
    	 int64
    Null Values Present = 0
    


    count    21419.000
    mean      1988.415
    std        685.538
    min        399.000
    25%       1490.000
    50%       1840.000
    75%       2370.000
    max       6210.000
    Name: sqft_living15, dtype: float64


    
    Value Counts:
    


    1540    193
    1560    190
    1440    190
    1500    178
    1460    168
    Name: sqft_living15, dtype: int64


    
    Correlation:
    


    0.58



![png](output_119_6.png)


> Looks very similar to the other square footage variables. I will need to keep one or more of them for sure, this will be a drop candidate due to the nature of what the variable itself captures. 

### 'Sqft_lot15'


```python
check_column(kc_house_newidx2, 'sqft_lot15', 'price');
```

    DataType:
    	 int64
    Null Values Present = 0
    


    count     21419.000
    mean      12776.095
    std       27346.205
    min         651.000
    25%        5100.000
    50%        7620.000
    75%       10086.500
    max      871200.000
    Name: sqft_lot15, dtype: float64


    
    Value Counts:
    


    5000    425
    4000    354
    6000    285
    7200    209
    4800    144
    Name: sqft_lot15, dtype: int64


    
    Correlation:
    


    0.08



![png](output_122_6.png)


> This one is even more skewed than the other lot variable and would need to have outliers dropped in order to be viable. Strong drop candidate.

### Finally removing 'id'

> Now that I have looked over all of the columns, I will drop the 'id' column that I was using as a way to ensure I have unique homes within the data set.


```python
kc_house_newidx2 = kc_house_newidx2.drop('id', axis=1)
```

## Removing the outliers


```python
#This will use the stats.zscore() to grab all abs(zscores) less than 3
    #and place them in a separate df
kc_no_outlier = kc_house_newidx2[(np.abs(stats.zscore(kc_house_newidx2)) < 3).all(axis=1)]
```


```python
#need to reset index
kc_no_outlier = kc_no_outlier.reset_index(drop=True)
kc_no_outlier.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 19925 entries, 0 to 19924
    Data columns (total 15 columns):
    price            19925 non-null float64
    bedrooms         19925 non-null int64
    bathrooms        19925 non-null float64
    sqft_living      19925 non-null int64
    sqft_lot         19925 non-null int64
    floors           19925 non-null float64
    waterfront       19925 non-null float64
    condition        19925 non-null int64
    grade            19925 non-null int64
    sqft_above       19925 non-null int64
    sqft_basement    19925 non-null float64
    yr_built         19925 non-null int64
    zipcode          19925 non-null int64
    sqft_living15    19925 non-null int64
    sqft_lot15       19925 non-null int64
    dtypes: float64(5), int64(10)
    memory usage: 2.3 MB
    

## Heatmap check for collinearity


```python
sns.set(style="white")

#Grabbing the correlation matrix
corr = kc_no_outlier.corr()

#Making a mask
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

#Setting the figure
fig, ax = plt.subplots(figsize=(11, 9))

#Generate the heatmap
sns.heatmap(corr, mask=mask, cmap='RdBu_r', vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True);
```


![png](output_131_0.png)


> Looking at the heatmap, there are high collinearity between:

* SQFT_LIVING & SQFT_ABOVE (0.85) & GRADE (0.71) & SQFT_LIVING15 (0.74) & BATHROOMS (0.71)
* SQFT_LOT & SQFT_LOT15 (0.82)
* SQFT_LIVING15 & SQFT_ABOVE (0.72)
* SQFT_ABOVE & GRADE (0.71)

> Price the target variable, shows collinearity with:

* SQFT_LIVING (0.63)
* GRADE (0.64)

> I will drop both of the sqft...15 features since they show high collinearity with the other sqft features and lot15 has a lower correlation with price than the sqft_lot does. Sqft_above will be dropped for similar reasoning as with sqft_lot15.


```python
kc_no_outlier.drop(['sqft_living15', 'sqft_lot15', 'sqft_above'], axis=1, inplace=True)
```


```python
#kc_no_outlier.drop('sqft_above', axis=1, inplace=True)
```


```python
#Grabbing the correlation matrix
corr2 = kc_no_outlier.corr()

#Making a mask
mask = np.zeros_like(corr2, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

#Setting the figure
fig, ax = plt.subplots(figsize=(11, 9))

#Generate the heatmap
sns.heatmap(corr2, mask=mask, cmap='RdBu_r', vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True);
```


![png](output_137_0.png)


> For right now I will leave the remaining hotspots in as none are beyond my threshold of 0.75 and may still provide good information.

## Creating dummy columns

#### Floors


```python
#creating a series of binary values to put into bins easier
flrs_binary = pd.Series([])
for idx, item in zip(list(range(0, len(kc_no_outlier['floors']))), kc_no_outlier['floors']):
    if item < 1.5:
        flrs_binary[idx] = 0
    else:
        flrs_binary[idx] = 1  
```


```python
bins = [0, 0.5, 1]
bins_flrs = pd.cut(flrs_binary, bins)
bins_flrs = bins_flrs.cat.as_unordered()
flrs_dummy = pd.get_dummies(bins_flrs, prefix='flrs', drop_first=True)
            #len(flrs_dummy)
```


```python
kc_no_outlier = kc_no_outlier.drop('floors', axis=1)
kc_no_outlier = pd.concat([kc_no_outlier, flrs_dummy], axis=1)
```

***************

> * At this point I will have two different iterations of the same dataset. **kc_no_outlier** will represent my __original__ attempt at the model. **kc_no_out2** will represent my __revisional__ changes for my model.


```python
kc_no_out2 = kc_no_outlier.copy()
```

*********************

#### Condition


```python
bins = [0, 1, 2, 3, 4, 5]
bins_cond = pd.cut(kc_no_outlier['condition'], bins)
bins_cond = bins_cond.cat.as_unordered()
cond_dummy = pd.get_dummies(bins_cond, prefix='cond', drop_first=True)
            #len(cond_dummy)
```


```python
kc_no_outlier = kc_no_outlier.drop('condition', axis=1)
kc_no_outlier = pd.concat([kc_no_outlier, cond_dummy], axis=1)
```

**************************

> * I will be leaving condition un-binned for _kc_no_out2_

#### Yr_built


```python
bins = [0, 1975, 2016]
bins_yrbuilt = pd.cut(kc_no_outlier['yr_built'], bins)
bins_yrbuilt = bins_yrbuilt.cat.as_unordered()
yrbuilt_dummy = pd.get_dummies(bins_yrbuilt, prefix='yr_built', drop_first=True)
            #len(yrbuilt_dummy)
```


```python
kc_no_outlier = kc_no_outlier.drop('yr_built', axis=1)
kc_no_outlier = pd.concat([kc_no_outlier, yrbuilt_dummy], axis=1)
```

***************************


```python
#same as above

bins = [0, 1975, 2016]
bins_yrbuilt = pd.cut(kc_no_out2['yr_built'], bins)
bins_yrbuilt = bins_yrbuilt.cat.as_unordered()
yrbuilt_dummy = pd.get_dummies(bins_yrbuilt, prefix='yr_built', drop_first=True)
```


```python
kc_no_out2 = kc_no_out2.drop('yr_built', axis=1)
kc_no_out2 = pd.concat([kc_no_out2, yrbuilt_dummy], axis=1)
```

#### Zipcode


```python
#creating a list of unique zipcodes to use as bins
ziplist = list(kc_no_outlier['zipcode'].unique())
bins = [0]
bins.extend(ziplist)
bins = sorted(bins)

bins_zipC = pd.cut(kc_no_outlier['zipcode'], bins)
bins_zipC = bins_zipC.cat.as_unordered()
zipC_dummy = pd.get_dummies(bins_zipC, prefix='zipcode', drop_first=True)
            #len(zipC_dummy)
```


```python
kc_no_outlier = kc_no_outlier.drop('zipcode', axis=1)
kc_no_outlier = pd.concat([kc_no_outlier, zipC_dummy], axis=1)
```

**************************


```python
#creating dummies without binning
zipC_dummy = pd.get_dummies(kc_no_out2['zipcode'], prefix='zipcode', drop_first=True)

#assigning to df
kc_no_out2 = kc_no_out2.drop('zipcode', axis=1)
kc_no_out2 = pd.concat([kc_no_out2, zipC_dummy], axis=1)
```

### Final-_ish_ Check


```python
kc_no_outlier.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 19925 entries, 0 to 19924
    Data columns (total 83 columns):
    price                     19925 non-null float64
    bedrooms                  19925 non-null int64
    bathrooms                 19925 non-null float64
    sqft_living               19925 non-null int64
    sqft_lot                  19925 non-null int64
    waterfront                19925 non-null float64
    grade                     19925 non-null int64
    sqft_basement             19925 non-null float64
    flrs_(0.5, 1.0]           19925 non-null uint8
    cond_(1, 2]               19925 non-null uint8
    cond_(2, 3]               19925 non-null uint8
    cond_(3, 4]               19925 non-null uint8
    cond_(4, 5]               19925 non-null uint8
    yr_built_(1975, 2016]     19925 non-null uint8
    zipcode_(98001, 98002]    19925 non-null uint8
    zipcode_(98002, 98003]    19925 non-null uint8
    zipcode_(98003, 98004]    19925 non-null uint8
    zipcode_(98004, 98005]    19925 non-null uint8
    zipcode_(98005, 98006]    19925 non-null uint8
    zipcode_(98006, 98007]    19925 non-null uint8
    zipcode_(98007, 98008]    19925 non-null uint8
    zipcode_(98008, 98010]    19925 non-null uint8
    zipcode_(98010, 98011]    19925 non-null uint8
    zipcode_(98011, 98014]    19925 non-null uint8
    zipcode_(98014, 98019]    19925 non-null uint8
    zipcode_(98019, 98022]    19925 non-null uint8
    zipcode_(98022, 98023]    19925 non-null uint8
    zipcode_(98023, 98024]    19925 non-null uint8
    zipcode_(98024, 98027]    19925 non-null uint8
    zipcode_(98027, 98028]    19925 non-null uint8
    zipcode_(98028, 98029]    19925 non-null uint8
    zipcode_(98029, 98030]    19925 non-null uint8
    zipcode_(98030, 98031]    19925 non-null uint8
    zipcode_(98031, 98032]    19925 non-null uint8
    zipcode_(98032, 98033]    19925 non-null uint8
    zipcode_(98033, 98034]    19925 non-null uint8
    zipcode_(98034, 98038]    19925 non-null uint8
    zipcode_(98038, 98039]    19925 non-null uint8
    zipcode_(98039, 98040]    19925 non-null uint8
    zipcode_(98040, 98042]    19925 non-null uint8
    zipcode_(98042, 98045]    19925 non-null uint8
    zipcode_(98045, 98052]    19925 non-null uint8
    zipcode_(98052, 98053]    19925 non-null uint8
    zipcode_(98053, 98055]    19925 non-null uint8
    zipcode_(98055, 98056]    19925 non-null uint8
    zipcode_(98056, 98058]    19925 non-null uint8
    zipcode_(98058, 98059]    19925 non-null uint8
    zipcode_(98059, 98065]    19925 non-null uint8
    zipcode_(98065, 98070]    19925 non-null uint8
    zipcode_(98070, 98072]    19925 non-null uint8
    zipcode_(98072, 98074]    19925 non-null uint8
    zipcode_(98074, 98075]    19925 non-null uint8
    zipcode_(98075, 98077]    19925 non-null uint8
    zipcode_(98077, 98092]    19925 non-null uint8
    zipcode_(98092, 98102]    19925 non-null uint8
    zipcode_(98102, 98103]    19925 non-null uint8
    zipcode_(98103, 98105]    19925 non-null uint8
    zipcode_(98105, 98106]    19925 non-null uint8
    zipcode_(98106, 98107]    19925 non-null uint8
    zipcode_(98107, 98108]    19925 non-null uint8
    zipcode_(98108, 98109]    19925 non-null uint8
    zipcode_(98109, 98112]    19925 non-null uint8
    zipcode_(98112, 98115]    19925 non-null uint8
    zipcode_(98115, 98116]    19925 non-null uint8
    zipcode_(98116, 98117]    19925 non-null uint8
    zipcode_(98117, 98118]    19925 non-null uint8
    zipcode_(98118, 98119]    19925 non-null uint8
    zipcode_(98119, 98122]    19925 non-null uint8
    zipcode_(98122, 98125]    19925 non-null uint8
    zipcode_(98125, 98126]    19925 non-null uint8
    zipcode_(98126, 98133]    19925 non-null uint8
    zipcode_(98133, 98136]    19925 non-null uint8
    zipcode_(98136, 98144]    19925 non-null uint8
    zipcode_(98144, 98146]    19925 non-null uint8
    zipcode_(98146, 98148]    19925 non-null uint8
    zipcode_(98148, 98155]    19925 non-null uint8
    zipcode_(98155, 98166]    19925 non-null uint8
    zipcode_(98166, 98168]    19925 non-null uint8
    zipcode_(98168, 98177]    19925 non-null uint8
    zipcode_(98177, 98178]    19925 non-null uint8
    zipcode_(98178, 98188]    19925 non-null uint8
    zipcode_(98188, 98198]    19925 non-null uint8
    zipcode_(98198, 98199]    19925 non-null uint8
    dtypes: float64(4), int64(4), uint8(75)
    memory usage: 2.6 MB
    


```python
kc_no_out2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 19925 entries, 0 to 19924
    Data columns (total 80 columns):
    price                    19925 non-null float64
    bedrooms                 19925 non-null int64
    bathrooms                19925 non-null float64
    sqft_living              19925 non-null int64
    sqft_lot                 19925 non-null int64
    waterfront               19925 non-null float64
    condition                19925 non-null int64
    grade                    19925 non-null int64
    sqft_basement            19925 non-null float64
    flrs_(0.5, 1.0]          19925 non-null uint8
    yr_built_(1975, 2016]    19925 non-null uint8
    zipcode_98002            19925 non-null uint8
    zipcode_98003            19925 non-null uint8
    zipcode_98004            19925 non-null uint8
    zipcode_98005            19925 non-null uint8
    zipcode_98006            19925 non-null uint8
    zipcode_98007            19925 non-null uint8
    zipcode_98008            19925 non-null uint8
    zipcode_98010            19925 non-null uint8
    zipcode_98011            19925 non-null uint8
    zipcode_98014            19925 non-null uint8
    zipcode_98019            19925 non-null uint8
    zipcode_98022            19925 non-null uint8
    zipcode_98023            19925 non-null uint8
    zipcode_98024            19925 non-null uint8
    zipcode_98027            19925 non-null uint8
    zipcode_98028            19925 non-null uint8
    zipcode_98029            19925 non-null uint8
    zipcode_98030            19925 non-null uint8
    zipcode_98031            19925 non-null uint8
    zipcode_98032            19925 non-null uint8
    zipcode_98033            19925 non-null uint8
    zipcode_98034            19925 non-null uint8
    zipcode_98038            19925 non-null uint8
    zipcode_98039            19925 non-null uint8
    zipcode_98040            19925 non-null uint8
    zipcode_98042            19925 non-null uint8
    zipcode_98045            19925 non-null uint8
    zipcode_98052            19925 non-null uint8
    zipcode_98053            19925 non-null uint8
    zipcode_98055            19925 non-null uint8
    zipcode_98056            19925 non-null uint8
    zipcode_98058            19925 non-null uint8
    zipcode_98059            19925 non-null uint8
    zipcode_98065            19925 non-null uint8
    zipcode_98070            19925 non-null uint8
    zipcode_98072            19925 non-null uint8
    zipcode_98074            19925 non-null uint8
    zipcode_98075            19925 non-null uint8
    zipcode_98077            19925 non-null uint8
    zipcode_98092            19925 non-null uint8
    zipcode_98102            19925 non-null uint8
    zipcode_98103            19925 non-null uint8
    zipcode_98105            19925 non-null uint8
    zipcode_98106            19925 non-null uint8
    zipcode_98107            19925 non-null uint8
    zipcode_98108            19925 non-null uint8
    zipcode_98109            19925 non-null uint8
    zipcode_98112            19925 non-null uint8
    zipcode_98115            19925 non-null uint8
    zipcode_98116            19925 non-null uint8
    zipcode_98117            19925 non-null uint8
    zipcode_98118            19925 non-null uint8
    zipcode_98119            19925 non-null uint8
    zipcode_98122            19925 non-null uint8
    zipcode_98125            19925 non-null uint8
    zipcode_98126            19925 non-null uint8
    zipcode_98133            19925 non-null uint8
    zipcode_98136            19925 non-null uint8
    zipcode_98144            19925 non-null uint8
    zipcode_98146            19925 non-null uint8
    zipcode_98148            19925 non-null uint8
    zipcode_98155            19925 non-null uint8
    zipcode_98166            19925 non-null uint8
    zipcode_98168            19925 non-null uint8
    zipcode_98177            19925 non-null uint8
    zipcode_98178            19925 non-null uint8
    zipcode_98188            19925 non-null uint8
    zipcode_98198            19925 non-null uint8
    zipcode_98199            19925 non-null uint8
    dtypes: float64(4), int64(5), uint8(71)
    memory usage: 2.7 MB
    

# EXPLORE

### Dealing with collinearity

> Now I will begin playing with the distributions to get something I feel comfortable with modeling. I will first split off the price variable into it's own series, leaving the remaining dataframe as my 'features'.


```python
#one for each
price_series = kc_no_outlier['price'].copy()
price_series2 = kc_no_out2['price'].copy()
```


```python
kc_house_feat1 = kc_no_outlier.drop('price', axis=1).copy()
kc_house_feat1_2 = kc_no_out2.drop('price', axis=1).copy()
```


```python
corr3 = np.abs(kc_house_feat1.corr().round(3))
corr3_2 = np.abs(kc_house_feat1_2.corr().round(3))
```


```python
#corr3[corr3 > 0.75]     #this was really large

(corr3[abs(corr3 > 0.75)].sum() > 1).value_counts()
```




    False    80
    True      2
    dtype: int64



****************


```python
(corr3_2[abs(corr3_2 > 0.75)].sum() > 1).value_counts()
```




    False    79
    dtype: int64



> COND_(2, 3] & COND_(3, 4] have collinearity, but I will keep for now since they came from the same category

> For the above, I looked at a very large dataframe that had a bunch of NaN values. To remedy this, I looked at a boolean series of the columns indicating which had a correlation greater than 0.75 with another column. This was acheived by asking python to return those columns with a sum of the non-null correlation values greater than 1 (every column has 1 due to matching with itself).

> Then I used the .value_counts() method to count up how many columns I need to target, since Jupyter did not display _every_ column in the dataframe.

* #### These columns are: 

>'cond_(2, 3]' , 'cond_(3, 4]'. 

* I will not remove one of these since I have already removed the first bin. If I see anything weird with the P-values, I will adjust accordingly


```python
#making it a copy for safety

kc_house_feat_nocorr = kc_house_feat1.copy()
kc_house_feat_nocorr2 = kc_house_feat1_2.copy()
```

### Transforming features

> I have decided to min-max scale the sqft features to allow them to be on a similar scale and not mess too much with the bed/bathroom features due to the large values within them currently.


```python
#making copies for safety
no_zip_feat = kc_house_feat_nocorr[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_basement']].copy()
no_zip_feat2 = no_zip_feat.copy()

#revised df
no_zip_feat_2 = kc_house_feat_nocorr2[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_basement']].copy()
no_zip_feat2_new = no_zip_feat_2.copy()
```

> For my revisions I will normalize the features. I will be adding bedrooms and bathrooms to make sure __all__ of the features are within the same range. I am hoping this will increase my predictive accuracy. I am interested to see how having 'negative' bedrooms, etc. will affect the output modeling as well. 


```python
tfsqftliving = minmaxscaler(no_zip_feat['sqft_living'])
tfsqftlot = minmaxscaler(no_zip_feat['sqft_lot'])
tfsqftbasement = minmaxscaler(no_zip_feat['sqft_basement'])

#revised df
tfsqftliving_2 = nrmlizer(no_zip_feat_2['sqft_living'])
tfsqftlot_2 = nrmlizer(no_zip_feat_2['sqft_lot'])
tfsqftbasement_2 = nrmlizer(no_zip_feat_2['sqft_basement'])
tfbedrooms_2 = nrmlizer(no_zip_feat_2['bedrooms'])
tfbathooms_2 = nrmlizer(no_zip_feat_2['bathrooms'])
```


```python
no_zip_feat2['sqft_living'] = tfsqftliving.copy()
no_zip_feat2['sqft_lot'] = tfsqftlot.copy()
no_zip_feat2['sqft_basement'] = tfsqftbasement.copy()

#revised df
no_zip_feat2_new['sqft_living'] = tfsqftliving_2.copy()
no_zip_feat2_new['sqft_lot'] = tfsqftlot_2.copy()
no_zip_feat2_new['sqft_basement'] = tfsqftbasement_2.copy()
no_zip_feat2_new['bedrooms'] = tfbedrooms_2.copy()
no_zip_feat2_new['bathrooms'] = tfbathooms_2.copy()
```


```python
kde_hist_plot(no_zip_feat2)
#for column in no_zip_feat2:
#    print(no_zip_feat2[column].describe().round(3))
```


![png](output_187_0.png)



![png](output_187_1.png)



![png](output_187_2.png)



![png](output_187_3.png)



![png](output_187_4.png)


> Since the sqft basement and lot are still highly skewed, I may drop them if this model doesn't quite work out.


```python
kde_hist_plot(no_zip_feat2_new)
```


![png](output_189_0.png)



![png](output_189_1.png)



![png](output_189_2.png)



![png](output_189_3.png)



![png](output_189_4.png)


> Visually, the revision of normalization hasn't changed much on the corresponding features. 


```python
#updating the min-maxed columns in 'no_corr' dataframe
kc_house_feat_nocorr.update(no_zip_feat2)
kc_house_feat_nocorr2.update(no_zip_feat2_new)

#sanity check
print(no_zip_feat2['sqft_living'].describe() == kc_house_feat_nocorr['sqft_living'].describe())
no_zip_feat2_new['sqft_living'].describe() == kc_house_feat_nocorr2['sqft_living'].describe()
```

    count    True
    mean     True
    std      True
    min      True
    25%      True
    50%      True
    75%      True
    max      True
    Name: sqft_living, dtype: bool
    




    count    True
    mean     True
    std      True
    min      True
    25%      True
    50%      True
    75%      True
    max      True
    Name: sqft_living, dtype: bool



# MODEL

###### Train-test split 1


```python
x = kc_house_feat_nocorr.copy()
y = pd.DataFrame(price_series)
```


```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
```


```python
linreg = LinearRegression()
linreg.fit(x_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



###### OLS


```python
x_cnst_trn = sm.add_constant(x_train)
model1 = sm.OLS(y_train, x_cnst_trn).fit()
model1.summary()
```

    C:\Users\d_ful\Anaconda3\lib\site-packages\numpy\core\fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.816</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.815</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   770.0</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 09 Dec 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>17:06:09</td>     <th>  Log-Likelihood:    </th> <td>-1.8094e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 13947</td>      <th>  AIC:               </th>  <td>3.620e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 13866</td>      <th>  BIC:               </th>  <td>3.627e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    80</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
             <td></td>               <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                  <td>-2.722e+05</td> <td> 9466.914</td> <td>  -28.756</td> <td> 0.000</td> <td>-2.91e+05</td> <td>-2.54e+05</td>
</tr>
<tr>
  <th>bedrooms</th>               <td>-1.102e+04</td> <td> 1392.416</td> <td>   -7.912</td> <td> 0.000</td> <td>-1.37e+04</td> <td>-8287.796</td>
</tr>
<tr>
  <th>bathrooms</th>              <td> 1.569e+04</td> <td> 2214.696</td> <td>    7.087</td> <td> 0.000</td> <td> 1.14e+04</td> <td>    2e+04</td>
</tr>
<tr>
  <th>sqft_living</th>            <td> 6.684e+05</td> <td> 1.13e+04</td> <td>   59.369</td> <td> 0.000</td> <td> 6.46e+05</td> <td>  6.9e+05</td>
</tr>
<tr>
  <th>sqft_lot</th>               <td> 1.183e+05</td> <td> 1.19e+04</td> <td>    9.965</td> <td> 0.000</td> <td>  9.5e+04</td> <td> 1.42e+05</td>
</tr>
<tr>
  <th>waterfront</th>             <td> 1.403e-09</td> <td> 3.45e-11</td> <td>   40.666</td> <td> 0.000</td> <td> 1.33e-09</td> <td> 1.47e-09</td>
</tr>
<tr>
  <th>grade</th>                  <td> 5.779e+04</td> <td> 1467.767</td> <td>   39.375</td> <td> 0.000</td> <td> 5.49e+04</td> <td> 6.07e+04</td>
</tr>
<tr>
  <th>sqft_basement</th>          <td>-6.719e+04</td> <td> 5110.231</td> <td>  -13.149</td> <td> 0.000</td> <td>-7.72e+04</td> <td>-5.72e+04</td>
</tr>
<tr>
  <th>flrs_(0.5, 1.0]</th>        <td>-1.213e+04</td> <td> 2609.426</td> <td>   -4.650</td> <td> 0.000</td> <td>-1.72e+04</td> <td>-7019.893</td>
</tr>
<tr>
  <th>cond_(1, 2]</th>            <td>-1.007e+05</td> <td> 9100.880</td> <td>  -11.064</td> <td> 0.000</td> <td>-1.19e+05</td> <td>-8.29e+04</td>
</tr>
<tr>
  <th>cond_(2, 3]</th>            <td>-8.369e+04</td> <td> 3581.974</td> <td>  -23.364</td> <td> 0.000</td> <td>-9.07e+04</td> <td>-7.67e+04</td>
</tr>
<tr>
  <th>cond_(3, 4]</th>            <td>-6.338e+04</td> <td> 3578.620</td> <td>  -17.711</td> <td> 0.000</td> <td>-7.04e+04</td> <td>-5.64e+04</td>
</tr>
<tr>
  <th>cond_(4, 5]</th>            <td>-2.447e+04</td> <td> 4202.311</td> <td>   -5.822</td> <td> 0.000</td> <td>-3.27e+04</td> <td>-1.62e+04</td>
</tr>
<tr>
  <th>yr_built_(1975, 2016]</th>  <td>-3.683e+04</td> <td> 2799.248</td> <td>  -13.156</td> <td> 0.000</td> <td>-4.23e+04</td> <td>-3.13e+04</td>
</tr>
<tr>
  <th>zipcode_(98001, 98002]</th> <td> 2.779e+04</td> <td> 1.14e+04</td> <td>    2.428</td> <td> 0.015</td> <td> 5352.776</td> <td> 5.02e+04</td>
</tr>
<tr>
  <th>zipcode_(98002, 98003]</th> <td>  644.8292</td> <td> 1.02e+04</td> <td>    0.063</td> <td> 0.950</td> <td>-1.94e+04</td> <td> 2.07e+04</td>
</tr>
<tr>
  <th>zipcode_(98003, 98004]</th> <td> 6.015e+05</td> <td>  1.1e+04</td> <td>   54.641</td> <td> 0.000</td> <td>  5.8e+05</td> <td> 6.23e+05</td>
</tr>
<tr>
  <th>zipcode_(98004, 98005]</th> <td> 3.128e+05</td> <td> 1.24e+04</td> <td>   25.316</td> <td> 0.000</td> <td> 2.89e+05</td> <td> 3.37e+05</td>
</tr>
<tr>
  <th>zipcode_(98005, 98006]</th> <td> 2.803e+05</td> <td> 9206.965</td> <td>   30.446</td> <td> 0.000</td> <td> 2.62e+05</td> <td> 2.98e+05</td>
</tr>
<tr>
  <th>zipcode_(98006, 98007]</th> <td> 2.506e+05</td> <td> 1.27e+04</td> <td>   19.732</td> <td> 0.000</td> <td> 2.26e+05</td> <td> 2.76e+05</td>
</tr>
<tr>
  <th>zipcode_(98007, 98008]</th> <td>   2.5e+05</td> <td> 1.02e+04</td> <td>   24.399</td> <td> 0.000</td> <td>  2.3e+05</td> <td>  2.7e+05</td>
</tr>
<tr>
  <th>zipcode_(98008, 98010]</th> <td> 8.185e+04</td> <td> 1.57e+04</td> <td>    5.198</td> <td> 0.000</td> <td>  5.1e+04</td> <td> 1.13e+05</td>
</tr>
<tr>
  <th>zipcode_(98010, 98011]</th> <td> 1.454e+05</td> <td> 1.12e+04</td> <td>   12.959</td> <td> 0.000</td> <td> 1.23e+05</td> <td> 1.67e+05</td>
</tr>
<tr>
  <th>zipcode_(98011, 98014]</th> <td> 9.482e+04</td> <td> 1.51e+04</td> <td>    6.263</td> <td> 0.000</td> <td> 6.51e+04</td> <td> 1.24e+05</td>
</tr>
<tr>
  <th>zipcode_(98014, 98019]</th> <td> 1.023e+05</td> <td> 1.17e+04</td> <td>    8.736</td> <td> 0.000</td> <td> 7.93e+04</td> <td> 1.25e+05</td>
</tr>
<tr>
  <th>zipcode_(98019, 98022]</th> <td> 1.341e+04</td> <td> 1.17e+04</td> <td>    1.145</td> <td> 0.252</td> <td>-9549.282</td> <td> 3.64e+04</td>
</tr>
<tr>
  <th>zipcode_(98022, 98023]</th> <td>-1.466e+04</td> <td> 8787.664</td> <td>   -1.669</td> <td> 0.095</td> <td>-3.19e+04</td> <td> 2561.028</td>
</tr>
<tr>
  <th>zipcode_(98023, 98024]</th> <td> 1.495e+05</td> <td>  1.9e+04</td> <td>    7.883</td> <td> 0.000</td> <td> 1.12e+05</td> <td> 1.87e+05</td>
</tr>
<tr>
  <th>zipcode_(98024, 98027]</th> <td> 1.932e+05</td> <td> 9348.282</td> <td>   20.665</td> <td> 0.000</td> <td> 1.75e+05</td> <td> 2.12e+05</td>
</tr>
<tr>
  <th>zipcode_(98027, 98028]</th> <td> 1.392e+05</td> <td> 1.04e+04</td> <td>   13.345</td> <td> 0.000</td> <td> 1.19e+05</td> <td>  1.6e+05</td>
</tr>
<tr>
  <th>zipcode_(98028, 98029]</th> <td> 2.272e+05</td> <td> 9799.849</td> <td>   23.183</td> <td> 0.000</td> <td> 2.08e+05</td> <td> 2.46e+05</td>
</tr>
<tr>
  <th>zipcode_(98029, 98030]</th> <td> 7979.9222</td> <td> 1.01e+04</td> <td>    0.787</td> <td> 0.432</td> <td>-1.19e+04</td> <td> 2.79e+04</td>
</tr>
<tr>
  <th>zipcode_(98030, 98031]</th> <td> 1.598e+04</td> <td>    1e+04</td> <td>    1.592</td> <td> 0.111</td> <td>-3699.663</td> <td> 3.57e+04</td>
</tr>
<tr>
  <th>zipcode_(98031, 98032]</th> <td> 6525.0559</td> <td> 1.42e+04</td> <td>    0.459</td> <td> 0.646</td> <td>-2.13e+04</td> <td> 3.44e+04</td>
</tr>
<tr>
  <th>zipcode_(98032, 98033]</th> <td> 3.585e+05</td> <td> 9079.068</td> <td>   39.484</td> <td> 0.000</td> <td> 3.41e+05</td> <td> 3.76e+05</td>
</tr>
<tr>
  <th>zipcode_(98033, 98034]</th> <td>  1.93e+05</td> <td> 8668.815</td> <td>   22.265</td> <td> 0.000</td> <td> 1.76e+05</td> <td>  2.1e+05</td>
</tr>
<tr>
  <th>zipcode_(98034, 98038]</th> <td> 4.024e+04</td> <td> 8664.925</td> <td>    4.644</td> <td> 0.000</td> <td> 2.33e+04</td> <td> 5.72e+04</td>
</tr>
<tr>
  <th>zipcode_(98038, 98039]</th> <td> 8.347e+05</td> <td>  3.1e+04</td> <td>   26.964</td> <td> 0.000</td> <td> 7.74e+05</td> <td> 8.95e+05</td>
</tr>
<tr>
  <th>zipcode_(98039, 98040]</th> <td> 4.933e+05</td> <td> 1.09e+04</td> <td>   45.264</td> <td> 0.000</td> <td> 4.72e+05</td> <td> 5.15e+05</td>
</tr>
<tr>
  <th>zipcode_(98040, 98042]</th> <td> 8518.5273</td> <td> 8704.417</td> <td>    0.979</td> <td> 0.328</td> <td>-8543.306</td> <td> 2.56e+04</td>
</tr>
<tr>
  <th>zipcode_(98042, 98045]</th> <td> 1.143e+05</td> <td> 1.11e+04</td> <td>   10.322</td> <td> 0.000</td> <td> 9.26e+04</td> <td> 1.36e+05</td>
</tr>
<tr>
  <th>zipcode_(98045, 98052]</th> <td> 2.466e+05</td> <td> 8602.569</td> <td>   28.668</td> <td> 0.000</td> <td>  2.3e+05</td> <td> 2.63e+05</td>
</tr>
<tr>
  <th>zipcode_(98052, 98053]</th> <td> 2.272e+05</td> <td> 9589.890</td> <td>   23.691</td> <td> 0.000</td> <td> 2.08e+05</td> <td> 2.46e+05</td>
</tr>
<tr>
  <th>zipcode_(98053, 98055]</th> <td>  4.94e+04</td> <td> 1.04e+04</td> <td>    4.736</td> <td> 0.000</td> <td>  2.9e+04</td> <td> 6.98e+04</td>
</tr>
<tr>
  <th>zipcode_(98055, 98056]</th> <td> 1.044e+05</td> <td> 9369.818</td> <td>   11.147</td> <td> 0.000</td> <td> 8.61e+04</td> <td> 1.23e+05</td>
</tr>
<tr>
  <th>zipcode_(98056, 98058]</th> <td> 3.397e+04</td> <td> 9010.134</td> <td>    3.770</td> <td> 0.000</td> <td> 1.63e+04</td> <td> 5.16e+04</td>
</tr>
<tr>
  <th>zipcode_(98058, 98059]</th> <td> 1.018e+05</td> <td> 9009.468</td> <td>   11.294</td> <td> 0.000</td> <td> 8.41e+04</td> <td> 1.19e+05</td>
</tr>
<tr>
  <th>zipcode_(98059, 98065]</th> <td> 1.302e+05</td> <td> 1.01e+04</td> <td>   12.910</td> <td> 0.000</td> <td>  1.1e+05</td> <td>  1.5e+05</td>
</tr>
<tr>
  <th>zipcode_(98065, 98070]</th> <td> 1.363e+05</td> <td> 1.78e+04</td> <td>    7.677</td> <td> 0.000</td> <td> 1.02e+05</td> <td> 1.71e+05</td>
</tr>
<tr>
  <th>zipcode_(98070, 98072]</th> <td> 1.652e+05</td> <td> 1.03e+04</td> <td>   15.979</td> <td> 0.000</td> <td> 1.45e+05</td> <td> 1.85e+05</td>
</tr>
<tr>
  <th>zipcode_(98072, 98074]</th> <td> 2.075e+05</td> <td> 9207.565</td> <td>   22.533</td> <td> 0.000</td> <td> 1.89e+05</td> <td> 2.26e+05</td>
</tr>
<tr>
  <th>zipcode_(98074, 98075]</th> <td> 2.222e+05</td> <td> 9759.176</td> <td>   22.767</td> <td> 0.000</td> <td> 2.03e+05</td> <td> 2.41e+05</td>
</tr>
<tr>
  <th>zipcode_(98075, 98077]</th> <td> 1.494e+05</td> <td> 1.19e+04</td> <td>   12.542</td> <td> 0.000</td> <td> 1.26e+05</td> <td> 1.73e+05</td>
</tr>
<tr>
  <th>zipcode_(98077, 98092]</th> <td>-2.146e+04</td> <td> 9824.939</td> <td>   -2.184</td> <td> 0.029</td> <td>-4.07e+04</td> <td>-2203.016</td>
</tr>
<tr>
  <th>zipcode_(98092, 98102]</th> <td> 4.551e+05</td> <td> 1.44e+04</td> <td>   31.666</td> <td> 0.000</td> <td> 4.27e+05</td> <td> 4.83e+05</td>
</tr>
<tr>
  <th>zipcode_(98102, 98103]</th> <td> 3.338e+05</td> <td> 8618.463</td> <td>   38.733</td> <td> 0.000</td> <td> 3.17e+05</td> <td> 3.51e+05</td>
</tr>
<tr>
  <th>zipcode_(98103, 98105]</th> <td> 4.446e+05</td> <td>  1.1e+04</td> <td>   40.248</td> <td> 0.000</td> <td> 4.23e+05</td> <td> 4.66e+05</td>
</tr>
<tr>
  <th>zipcode_(98105, 98106]</th> <td> 1.277e+05</td> <td> 9834.021</td> <td>   12.981</td> <td> 0.000</td> <td> 1.08e+05</td> <td> 1.47e+05</td>
</tr>
<tr>
  <th>zipcode_(98106, 98107]</th> <td> 3.297e+05</td> <td> 1.04e+04</td> <td>   31.689</td> <td> 0.000</td> <td> 3.09e+05</td> <td>  3.5e+05</td>
</tr>
<tr>
  <th>zipcode_(98107, 98108]</th> <td> 1.313e+05</td> <td> 1.15e+04</td> <td>   11.453</td> <td> 0.000</td> <td> 1.09e+05</td> <td> 1.54e+05</td>
</tr>
<tr>
  <th>zipcode_(98108, 98109]</th> <td> 4.933e+05</td> <td> 1.45e+04</td> <td>   34.000</td> <td> 0.000</td> <td> 4.65e+05</td> <td> 5.22e+05</td>
</tr>
<tr>
  <th>zipcode_(98109, 98112]</th> <td> 5.253e+05</td> <td> 1.12e+04</td> <td>   46.838</td> <td> 0.000</td> <td> 5.03e+05</td> <td> 5.47e+05</td>
</tr>
<tr>
  <th>zipcode_(98112, 98115]</th> <td> 3.442e+05</td> <td> 8614.506</td> <td>   39.960</td> <td> 0.000</td> <td> 3.27e+05</td> <td> 3.61e+05</td>
</tr>
<tr>
  <th>zipcode_(98115, 98116]</th> <td>  3.21e+05</td> <td> 9756.845</td> <td>   32.897</td> <td> 0.000</td> <td> 3.02e+05</td> <td>  3.4e+05</td>
</tr>
<tr>
  <th>zipcode_(98116, 98117]</th> <td> 3.268e+05</td> <td> 8701.084</td> <td>   37.564</td> <td> 0.000</td> <td>  3.1e+05</td> <td> 3.44e+05</td>
</tr>
<tr>
  <th>zipcode_(98117, 98118]</th> <td> 1.847e+05</td> <td> 8969.863</td> <td>   20.589</td> <td> 0.000</td> <td> 1.67e+05</td> <td> 2.02e+05</td>
</tr>
<tr>
  <th>zipcode_(98118, 98119]</th> <td> 4.733e+05</td> <td> 1.14e+04</td> <td>   41.590</td> <td> 0.000</td> <td> 4.51e+05</td> <td> 4.96e+05</td>
</tr>
<tr>
  <th>zipcode_(98119, 98122]</th> <td> 3.287e+05</td> <td> 1.03e+04</td> <td>   31.929</td> <td> 0.000</td> <td> 3.09e+05</td> <td> 3.49e+05</td>
</tr>
<tr>
  <th>zipcode_(98122, 98125]</th> <td> 2.009e+05</td> <td> 9393.389</td> <td>   21.384</td> <td> 0.000</td> <td> 1.82e+05</td> <td> 2.19e+05</td>
</tr>
<tr>
  <th>zipcode_(98125, 98126]</th> <td> 2.076e+05</td> <td> 9550.069</td> <td>   21.742</td> <td> 0.000</td> <td> 1.89e+05</td> <td> 2.26e+05</td>
</tr>
<tr>
  <th>zipcode_(98126, 98133]</th> <td> 1.572e+05</td> <td> 8810.901</td> <td>   17.838</td> <td> 0.000</td> <td>  1.4e+05</td> <td> 1.74e+05</td>
</tr>
<tr>
  <th>zipcode_(98133, 98136]</th> <td> 2.752e+05</td> <td> 1.07e+04</td> <td>   25.627</td> <td> 0.000</td> <td> 2.54e+05</td> <td> 2.96e+05</td>
</tr>
<tr>
  <th>zipcode_(98136, 98144]</th> <td>   2.7e+05</td> <td> 9762.938</td> <td>   27.659</td> <td> 0.000</td> <td> 2.51e+05</td> <td> 2.89e+05</td>
</tr>
<tr>
  <th>zipcode_(98144, 98146]</th> <td> 1.364e+05</td> <td> 1.02e+04</td> <td>   13.379</td> <td> 0.000</td> <td> 1.16e+05</td> <td> 1.56e+05</td>
</tr>
<tr>
  <th>zipcode_(98146, 98148]</th> <td> 5.694e+04</td> <td> 1.81e+04</td> <td>    3.151</td> <td> 0.002</td> <td> 2.15e+04</td> <td> 9.24e+04</td>
</tr>
<tr>
  <th>zipcode_(98148, 98155]</th> <td> 1.409e+05</td> <td> 9029.205</td> <td>   15.607</td> <td> 0.000</td> <td> 1.23e+05</td> <td> 1.59e+05</td>
</tr>
<tr>
  <th>zipcode_(98155, 98166]</th> <td> 1.128e+05</td> <td> 1.08e+04</td> <td>   10.400</td> <td> 0.000</td> <td> 9.15e+04</td> <td> 1.34e+05</td>
</tr>
<tr>
  <th>zipcode_(98166, 98168]</th> <td> 6.402e+04</td> <td> 1.04e+04</td> <td>    6.143</td> <td> 0.000</td> <td> 4.36e+04</td> <td> 8.44e+04</td>
</tr>
<tr>
  <th>zipcode_(98168, 98177]</th> <td> 2.436e+05</td> <td> 1.05e+04</td> <td>   23.167</td> <td> 0.000</td> <td> 2.23e+05</td> <td> 2.64e+05</td>
</tr>
<tr>
  <th>zipcode_(98177, 98178]</th> <td> 6.524e+04</td> <td> 1.04e+04</td> <td>    6.289</td> <td> 0.000</td> <td> 4.49e+04</td> <td> 8.56e+04</td>
</tr>
<tr>
  <th>zipcode_(98178, 98188]</th> <td> 3.848e+04</td> <td> 1.28e+04</td> <td>    3.013</td> <td> 0.003</td> <td> 1.34e+04</td> <td> 6.35e+04</td>
</tr>
<tr>
  <th>zipcode_(98188, 98198]</th> <td> 4.624e+04</td> <td> 1.03e+04</td> <td>    4.507</td> <td> 0.000</td> <td> 2.61e+04</td> <td> 6.64e+04</td>
</tr>
<tr>
  <th>zipcode_(98198, 98199]</th> <td> 3.892e+05</td> <td> 1.01e+04</td> <td>   38.514</td> <td> 0.000</td> <td> 3.69e+05</td> <td> 4.09e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>3616.615</td> <th>  Durbin-Watson:     </th> <td>   1.983</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>20574.704</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 1.126</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 8.508</td>  <th>  Cond. No.          </th> <td>1.06e+16</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 9.37e-27. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.




```python
residuals1 = model1.resid
fig1 = sm.graphics.qqplot(residuals1, dist=stats.norm, line='45', fit=True)
plt.title('Model1 QQ-plot')
fig1.show();
```

    C:\Users\d_ful\Anaconda3\lib\site-packages\ipykernel_launcher.py:4: UserWarning: Matplotlib is currently using module://ipykernel.pylab.backend_inline, which is a non-GUI backend, so cannot show the figure.
      after removing the cwd from sys.path.
    


![png](output_199_1.png)


* There is a clear right-tailed skew prevalent within my _model1_

> I am going to drop anything with a p-value > 0.05 and re-run


```python
x = x.drop('zipcode_(98001, 98002]', axis=1).copy()
x = x.drop('zipcode_(98002, 98003]', axis=1).copy()
x = x.drop('zipcode_(98019, 98022]', axis=1).copy()
x = x.drop('zipcode_(98022, 98023]', axis=1).copy()
x = x.drop('zipcode_(98029, 98030]', axis=1).copy()
x = x.drop('zipcode_(98030, 98031]', axis=1).copy()
x = x.drop('zipcode_(98031, 98032]', axis=1).copy()
x = x.drop('zipcode_(98040, 98042]', axis=1).copy()
```

###### Train-test split 2


```python
x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, test_size=0.30, random_state=43)
```

###### OLS 2


```python
x_cnst_trn2 = sm.add_constant(x_train2)
model2 = sm.OLS(y_train2, x_cnst_trn2).fit()
model2.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.813</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.812</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   839.9</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 09 Dec 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>17:06:22</td>     <th>  Log-Likelihood:    </th> <td>-1.8090e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 13947</td>      <th>  AIC:               </th>  <td>3.620e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 13874</td>      <th>  BIC:               </th>  <td>3.625e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    72</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
             <td></td>               <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                  <td> -2.61e+05</td> <td> 8070.004</td> <td>  -32.345</td> <td> 0.000</td> <td>-2.77e+05</td> <td>-2.45e+05</td>
</tr>
<tr>
  <th>bedrooms</th>               <td>-1.025e+04</td> <td> 1388.444</td> <td>   -7.382</td> <td> 0.000</td> <td> -1.3e+04</td> <td>-7528.110</td>
</tr>
<tr>
  <th>bathrooms</th>              <td> 1.359e+04</td> <td> 2187.541</td> <td>    6.212</td> <td> 0.000</td> <td> 9300.209</td> <td> 1.79e+04</td>
</tr>
<tr>
  <th>sqft_living</th>            <td> 6.535e+05</td> <td> 1.13e+04</td> <td>   58.033</td> <td> 0.000</td> <td> 6.31e+05</td> <td> 6.76e+05</td>
</tr>
<tr>
  <th>sqft_lot</th>               <td> 1.312e+05</td> <td> 1.19e+04</td> <td>   10.986</td> <td> 0.000</td> <td> 1.08e+05</td> <td> 1.55e+05</td>
</tr>
<tr>
  <th>waterfront</th>             <td>-2.415e-09</td> <td> 4.23e-11</td> <td>  -57.060</td> <td> 0.000</td> <td> -2.5e-09</td> <td>-2.33e-09</td>
</tr>
<tr>
  <th>grade</th>                  <td> 5.751e+04</td> <td> 1464.404</td> <td>   39.274</td> <td> 0.000</td> <td> 5.46e+04</td> <td> 6.04e+04</td>
</tr>
<tr>
  <th>sqft_basement</th>          <td>-6.156e+04</td> <td> 5081.824</td> <td>  -12.114</td> <td> 0.000</td> <td>-7.15e+04</td> <td>-5.16e+04</td>
</tr>
<tr>
  <th>flrs_(0.5, 1.0]</th>        <td>-6825.1080</td> <td> 2602.158</td> <td>   -2.623</td> <td> 0.009</td> <td>-1.19e+04</td> <td>-1724.527</td>
</tr>
<tr>
  <th>cond_(1, 2]</th>            <td>-9.283e+04</td> <td> 8598.535</td> <td>  -10.796</td> <td> 0.000</td> <td> -1.1e+05</td> <td> -7.6e+04</td>
</tr>
<tr>
  <th>cond_(2, 3]</th>            <td> -8.15e+04</td> <td> 3300.316</td> <td>  -24.696</td> <td> 0.000</td> <td> -8.8e+04</td> <td> -7.5e+04</td>
</tr>
<tr>
  <th>cond_(3, 4]</th>            <td>-6.076e+04</td> <td> 3256.468</td> <td>  -18.660</td> <td> 0.000</td> <td>-6.71e+04</td> <td>-5.44e+04</td>
</tr>
<tr>
  <th>cond_(4, 5]</th>            <td>-2.593e+04</td> <td> 3892.711</td> <td>   -6.661</td> <td> 0.000</td> <td>-3.36e+04</td> <td>-1.83e+04</td>
</tr>
<tr>
  <th>yr_built_(1975, 2016]</th>  <td>-4.143e+04</td> <td> 2756.310</td> <td>  -15.033</td> <td> 0.000</td> <td>-4.68e+04</td> <td> -3.6e+04</td>
</tr>
<tr>
  <th>zipcode_(98003, 98004]</th> <td> 6.005e+05</td> <td> 8800.746</td> <td>   68.231</td> <td> 0.000</td> <td> 5.83e+05</td> <td> 6.18e+05</td>
</tr>
<tr>
  <th>zipcode_(98004, 98005]</th> <td> 3.141e+05</td> <td> 1.11e+04</td> <td>   28.217</td> <td> 0.000</td> <td> 2.92e+05</td> <td> 3.36e+05</td>
</tr>
<tr>
  <th>zipcode_(98005, 98006]</th> <td> 2.758e+05</td> <td> 6701.463</td> <td>   41.157</td> <td> 0.000</td> <td> 2.63e+05</td> <td> 2.89e+05</td>
</tr>
<tr>
  <th>zipcode_(98006, 98007]</th> <td> 2.468e+05</td> <td> 1.09e+04</td> <td>   22.605</td> <td> 0.000</td> <td> 2.25e+05</td> <td> 2.68e+05</td>
</tr>
<tr>
  <th>zipcode_(98007, 98008]</th> <td> 2.408e+05</td> <td> 7976.415</td> <td>   30.188</td> <td> 0.000</td> <td> 2.25e+05</td> <td> 2.56e+05</td>
</tr>
<tr>
  <th>zipcode_(98008, 98010]</th> <td> 7.541e+04</td> <td> 1.41e+04</td> <td>    5.363</td> <td> 0.000</td> <td> 4.78e+04</td> <td> 1.03e+05</td>
</tr>
<tr>
  <th>zipcode_(98010, 98011]</th> <td> 1.365e+05</td> <td> 9623.263</td> <td>   14.179</td> <td> 0.000</td> <td> 1.18e+05</td> <td> 1.55e+05</td>
</tr>
<tr>
  <th>zipcode_(98011, 98014]</th> <td> 1.072e+05</td> <td> 1.34e+04</td> <td>    8.026</td> <td> 0.000</td> <td>  8.1e+04</td> <td> 1.33e+05</td>
</tr>
<tr>
  <th>zipcode_(98014, 98019]</th> <td> 9.208e+04</td> <td> 9876.244</td> <td>    9.324</td> <td> 0.000</td> <td> 7.27e+04</td> <td> 1.11e+05</td>
</tr>
<tr>
  <th>zipcode_(98023, 98024]</th> <td> 1.445e+05</td> <td> 1.81e+04</td> <td>    7.977</td> <td> 0.000</td> <td> 1.09e+05</td> <td>  1.8e+05</td>
</tr>
<tr>
  <th>zipcode_(98024, 98027]</th> <td> 1.843e+05</td> <td> 7040.860</td> <td>   26.177</td> <td> 0.000</td> <td> 1.71e+05</td> <td> 1.98e+05</td>
</tr>
<tr>
  <th>zipcode_(98027, 98028]</th> <td> 1.315e+05</td> <td> 7784.467</td> <td>   16.891</td> <td> 0.000</td> <td> 1.16e+05</td> <td> 1.47e+05</td>
</tr>
<tr>
  <th>zipcode_(98028, 98029]</th> <td> 2.222e+05</td> <td> 7534.195</td> <td>   29.489</td> <td> 0.000</td> <td> 2.07e+05</td> <td> 2.37e+05</td>
</tr>
<tr>
  <th>zipcode_(98032, 98033]</th> <td> 3.499e+05</td> <td> 6916.088</td> <td>   50.593</td> <td> 0.000</td> <td> 3.36e+05</td> <td> 3.63e+05</td>
</tr>
<tr>
  <th>zipcode_(98033, 98034]</th> <td> 1.892e+05</td> <td> 6051.058</td> <td>   31.260</td> <td> 0.000</td> <td> 1.77e+05</td> <td> 2.01e+05</td>
</tr>
<tr>
  <th>zipcode_(98034, 98038]</th> <td> 3.555e+04</td> <td> 6042.130</td> <td>    5.884</td> <td> 0.000</td> <td> 2.37e+04</td> <td> 4.74e+04</td>
</tr>
<tr>
  <th>zipcode_(98038, 98039]</th> <td> 8.131e+05</td> <td> 3.16e+04</td> <td>   25.763</td> <td> 0.000</td> <td> 7.51e+05</td> <td> 8.75e+05</td>
</tr>
<tr>
  <th>zipcode_(98039, 98040]</th> <td> 4.775e+05</td> <td> 9042.576</td> <td>   52.803</td> <td> 0.000</td> <td>  4.6e+05</td> <td> 4.95e+05</td>
</tr>
<tr>
  <th>zipcode_(98042, 98045]</th> <td> 1.117e+05</td> <td> 9302.526</td> <td>   12.011</td> <td> 0.000</td> <td> 9.35e+04</td> <td>  1.3e+05</td>
</tr>
<tr>
  <th>zipcode_(98045, 98052]</th> <td> 2.438e+05</td> <td> 5935.226</td> <td>   41.083</td> <td> 0.000</td> <td> 2.32e+05</td> <td> 2.55e+05</td>
</tr>
<tr>
  <th>zipcode_(98052, 98053]</th> <td> 2.305e+05</td> <td> 7571.788</td> <td>   30.442</td> <td> 0.000</td> <td> 2.16e+05</td> <td> 2.45e+05</td>
</tr>
<tr>
  <th>zipcode_(98053, 98055]</th> <td>  4.07e+04</td> <td> 7970.246</td> <td>    5.107</td> <td> 0.000</td> <td> 2.51e+04</td> <td> 5.63e+04</td>
</tr>
<tr>
  <th>zipcode_(98055, 98056]</th> <td> 1.035e+05</td> <td> 6847.756</td> <td>   15.122</td> <td> 0.000</td> <td> 9.01e+04</td> <td> 1.17e+05</td>
</tr>
<tr>
  <th>zipcode_(98056, 98058]</th> <td> 2.411e+04</td> <td> 6613.841</td> <td>    3.645</td> <td> 0.000</td> <td> 1.11e+04</td> <td> 3.71e+04</td>
</tr>
<tr>
  <th>zipcode_(98058, 98059]</th> <td> 8.833e+04</td> <td> 6526.302</td> <td>   13.535</td> <td> 0.000</td> <td> 7.55e+04</td> <td> 1.01e+05</td>
</tr>
<tr>
  <th>zipcode_(98059, 98065]</th> <td> 1.269e+05</td> <td> 7998.076</td> <td>   15.872</td> <td> 0.000</td> <td> 1.11e+05</td> <td> 1.43e+05</td>
</tr>
<tr>
  <th>zipcode_(98065, 98070]</th> <td> 1.136e+05</td> <td> 1.66e+04</td> <td>    6.847</td> <td> 0.000</td> <td> 8.11e+04</td> <td> 1.46e+05</td>
</tr>
<tr>
  <th>zipcode_(98070, 98072]</th> <td> 1.646e+05</td> <td> 8293.965</td> <td>   19.850</td> <td> 0.000</td> <td> 1.48e+05</td> <td> 1.81e+05</td>
</tr>
<tr>
  <th>zipcode_(98072, 98074]</th> <td> 2.107e+05</td> <td> 6882.054</td> <td>   30.612</td> <td> 0.000</td> <td> 1.97e+05</td> <td> 2.24e+05</td>
</tr>
<tr>
  <th>zipcode_(98074, 98075]</th> <td> 2.291e+05</td> <td> 7527.465</td> <td>   30.441</td> <td> 0.000</td> <td> 2.14e+05</td> <td> 2.44e+05</td>
</tr>
<tr>
  <th>zipcode_(98075, 98077]</th> <td> 1.468e+05</td> <td> 1.05e+04</td> <td>   13.971</td> <td> 0.000</td> <td> 1.26e+05</td> <td> 1.67e+05</td>
</tr>
<tr>
  <th>zipcode_(98077, 98092]</th> <td>-2.349e+04</td> <td> 7597.061</td> <td>   -3.092</td> <td> 0.002</td> <td>-3.84e+04</td> <td>-8601.065</td>
</tr>
<tr>
  <th>zipcode_(98092, 98102]</th> <td>  4.52e+05</td> <td>  1.4e+04</td> <td>   32.323</td> <td> 0.000</td> <td> 4.25e+05</td> <td> 4.79e+05</td>
</tr>
<tr>
  <th>zipcode_(98102, 98103]</th> <td>  3.25e+05</td> <td> 5878.562</td> <td>   55.285</td> <td> 0.000</td> <td> 3.13e+05</td> <td> 3.37e+05</td>
</tr>
<tr>
  <th>zipcode_(98103, 98105]</th> <td> 4.345e+05</td> <td> 9325.642</td> <td>   46.591</td> <td> 0.000</td> <td> 4.16e+05</td> <td> 4.53e+05</td>
</tr>
<tr>
  <th>zipcode_(98105, 98106]</th> <td> 1.199e+05</td> <td> 7275.951</td> <td>   16.484</td> <td> 0.000</td> <td> 1.06e+05</td> <td> 1.34e+05</td>
</tr>
<tr>
  <th>zipcode_(98106, 98107]</th> <td> 3.257e+05</td> <td> 8119.427</td> <td>   40.118</td> <td> 0.000</td> <td>  3.1e+05</td> <td> 3.42e+05</td>
</tr>
<tr>
  <th>zipcode_(98107, 98108]</th> <td> 1.227e+05</td> <td> 9303.048</td> <td>   13.190</td> <td> 0.000</td> <td> 1.04e+05</td> <td> 1.41e+05</td>
</tr>
<tr>
  <th>zipcode_(98108, 98109]</th> <td> 4.942e+05</td> <td> 1.24e+04</td> <td>   39.897</td> <td> 0.000</td> <td>  4.7e+05</td> <td> 5.18e+05</td>
</tr>
<tr>
  <th>zipcode_(98109, 98112]</th> <td> 5.044e+05</td> <td> 8905.931</td> <td>   56.638</td> <td> 0.000</td> <td> 4.87e+05</td> <td> 5.22e+05</td>
</tr>
<tr>
  <th>zipcode_(98112, 98115]</th> <td> 3.292e+05</td> <td> 5967.504</td> <td>   55.169</td> <td> 0.000</td> <td> 3.18e+05</td> <td> 3.41e+05</td>
</tr>
<tr>
  <th>zipcode_(98115, 98116]</th> <td> 3.023e+05</td> <td> 7794.184</td> <td>   38.788</td> <td> 0.000</td> <td> 2.87e+05</td> <td> 3.18e+05</td>
</tr>
<tr>
  <th>zipcode_(98116, 98117]</th> <td> 3.163e+05</td> <td> 6073.567</td> <td>   52.079</td> <td> 0.000</td> <td> 3.04e+05</td> <td> 3.28e+05</td>
</tr>
<tr>
  <th>zipcode_(98117, 98118]</th> <td> 1.763e+05</td> <td> 6304.703</td> <td>   27.966</td> <td> 0.000</td> <td> 1.64e+05</td> <td> 1.89e+05</td>
</tr>
<tr>
  <th>zipcode_(98118, 98119]</th> <td> 4.577e+05</td> <td> 9404.944</td> <td>   48.667</td> <td> 0.000</td> <td> 4.39e+05</td> <td> 4.76e+05</td>
</tr>
<tr>
  <th>zipcode_(98119, 98122]</th> <td> 3.333e+05</td> <td> 8094.551</td> <td>   41.176</td> <td> 0.000</td> <td> 3.17e+05</td> <td> 3.49e+05</td>
</tr>
<tr>
  <th>zipcode_(98122, 98125]</th> <td> 1.949e+05</td> <td> 6843.637</td> <td>   28.483</td> <td> 0.000</td> <td> 1.82e+05</td> <td> 2.08e+05</td>
</tr>
<tr>
  <th>zipcode_(98125, 98126]</th> <td> 2.019e+05</td> <td> 7344.532</td> <td>   27.484</td> <td> 0.000</td> <td> 1.87e+05</td> <td> 2.16e+05</td>
</tr>
<tr>
  <th>zipcode_(98126, 98133]</th> <td> 1.472e+05</td> <td> 6470.668</td> <td>   22.749</td> <td> 0.000</td> <td> 1.35e+05</td> <td>  1.6e+05</td>
</tr>
<tr>
  <th>zipcode_(98133, 98136]</th> <td> 2.694e+05</td> <td> 8127.482</td> <td>   33.141</td> <td> 0.000</td> <td> 2.53e+05</td> <td> 2.85e+05</td>
</tr>
<tr>
  <th>zipcode_(98136, 98144]</th> <td> 2.671e+05</td> <td> 7412.193</td> <td>   36.031</td> <td> 0.000</td> <td> 2.53e+05</td> <td> 2.82e+05</td>
</tr>
<tr>
  <th>zipcode_(98144, 98146]</th> <td> 1.274e+05</td> <td> 7995.199</td> <td>   15.933</td> <td> 0.000</td> <td> 1.12e+05</td> <td> 1.43e+05</td>
</tr>
<tr>
  <th>zipcode_(98146, 98148]</th> <td> 5.585e+04</td> <td> 1.78e+04</td> <td>    3.130</td> <td> 0.002</td> <td> 2.09e+04</td> <td> 9.08e+04</td>
</tr>
<tr>
  <th>zipcode_(98148, 98155]</th> <td> 1.342e+05</td> <td> 6670.732</td> <td>   20.113</td> <td> 0.000</td> <td> 1.21e+05</td> <td> 1.47e+05</td>
</tr>
<tr>
  <th>zipcode_(98155, 98166]</th> <td> 1.051e+05</td> <td> 8610.820</td> <td>   12.210</td> <td> 0.000</td> <td> 8.83e+04</td> <td> 1.22e+05</td>
</tr>
<tr>
  <th>zipcode_(98166, 98168]</th> <td> 5.576e+04</td> <td> 8049.042</td> <td>    6.928</td> <td> 0.000</td> <td>    4e+04</td> <td> 7.15e+04</td>
</tr>
<tr>
  <th>zipcode_(98168, 98177]</th> <td> 2.314e+05</td> <td> 9002.468</td> <td>   25.699</td> <td> 0.000</td> <td> 2.14e+05</td> <td> 2.49e+05</td>
</tr>
<tr>
  <th>zipcode_(98177, 98178]</th> <td> 5.769e+04</td> <td> 8618.794</td> <td>    6.694</td> <td> 0.000</td> <td> 4.08e+04</td> <td> 7.46e+04</td>
</tr>
<tr>
  <th>zipcode_(98178, 98188]</th> <td> 2.844e+04</td> <td> 1.09e+04</td> <td>    2.617</td> <td> 0.009</td> <td> 7141.271</td> <td> 4.97e+04</td>
</tr>
<tr>
  <th>zipcode_(98188, 98198]</th> <td> 2.856e+04</td> <td> 8187.975</td> <td>    3.488</td> <td> 0.000</td> <td> 1.25e+04</td> <td> 4.46e+04</td>
</tr>
<tr>
  <th>zipcode_(98198, 98199]</th> <td>  3.83e+05</td> <td> 8007.162</td> <td>   47.831</td> <td> 0.000</td> <td> 3.67e+05</td> <td> 3.99e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>3786.374</td> <th>  Durbin-Watson:     </th> <td>   2.025</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>20805.041</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 1.195</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 8.485</td>  <th>  Cond. No.          </th> <td>2.36e+16</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 1.9e-27. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.




```python
residuals2 = model2.resid
fig2 = sm.graphics.qqplot(residuals2, dist=stats.norm, line='45', fit=True)
plt.title('Model2 QQ-plot')
fig2.show();
```

    C:\Users\d_ful\Anaconda3\lib\site-packages\ipykernel_launcher.py:4: UserWarning: Matplotlib is currently using module://ipykernel.pylab.backend_inline, which is a non-GUI backend, so cannot show the figure.
      after removing the cwd from sys.path.
    


![png](output_207_1.png)


> I want to take a look at the MSE for this model, since all of the coefficients fall within our threshold of 0.05 p-value (the closest is 'FLRS_(0.5, 1.0]' at 0.019)


```python
linreg.fit(x_train2, y_train2)

y_hat_train2 = linreg.predict(x_train2)
y_hat_test2 = linreg.predict(x_test2)
```


```python
test2_resid = y_hat_test2 - y_test2
train2_resid = y_hat_train2 - y_train2
```


```python
test2_mse = mean_squared_error(y_test2, y_hat_test2)
train2_mse = mean_squared_error(y_train2, y_hat_train2)
```


```python
print(test2_mse, train2_mse)
```

    11650556325.241385 10810424178.098965
    

> In this case, the MSE isn't telling me too much since it is the _squared_ error against prices in the 100s of thousands of dollars. If I square root these numbers I get (respectively): [104,783.47 & 105,335.00]. This means that the difference of prediction error isn't too different and actually better on the test data. 

* Due to the negligible difference in R-squared between the two models, I will go with model1. I will check the MSE just to make sure it translates to test, and therefore other, data with the same predictability.


```python
linreg.fit(x_train, y_train)

y_hat_train1 = linreg.predict(x_train)
y_hat_test1 = linreg.predict(x_test)
```


```python
test1_resid = y_hat_test1 - y_test
train1_resid = y_hat_train1 - y_train
```


```python
test1_mse = mean_squared_error(y_test, y_hat_test1)
train1_mse = mean_squared_error(y_train, y_hat_train1)
```


```python
print(test1_mse, train1_mse)
```

    11445545357.753975 10864118271.325392
    

> This has almost identical MSEs of (respectively): [105,059.56 & 105,089.41]

## Revised modeling


```python
#setting variables
x = kc_house_feat_nocorr2.copy()
y = pd.DataFrame(price_series2)
```


```python
#t-t split
x_train3, x_test3, y_train3, y_test3 = train_test_split(x, y, test_size=0.30, random_state=42)
```


```python
x_cnst_trn3 = sm.add_constant(x_train3)
model3 = sm.OLS(y_train3, x_cnst_trn3).fit()
model3.summary()
```

    C:\Users\d_ful\Anaconda3\lib\site-packages\numpy\core\fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.816</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.815</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   788.8</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 09 Dec 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>17:14:08</td>     <th>  Log-Likelihood:    </th> <td>-1.8095e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 13947</td>      <th>  AIC:               </th>  <td>3.621e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 13868</td>      <th>  BIC:               </th>  <td>3.626e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    78</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
            <td></td>               <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                 <td>-1.981e+05</td> <td> 1.39e+04</td> <td>  -14.227</td> <td> 0.000</td> <td>-2.25e+05</td> <td>-1.71e+05</td>
</tr>
<tr>
  <th>bedrooms</th>              <td>-5.545e+04</td> <td> 6963.980</td> <td>   -7.963</td> <td> 0.000</td> <td>-6.91e+04</td> <td>-4.18e+04</td>
</tr>
<tr>
  <th>bathrooms</th>             <td> 6.069e+04</td> <td> 8294.357</td> <td>    7.317</td> <td> 0.000</td> <td> 4.44e+04</td> <td> 7.69e+04</td>
</tr>
<tr>
  <th>sqft_living</th>           <td> 6.684e+05</td> <td> 1.13e+04</td> <td>   59.343</td> <td> 0.000</td> <td> 6.46e+05</td> <td>  6.9e+05</td>
</tr>
<tr>
  <th>sqft_lot</th>              <td> 1.171e+05</td> <td> 1.18e+04</td> <td>    9.889</td> <td> 0.000</td> <td> 9.39e+04</td> <td>  1.4e+05</td>
</tr>
<tr>
  <th>waterfront</th>            <td>-2.739e-09</td> <td> 1.06e-10</td> <td>  -25.889</td> <td> 0.000</td> <td>-2.95e-09</td> <td>-2.53e-09</td>
</tr>
<tr>
  <th>condition</th>             <td> 2.602e+04</td> <td> 1559.528</td> <td>   16.685</td> <td> 0.000</td> <td>  2.3e+04</td> <td> 2.91e+04</td>
</tr>
<tr>
  <th>grade</th>                 <td> 5.776e+04</td> <td> 1466.285</td> <td>   39.393</td> <td> 0.000</td> <td> 5.49e+04</td> <td> 6.06e+04</td>
</tr>
<tr>
  <th>sqft_basement</th>         <td>-6.739e+04</td> <td> 5112.338</td> <td>  -13.183</td> <td> 0.000</td> <td>-7.74e+04</td> <td>-5.74e+04</td>
</tr>
<tr>
  <th>flrs_(0.5, 1.0]</th>       <td> -1.18e+04</td> <td> 2608.919</td> <td>   -4.523</td> <td> 0.000</td> <td>-1.69e+04</td> <td>-6687.503</td>
</tr>
<tr>
  <th>yr_built_(1975, 2016]</th> <td>-3.668e+04</td> <td> 2793.753</td> <td>  -13.128</td> <td> 0.000</td> <td>-4.22e+04</td> <td>-3.12e+04</td>
</tr>
<tr>
  <th>zipcode_98002</th>         <td>  2.71e+04</td> <td> 1.14e+04</td> <td>    2.367</td> <td> 0.018</td> <td> 4661.280</td> <td> 4.95e+04</td>
</tr>
<tr>
  <th>zipcode_98003</th>         <td>   29.5693</td> <td> 1.02e+04</td> <td>    0.003</td> <td> 0.998</td> <td>   -2e+04</td> <td> 2.01e+04</td>
</tr>
<tr>
  <th>zipcode_98004</th>         <td> 6.007e+05</td> <td>  1.1e+04</td> <td>   54.558</td> <td> 0.000</td> <td> 5.79e+05</td> <td> 6.22e+05</td>
</tr>
<tr>
  <th>zipcode_98005</th>         <td> 3.121e+05</td> <td> 1.24e+04</td> <td>   25.257</td> <td> 0.000</td> <td> 2.88e+05</td> <td> 3.36e+05</td>
</tr>
<tr>
  <th>zipcode_98006</th>         <td>   2.8e+05</td> <td> 9207.709</td> <td>   30.413</td> <td> 0.000</td> <td> 2.62e+05</td> <td> 2.98e+05</td>
</tr>
<tr>
  <th>zipcode_98007</th>         <td> 2.492e+05</td> <td> 1.27e+04</td> <td>   19.618</td> <td> 0.000</td> <td> 2.24e+05</td> <td> 2.74e+05</td>
</tr>
<tr>
  <th>zipcode_98008</th>         <td> 2.489e+05</td> <td> 1.02e+04</td> <td>   24.298</td> <td> 0.000</td> <td> 2.29e+05</td> <td> 2.69e+05</td>
</tr>
<tr>
  <th>zipcode_98010</th>         <td> 8.195e+04</td> <td> 1.58e+04</td> <td>    5.202</td> <td> 0.000</td> <td> 5.11e+04</td> <td> 1.13e+05</td>
</tr>
<tr>
  <th>zipcode_98011</th>         <td> 1.457e+05</td> <td> 1.12e+04</td> <td>   12.979</td> <td> 0.000</td> <td> 1.24e+05</td> <td> 1.68e+05</td>
</tr>
<tr>
  <th>zipcode_98014</th>         <td> 9.494e+04</td> <td> 1.51e+04</td> <td>    6.268</td> <td> 0.000</td> <td> 6.53e+04</td> <td> 1.25e+05</td>
</tr>
<tr>
  <th>zipcode_98019</th>         <td> 1.028e+05</td> <td> 1.17e+04</td> <td>    8.777</td> <td> 0.000</td> <td> 7.99e+04</td> <td> 1.26e+05</td>
</tr>
<tr>
  <th>zipcode_98022</th>         <td> 1.411e+04</td> <td> 1.17e+04</td> <td>    1.205</td> <td> 0.228</td> <td>-8852.141</td> <td> 3.71e+04</td>
</tr>
<tr>
  <th>zipcode_98023</th>         <td>-1.549e+04</td> <td> 8785.130</td> <td>   -1.763</td> <td> 0.078</td> <td>-3.27e+04</td> <td> 1732.276</td>
</tr>
<tr>
  <th>zipcode_98024</th>         <td> 1.496e+05</td> <td>  1.9e+04</td> <td>    7.889</td> <td> 0.000</td> <td> 1.12e+05</td> <td> 1.87e+05</td>
</tr>
<tr>
  <th>zipcode_98027</th>         <td>  1.93e+05</td> <td> 9352.032</td> <td>   20.632</td> <td> 0.000</td> <td> 1.75e+05</td> <td> 2.11e+05</td>
</tr>
<tr>
  <th>zipcode_98028</th>         <td> 1.394e+05</td> <td> 1.04e+04</td> <td>   13.360</td> <td> 0.000</td> <td> 1.19e+05</td> <td>  1.6e+05</td>
</tr>
<tr>
  <th>zipcode_98029</th>         <td> 2.273e+05</td> <td> 9803.834</td> <td>   23.184</td> <td> 0.000</td> <td> 2.08e+05</td> <td> 2.47e+05</td>
</tr>
<tr>
  <th>zipcode_98030</th>         <td> 7557.7923</td> <td> 1.01e+04</td> <td>    0.745</td> <td> 0.456</td> <td>-1.23e+04</td> <td> 2.75e+04</td>
</tr>
<tr>
  <th>zipcode_98031</th>         <td> 1.552e+04</td> <td>    1e+04</td> <td>    1.545</td> <td> 0.122</td> <td>-4170.527</td> <td> 3.52e+04</td>
</tr>
<tr>
  <th>zipcode_98032</th>         <td> 5960.7674</td> <td> 1.42e+04</td> <td>    0.420</td> <td> 0.675</td> <td>-2.19e+04</td> <td> 3.38e+04</td>
</tr>
<tr>
  <th>zipcode_98033</th>         <td> 3.583e+05</td> <td> 9081.739</td> <td>   39.458</td> <td> 0.000</td> <td> 3.41e+05</td> <td> 3.76e+05</td>
</tr>
<tr>
  <th>zipcode_98034</th>         <td> 1.931e+05</td> <td> 8672.687</td> <td>   22.270</td> <td> 0.000</td> <td> 1.76e+05</td> <td>  2.1e+05</td>
</tr>
<tr>
  <th>zipcode_98038</th>         <td> 4.047e+04</td> <td> 8668.746</td> <td>    4.668</td> <td> 0.000</td> <td> 2.35e+04</td> <td> 5.75e+04</td>
</tr>
<tr>
  <th>zipcode_98039</th>         <td> 8.337e+05</td> <td>  3.1e+04</td> <td>   26.920</td> <td> 0.000</td> <td> 7.73e+05</td> <td> 8.94e+05</td>
</tr>
<tr>
  <th>zipcode_98040</th>         <td> 4.925e+05</td> <td> 1.09e+04</td> <td>   45.208</td> <td> 0.000</td> <td> 4.71e+05</td> <td> 5.14e+05</td>
</tr>
<tr>
  <th>zipcode_98042</th>         <td> 8486.7820</td> <td> 8705.497</td> <td>    0.975</td> <td> 0.330</td> <td>-8577.168</td> <td> 2.56e+04</td>
</tr>
<tr>
  <th>zipcode_98045</th>         <td> 1.149e+05</td> <td> 1.11e+04</td> <td>   10.371</td> <td> 0.000</td> <td> 9.32e+04</td> <td> 1.37e+05</td>
</tr>
<tr>
  <th>zipcode_98052</th>         <td> 2.461e+05</td> <td> 8603.854</td> <td>   28.603</td> <td> 0.000</td> <td> 2.29e+05</td> <td> 2.63e+05</td>
</tr>
<tr>
  <th>zipcode_98053</th>         <td>  2.28e+05</td> <td> 9592.237</td> <td>   23.767</td> <td> 0.000</td> <td> 2.09e+05</td> <td> 2.47e+05</td>
</tr>
<tr>
  <th>zipcode_98055</th>         <td> 4.967e+04</td> <td> 1.04e+04</td> <td>    4.760</td> <td> 0.000</td> <td> 2.92e+04</td> <td> 7.01e+04</td>
</tr>
<tr>
  <th>zipcode_98056</th>         <td> 1.055e+05</td> <td> 9369.436</td> <td>   11.259</td> <td> 0.000</td> <td> 8.71e+04</td> <td> 1.24e+05</td>
</tr>
<tr>
  <th>zipcode_98058</th>         <td> 3.387e+04</td> <td> 9010.555</td> <td>    3.759</td> <td> 0.000</td> <td> 1.62e+04</td> <td> 5.15e+04</td>
</tr>
<tr>
  <th>zipcode_98059</th>         <td> 1.019e+05</td> <td> 9012.383</td> <td>   11.304</td> <td> 0.000</td> <td> 8.42e+04</td> <td>  1.2e+05</td>
</tr>
<tr>
  <th>zipcode_98065</th>         <td> 1.306e+05</td> <td> 1.01e+04</td> <td>   12.942</td> <td> 0.000</td> <td> 1.11e+05</td> <td>  1.5e+05</td>
</tr>
<tr>
  <th>zipcode_98070</th>         <td> 1.359e+05</td> <td> 1.78e+04</td> <td>    7.650</td> <td> 0.000</td> <td> 1.01e+05</td> <td> 1.71e+05</td>
</tr>
<tr>
  <th>zipcode_98072</th>         <td>  1.65e+05</td> <td> 1.03e+04</td> <td>   15.954</td> <td> 0.000</td> <td> 1.45e+05</td> <td> 1.85e+05</td>
</tr>
<tr>
  <th>zipcode_98074</th>         <td>  2.08e+05</td> <td> 9209.852</td> <td>   22.581</td> <td> 0.000</td> <td>  1.9e+05</td> <td> 2.26e+05</td>
</tr>
<tr>
  <th>zipcode_98075</th>         <td> 2.222e+05</td> <td> 9763.093</td> <td>   22.762</td> <td> 0.000</td> <td> 2.03e+05</td> <td> 2.41e+05</td>
</tr>
<tr>
  <th>zipcode_98077</th>         <td> 1.489e+05</td> <td> 1.19e+04</td> <td>   12.493</td> <td> 0.000</td> <td> 1.26e+05</td> <td> 1.72e+05</td>
</tr>
<tr>
  <th>zipcode_98092</th>         <td>-2.155e+04</td> <td> 9828.648</td> <td>   -2.193</td> <td> 0.028</td> <td>-4.08e+04</td> <td>-2284.433</td>
</tr>
<tr>
  <th>zipcode_98102</th>         <td> 4.554e+05</td> <td> 1.44e+04</td> <td>   31.670</td> <td> 0.000</td> <td> 4.27e+05</td> <td> 4.84e+05</td>
</tr>
<tr>
  <th>zipcode_98103</th>         <td> 3.346e+05</td> <td> 8619.743</td> <td>   38.821</td> <td> 0.000</td> <td> 3.18e+05</td> <td> 3.52e+05</td>
</tr>
<tr>
  <th>zipcode_98105</th>         <td> 4.449e+05</td> <td> 1.11e+04</td> <td>   40.262</td> <td> 0.000</td> <td> 4.23e+05</td> <td> 4.67e+05</td>
</tr>
<tr>
  <th>zipcode_98106</th>         <td> 1.284e+05</td> <td> 9836.630</td> <td>   13.050</td> <td> 0.000</td> <td> 1.09e+05</td> <td> 1.48e+05</td>
</tr>
<tr>
  <th>zipcode_98107</th>         <td> 3.302e+05</td> <td> 1.04e+04</td> <td>   31.724</td> <td> 0.000</td> <td>  3.1e+05</td> <td> 3.51e+05</td>
</tr>
<tr>
  <th>zipcode_98108</th>         <td> 1.322e+05</td> <td> 1.15e+04</td> <td>   11.534</td> <td> 0.000</td> <td>  1.1e+05</td> <td> 1.55e+05</td>
</tr>
<tr>
  <th>zipcode_98109</th>         <td> 4.945e+05</td> <td> 1.45e+04</td> <td>   34.076</td> <td> 0.000</td> <td> 4.66e+05</td> <td> 5.23e+05</td>
</tr>
<tr>
  <th>zipcode_98112</th>         <td> 5.261e+05</td> <td> 1.12e+04</td> <td>   46.904</td> <td> 0.000</td> <td> 5.04e+05</td> <td> 5.48e+05</td>
</tr>
<tr>
  <th>zipcode_98115</th>         <td> 3.453e+05</td> <td> 8613.849</td> <td>   40.090</td> <td> 0.000</td> <td> 3.28e+05</td> <td> 3.62e+05</td>
</tr>
<tr>
  <th>zipcode_98116</th>         <td> 3.218e+05</td> <td> 9758.923</td> <td>   32.974</td> <td> 0.000</td> <td> 3.03e+05</td> <td> 3.41e+05</td>
</tr>
<tr>
  <th>zipcode_98117</th>         <td> 3.271e+05</td> <td> 8704.277</td> <td>   37.583</td> <td> 0.000</td> <td>  3.1e+05</td> <td> 3.44e+05</td>
</tr>
<tr>
  <th>zipcode_98118</th>         <td> 1.858e+05</td> <td> 8967.124</td> <td>   20.715</td> <td> 0.000</td> <td> 1.68e+05</td> <td> 2.03e+05</td>
</tr>
<tr>
  <th>zipcode_98119</th>         <td> 4.741e+05</td> <td> 1.14e+04</td> <td>   41.643</td> <td> 0.000</td> <td> 4.52e+05</td> <td> 4.96e+05</td>
</tr>
<tr>
  <th>zipcode_98122</th>         <td> 3.296e+05</td> <td> 1.03e+04</td> <td>   32.005</td> <td> 0.000</td> <td> 3.09e+05</td> <td>  3.5e+05</td>
</tr>
<tr>
  <th>zipcode_98125</th>         <td> 2.017e+05</td> <td> 9395.126</td> <td>   21.472</td> <td> 0.000</td> <td> 1.83e+05</td> <td>  2.2e+05</td>
</tr>
<tr>
  <th>zipcode_98126</th>         <td> 2.087e+05</td> <td> 9550.835</td> <td>   21.848</td> <td> 0.000</td> <td>  1.9e+05</td> <td> 2.27e+05</td>
</tr>
<tr>
  <th>zipcode_98133</th>         <td> 1.573e+05</td> <td> 8814.924</td> <td>   17.849</td> <td> 0.000</td> <td>  1.4e+05</td> <td> 1.75e+05</td>
</tr>
<tr>
  <th>zipcode_98136</th>         <td> 2.757e+05</td> <td> 1.07e+04</td> <td>   25.662</td> <td> 0.000</td> <td> 2.55e+05</td> <td> 2.97e+05</td>
</tr>
<tr>
  <th>zipcode_98144</th>         <td> 2.707e+05</td> <td> 9765.997</td> <td>   27.716</td> <td> 0.000</td> <td> 2.52e+05</td> <td>  2.9e+05</td>
</tr>
<tr>
  <th>zipcode_98146</th>         <td> 1.372e+05</td> <td> 1.02e+04</td> <td>   13.456</td> <td> 0.000</td> <td> 1.17e+05</td> <td> 1.57e+05</td>
</tr>
<tr>
  <th>zipcode_98148</th>         <td> 5.869e+04</td> <td> 1.81e+04</td> <td>    3.247</td> <td> 0.001</td> <td> 2.33e+04</td> <td> 9.41e+04</td>
</tr>
<tr>
  <th>zipcode_98155</th>         <td> 1.417e+05</td> <td> 9031.290</td> <td>   15.687</td> <td> 0.000</td> <td> 1.24e+05</td> <td> 1.59e+05</td>
</tr>
<tr>
  <th>zipcode_98166</th>         <td> 1.125e+05</td> <td> 1.08e+04</td> <td>   10.371</td> <td> 0.000</td> <td> 9.12e+04</td> <td> 1.34e+05</td>
</tr>
<tr>
  <th>zipcode_98168</th>         <td> 6.525e+04</td> <td> 1.04e+04</td> <td>    6.262</td> <td> 0.000</td> <td> 4.48e+04</td> <td> 8.57e+04</td>
</tr>
<tr>
  <th>zipcode_98177</th>         <td> 2.439e+05</td> <td> 1.05e+04</td> <td>   23.183</td> <td> 0.000</td> <td> 2.23e+05</td> <td> 2.65e+05</td>
</tr>
<tr>
  <th>zipcode_98178</th>         <td> 6.656e+04</td> <td> 1.04e+04</td> <td>    6.417</td> <td> 0.000</td> <td> 4.62e+04</td> <td> 8.69e+04</td>
</tr>
<tr>
  <th>zipcode_98188</th>         <td> 3.945e+04</td> <td> 1.28e+04</td> <td>    3.088</td> <td> 0.002</td> <td> 1.44e+04</td> <td> 6.45e+04</td>
</tr>
<tr>
  <th>zipcode_98198</th>         <td> 4.587e+04</td> <td> 1.03e+04</td> <td>    4.470</td> <td> 0.000</td> <td> 2.58e+04</td> <td>  6.6e+04</td>
</tr>
<tr>
  <th>zipcode_98199</th>         <td> 3.901e+05</td> <td> 1.01e+04</td> <td>   38.593</td> <td> 0.000</td> <td>  3.7e+05</td> <td>  4.1e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>3610.277</td> <th>  Durbin-Watson:     </th> <td>   1.983</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>20487.040</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 1.124</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 8.495</td>  <th>  Cond. No.          </th> <td>2.25e+15</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 1.96e-25. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.



> * My changes have not altered the R-squared figure considerably. They, after rounding, are indentical in fact. 
* However, the bedroom and bathroom features have had their coefficients become polarized. Bedrooms has decreased by 4.443e+04 while bathrooms has increased by 4.500e+04.
* This may be explained by my features now including negative values, forcing the model to account for this with the coefficients.


```python
residuals3 = model3.resid
fig3 = sm.graphics.qqplot(residuals3, dist=stats.norm, line='45', fit=True)
plt.title('Model3 QQ-plot')
fig3.show();
```

    C:\Users\d_ful\Anaconda3\lib\site-packages\ipykernel_launcher.py:4: UserWarning: Matplotlib is currently using module://ipykernel.pylab.backend_inline, which is a non-GUI backend, so cannot show the figure.
      after removing the cwd from sys.path.
    


![png](output_225_1.png)


> My revisions still show a similar skew to the first model I did. There is a potential to re-work the data to be more normal still.

# iNTERPRET


```python
model1.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.816</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.815</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   770.0</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 09 Dec 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>17:06:44</td>     <th>  Log-Likelihood:    </th> <td>-1.8094e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 13947</td>      <th>  AIC:               </th>  <td>3.620e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 13866</td>      <th>  BIC:               </th>  <td>3.627e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    80</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
             <td></td>               <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                  <td>-2.722e+05</td> <td> 9466.914</td> <td>  -28.756</td> <td> 0.000</td> <td>-2.91e+05</td> <td>-2.54e+05</td>
</tr>
<tr>
  <th>bedrooms</th>               <td>-1.102e+04</td> <td> 1392.416</td> <td>   -7.912</td> <td> 0.000</td> <td>-1.37e+04</td> <td>-8287.796</td>
</tr>
<tr>
  <th>bathrooms</th>              <td> 1.569e+04</td> <td> 2214.696</td> <td>    7.087</td> <td> 0.000</td> <td> 1.14e+04</td> <td>    2e+04</td>
</tr>
<tr>
  <th>sqft_living</th>            <td> 6.684e+05</td> <td> 1.13e+04</td> <td>   59.369</td> <td> 0.000</td> <td> 6.46e+05</td> <td>  6.9e+05</td>
</tr>
<tr>
  <th>sqft_lot</th>               <td> 1.183e+05</td> <td> 1.19e+04</td> <td>    9.965</td> <td> 0.000</td> <td>  9.5e+04</td> <td> 1.42e+05</td>
</tr>
<tr>
  <th>waterfront</th>             <td> 1.403e-09</td> <td> 3.45e-11</td> <td>   40.666</td> <td> 0.000</td> <td> 1.33e-09</td> <td> 1.47e-09</td>
</tr>
<tr>
  <th>grade</th>                  <td> 5.779e+04</td> <td> 1467.767</td> <td>   39.375</td> <td> 0.000</td> <td> 5.49e+04</td> <td> 6.07e+04</td>
</tr>
<tr>
  <th>sqft_basement</th>          <td>-6.719e+04</td> <td> 5110.231</td> <td>  -13.149</td> <td> 0.000</td> <td>-7.72e+04</td> <td>-5.72e+04</td>
</tr>
<tr>
  <th>flrs_(0.5, 1.0]</th>        <td>-1.213e+04</td> <td> 2609.426</td> <td>   -4.650</td> <td> 0.000</td> <td>-1.72e+04</td> <td>-7019.893</td>
</tr>
<tr>
  <th>cond_(1, 2]</th>            <td>-1.007e+05</td> <td> 9100.880</td> <td>  -11.064</td> <td> 0.000</td> <td>-1.19e+05</td> <td>-8.29e+04</td>
</tr>
<tr>
  <th>cond_(2, 3]</th>            <td>-8.369e+04</td> <td> 3581.974</td> <td>  -23.364</td> <td> 0.000</td> <td>-9.07e+04</td> <td>-7.67e+04</td>
</tr>
<tr>
  <th>cond_(3, 4]</th>            <td>-6.338e+04</td> <td> 3578.620</td> <td>  -17.711</td> <td> 0.000</td> <td>-7.04e+04</td> <td>-5.64e+04</td>
</tr>
<tr>
  <th>cond_(4, 5]</th>            <td>-2.447e+04</td> <td> 4202.311</td> <td>   -5.822</td> <td> 0.000</td> <td>-3.27e+04</td> <td>-1.62e+04</td>
</tr>
<tr>
  <th>yr_built_(1975, 2016]</th>  <td>-3.683e+04</td> <td> 2799.248</td> <td>  -13.156</td> <td> 0.000</td> <td>-4.23e+04</td> <td>-3.13e+04</td>
</tr>
<tr>
  <th>zipcode_(98001, 98002]</th> <td> 2.779e+04</td> <td> 1.14e+04</td> <td>    2.428</td> <td> 0.015</td> <td> 5352.776</td> <td> 5.02e+04</td>
</tr>
<tr>
  <th>zipcode_(98002, 98003]</th> <td>  644.8292</td> <td> 1.02e+04</td> <td>    0.063</td> <td> 0.950</td> <td>-1.94e+04</td> <td> 2.07e+04</td>
</tr>
<tr>
  <th>zipcode_(98003, 98004]</th> <td> 6.015e+05</td> <td>  1.1e+04</td> <td>   54.641</td> <td> 0.000</td> <td>  5.8e+05</td> <td> 6.23e+05</td>
</tr>
<tr>
  <th>zipcode_(98004, 98005]</th> <td> 3.128e+05</td> <td> 1.24e+04</td> <td>   25.316</td> <td> 0.000</td> <td> 2.89e+05</td> <td> 3.37e+05</td>
</tr>
<tr>
  <th>zipcode_(98005, 98006]</th> <td> 2.803e+05</td> <td> 9206.965</td> <td>   30.446</td> <td> 0.000</td> <td> 2.62e+05</td> <td> 2.98e+05</td>
</tr>
<tr>
  <th>zipcode_(98006, 98007]</th> <td> 2.506e+05</td> <td> 1.27e+04</td> <td>   19.732</td> <td> 0.000</td> <td> 2.26e+05</td> <td> 2.76e+05</td>
</tr>
<tr>
  <th>zipcode_(98007, 98008]</th> <td>   2.5e+05</td> <td> 1.02e+04</td> <td>   24.399</td> <td> 0.000</td> <td>  2.3e+05</td> <td>  2.7e+05</td>
</tr>
<tr>
  <th>zipcode_(98008, 98010]</th> <td> 8.185e+04</td> <td> 1.57e+04</td> <td>    5.198</td> <td> 0.000</td> <td>  5.1e+04</td> <td> 1.13e+05</td>
</tr>
<tr>
  <th>zipcode_(98010, 98011]</th> <td> 1.454e+05</td> <td> 1.12e+04</td> <td>   12.959</td> <td> 0.000</td> <td> 1.23e+05</td> <td> 1.67e+05</td>
</tr>
<tr>
  <th>zipcode_(98011, 98014]</th> <td> 9.482e+04</td> <td> 1.51e+04</td> <td>    6.263</td> <td> 0.000</td> <td> 6.51e+04</td> <td> 1.24e+05</td>
</tr>
<tr>
  <th>zipcode_(98014, 98019]</th> <td> 1.023e+05</td> <td> 1.17e+04</td> <td>    8.736</td> <td> 0.000</td> <td> 7.93e+04</td> <td> 1.25e+05</td>
</tr>
<tr>
  <th>zipcode_(98019, 98022]</th> <td> 1.341e+04</td> <td> 1.17e+04</td> <td>    1.145</td> <td> 0.252</td> <td>-9549.282</td> <td> 3.64e+04</td>
</tr>
<tr>
  <th>zipcode_(98022, 98023]</th> <td>-1.466e+04</td> <td> 8787.664</td> <td>   -1.669</td> <td> 0.095</td> <td>-3.19e+04</td> <td> 2561.028</td>
</tr>
<tr>
  <th>zipcode_(98023, 98024]</th> <td> 1.495e+05</td> <td>  1.9e+04</td> <td>    7.883</td> <td> 0.000</td> <td> 1.12e+05</td> <td> 1.87e+05</td>
</tr>
<tr>
  <th>zipcode_(98024, 98027]</th> <td> 1.932e+05</td> <td> 9348.282</td> <td>   20.665</td> <td> 0.000</td> <td> 1.75e+05</td> <td> 2.12e+05</td>
</tr>
<tr>
  <th>zipcode_(98027, 98028]</th> <td> 1.392e+05</td> <td> 1.04e+04</td> <td>   13.345</td> <td> 0.000</td> <td> 1.19e+05</td> <td>  1.6e+05</td>
</tr>
<tr>
  <th>zipcode_(98028, 98029]</th> <td> 2.272e+05</td> <td> 9799.849</td> <td>   23.183</td> <td> 0.000</td> <td> 2.08e+05</td> <td> 2.46e+05</td>
</tr>
<tr>
  <th>zipcode_(98029, 98030]</th> <td> 7979.9222</td> <td> 1.01e+04</td> <td>    0.787</td> <td> 0.432</td> <td>-1.19e+04</td> <td> 2.79e+04</td>
</tr>
<tr>
  <th>zipcode_(98030, 98031]</th> <td> 1.598e+04</td> <td>    1e+04</td> <td>    1.592</td> <td> 0.111</td> <td>-3699.663</td> <td> 3.57e+04</td>
</tr>
<tr>
  <th>zipcode_(98031, 98032]</th> <td> 6525.0559</td> <td> 1.42e+04</td> <td>    0.459</td> <td> 0.646</td> <td>-2.13e+04</td> <td> 3.44e+04</td>
</tr>
<tr>
  <th>zipcode_(98032, 98033]</th> <td> 3.585e+05</td> <td> 9079.068</td> <td>   39.484</td> <td> 0.000</td> <td> 3.41e+05</td> <td> 3.76e+05</td>
</tr>
<tr>
  <th>zipcode_(98033, 98034]</th> <td>  1.93e+05</td> <td> 8668.815</td> <td>   22.265</td> <td> 0.000</td> <td> 1.76e+05</td> <td>  2.1e+05</td>
</tr>
<tr>
  <th>zipcode_(98034, 98038]</th> <td> 4.024e+04</td> <td> 8664.925</td> <td>    4.644</td> <td> 0.000</td> <td> 2.33e+04</td> <td> 5.72e+04</td>
</tr>
<tr>
  <th>zipcode_(98038, 98039]</th> <td> 8.347e+05</td> <td>  3.1e+04</td> <td>   26.964</td> <td> 0.000</td> <td> 7.74e+05</td> <td> 8.95e+05</td>
</tr>
<tr>
  <th>zipcode_(98039, 98040]</th> <td> 4.933e+05</td> <td> 1.09e+04</td> <td>   45.264</td> <td> 0.000</td> <td> 4.72e+05</td> <td> 5.15e+05</td>
</tr>
<tr>
  <th>zipcode_(98040, 98042]</th> <td> 8518.5273</td> <td> 8704.417</td> <td>    0.979</td> <td> 0.328</td> <td>-8543.306</td> <td> 2.56e+04</td>
</tr>
<tr>
  <th>zipcode_(98042, 98045]</th> <td> 1.143e+05</td> <td> 1.11e+04</td> <td>   10.322</td> <td> 0.000</td> <td> 9.26e+04</td> <td> 1.36e+05</td>
</tr>
<tr>
  <th>zipcode_(98045, 98052]</th> <td> 2.466e+05</td> <td> 8602.569</td> <td>   28.668</td> <td> 0.000</td> <td>  2.3e+05</td> <td> 2.63e+05</td>
</tr>
<tr>
  <th>zipcode_(98052, 98053]</th> <td> 2.272e+05</td> <td> 9589.890</td> <td>   23.691</td> <td> 0.000</td> <td> 2.08e+05</td> <td> 2.46e+05</td>
</tr>
<tr>
  <th>zipcode_(98053, 98055]</th> <td>  4.94e+04</td> <td> 1.04e+04</td> <td>    4.736</td> <td> 0.000</td> <td>  2.9e+04</td> <td> 6.98e+04</td>
</tr>
<tr>
  <th>zipcode_(98055, 98056]</th> <td> 1.044e+05</td> <td> 9369.818</td> <td>   11.147</td> <td> 0.000</td> <td> 8.61e+04</td> <td> 1.23e+05</td>
</tr>
<tr>
  <th>zipcode_(98056, 98058]</th> <td> 3.397e+04</td> <td> 9010.134</td> <td>    3.770</td> <td> 0.000</td> <td> 1.63e+04</td> <td> 5.16e+04</td>
</tr>
<tr>
  <th>zipcode_(98058, 98059]</th> <td> 1.018e+05</td> <td> 9009.468</td> <td>   11.294</td> <td> 0.000</td> <td> 8.41e+04</td> <td> 1.19e+05</td>
</tr>
<tr>
  <th>zipcode_(98059, 98065]</th> <td> 1.302e+05</td> <td> 1.01e+04</td> <td>   12.910</td> <td> 0.000</td> <td>  1.1e+05</td> <td>  1.5e+05</td>
</tr>
<tr>
  <th>zipcode_(98065, 98070]</th> <td> 1.363e+05</td> <td> 1.78e+04</td> <td>    7.677</td> <td> 0.000</td> <td> 1.02e+05</td> <td> 1.71e+05</td>
</tr>
<tr>
  <th>zipcode_(98070, 98072]</th> <td> 1.652e+05</td> <td> 1.03e+04</td> <td>   15.979</td> <td> 0.000</td> <td> 1.45e+05</td> <td> 1.85e+05</td>
</tr>
<tr>
  <th>zipcode_(98072, 98074]</th> <td> 2.075e+05</td> <td> 9207.565</td> <td>   22.533</td> <td> 0.000</td> <td> 1.89e+05</td> <td> 2.26e+05</td>
</tr>
<tr>
  <th>zipcode_(98074, 98075]</th> <td> 2.222e+05</td> <td> 9759.176</td> <td>   22.767</td> <td> 0.000</td> <td> 2.03e+05</td> <td> 2.41e+05</td>
</tr>
<tr>
  <th>zipcode_(98075, 98077]</th> <td> 1.494e+05</td> <td> 1.19e+04</td> <td>   12.542</td> <td> 0.000</td> <td> 1.26e+05</td> <td> 1.73e+05</td>
</tr>
<tr>
  <th>zipcode_(98077, 98092]</th> <td>-2.146e+04</td> <td> 9824.939</td> <td>   -2.184</td> <td> 0.029</td> <td>-4.07e+04</td> <td>-2203.016</td>
</tr>
<tr>
  <th>zipcode_(98092, 98102]</th> <td> 4.551e+05</td> <td> 1.44e+04</td> <td>   31.666</td> <td> 0.000</td> <td> 4.27e+05</td> <td> 4.83e+05</td>
</tr>
<tr>
  <th>zipcode_(98102, 98103]</th> <td> 3.338e+05</td> <td> 8618.463</td> <td>   38.733</td> <td> 0.000</td> <td> 3.17e+05</td> <td> 3.51e+05</td>
</tr>
<tr>
  <th>zipcode_(98103, 98105]</th> <td> 4.446e+05</td> <td>  1.1e+04</td> <td>   40.248</td> <td> 0.000</td> <td> 4.23e+05</td> <td> 4.66e+05</td>
</tr>
<tr>
  <th>zipcode_(98105, 98106]</th> <td> 1.277e+05</td> <td> 9834.021</td> <td>   12.981</td> <td> 0.000</td> <td> 1.08e+05</td> <td> 1.47e+05</td>
</tr>
<tr>
  <th>zipcode_(98106, 98107]</th> <td> 3.297e+05</td> <td> 1.04e+04</td> <td>   31.689</td> <td> 0.000</td> <td> 3.09e+05</td> <td>  3.5e+05</td>
</tr>
<tr>
  <th>zipcode_(98107, 98108]</th> <td> 1.313e+05</td> <td> 1.15e+04</td> <td>   11.453</td> <td> 0.000</td> <td> 1.09e+05</td> <td> 1.54e+05</td>
</tr>
<tr>
  <th>zipcode_(98108, 98109]</th> <td> 4.933e+05</td> <td> 1.45e+04</td> <td>   34.000</td> <td> 0.000</td> <td> 4.65e+05</td> <td> 5.22e+05</td>
</tr>
<tr>
  <th>zipcode_(98109, 98112]</th> <td> 5.253e+05</td> <td> 1.12e+04</td> <td>   46.838</td> <td> 0.000</td> <td> 5.03e+05</td> <td> 5.47e+05</td>
</tr>
<tr>
  <th>zipcode_(98112, 98115]</th> <td> 3.442e+05</td> <td> 8614.506</td> <td>   39.960</td> <td> 0.000</td> <td> 3.27e+05</td> <td> 3.61e+05</td>
</tr>
<tr>
  <th>zipcode_(98115, 98116]</th> <td>  3.21e+05</td> <td> 9756.845</td> <td>   32.897</td> <td> 0.000</td> <td> 3.02e+05</td> <td>  3.4e+05</td>
</tr>
<tr>
  <th>zipcode_(98116, 98117]</th> <td> 3.268e+05</td> <td> 8701.084</td> <td>   37.564</td> <td> 0.000</td> <td>  3.1e+05</td> <td> 3.44e+05</td>
</tr>
<tr>
  <th>zipcode_(98117, 98118]</th> <td> 1.847e+05</td> <td> 8969.863</td> <td>   20.589</td> <td> 0.000</td> <td> 1.67e+05</td> <td> 2.02e+05</td>
</tr>
<tr>
  <th>zipcode_(98118, 98119]</th> <td> 4.733e+05</td> <td> 1.14e+04</td> <td>   41.590</td> <td> 0.000</td> <td> 4.51e+05</td> <td> 4.96e+05</td>
</tr>
<tr>
  <th>zipcode_(98119, 98122]</th> <td> 3.287e+05</td> <td> 1.03e+04</td> <td>   31.929</td> <td> 0.000</td> <td> 3.09e+05</td> <td> 3.49e+05</td>
</tr>
<tr>
  <th>zipcode_(98122, 98125]</th> <td> 2.009e+05</td> <td> 9393.389</td> <td>   21.384</td> <td> 0.000</td> <td> 1.82e+05</td> <td> 2.19e+05</td>
</tr>
<tr>
  <th>zipcode_(98125, 98126]</th> <td> 2.076e+05</td> <td> 9550.069</td> <td>   21.742</td> <td> 0.000</td> <td> 1.89e+05</td> <td> 2.26e+05</td>
</tr>
<tr>
  <th>zipcode_(98126, 98133]</th> <td> 1.572e+05</td> <td> 8810.901</td> <td>   17.838</td> <td> 0.000</td> <td>  1.4e+05</td> <td> 1.74e+05</td>
</tr>
<tr>
  <th>zipcode_(98133, 98136]</th> <td> 2.752e+05</td> <td> 1.07e+04</td> <td>   25.627</td> <td> 0.000</td> <td> 2.54e+05</td> <td> 2.96e+05</td>
</tr>
<tr>
  <th>zipcode_(98136, 98144]</th> <td>   2.7e+05</td> <td> 9762.938</td> <td>   27.659</td> <td> 0.000</td> <td> 2.51e+05</td> <td> 2.89e+05</td>
</tr>
<tr>
  <th>zipcode_(98144, 98146]</th> <td> 1.364e+05</td> <td> 1.02e+04</td> <td>   13.379</td> <td> 0.000</td> <td> 1.16e+05</td> <td> 1.56e+05</td>
</tr>
<tr>
  <th>zipcode_(98146, 98148]</th> <td> 5.694e+04</td> <td> 1.81e+04</td> <td>    3.151</td> <td> 0.002</td> <td> 2.15e+04</td> <td> 9.24e+04</td>
</tr>
<tr>
  <th>zipcode_(98148, 98155]</th> <td> 1.409e+05</td> <td> 9029.205</td> <td>   15.607</td> <td> 0.000</td> <td> 1.23e+05</td> <td> 1.59e+05</td>
</tr>
<tr>
  <th>zipcode_(98155, 98166]</th> <td> 1.128e+05</td> <td> 1.08e+04</td> <td>   10.400</td> <td> 0.000</td> <td> 9.15e+04</td> <td> 1.34e+05</td>
</tr>
<tr>
  <th>zipcode_(98166, 98168]</th> <td> 6.402e+04</td> <td> 1.04e+04</td> <td>    6.143</td> <td> 0.000</td> <td> 4.36e+04</td> <td> 8.44e+04</td>
</tr>
<tr>
  <th>zipcode_(98168, 98177]</th> <td> 2.436e+05</td> <td> 1.05e+04</td> <td>   23.167</td> <td> 0.000</td> <td> 2.23e+05</td> <td> 2.64e+05</td>
</tr>
<tr>
  <th>zipcode_(98177, 98178]</th> <td> 6.524e+04</td> <td> 1.04e+04</td> <td>    6.289</td> <td> 0.000</td> <td> 4.49e+04</td> <td> 8.56e+04</td>
</tr>
<tr>
  <th>zipcode_(98178, 98188]</th> <td> 3.848e+04</td> <td> 1.28e+04</td> <td>    3.013</td> <td> 0.003</td> <td> 1.34e+04</td> <td> 6.35e+04</td>
</tr>
<tr>
  <th>zipcode_(98188, 98198]</th> <td> 4.624e+04</td> <td> 1.03e+04</td> <td>    4.507</td> <td> 0.000</td> <td> 2.61e+04</td> <td> 6.64e+04</td>
</tr>
<tr>
  <th>zipcode_(98198, 98199]</th> <td> 3.892e+05</td> <td> 1.01e+04</td> <td>   38.514</td> <td> 0.000</td> <td> 3.69e+05</td> <td> 4.09e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>3616.615</td> <th>  Durbin-Watson:     </th> <td>   1.983</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>20574.704</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 1.126</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 8.508</td>  <th>  Cond. No.          </th> <td>1.06e+16</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 9.37e-27. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.



* ###### CONSTANT = -$270,000.00

* BEDROOMS = -$9,762.65

* BATHROOMS = $14,600.00

* SQFT_LIVING = $664,500.00

* SQFT_LOT = $121,300.00

* WATERFRONT = $0.0000000003174

* GRADE = $57,740.00

* SQFT_BASEMENT = -$67,720.00

* FLOOR DUMMY = -$9,794.43

* COND 2 = -$100,000.00

* COND 3 = -$83,380.00

* COND 4 = -$60,900.00

* COND 5 = -$25,760.00

* YR BUILT DUMMY = -$39,260.00

* ZIPCODE DUMMIES = A majority of them are positive. Only one will come into play when predicting any given home so I will not list them all out.

> Interestingly, the constant demonstrates a negative y-intercept for my model. In addition, a few of the features such as **bedrooms** and **floors dummy** actually _reduce_ the price of the home.

> The square footage features carry different influences on the price and seem to indicate living and lot square footage increases allowing space for the other features to subtract from. Additionally grade, although not much comparatively, contributes postiviely to price.

> Zipcodes can contribute as much as 841,200 or -27,500 to the home price depending on the zipcode it falls in

# Plots for presentation


```python
x1 = x_train['sqft_living'].copy()
x3 = x_train['grade'].copy()
x4 = x_train['sqft_lot'].copy()
x5 = x_train['bathrooms'].copy() 
x6 = x_train['bedrooms'].copy()
x7 = x_train['waterfront'].copy()
x8 = x_train['flrs_(0.5, 1.0]'].copy()
y1 = y_train.copy()
x2 = [x8.min(),x8.max()]
y2 = pd.Series(data=[y_hat_train1.min(), y_hat_train1.max()])
```


```python
y2
```




    0       6953.5
    1    1454560.0
    dtype: float64




```python
plt.scatter(x6, y1, label='Actual')
plt.plot(x2, y2, c='red', label='Predicted')
plt.xlabel('Bedrooms')
plt.ylabel('Price ($)')
plt.title('Bedrooms vs Price')
plt.grid()
plt.legend();
```


![png](output_235_0.png)



```python
plt.scatter(x5, y1, label='Actual')
plt.plot(x2, y2, c='red', label='Predicted')
plt.xlabel('Bathrooms')
plt.ylabel('Price ($)')
plt.title('Bathrooms vs Price')
plt.grid()
plt.legend();
```


![png](output_236_0.png)



```python
plt.scatter(x4, y1, label='Actual')
plt.plot(x2, y2, c='red', label='Predicted')
plt.xlabel('Total Square Feet Lot')
plt.ylabel('Price ($)')
plt.title('Sqft Lot vs Price')
plt.grid()
plt.legend();
```


![png](output_237_0.png)


> In this graph, 0 represents **~520 sqft** and 1 represents **~137,214 sqft**. 

> Each 0.2 tick represents **~876 sqft**.


```python
plt.scatter(x3, y1, label='Actual')
plt.plot(x2, y2, c='red', label='Predicted')
plt.xlabel('King Cty Grade')
plt.ylabel('Price ($)')
plt.title('Grade vs Price')
plt.grid()
plt.legend();
```


![png](output_239_0.png)



```python
plt.scatter(x1, y1, label='Actual')
plt.plot(x2, y2, c='red', label='Predicted')
plt.xlabel('Total Square Feet')
plt.ylabel('Price ($)')
plt.title('Sqft vs Price')
plt.grid()
plt.legend();
```


![png](output_240_0.png)


> In this graph, 0 represents **~370 sqft** and 1 represents **~4750 sqft**. 

> Each 0.2 tick represents **~876 sqft**.


```python
plt.scatter(kc_house_newidx2['waterfront'], kc_house_newidx2['price'], label='Actual')
plt.xlabel('Waterfront or not')
plt.ylabel('Price ($)')
plt.title('Waterfront vs Price')
plt.grid()
plt.legend(loc='upper center');
```


![png](output_242_0.png)



```python
plt.scatter(x8, y1, label='Actual')
plt.xlabel('Multi-level or not')
plt.ylabel('Price ($)')
plt.title('Multi-level vs Price')
plt.grid()
plt.legend();
```


![png](output_243_0.png)


# CONCLUSIONS & RECOMMENDATIONS

> Summarize your conclusions and bullet-point your list of recommendations, which are based on your modeling results.

* To answer my questions, my model indicates that all else held equal:

> A home with a waterfront view does not have a real effect on the price of a home. Not even a single penny.

> A multi-leveled home is almost ten-thousand dollars cheaper than a single-level. This is suprising but really isn't much proportionally since the average home price for this data was 492,726. This works out to approximately 2% of the mean house price.

> The top three positive contributors are (most to least):
* Sqft_living
* Sqft_lot
* Grade

> The top three negative contributors are (most to least):
* Condition level two
* Condition level three
* Sqft_basement


```python
sns.distplot(price_series);
plt.vlines(950000, ymin=0, ymax=0.0000025, label='95th %tile', colors='r')
plt.xlabel('Price')
plt.ylabel('Frequency');
plt.title('Distribution of Prices');
plt.grid()
plt.legend();
```


![png](output_247_0.png)



```python
price_series.quantile(q=.95)
```




    950000.0



### My Recommendations

* For someone looking to sell their home, they **definitely** need to ensure that the condition of the home is not subpar and do whatever to ensure King County gives the home a grade as high as possible. Each grade of improvement will raise the price by 57k and for condition the reduction can go from -100k down to -25k.


* I would also recommend them to maximize the amount of square footage within the home by **any means**. Otherwise, a home with a large lot will help (but is harder to control).


* If they want to add any rooms to the home, I would recommend adding a bathroom over a bedroom, as my model suggests (although small) a decrease dor each bedroom added to a home. In contrast, the bathrooms give a small boost. 


* I would not recommend when expanding a home to add a floor or expand the basement level (if one applies). In both cases my model shows a decrease for the overall price of a home. The basement feature demonstrates -67k as a maximum loss to a home's value.
