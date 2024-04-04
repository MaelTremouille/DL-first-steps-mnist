<style>
body {
    font-family: 'Times New Roman', Times, serif;
    margin: 40px;
    line-height: 1.6;
    color: #333;
}
h1, h2, h3 {
    color: #354458;
}
</style>
# <u>Creating, Reading and Writing</u>
## <p style='color:blue'>1. Creating a DataFrame</p> 

```python
import pandas as pd
pd.DataFrame({'col_A':[1,2],
              'col_B':[3,4]})
```
We can precise indexes this way:
```python
import pandas as pd
pd.DataFrame({'col_A':[1,2],
              'col_B':[3,4]},
              index = ['idx_1','idx_2'])
```

## <p style='color:blue'>2. Series</p> 
It's basically a single column DataFrame

```python
pd.Series([30, 35, 40], 
           index=['2015 Sales', '2016 Sales', '2017 Sales'], 
           name='Product A')
```

## <p style='color:blue'>3. Reading data files</p> 

```python
data = pd.read_csv("data.csv", index_col=0)
```
We can easily get the size of a DataFrame in the format (row, columns) (it's an attribute)

```python
data.shape
```

# <u>Indexing, Selecting & Assigning</u>
## <p style='color:blue'>1. Native accessors</p> 
We can access a column easily:

```python
data.col_name
data["col_name"]
# we can also get a precise datum:
data["col_name"][idx]
```

## <p style='color:blue'>2. Indexing in pandas</p> 
Whe can access to the data with iloc:
```python
data.iloc[[rows_idx], [columns_idx]]
```

Whe can also access to the data with loc:
```python
data.iloc[[rows_idx], [columns_names]]
```
iloc: 0:10 will select entries 0,...,9
loc: 0:10 will select entries 0,...,10


## <p style='color:blue'>3. Manipulating the index</p>
```python
data.set_index("column_name")
```

## <p style='color:blue'>4. Conditional selection</p>
We have to use loc:
```python
data.loc[bool_to_select_rows]
```

# <u>Summary Functions and Maps</u>
## <p style='color:blue'>1. Summary functions</p>
```python
data.col_name.describe() # Overall description
data.col_name.mean() # Get the mean
data.col_name.unique() # So that to get the unique values
```

## <p style='color:blue'>2. Maps</p>
An example of use to remean a column

```python
# using map()
data_col_name_mean = data.col_name.mean()
data.col_name.map(lambda p: p - data_col_name_mean)

# using apply()
def remean_col_name(row):
    row.col_name = row.col_name - data_col_name_mean
    return row

data.apply(remean_col_name, axis='columns')
```
Note that map() and apply() return new, transformed Series and DataFrames, respectively. They don't modify the original data they're called on.

# <u>Grouping and Sorting</u>
## <p style='color:blue'>1. Groupwise analysis</p>
```python
data.groupby('col_name1').col_name2.count()
data.groupby(['col_name1']).col_name2.agg([len, min, max])
```