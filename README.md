![image](https://github.com/user-attachments/assets/e7b33845-e1d6-4eca-9291-c21d204c8618)![image](https://github.com/user-attachments/assets/ece5e5b4-46fe-4973-9d4c-6ff65a84092b)## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
Name: RAMYA R
Reg No: 212223230169
```

```
import pandas as pd
df = pd.read_csv("Encoding Data.csv")
df
```

![image](https://github.com/user-attachments/assets/475a52c6-b681-4229-a508-6a4de86418f2)

```
 from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
 pm=['Hot','Warm','Cold']
 e1=OrdinalEncoder(categories=[pm])
 e1.fit_transform(df[["ord_2"]])
```

![image](https://github.com/user-attachments/assets/ddb1f7a9-153a-4aea-8e84-b986471376af)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

![image](https://github.com/user-attachments/assets/a67dddc4-72ff-4f27-8f53-88c82b29a205)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

![image](https://github.com/user-attachments/assets/aaa538ee-12ad-43b5-80f0-cc6ffe3a814d)

```
# ONE HOT ENCODING
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```

```
df2=pd.concat([df2,enc],axis=1)
df2
```

![image](https://github.com/user-attachments/assets/11f997e1-93d3-4b62-8f62-62c9712cc51f)

```
pd.get_dummies(df2,columns=["nom_0"])
```

![image](https://github.com/user-attachments/assets/ea1c12e8-722f-4cd6-99b6-b3a87445983e)

```
pip install --upgrade category_encoders
```

![image](https://github.com/user-attachments/assets/8562b7d2-82e3-4e57-ba68-1a5c43c054fa)

```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```

![image](https://github.com/user-attachments/assets/245d72d9-ea19-4ae1-865e-f39a8fb2a81f)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
df
```

![image](https://github.com/user-attachments/assets/14dfa824-f337-49e7-8d2c-9a54e822cfdb)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

![image](https://github.com/user-attachments/assets/dbe26520-9a9f-4a82-9112-7751631078f6)

```
 import pandas as pd
 from scipy import stats
 import numpy as np
 df=pd.read_csv("Data_to_Transform.csv")
 df
```

![image](https://github.com/user-attachments/assets/03da2a0f-688c-4ab7-b726-b1bbcb90b8ed)

```
df.skew()
```

![image](https://github.com/user-attachments/assets/64b6129c-a312-41a6-907a-7df381f0ac52)

```
np.log(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/7df21295-cfa8-4fc3-8c40-2e3fac314e49)

```
 np.reciprocal(df["Moderate Positive Skew"])
```

![image](https://github.com/user-attachments/assets/c7cd77ac-a518-4ab9-a10b-b9a15113e6e5)

```
 np.sqrt(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/d9bc200b-59dc-4ab7-ab9b-14788a47d8ee)

```
#BOX - Cos
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![image](https://github.com/user-attachments/assets/3f5e2cfc-87a0-4eac-909b-8735edb5b7ea)

```
df.skew()
```

![image](https://github.com/user-attachments/assets/264306b6-45c8-49fa-91d0-a8cf6708811d)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

![image](https://github.com/user-attachments/assets/c61ea310-a704-4282-8f47-8f87f2446bf4)

```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal')
 df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 df
```

![image](https://github.com/user-attachments/assets/63860ec8-3c67-4cf1-a5ab-571be236b5b6)

```
 import seaborn as sns
 import statsmodels.api as sm
 import matplotlib.pyplot as plt
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()
```

![image](https://github.com/user-attachments/assets/48fd034f-0aa5-4a5e-a7b7-84a070335490)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/eb5def21-0711-4572-ba81-c5c442067b2a)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/046b8a9b-c919-4256-8d55-5e97fbfa5136)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/3d5f428d-2905-48e7-a4cc-049682f62a52)

```
dt=pd.read_csv("titanic_dataset.csv")
dt
```

![image](https://github.com/user-attachments/assets/4268c045-6fde-41f0-95ab-4b3f29c08263)

```
from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```

![image](https://github.com/user-attachments/assets/9d74ef94-ef1d-4394-b3d9-0a0bea00aa60)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/3381b60d-788a-4c93-9112-886977683f70)

# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       
