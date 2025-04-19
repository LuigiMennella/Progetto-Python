```python
# Importiamo le librerie occorrenti
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

from sklearn.compose import make_column_transformer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
```


```python
#Importiamo il db
titanic=sns.load_dataset("titanic")
```


```python
#Diamo un'occhiata alle variabili. Stabiliamo come target la variabile survived.
#Le altre saranno le nostre features
titanic.head()
```



```python
#Verifichiamo per alcune variabili meno intuitive come si esprimono (es.ponte della nave)
data = titanic.groupby(by=["deck"],
                as_index=False, 
                dropna=False) \
       .size()
data
```

    


```python
#Verifichiamo la ridondanza tra alcune variabili
#Innanzitutto c'è ridondanza della variabile "alive" rispetto alla terget, "survived"
titanic.groupby(by=["survived","alive"],
                as_index=False, 
                dropna=False) \
       .agg(conteggio   = ("survived",
                           np.size)
            )
```






```python
#Ridondanza anche tra class e pclass
pd.crosstab(titanic["class"], titanic["pclass"])
```





```python
#Ridondanza tra embraked e embrark_town
titanic.groupby(by=["embarked","embark_town"],
                as_index=False, 
                dropna=False) \
       .agg(conteggio   = ("survived",
                           np.size)
            )
```



```python
#Correlazione tra sex e who
titanic.groupby(by=["sex","who"],
                as_index=False, 
                dropna=False) \
       .agg(conteggio   = ("sex",
                           np.size)
            )
```

```python
#Correlazione tra adult_male e who
titanic.groupby(by=["adult_male","who"],
                as_index=False, 
                dropna=False) \
       .agg(conteggio   = ("adult_male",
                           np.size)
            )
```



```python
#Eseguiamo anche un grafico a ulteriore supporto delle analisi
g = sns.catplot (data=titanic, x="who", hue="sex", kind="count")
```


    
![png](output_9_0.png)
    



```python
#Correlazione tra alone e sibsp
titanic.groupby(by=["alone","sibsp"],
                as_index=False, 
                dropna=False) \
       .agg(conteggio   = ("survived",
                           np.size)
            )
```






```python
#Ridondanza tra alone e parch
titanic.groupby(by=["alone","parch"],
                as_index=False, 
                dropna=False) \
       .agg(conteggio   = ("survived",
                           np.size)
            )
```




```python
#Verifichiamo le correlazioni tra le variabili numeriche
titanic.corr(numeric_only=True)
```




```python
g = sns.heatmap(data = titanic.corr(numeric_only=True), cmap='RdBu', vmin=-1, vmax=1, annot=True, fmt=".1f", linewidth=.5)
```


    
![png](output_13_0.png)
    



```python
g = sns.heatmap(data = titanic.corr(numeric_only=True, method="spearman"), cmap='RdBu', vmin=-1, vmax=1, annot=True, fmt=".1f", linewidth=.5)
#è sempre consigliabile usare anche spearman perché considera anche correlazioni non lineari
#in questo caso non ci sono grandi differenze
```


    
![png](output_14_0.png)
    



```python
g=sns.pairplot(data = titanic [["alone", "sibsp", "parch"]])
```


    
![png](output_15_0.png)
    



```python
# Il comando info ci permette di valutare le tipologie di variabili e il conteggio dei nulli
titanic.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 15 columns):
     #   Column       Non-Null Count  Dtype   
    ---  ------       --------------  -----   
     0   survived     891 non-null    int64   
     1   pclass       891 non-null    int64   
     2   sex          891 non-null    object  
     3   age          714 non-null    float64 
     4   sibsp        891 non-null    int64   
     5   parch        891 non-null    int64   
     6   fare         891 non-null    float64 
     7   embarked     889 non-null    object  
     8   class        891 non-null    category
     9   who          891 non-null    object  
     10  adult_male   891 non-null    bool    
     11  deck         203 non-null    category
     12  embark_town  889 non-null    object  
     13  alive        891 non-null    object  
     14  alone        891 non-null    bool    
    dtypes: bool(2), category(2), float64(2), int64(4), object(5)
    memory usage: 80.7+ KB
    


```python
#Procediamo all'eliminazione delle features ridondanti o carenti
titanic = titanic.drop(["alive","sex","adult_male","pclass","embark_town","alone","deck"], axis=1) 


```


```python
#Verifichiamo il nuovo db titanic
titanic.head()
```




```python
# Tra la variabili rimaste, solo due, embarked e age, presentano dei nulli.
# Procediamo ad analizzarle per verificare il modo sistema migliore per sostituire i nulli
data_embarked = titanic.groupby(by=["embarked"],
                as_index=False, 
                dropna=False) \
       .size()
```


```python
data_embarked
```






```python
#Informazioni sulla statistica monovariata
titanic.describe(include='all')
```




```python
titanic["age"].describe()
```




    count    714.000000
    mean      29.699118
    std       14.526497
    min        0.420000
    25%       20.125000
    50%       28.000000
    75%       38.000000
    max       80.000000
    Name: age, dtype: float64




```python
g = titanic.boxplot(column = ["age"], figsize = (8,6))
```


    
![png](output_23_0.png)
    


Da quanto osservato sia per embarked che per age è preferibile intervenire sostituendo i nulli, 
con la moda nel primo caso e con la mediana nel secondo.
Infatti, embarked, che presenta solo due nulli, è una variabile qualitativa che può assumere tre categorie, delle quali "S" è nettamente prevalente.
Age, invece, è una variabile quantitativa che presenta alcuni outlier nei valori più che influenzano leggermente la distribuzione.
Perciò, in questo caso si ritiene preferibile l'uso della mediana invece della media.


```python
#Trasformiamo le variabili qualitative in quantitative
#quasi sempre gli algortimi richiedono soltanto numeri
# In realtà si può anche effettuarlo direttamente nel pre-processing 
titanic = pd.get_dummies(titanic) 
```


```python
titanic
```







```python
#Definiamo training e test
# non necessario usare label encorder perché y è già binomiale
X = titanic.drop("survived",axis=1).values
Xcolumns = titanic.drop("survived",axis=1).columns
y = titanic["survived"].values
```


```python
# creiamo i db di training e test
x1,x2,y1,y2 = train_test_split(X,
                               y,
                               test_size = 0.3,
                               shuffle = True,
                               random_state = 1)
```


```python
Xcolumns
```




    Index(['age', 'sibsp', 'parch', 'fare', 'embarked', 'class', 'who'], dtype='object')




```python
x1
```




    array([[17.0, 0, 0, ..., 'C', 'Third', 'woman'],
           [28.0, 1, 0, ..., 'C', 'Second', 'woman'],
           [nan, 0, 0, ..., 'S', 'Third', 'man'],
           ...,
           [21.0, 0, 0, ..., 'S', 'Second', 'man'],
           [nan, 0, 0, ..., 'S', 'Third', 'woman'],
           [21.0, 0, 0, ..., 'S', 'Third', 'man']], dtype=object)




```python
#effettuiamo una copia prima del preprocessing
x1_orig = x1.copy()
x2_orig = x2.copy()
```


```python
#valorizziamo i null con varie strategie
#in questo caso anziché pandas usiamo sklearn
#importante valorizzare il remainder, altrimenti le variabili non precisate vengono eliminate
#bisogna indicare la riga della variabile, si conta partendo da zero
#l'ordinne indicato nel comando sarà poi quello di output
#Si considerano tutte le variaibli anche quelle che al momento non hanno nulli (non sarebbe necessario in questo caso specifico)
null_handling = make_column_transformer(
        (SimpleImputer(strategy='median'), [0,1,2,3]),
        (SimpleImputer(strategy='most_frequent'), [4,5,6]),    
        remainder='passthrough')
```


```python
#si allena su x1 e si applica a x1 e x2
null_handling.fit(x1)
x1 = null_handling.transform(x1)
x2 = null_handling.transform(x2)
```


```python
Standardizziamo le features quantitative, per avere dimensioni confrontabii, in questo caso usiamo per tutte lo standard scaler,
ma volendo è possibile differneziare

Rappresentiamo le qualitative come numeriche, con OrdinalEncoder per le ordinabili e 
OneHotEncoder per le non ordinabili
```


```python
preprocessing = make_column_transformer(
        (StandardScaler(), [0,1,2,3]),
        (OneHotEncoder(handle_unknown='ignore',drop = 'first'), [4,6]), 
        (OrdinalEncoder(categories=[["First","Second","Third"]],handle_unknown="use_encoded_value",unknown_value=4),[5]),       
        remainder='passthrough')


```


```python
preprocessing.fit(x1)
x1 = preprocessing.transform(x1)
x2 = preprocessing.transform(x2)
```


```python
#osserviamo le dimensioni pre e post
#il numero di colonne aumenta a causa della duplicazione delle colonne qualitative "embarked" e "who"
#per entrambi le categorie sono 3, per cui le nuove colonne sono due (n-1), in quanto rappresentare anche l'altra sarebbe ridondante
x1_orig.shape
```




    (623, 7)




```python
x1.shape
```




    (623, 9)




```python
#Variabile age
x1_orig[0:20:,0]
```




    array([17.0, 28.0, nan, 20.0, nan, 47.0, 36.0, 16.0, 5.0, 25.0, 39.0,
           18.0, 40.0, 24.0, 18.0, 19.0, 17.0, 30.0, 24.0, nan], dtype=object)




```python
x1[0:20]
```




    array([[-0.96517045, -0.47149154, -0.4764597 , -0.35793161,  0.        ,
             0.        ,  0.        ,  1.        ,  2.        ],
           [-0.13643897,  0.4915879 , -0.4764597 , -0.1527973 ,  0.        ,
             0.        ,  0.        ,  1.        ,  1.        ],
           [-0.07993455, -0.47149154, -0.4764597 , -0.49901694,  0.        ,
             1.        ,  1.        ,  0.        ,  2.        ],
           [-0.73915277, -0.47149154, -0.4764597 , -0.45709444,  0.        ,
             1.        ,  1.        ,  0.        ,  2.        ],
           [-0.07993455, -0.47149154, -0.4764597 , -0.66876651,  0.        ,
             1.        ,  1.        ,  0.        ,  1.        ],
           [ 1.2950063 ,  0.4915879 , -0.4764597 , -0.35703511,  0.        ,
             1.        ,  0.        ,  1.        ,  2.        ],
           [ 0.46627483,  0.4915879 , -0.4764597 , -0.33446146,  0.        ,
             1.        ,  1.        ,  0.        ,  2.        ],
           [-1.04050967, -0.47149154, -0.4764597 , -0.10979987,  0.        ,
             1.        ,  1.        ,  0.        ,  1.        ],
           [-1.86924115,  0.4915879 ,  2.07147502, -0.07217711,  0.        ,
             1.        ,  0.        ,  0.        ,  1.        ],
           [-0.36245665, -0.47149154, -0.4764597 , -0.50161398,  0.        ,
             1.        ,  0.        ,  1.        ,  2.        ],
           [ 0.6922925 ,  0.4915879 ,  0.79750766,  1.04360629,  0.        ,
             1.        ,  0.        ,  1.        ,  0.        ],
           [-0.88983122, -0.47149154, -0.4764597 , -0.50215145,  0.        ,
             1.        ,  1.        ,  0.        ,  2.        ],
           [ 0.76763172,  0.4915879 , -0.4764597 , -0.46506616,  0.        ,
             1.        ,  0.        ,  1.        ,  2.        ],
           [-0.43779587, -0.47149154, -0.4764597 , -0.49570184,  0.        ,
             1.        ,  1.        ,  0.        ,  2.        ],
           [-0.88983122, -0.47149154, -0.4764597 , -0.45718258,  0.        ,
             1.        ,  0.        ,  1.        ,  2.        ],
           [-0.814492  , -0.47149154, -0.4764597 , -0.49570184,  0.        ,
             1.        ,  1.        ,  0.        ,  2.        ],
           [-0.96517045,  3.3808262 ,  2.07147502, -0.49838917,  0.        ,
             1.        ,  0.        ,  1.        ,  2.        ],
           [ 0.01423948, -0.47149154, -0.4764597 , -0.4645287 ,  0.        ,
             1.        ,  1.        ,  0.        ,  2.        ],
           [-0.43779587, -0.47149154, -0.4764597 ,  1.03393187,  0.        ,
             0.        ,  1.        ,  0.        ,  0.        ],
           [-0.07993455, -0.47149154, -0.4764597 , -0.51343828,  0.        ,
             0.        ,  1.        ,  0.        ,  2.        ]])




```python
#Variabile who
x1_orig[0:20:,6]
```




    array(['woman', 'woman', 'man', 'man', 'man', 'woman', 'man', 'man',
           'child', 'woman', 'woman', 'man', 'woman', 'man', 'woman', 'man',
           'woman', 'man', 'man', 'man'], dtype=object)




```python
x1[0:20:,6:8]
#woman = 0,1
#man = 1,0
#child=1,1
```




    array([[0., 1.],
           [0., 1.],
           [1., 0.],
           [1., 0.],
           [1., 0.],
           [0., 1.],
           [1., 0.],
           [1., 0.],
           [0., 0.],
           [0., 1.],
           [0., 1.],
           [1., 0.],
           [0., 1.],
           [1., 0.],
           [0., 1.],
           [1., 0.],
           [0., 1.],
           [1., 0.],
           [1., 0.],
           [1., 0.]])




```python
#Variabile embarked
x1_orig[0:20:,4]
```




    array(['C', 'C', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S',
           'S', 'S', 'S', 'S', 'S', 'C', 'C'], dtype=object)




```python
x1[0:20:,4:6]
```




    array([[0., 0.],
           [0., 0.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 0.],
           [0., 0.]])




```python
#Variabile pclass
x1_orig[0:20:,5]
```




    array(['Third', 'Second', 'Third', 'Third', 'Second', 'Third', 'Third',
           'Second', 'Second', 'Third', 'First', 'Third', 'Third', 'Third',
           'Third', 'Third', 'Third', 'Third', 'First', 'Third'], dtype=object)




```python
x1[0:20:,8]
```




    array([2., 1., 2., 2., 1., 2., 2., 1., 1., 2., 0., 2., 2., 2., 2., 2., 2.,
           2., 0., 2.])




```python
#testiamo gli algoritmi
lr = LogisticRegression(random_state=0)

lr.fit(x1,y1)

lr.score(x2,y2)
```




    0.7910447761194029




```python
knn = KNeighborsClassifier()
knn.fit(x1,y1)

knn.score(x2,y2)
```




    0.7611940298507462




```python
sgd = SGDClassifier(random_state=0)

sgd.fit(x1,y1)

sgd.score(x2,y2)
```




    0.7201492537313433




```python
#Mettiamo insiemi le esecuzioni dei singoli step in una pipeline
#composta dai due processi appena descritti più un metodo prescelto (es.sdg)
pipeline = Pipeline([('nh',null_handling),
                              ('pr',preprocessing),
                              ('lr', LogisticRegression(random_state=0))
                               ])
```


```python
#Non bisogna basarsi solo sull'accuratezza, sia perché può dipendere dagli effetti randomici
# ma non in questo caso, ma perché può dipendere anche dal campione scelto
#per effettuare la scelta migliore bisogna vedere la tecnica della convalida incrociata
#che confronta un maggior numero di risultati ottenuti dai vdiversi metodi.
#Scegliamo comunque la regressione logistica
```


```python
la prossima istruzione pipeline.fit(x1,y1) è equivalente a lanciare questa sequenza di cinque istruzioni

valorizza_null.fit(x1)

x1 = valorizza_null.transform(x1)

preprocessing.fit(x1)

x1 = preprocessing.transform(x1)

LogisticRegression.fit(x1,y1)

Comunque dopo il fit della pipeline, x1 conterrà ancora i dati originali
```


```python
pipeline.fit(x1,y1)
```


    ---------------------------------------------------------------------------



```python
#Reimportiamo i dati, cancelliamo le colonne e dividiamo in training e test
#Attenzione! Ovviamente immportare per prima cosa le librerie

#caricamento del db, eliminazione variabili non necessarie o ridondanti
titanic=sns.load_dataset("titanic")
titanic = titanic.drop(["alive"],axis=1)
titanic = titanic.drop(["pclass","embark_town","deck","adult_male","sex","alone"],axis=1)

#definizione delle features e della variaible dipendente
X = titanic.drop("survived",axis=1).values
Xcolumns = titanic.drop("survived",axis=1).columns
y = titanic["survived"].values

# suddivisione in training e test, col comando train_test_split di sklearn
x1,x2,y1,y2 = train_test_split(X,
                               y,
                               test_size = 0.3,
                               shuffle = True,
                               random_state = 1)
```


```python
Xcolumns
```




    Index(['age', 'sibsp', 'parch', 'fare', 'embarked', 'class', 'who'], dtype='object')




```python
titanic.head()
```






```python
#Ridefiniamo i passi per valorizzare i null e trasformare le variabili
```


```python
x1_orig = x1.copy()
x2_orig = x2.copy()

valorizza_null = make_column_transformer(
        (SimpleImputer(strategy='median'), [0,1,2,3]),
        (SimpleImputer(strategy='most_frequent'), [4,5,6]),    
        remainder='passthrough')

preprocessing = make_column_transformer(
        (StandardScaler(), [0,1,2,3]),
        (OrdinalEncoder(categories=[["First","Second","Third"]],handle_unknown="use_encoded_value",unknown_value=4),[5]),
        (OneHotEncoder(handle_unknown='ignore',drop = 'first'), [4,6]),    
        remainder='passthrough')
#sostituzione dei valori nulli e trasformazione delle varabili
```


```python
#Mettiamo insiemi le esecuzioni dei singoli step in una pipeline
#composta dai due processi appena descritti più un metodo prescelto (es.sdg)
pipeline = Pipeline([('vn',valorizza_null),
                              ('pr',preprocessing),
                              ('lr', LogisticRegression(random_state=0))
                               ])
```


```python
pipeline.fit(x1,y1)
```






```python
#calcoliamo l'accuratezza del processo sui dati di test
pipeline.score(x2,y2)
```




    0.7910447761194029



Per valutare se questo modello è migliore del precedente, occorre valutarlo con il metodo della cross validation. Questo medoto coinvolgerà solo i dati di training e prevede di:

dividere i dati di training in 10 parti

eseguire 10 volte algoritmo scegliendo come test ogni volta una delle 10 parti diverse

fare una media delle 10 accuratezze ottenute


```python
x1,x2,y1,y2 = train_test_split(X,
                               y,
                               test_size = 0.3,
                               shuffle = True,
                               random_state = 1)
scores = cross_val_score(estimator = pipeline,
                         X = x1,
                         y = y1,
                         cv = 10)
scores
```




    array([0.79365079, 0.82539683, 0.84126984, 0.82258065, 0.88709677,
           0.85483871, 0.87096774, 0.87096774, 0.82258065, 0.77419355])




```python
np.mean(scores)
```




    np.float64(0.8363543266769072)




```python
#Ripetiamo la cross validation sulla pipeline basilare
titanic = sns.load_dataset("titanic")
titanic = titanic.drop(["alive"],axis=1)

titanic = pd.get_dummies(titanic) 

X = titanic.drop("survived",axis=1).values
y = titanic["survived"].values

basic_pipeline = Pipeline([('si',SimpleImputer()),
                            ('lr', LogisticRegression(random_state=0))
                            ])
```


```python
x1,x2,y1,y2 = train_test_split(X,
                               y,
                               test_size = 0.3,
                               shuffle = True,
                               random_state = 1)
scores = cross_val_score(estimator = basic_pipeline,
                         X = x1,
                         y = y1,
                         cv = 10)
scores
```

    array([0.73015873, 0.82539683, 0.85714286, 0.79032258, 0.90322581,
           0.87096774, 0.83870968, 0.88709677, 0.87096774, 0.75806452])




```python
np.mean(scores)
```




    np.float64(0.8332053251408089)




```python
#stavolta la media dello score è più basso, anche se di poco, per cui giustica i passaggi del preprocessing
#tuttavia occorre ugualmente testare sempre i metodi anche sui dati di test
#per evitare errori, in paticolare legati all'overfitting di alcuni metodi (alberi decisionali)
```

Solo a questo punto scelgo la prima implementazione, rilancio il metodo fit sull'intero set di training e valuto l'accuratezza sul set di test che non avevo utilizzato durante la convalida. In questo modo posso valutare se l'algoritmo lavora bene anche su nuovi dati o se ha problemi di overfitting


```python
#caricamento del db, eliminazione variabili non necessarie o ridondanti
titanic=sns.load_dataset("titanic")
titanic = titanic.drop(["alive"],axis=1)
titanic = titanic.drop(["pclass","embark_town","deck","adult_male","sex","alone"],axis=1)

X = titanic.drop("survived",axis=1).values
y = titanic["survived"].values

x1,x2,y1,y2 = train_test_split(X,
                               y,
                               test_size = 0.3,
                               shuffle = True,
                               random_state = 1)

pipeline.fit(x1,y1)
pipeline.score(x2,y2)
```




    0.7910447761194029



L'adattamento ai dati nuovi è buono ma comunque più basso rispetto al training


```python
#Sperimento una nuova pipeline, sostituendo la regressione logistica con lo sgd. 
# I risultati confermano la maggiore bontà del primo metodo.
pipeline2 = Pipeline([('vn',valorizza_null),
                              ('pr',preprocessing),
                              ('sgd', SGDClassifier(random_state=0))
                               ])
```


```python
pipeline2.fit(x1,y1)
```





```python
#calcoliamo l'accuratezza del processo sui dati di test
pipeline2.score(x2,y2)
```




    0.7201492537313433




```python
x1,x2,y1,y2 = train_test_split(X,
                               y,
                               test_size = 0.3,
                               shuffle = True,
                               random_state = 1)
scores = cross_val_score(estimator = pipeline2,
                         X = x1,
                         y = y1,
                         cv = 10)
scores
```




    array([0.80952381, 0.73015873, 0.88888889, 0.80645161, 0.72580645,
           0.79032258, 0.83870968, 0.82258065, 0.82258065, 0.72580645])




```python
np.mean(scores)
```




    np.float64(0.7960829493087557)




```python
#le mie pipeline hanno dei parametri per i quali è possibile verificare le migliori opzioni.
#In realtà andrebbe fatto prima di definire le pipeline. Verifichiamo comunque se la scelata di sostituire i nulli delle variabili quantitative
#con la mediana è stata corretta
pipeline.get_params()
```




    {'memory': None,
     'steps': [('vn', ColumnTransformer(remainder='passthrough',
                         transformers=[('simpleimputer-1',
                                        SimpleImputer(strategy='median'),
                                        [0, 1, 2, 3]),
                                       ('simpleimputer-2',
                                        SimpleImputer(strategy='most_frequent'),
                                        [4, 5, 6])])),
      ('pr',
       ColumnTransformer(remainder='passthrough',
                         transformers=[('standardscaler', StandardScaler(),
                                        [0, 1, 2, 3]),
                                       ('ordinalencoder',
                                        OrdinalEncoder(categories=[['First', 'Second',
                                                                    'Third']],
                                                       handle_unknown='use_encoded_value',
                                                       unknown_value=4),
                                        [5]),
                                       ('onehotencoder',
                                        OneHotEncoder(drop='first',
                                                      handle_unknown='ignore'),
                                        [4, 6])])),
      ('lr', LogisticRegression(random_state=0))],
     'verbose': False,
     'vn': ColumnTransformer(remainder='passthrough',
                       transformers=[('simpleimputer-1',
                                      SimpleImputer(strategy='median'),
                                      [0, 1, 2, 3]),
                                     ('simpleimputer-2',
                                      SimpleImputer(strategy='most_frequent'),
                                      [4, 5, 6])]),
     'pr': ColumnTransformer(remainder='passthrough',
                       transformers=[('standardscaler', StandardScaler(),
                                      [0, 1, 2, 3]),
                                     ('ordinalencoder',
                                      OrdinalEncoder(categories=[['First', 'Second',
                                                                  'Third']],
                                                     handle_unknown='use_encoded_value',
                                                     unknown_value=4),
                                      [5]),
                                     ('onehotencoder',
                                      OneHotEncoder(drop='first',
                                                    handle_unknown='ignore'),
                                      [4, 6])]),
     'lr': LogisticRegression(random_state=0),
     'vn__force_int_remainder_cols': True,
     'vn__n_jobs': None,
     'vn__remainder': 'passthrough',
     'vn__sparse_threshold': 0.3,
     'vn__transformer_weights': None,
     'vn__transformers': [('simpleimputer-1',
       SimpleImputer(strategy='median'),
       [0, 1, 2, 3]),
      ('simpleimputer-2', SimpleImputer(strategy='most_frequent'), [4, 5, 6])],
     'vn__verbose': False,
     'vn__verbose_feature_names_out': True,
     'vn__simpleimputer-1': SimpleImputer(strategy='median'),
     'vn__simpleimputer-2': SimpleImputer(strategy='most_frequent'),
     'vn__simpleimputer-1__add_indicator': False,
     'vn__simpleimputer-1__copy': True,
     'vn__simpleimputer-1__fill_value': None,
     'vn__simpleimputer-1__keep_empty_features': False,
     'vn__simpleimputer-1__missing_values': nan,
     'vn__simpleimputer-1__strategy': 'median',
     'vn__simpleimputer-2__add_indicator': False,
     'vn__simpleimputer-2__copy': True,
     'vn__simpleimputer-2__fill_value': None,
     'vn__simpleimputer-2__keep_empty_features': False,
     'vn__simpleimputer-2__missing_values': nan,
     'vn__simpleimputer-2__strategy': 'most_frequent',
     'pr__force_int_remainder_cols': True,
     'pr__n_jobs': None,
     'pr__remainder': 'passthrough',
     'pr__sparse_threshold': 0.3,
     'pr__transformer_weights': None,
     'pr__transformers': [('standardscaler', StandardScaler(), [0, 1, 2, 3]),
      ('ordinalencoder',
       OrdinalEncoder(categories=[['First', 'Second', 'Third']],
                      handle_unknown='use_encoded_value', unknown_value=4),
       [5]),
      ('onehotencoder',
       OneHotEncoder(drop='first', handle_unknown='ignore'),
       [4, 6])],
     'pr__verbose': False,
     'pr__verbose_feature_names_out': True,
     'pr__standardscaler': StandardScaler(),
     'pr__ordinalencoder': OrdinalEncoder(categories=[['First', 'Second', 'Third']],
                    handle_unknown='use_encoded_value', unknown_value=4),
     'pr__onehotencoder': OneHotEncoder(drop='first', handle_unknown='ignore'),
     'pr__standardscaler__copy': True,
     'pr__standardscaler__with_mean': True,
     'pr__standardscaler__with_std': True,
     'pr__ordinalencoder__categories': [['First', 'Second', 'Third']],
     'pr__ordinalencoder__dtype': numpy.float64,
     'pr__ordinalencoder__encoded_missing_value': nan,
     'pr__ordinalencoder__handle_unknown': 'use_encoded_value',
     'pr__ordinalencoder__max_categories': None,
     'pr__ordinalencoder__min_frequency': None,
     'pr__ordinalencoder__unknown_value': 4,
     'pr__onehotencoder__categories': 'auto',
     'pr__onehotencoder__drop': 'first',
     'pr__onehotencoder__dtype': numpy.float64,
     'pr__onehotencoder__feature_name_combiner': 'concat',
     'pr__onehotencoder__handle_unknown': 'ignore',
     'pr__onehotencoder__max_categories': None,
     'pr__onehotencoder__min_frequency': None,
     'pr__onehotencoder__sparse_output': True,
     'lr__C': 1.0,
     'lr__class_weight': None,
     'lr__dual': False,
     'lr__fit_intercept': True,
     'lr__intercept_scaling': 1,
     'lr__l1_ratio': None,
     'lr__max_iter': 100,
     'lr__multi_class': 'deprecated',
     'lr__n_jobs': None,
     'lr__penalty': 'l2',
     'lr__random_state': 0,
     'lr__solver': 'lbfgs',
     'lr__tol': 0.0001,
     'lr__verbose': 0,
     'lr__warm_start': False}




```python
param_grid = [{'vn__simpleimputer-1__strategy':['median','mean']
              }]
```


```python
my_grid = GridSearchCV(estimator = pipeline,
                  param_grid = param_grid,
                  cv = 10)
```

Tramite GridSearchCV creo la griglia con tutte le possibili combinazioni di parametri, che saranno valutati con una convalida incrociata con 10 suddivisioni, per un totale di 20 esecuzioni Siamo sempre all'interno di sklearn


```python
my_grid.fit(x1,y1)
```


    


```python
#Verifichiamo il punteggio migliore
my_grid.best_score_
```




    np.float64(0.8363543266769072)




```python
my_grid.best_params_
#in effetti i risultaati usati erano i migliori
```




    {'vn__simpleimputer-1__strategy': 'median'}




```python
my_grid.best_estimator_
```





```python
my_grid.score(x2,y2)
#equivalene a creare come pipeline la migliore possibile
```




    0.7910447761194029




```python
best = my_grid.best_estimator_
best.fit(x1,y1)
best.score(x2,y2)
```




    0.7910447761194029




```python
#Un ulteriore elemento di valutazione è la matrice di confusione
from sklearn.metrics import confusion_matrix
```


```python
y_true = y2
y_pred = best.predict(x2)
confusion_matrix(y_true, y_pred)
#sulla diagonale abbiamo le previsioni corrette (ovvero 0;0, 1;1), mentre 0;1 e 1;0 rappresentano gli errori.
#l'accuratezza potrebbe essere stimata sommando la diagonale sul totale delle previsioni
#in questo caso sarebbe circa 0,795, ovvero 212/268, (133+79)/(133+79+20+36), ovvero l'adattamenento
#va detto che non necessariamente i due tipi di errori hanno uguale peso
#ad esempio, in ambito medico, un falso negativo è più grave di un falso positivo
```




    array([[133,  20],
           [ 36,  79]])




```python
m=(133+79)/(133+79+20+36)
```


```python
m
```




    0.7910447761194029




```python
n=y_true-y_pred
```


```python
n
```




    array([ 0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,
            0,  0,  0,  0,  0,  0,  0, -1,  1,  0,  0,  0,  0, -1,  1,  0, -1,
            0,  1,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  1,  0,  0,  0,  0, -1,  0,  0,  0,  1,  1,  0,  1,  1,
           -1,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0, -1,  0,
            1, -1,  0,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  1,  0,
            0,  0,  0,  0, -1,  0,  0, -1,  1, -1,  0,  0, -1,  0,  0,  0,  0,
            0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            1,  0,  0,  0,  1,  0,  0,  0,  0, -1,  1,  0,  0,  0,  0,  0,  1,
            0,  1,  0,  0,  1,  0,  0, -1,  1,  0,  0,  0,  0,  0, -1,  0, -1,
            0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,
           -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0])




```python
#Proviamo a ricostruire un df di confronto tra realtà e previsioni sul campione di test
#tradfrormiamo le previsioni in df
dfp = pd.DataFrame(data = y_pred, columns = ["prediction"])
dfp
```




```python
#Trasforminano in df anche i risultati reali
dfy=pd.DataFrame(y2, columns = ["true"])
dfy
```






```python
#Infine treaformiamo in df anche le features originali
dfx=pd.DataFrame(x2_orig, columns = [['age', 'sibsp', 'parch', 'fare', 'embarked', 'class', 'who']])
dfx
```





```python
Xcolumns
```




    Index(['age', 'sibsp', 'parch', 'fare', 'embarked', 'class', 'who'], dtype='object')




```python
#A questo punto costituiiamo un nuovo df, concatenando quanto prima ottenuto
dft = pd.concat([dfx,dfy,dfp],axis=1)
```


```python
dft
```




```python
dft[dft["true"] != dft["prediction"] ]
#infine mostriamo le righe dove la predizione è errata
```



```
