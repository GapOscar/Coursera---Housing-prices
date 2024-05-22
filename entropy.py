# Tratamiento de datos
# ==============================================================================
import warnings, pandas as pd, numpy as np
warnings.filterwarnings('ignore')

# ==============================================================================
# Generación de un archivo con datos simulados
np.random.seed(123)
df = pd.DataFrame({'num_edad': np.random.randint(20, 60, 100), 'obj': np.random.randint(0, 2, 100)})

# ==============================================================================
# definición propia de entropía
def f_entropy(df, var, obj):     # var: variable para particionar el df, obj: variable objetivo
        n = df.shape[0]
        df0 = df.copy()
        df0['obj0'] = df0[obj].apply(lambda x: 1 if x == 0 else 0)
        df0 = df0.groupby(var).agg(obj1=(obj, 'sum'), obj0=('obj0', 'sum'), total=(var, 'count'),).reset_index()
        df0['d'] = df0['total']/n
        df0['p'] = df0['obj1']/df0['total']
        df0['h'] = np.where((df0.p == 0) | (df0.p == 1), 0, -df0.d*(df0.p*np.log(df0.p) + (1-df0.p)*np.log(1-df0.p)))
        return df0.h.sum()

print("La entropía del conjunto es: ", f_entropy(df, 'num_edad', 'obj'))
# ==============================================================================
