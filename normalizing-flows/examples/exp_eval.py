# %%
import numpy as np
import pandas as pd
from scipy.stats import t
import torch
import numpy as np
from matplotlib import gridspec
import copy
import normflows as nf

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import copy
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import gc
import sys

from normflows.experiments.flowslib import planar, radial, nice, rnvp, nsp, iaf, residual

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
from skbio import DistanceMatrix
from skbio.stats.distance import permanova
import gower
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ks_2samp, mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pandas as pd
import pandas as pd
import numpy as np
import gower
from skbio.stats.distance import permanova
from skbio import DistanceMatrix
from sklearn.ensemble import GradientBoostingClassifier
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from scipy import stats
import numpy as np
import scipy
import torch
import torch
import sdmetrics

from sklearn.metrics import f1_score,accuracy_score,auc,precision_score,recall_score,roc_auc_score,fbeta_score,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score,accuracy_score,auc,precision_score,recall_score,roc_auc_score,fbeta_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import glob



# Get the filenames of synthetic datasets
synthetic_files = glob.glob('miniboone_*.csv')

# Define the classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    # Add more classifiers as needed
}
real = pd.read_csv('miniboone.csv').drop(['Unnamed: 0'],axis=1)

for column in real.columns:
    unique_values = real[column].nunique()
    print(f"Column '{column}' has {unique_values} unique values.")    

for synthetic_file in synthetic_files:
    # Load synthetic dataset
    real = pd.read_csv('miniboone.csv').drop(['Unnamed: 0'],axis=1)
    synthetic = pd.read_csv(synthetic_file).drop(['Unnamed: 0'],axis=1)
    
    # Perform KS test
    ks_results = []
    for feature in real.columns:
        ks_stat, p_value = ks_2samp(real[feature], synthetic[feature])
        ks_results.append(ks_stat)

    average_ks = np.mean(ks_results)
    print("KS statistic for", synthetic_file, ":", average_ks)
    
    # Create target labels for detection test
    real['label'] = 1
    synthetic['label'] = 0
    
    # Perform sampling on real and synthetic datasets
    sampled_real = real.sample(n=len(synthetic), random_state=42)
    sampled_synthetic = synthetic.sample(n=len(synthetic), random_state=42)
    
    # Concatenate sampled real and synthetic datasets
    combined_data = pd.concat([sampled_real, sampled_synthetic], ignore_index=True)
    
    # Split the combined dataset into train and test sets
    X = combined_data.drop('label', axis=1)
    y = combined_data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Train and evaluate each classifier
    for clf_name, classifier in classifiers.items():
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(clf_name, "accuracy for", synthetic_file, ":", accuracy)
    
    print()