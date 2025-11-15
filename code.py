import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import ElasticNetCV, MultiTaskElasticNetCV
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
import category_encoders as ce
from scipy.stats import boxcox, skew
from scipy.special import inv_boxcox  
from sklearn.metrics import (mean_squared_log_error, 
                            r2_score,
                            mean_absolute_error,
                            mean_squared_error)

data = pd.read_csv(r"C:\Users\Dosya\Downloads\spotifyMusic\Spotify_Youtube.csv")
data = data[(data['Licensed'] == True) & (data['official_video'] == True)].copy()

categorical = [col for col in data.columns if data[col].dtype == object]
numerical = [col for col in data.columns if data[col].dtype in [int, float] and col not in ['Views', 'Likes']]

randoming = 312
y_col = ['Views', 'Likes']


X = data.drop(columns=y_col)
y = data[y_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=randoming
)

X_train_cat = X_train[categorical]
X_train_num = X_train[numerical]

X_test_cat = X_test[categorical]
X_test_num = X_test[numerical]

my_encoder = ce.HashingEncoder(cols=categorical, n_components=1000)  
X_categorical_train_encoded = my_encoder.fit_transform(X_train_cat).reset_index(drop=True)
X_categorical_test_encoded = my_encoder.transform(X_test_cat).reset_index(drop=True)


num_imputer = SimpleImputer(strategy='mean')
X_train_num_imputed = num_imputer.fit_transform(X_train_num)
X_test_num_imputed = num_imputer.transform(X_test_num)

poly = PolynomialFeatures(degree=2, include_bias=True)
X_numerical_train_poly = poly.fit_transform(X_train_num_imputed)
X_numerical_test_poly = poly.transform(X_test_num_imputed)

scaler = StandardScaler()
X_numerical_train_s = scaler.fit_transform(X_numerical_train_poly)
X_numerical_test_s = scaler.transform(X_numerical_test_poly)

scaler_cat_encoded = StandardScaler()
X_categorical_train_s = scaler_cat_encoded.fit_transform(X_categorical_train_encoded)
X_categorical_test_s = scaler_cat_encoded.transform(X_categorical_test_encoded)


X_train_final = np.hstack([X_categorical_train_s, X_numerical_train_s])
X_test_final = np.hstack([X_categorical_test_s, X_numerical_test_s])

print("X_train_final shape:", X_train_final.shape)
print("X_test_final shape: ", X_test_final.shape)




def safe_boxcox_series(train_series, test_series):
    
    train_series = train_series.fillna(train_series.median()).astype(float)
    
    test_series = test_series.fillna(train_series.median()).astype(float)

    
    shift = 0.0
    if train_series.min() <= 0:
        shift = abs(train_series.min()) + 1e-6

    train_shifted = train_series + shift
    test_shifted = test_series + shift

  
    if (train_shifted <= 0).any():
        raise ValueError("Shift failed: still <= 0 values in train.")

    
    train_bc, lam = boxcox(train_shifted)
    test_bc = boxcox(test_shifted, lam)

    return train_bc, test_bc, lam, shift, train_series.index


y_train_views_bc, y_test_views_bc, lambda_views, shift_views, idx_train_views = safe_boxcox_series(y_train['Views'], y_test['Views'])
print("Skew train views:", skew(y_train_views_bc))
print("Skew test views: ", skew(y_test_views_bc))


y_train_likes_bc, y_test_likes_bc, lambda_likes, shift_likes, idx_train_likes = safe_boxcox_series(y_train['Likes'], y_test['Likes'])
print("Skew train likes:", skew(y_train_likes_bc))
print("Skew test likes: ", skew(y_test_likes_bc))


X_train_final_df = pd.DataFrame(X_train_final)
X_test_final_df = pd.DataFrame(X_test_final)


X_train_reset = X_train.reset_index(drop=True)

train_pos = np.arange(len(X_train_reset))

mask_notna = ~y_train['Views'].isna().reset_index(drop=True)

X_train_proc = X_train_final_df[mask_notna.values].values


print("X_train_proc shape:", X_train_proc.shape)
print("y_train_views_bc shape:", y_train_views_bc.shape)
print("y_train_likes_bc shape:", y_train_likes_bc.shape)


x_imputer = SimpleImputer(strategy="mean")
X_train_imp = x_imputer.fit_transform(X_train_proc)
X_test_imp = x_imputer.transform(X_test_final_df.values)

print("NaN in X_train_imp:", np.isnan(X_train_imp).sum())
print("NaN in X_test_imp:", np.isnan(X_test_imp).sum())


model = MultiTaskElasticNetCV(
    alphas=[0.01, 0.1, 0.5, 1, 5, 10],
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 1],
    cv=6,
    random_state=randoming,
    max_iter=5000 
)



y_train_bc = np.column_stack([y_train_views_bc, y_train_likes_bc])

print("Final shapes -> X:", X_train_imp.shape, " y:", y_train_bc.shape)

model.fit(X_train_imp, y_train_bc)

print("NaN in y_train_bc:", np.isnan(y_train_bc).sum())


y_pred_bc = model.predict(X_test_imp)
print("y_pred_bc shape:", y_pred_bc.shape)


y_pred_views_orig = inv_boxcox(y_pred_bc[:, 0], lambda_views) - shift_views
y_pred_likes_orig = inv_boxcox(y_pred_bc[:, 1], lambda_likes) - shift_likes


print("Shapes check:")
print("y_test Views:", y_test['Views'].shape)
print("y_pred Views:", y_pred_views_orig.shape)

print("First 5 true vs predicted:")
print(list(zip(y_test['Views'].head(5), y_pred_views_orig[:5])))


for i in range(5):
    print(f"Predicted (Views, Likes) #{i}:",
          int(y_pred_views_orig[i]), int(y_pred_likes_orig[i]))
    

r2_score_views = r2_score(y_test['Views'], y_pred_views_orig)
r2_score_likes = r2_score(y_test['Likes'], y_pred_likes_orig)

print(f'The quality of views prediction: {r2_score_views}')
print(f'The quality of likes prediction: {r2_score_likes}')

mae_views = mean_absolute_error(y_test['Views'], y_pred_views_orig)
rmse_views = mean_squared_error(y_test['Views'], y_pred_views_orig, squared=False)

print(f"MAE Views: {mae_views}, RMSE Views: {rmse_views}")    
