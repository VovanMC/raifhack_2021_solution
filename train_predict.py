IS_GPU = False
#Импорт нужных библиотек
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import time
from scipy.optimize import minimize

def reset_tensorflow_session():
    tf.keras.backend.clear_session()
    tf.random.set_seed(41)
    np.random.seed(41)    

THRESHOLD = 0.15
NEGATIVE_WEIGHT = 1.1

def deviation_metric_one_sample(y_true, y_pred):
    """
    Реализация кастомной метрики для хакатона.
    :param y_true: float, реальная цена
    :param y_pred: float, предсказанная цена
    :return: float, значение метрики
    """
    deviation = (y_pred - y_true) / np.maximum(1e-8, y_true)
    if np.abs(deviation) <= THRESHOLD:
        return 0
    elif deviation <= - 4 * THRESHOLD:
        return 9 * NEGATIVE_WEIGHT
    elif deviation < -THRESHOLD:
        return NEGATIVE_WEIGHT * ((deviation / THRESHOLD) + 1) ** 2
    elif deviation < 4 * THRESHOLD:
        return ((deviation / THRESHOLD) - 1) ** 2
    else:
        return 9
def deviation_metric(y_true, y_pred):
    return np.array([deviation_metric_one_sample(y_true[n], y_pred[n]) for n in range(len(y_true))]).mean()

#Категориальные данные
CATEGORICAL_FEATURES_COLUMNS = ['region', 'city', 'realty_type', 'floor']
#Численные данные
NUM_FEATURES_COLUMNS = ['lat', 'lng', 'osm_amenity_points_in_0.001',
       'osm_amenity_points_in_0.005', 'osm_amenity_points_in_0.0075',
       'osm_amenity_points_in_0.01', 'osm_building_points_in_0.001',
       'osm_building_points_in_0.005', 'osm_building_points_in_0.0075',
       'osm_building_points_in_0.01', 'osm_catering_points_in_0.001',
       'osm_catering_points_in_0.005', 'osm_catering_points_in_0.0075',
       'osm_catering_points_in_0.01', 'osm_city_closest_dist',
      'osm_city_nearest_population',
       'osm_crossing_closest_dist', 'osm_crossing_points_in_0.001',
       'osm_crossing_points_in_0.005', 'osm_crossing_points_in_0.0075',
       'osm_crossing_points_in_0.01', 'osm_culture_points_in_0.001',
       'osm_culture_points_in_0.005', 'osm_culture_points_in_0.0075',
       'osm_culture_points_in_0.01', 'osm_finance_points_in_0.001',
       'osm_finance_points_in_0.005', 'osm_finance_points_in_0.0075',
       'osm_finance_points_in_0.01', 'osm_healthcare_points_in_0.005',
       'osm_healthcare_points_in_0.0075', 'osm_healthcare_points_in_0.01',
       'osm_historic_points_in_0.005', 'osm_historic_points_in_0.0075',
       'osm_historic_points_in_0.01', 'osm_hotels_points_in_0.005',
       'osm_hotels_points_in_0.0075', 'osm_hotels_points_in_0.01',
       'osm_leisure_points_in_0.005', 'osm_leisure_points_in_0.0075',
       'osm_leisure_points_in_0.01', 'osm_offices_points_in_0.001',
       'osm_offices_points_in_0.005', 'osm_offices_points_in_0.0075',
       'osm_offices_points_in_0.01', 'osm_shops_points_in_0.001',
       'osm_shops_points_in_0.005', 'osm_shops_points_in_0.0075',
       'osm_shops_points_in_0.01', 'osm_subway_closest_dist',
       'osm_train_stop_closest_dist', 'osm_train_stop_points_in_0.005',
       'osm_train_stop_points_in_0.0075', 'osm_train_stop_points_in_0.01',
       'osm_transport_stop_closest_dist', 'osm_transport_stop_points_in_0.005',
       'osm_transport_stop_points_in_0.0075',
       'osm_transport_stop_points_in_0.01',
       'reform_count_of_houses_1000', 'reform_count_of_houses_500',
       'reform_house_population_1000', 'reform_house_population_500',
       'reform_mean_floor_count_1000', 'reform_mean_floor_count_500',
       'reform_mean_year_building_1000', 'reform_mean_year_building_500','total_square']
#Таргет
TARGET_COLUMNS = ['per_square_meter_price']

#Считываем данные
def read_train_test():
  train = pd.read_csv('train.csv')
  test = pd.read_csv('test.csv')
  test_submission = pd.read_csv('test_submission.csv')
  return train, test

#Encoder категориальных фичей
def encode_categorical_features(df, categorical_columns):
  for column in categorical_columns:    
    dict_encoding = {key : val for val, key in enumerate(df[column].unique())}
    df[column] = df[column].map(dict_encoding)
  return df

#Квантильное преобразование данных
def get_quantile_transform(_df, columns_for_quantilization, random_state = 41, n_quantiles = 100, output_distribution = 'normal'):
  df = _df.copy()
  for col in columns_for_quantilization:    
    qt = QuantileTransformer(random_state = random_state, n_quantiles = n_quantiles, output_distribution = output_distribution)
    df[col] = qt.fit_transform(df[[col]])
  return df

#МинМакс преобразование данных
def get_minmax_transform(_df, columns_for_quantilization, min_value = -1, max_value = 1):
  df = _df.copy()
  for col in columns_for_quantilization:    
    scaler = MinMaxScaler(feature_range=(min_value, max_value))
    df[col] = scaler.fit_transform(df[[col]])
  return df    
  
#Подготавливаем данные для модельки
def preprocess_data(train, test):
  train = train[train.price_type == 1].reset_index(drop=True)
  train['is_train'] = 1
  test['is_train'] = 0
  data = pd.concat([train, test]).reset_index(drop=True)
  #Hotencoding для категориальных фичей
  data = encode_categorical_features(data, CATEGORICAL_FEATURES_COLUMNS)  
  #Нормализация численных данных
  data = get_quantile_transform(data, NUM_FEATURES_COLUMNS)
  data = get_minmax_transform(data, NUM_FEATURES_COLUMNS)
  #Заполняем NaN значения
  data = data.fillna(data.mean())
  train = data[data.is_train == 1].reset_index(drop=True)
  test = data[data.is_train == 0].reset_index(drop=True)
  train = train.drop(columns = ['is_train'])
  test = test.drop(columns = ['is_train'])
  return train, test

#Стандартное разбиение данных на 5 фолдов случайным образом
def get_standart_split(data, n_splits = 5, seed = 41):
  kf = KFold(n_splits = n_splits, random_state = seed, shuffle = True)
  split_list = []
  for train_index, test_index in kf.split(data):
    split_list += [(train_index, test_index)]
  return split_list

#Создаем tf.Dataset по массивам данных
def get_dataset(arr_features, arr_target, arr_region, arr_city, arr_realty, batch_size):
    return tf.data.Dataset.from_tensor_slices(
        (
            {
                "model_features_input" : arr_features,
             "model_region_input" : arr_region,
             "model_city_input" : arr_city,
             "model_realty_input" : arr_realty,             
            },
            {
                "model_output" : arr_target,             
            },
        )
    ).batch(batch_size)

#Фиксируем поряд фичей в dataframe
def get_columns_order(columns):
  columns_order = sorted([x for x in columns if not x in (CATEGORICAL_FEATURES_COLUMNS + TARGET_COLUMNS)])
  return columns_order + CATEGORICAL_FEATURES_COLUMNS + TARGET_COLUMNS

#Коллбэк, для отслеживания целевой метрики
class CustomCallback(keras.callbacks.Callback):    
    def __init__(self, val_dataset, val_targets):      
        super(CustomCallback, self).__init__()
        self.val_targets = val_targets
        self.val_dataset = val_dataset        

    def on_epoch_end(self, epoch, logs=None):
        predicts = self.model.predict(self.val_dataset)[:,0]
        targets = self.val_targets[:,0]
        print(f"Текущий реальный скор(валидационная часть): {np.round(deviation_metric(targets, predicts), 4)}")

def Dropout(x):
  return keras.layers.Dropout(x)

def Flatten():
    return keras.layers.Flatten()

def Concatenate():
    return keras.layers.Concatenate()

#Функция обучения модели
def fit(model, epochs, train_dataset, val_dataset, val_targets, verbose = True):        
  if IS_GPU:
    print(f"Начинаю обучение модели (GPU) количество эпох = {epochs}")
    with tf.device('/device:GPU:0'):
      #Коллбэк для остановки, если модель перестала обучаться          
      early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta = 2.5e-6, patience = 100, restore_best_weights = True, mode = 'min')
      #Коллбэк для уменьшения скорости обучения
      lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = 0.5, patience = 10, min_lr = 1e-9, mode = 'min')
      #Кастомный коллбэк для отображения скора по целевой метрике
      metric_callback = CustomCallback(val_dataset, val_targets)
      history = model.fit(train_dataset, epochs = epochs, validation_data = val_dataset, verbose = verbose, shuffle=True, callbacks=[early_stopping_callback, lr_callback, metric_callback], workers = -1)
      return history
  else:        
      print(f"Начинаю обучение модели (СPU) количество эпох = {epochs}")        
      #Коллбэк для остановки, если модель перестала обучаться          
      early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta = 2.5e-6, patience = 100, restore_best_weights = True, mode = 'min')
      #Коллбэк для уменьшения скорости обучения
      lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = 0.5, patience = 10, min_lr = 1e-9, mode = 'min')
      #Кастомный коллбэк для отображения скора по целевой метрике
      metric_callback = CustomCallback(val_dataset, val_targets)
      history = model.fit(train_dataset, epochs = epochs, validation_data = val_dataset, verbose = verbose, shuffle=True, callbacks=[early_stopping_callback, lr_callback, metric_callback], workers = -1)
      return history

#Реализация кастомной функции потерь для обучения
def tf_custom_loss(y_true, y_pred):     
    threshold = 0.6     
    error = tf.abs(y_true - y_pred) / y_true
    is_small_error = error <= threshold     
    small_error_loss = tf.square(error / 0.15 - 1)    
    big_error_loss = 9.0 * tf.ones_like(small_error_loss) + tf.abs(error)
    #big_error_loss = (3.0 * tf.ones_like(small_error_loss) + tf.abs(error)) ** 2
    return tf.where(is_small_error, small_error_loss, big_error_loss)

#Компиляция текущей модели
def compile_model(train_dataset, val_dataset, num_features, max_realty, max_region, max_city, lr = 5e-4):  
  reset_tensorflow_session()       
  optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
          
  model_input_layer = tf.keras.Input(shape = (num_features), name = "model_features_input")
  model_input_realty = tf.keras.Input(shape = (1), name = "model_realty_input")
  model_input_region = tf.keras.Input(shape = (1), name = "model_region_input")
  model_input_city = tf.keras.Input(shape = (1), name = "model_city_input")

  model_embedding_layer_realty = keras.layers.Embedding(max_realty + 1, 4, input_length=1, dtype=tf.float64)(model_input_realty)
  model_embedding_layer_region = keras.layers.Embedding(max_region + 1, 32, input_length=1, dtype=tf.float64)(model_input_region)
  model_embedding_layer_city = keras.layers.Embedding(max_city + 1, 32, input_length=1, dtype=tf.float64)(model_input_city)

  concatenated_input_layer = Concatenate()([Flatten()(model_embedding_layer_realty), Flatten()(model_embedding_layer_region), Flatten()(model_embedding_layer_city), Flatten()(model_input_layer)])

  layer_0 = keras.layers.Dense(128, activation="relu")(concatenated_input_layer)  
  layer_1 = keras.layers.Dense(64, activation="relu")(layer_0)  
  layer_2 = keras.layers.Dense(32, activation="relu")(layer_1)
  model_output_layer = keras.layers.Dense(1, activation="relu", name = "model_output")(layer_2)
          
  cur_model = keras.Model(
      inputs = [
          model_input_layer,
          model_input_realty,          
          model_input_region,
          model_input_city,
      ], 
      outputs = [
          model_output_layer,            
      ])                                
  
  print(f"Модель: input_shape = {cur_model.input_shape} output_shape = {cur_model.output_shape}")
  cur_model.compile(loss = tf_custom_loss, optimizer = optimizer)#, run_eagerly=True)
  #cur_model.compile(loss = my_huber_loss, optimizer = optimizer)#, run_eagerly=True)
  #cur_model.compile(loss = tf.keras.losses.MeanAbsoluteError(), optimizer = optimizer)#, run_eagerly=True)
  #  
  return cur_model

#Считываем данные и подготавливаем их
train, test = read_train_test()
train, test = preprocess_data(train, test)
features_columns_order = get_columns_order(train.columns.values.tolist())
split_list = get_standart_split(train)

start_train_model_time = time.time()
#Размер батча для Dataset
BATCH_SIZE = int(2 ** 5)
#Количество эпох обучения
EPOCHS = 500
#Количество численных входных переменных модели
NUM_FEATURES = len(NUM_FEATURES_COLUMNS)
#Макс. значения категориалных фичей
MAX_REALTY = max(train['realty_type'].max(), test['realty_type'].max())
MAX_REGION = max(train['region'].max(), test['region'].max())
MAX_CITY = max(train['city'].max(), test['city'].max())
#Коэффициент домножения таргета, с целью быстрейшего сходимости модельки и лучшего обучения
MUL_TARGET = 5e-5

scores = []
nn_predicts = np.zeros(len(train))
models_nn = []

for fold_num, (train_indexes, valid_indexes) in enumerate(split_list):  
  start_time = time.time()
  print(f"Фолд: {fold_num}")

  train_sub_df = train[features_columns_order].loc[train_indexes].reset_index(drop=True)    
  valid_sub_df = train[features_columns_order].loc[valid_indexes].reset_index(drop=True)    

  print(f"Размер трейна = {train_sub_df.shape} Размер валидации = {valid_sub_df.shape}")
  
  #Строим датасеты
  train_ds = get_dataset(
      train_sub_df[NUM_FEATURES_COLUMNS].values, 
      train_sub_df[TARGET_COLUMNS].values * MUL_TARGET, 
      train_sub_df[['region']].values, 
      train_sub_df[['city']].values, 
      train_sub_df[['realty_type']].values, 
      BATCH_SIZE)
  valid_ds = get_dataset(
      valid_sub_df[NUM_FEATURES_COLUMNS].values, 
      valid_sub_df[TARGET_COLUMNS].values * MUL_TARGET, 
      valid_sub_df[['region']].values, 
      valid_sub_df[['city']].values, 
      valid_sub_df[['realty_type']].values, 
      len(valid_sub_df))

  #Компилируем модель
  model = compile_model(train_ds, valid_ds, NUM_FEATURES, MAX_REALTY, MAX_REGION, MAX_CITY)  
  #Обучаем модель
  fit(model, EPOCHS, train_ds, valid_ds, valid_sub_df[TARGET_COLUMNS].values * MUL_TARGET)

  predict_on_validation = model.predict(valid_ds)[:,0] / MUL_TARGET
  nn_predicts[valid_indexes] = predict_on_validation
  targets_for_validation = valid_sub_df[TARGET_COLUMNS].values[:,0]
  current_score = deviation_metric(targets_for_validation, predict_on_validation)
  scores += [current_score]
  models_nn  += [model]
  print(f"Скор для фолда({fold_num}) : {np.round(current_score, 4)} средний скор на префиксе = {np.round(np.mean(scores), 4)} это заняло = {int(time.time() - start_time)} сек.")
print(f"Процесс обучения модели занял = {int(time.time() - start_train_model_time)} секунд")

#Предикт нейронной сетью на test
def get_nn_predict(models, test):
  result = np.zeros(len(test))  
  test_ds = get_dataset(
      test[NUM_FEATURES_COLUMNS].values, 
      np.zeros(len(test)), 
      test[['region']].values, 
      test[['city']].values, 
      test[['realty_type']].values, 
      len(test))
  for model in models:
    predict = model.predict(test_ds)[:,0]    
    result += (predict / MUL_TARGET) / len(models)
  return result

test_nn_predict = get_nn_predict(models_nn, test)

#LightGBM кастомная метрика
def feval_deviation(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'deviation_error', deviation_metric(y_true, y_pred), False

#Функция для обучения модели LightGBM
def train_lgb(train, valid, num_features, categorical_features, target_train, target_valid, EPOCHS, params):                
    #feature_importances = np.zeros(len(features))                
    train_dataset = lgb.Dataset(train[num_features + categorical_features], target_train, weight = (1.0 / target_train), categorical_feature = categorical_features)
    valid_dataset = lgb.Dataset(valid[num_features + categorical_features], target_valid, weight = (1.0 / target_valid), categorical_feature = categorical_features)
    model = lgb.train(
        params = params, 
        num_boost_round = EPOCHS,
        train_set = train_dataset, 
        valid_sets = [train_dataset, valid_dataset], 
        verbose_eval = 100,
        early_stopping_rounds = int(5 / params['learning_rate']),                          
        feval = feval_deviation)
        
    y_valid = model.predict(valid[num_features + categorical_features])
    #feature_importances = model.feature_importance(importance_type='gain') / 5.0            
    #lgb.plot_importance(model,max_num_features = 41)        
    
    return model, y_valid

start_train_model_time = time.time()

boosting_seed = 41
boosting_params = {
    'bagging_fraction': 0.9, 
    'bagging_freq': 1, 
    'boost': 'gbdt', 
    'feature_fraction': 0.9, 
    'max_depth':3,
    'learning_rate': 0.05, 
    'metric': 'custom', 
    'objective': 'regression_l1', 
    'verbose': -1,
    'n_jobs': -1,
    'seed':boosting_seed,
    'feature_fraction_seed': boosting_seed,
    'bagging_seed': boosting_seed,
    'drop_seed': boosting_seed,
    'data_random_seed': boosting_seed,
}

#Количество эпох обучения
EPOCHS = 10000
scores = []
lgb_predicts = np.zeros(len(train))

lgb_models = []
for fold_num, (train_indexes, valid_indexes) in enumerate(split_list):  
  start_time = time.time()
  print(f"Фолд: {fold_num}")

  train_sub_df = train[features_columns_order].loc[train_indexes].reset_index(drop=True)    
  valid_sub_df = train[features_columns_order].loc[valid_indexes].reset_index(drop=True)    

  print(f"Размер трейна = {train_sub_df.shape} Размер валидации = {valid_sub_df.shape}")  
  #Обучаем LightGBM и делаем предикт на валидационной выборке  
  model, predict_validation = train_lgb(
      train_sub_df,
      valid_sub_df,      
      NUM_FEATURES_COLUMNS,
      CATEGORICAL_FEATURES_COLUMNS,
      train_sub_df[TARGET_COLUMNS[0]].values,
      valid_sub_df[TARGET_COLUMNS[0]].values,      
      EPOCHS,
      boosting_params)    

  lgb_models += [model]
  predict_on_validation = model.predict(valid_sub_df[NUM_FEATURES_COLUMNS + CATEGORICAL_FEATURES_COLUMNS])
  lgb_predicts[valid_indexes] = predict_on_validation
  targets_for_validation = valid_sub_df[TARGET_COLUMNS].values[:,0]
  current_score = deviation_metric(targets_for_validation, predict_on_validation)
  scores += [current_score]
  print(f"Скор для фолда({fold_num}) : {np.round(current_score, 4)} средний скор на префиксе = {np.round(np.mean(scores), 4)} это заняло = {int(time.time() - start_time)} сек.")
print(f"Процесс обучения модели занял = {int(time.time() - start_train_model_time)} секунд")

#Предикт lgb на test
def get_lgb_predict(models, test):
  result = np.zeros(len(test))    
  for model in models:
    predict = model.predict(test[NUM_FEATURES_COLUMNS + CATEGORICAL_FEATURES_COLUMNS]) 
    result += predict / len(models)
  return result

test_lgb_predict = get_lgb_predict(lgb_models, test)

test_lgb_predict.min(), test_lgb_predict.max(), test_lgb_predict.mean()

#Кастомная метрика для xgboost
def xbg_error(preds, dtrain):
    labels = dtrain.get_label()
    err = deviation_metric(labels, preds)
    return 'deviation_error', err

def train_xgb(train, valid, num_features, categorical_features, target_train, target_valid, EPOCHS, params):        

    dtest = xgb.DMatrix(test[num_features + categorical_features])
    y_valid = np.zeros(len(valid))    
    
    dtrain = xgb.DMatrix(train[num_features + categorical_features], target_train, weight = 1.0 / target_train)
    dvalid = xgb.DMatrix(valid[num_features + categorical_features], target_valid, weight = 1.0 / target_valid)
    model = xgb.train(
        params,
        dtrain,
        EPOCHS,        
        [(dvalid, "valid")],
        verbose_eval=250,
        early_stopping_rounds=500,
        feval=xbg_error,
    )
    y_valid = model.predict(dvalid)    
            
    return model, y_valid

start_train_model_time = time.time()

xgboost_seed = 41
xgboost_params = {
        "subsample": 0.60,
        "colsample_bytree": 0.40,
        "max_depth": 7,
        "learning_rate": 0.01,
        "objective": "reg:squarederror",
        'disable_default_eval_metric': 1, 
        "nthread": -1,                
        "max_bin": 64, 
        'min_child_weight': 0.0,
        'reg_lambda': 0.0,
        'reg_alpha': 0.0, 
        'seed' : xgboost_seed,
    }

#Количество эпох обучения
EPOCHS = 10000
scores = []
xgb_predicts = np.zeros(len(train))

xgb_models = []
for fold_num, (train_indexes, valid_indexes) in enumerate(split_list):  
  start_time = time.time()
  print(f"Фолд: {fold_num}")

  train_sub_df = train[features_columns_order].loc[train_indexes].reset_index(drop=True)    
  valid_sub_df = train[features_columns_order].loc[valid_indexes].reset_index(drop=True)    

  print(f"Размер трейна = {train_sub_df.shape} Размер валидации = {valid_sub_df.shape}")  
  #Обучаем Xgboost и делаем предикт на валидационной выборке  
  model, predict_validation = train_xgb(
      train_sub_df,
      valid_sub_df,      
      NUM_FEATURES_COLUMNS,
      CATEGORICAL_FEATURES_COLUMNS,
      train_sub_df[TARGET_COLUMNS[0]].values,
      valid_sub_df[TARGET_COLUMNS[0]].values,      
      EPOCHS,
      xgboost_params)    
  
  xgb_models += [model]
  predict_on_validation = model.predict(xgb.DMatrix(valid_sub_df[NUM_FEATURES_COLUMNS + CATEGORICAL_FEATURES_COLUMNS]))
  xgb_predicts[valid_indexes] = predict_on_validation
  targets_for_validation = valid_sub_df[TARGET_COLUMNS].values[:,0]
  current_score = deviation_metric(targets_for_validation, predict_on_validation)
  scores += [current_score]
  print(f"Скор для фолда({fold_num}) : {np.round(current_score, 4)} средний скор на префиксе = {np.round(np.mean(scores), 4)} это заняло = {int(time.time() - start_time)} сек.")
print(f"Процесс обучения модели занял = {int(time.time() - start_train_model_time)} секунд")

#Предикт xgb на test
def get_xgb_predict(models, test):
  result = np.zeros(len(test))    
  for model in models:
    predict = model.predict(xgb.DMatrix(test[NUM_FEATURES_COLUMNS + CATEGORICAL_FEATURES_COLUMNS]))
    result += predict / len(models)
  return result

test_xgb_predict = get_xgb_predict(xgb_models, test)

test_xgb_predict.min(), test_xgb_predict.max(), test_xgb_predict.mean()

train_targets = train[TARGET_COLUMNS[0]].values

def minimize_arit(W):
    ypred = W[0] * nn_predicts + W[1] * lgb_predicts + W[2] * xgb_predicts
    return deviation_metric(train_targets, ypred)

W = minimize(minimize_arit, [1.0 / 3] * 3, options={'gtol': 1e-6, 'disp': True}).x
print('Weights arit:', W)

print(nn_predicts.min(), nn_predicts.max(), nn_predicts.mean())
print(lgb_predicts.min(), lgb_predicts.max(), lgb_predicts.mean())
print(xgb_predicts.min(), xgb_predicts.max(), xgb_predicts.mean())

test_submission = pd.read_csv('test_submission.csv')
test_submission['per_square_meter_price'] = test_nn_predict * W[0] + test_lgb_predict * W[1] + test_xgb_predict * W[2]
test_submission['per_square_meter_price'] = test_submission['per_square_meter_price'].apply(lambda x: max(0.0, x))
test_submission.to_csv('submission.csv', index = False)