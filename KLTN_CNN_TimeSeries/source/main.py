import time
import io
import math 
import torch
import pandas as pd
import numpy as np
from EDA import EDA
from Multiple_Lines import MultipleLines
import streamlit as st
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
from scikeras.wrappers import KerasRegressor, KerasClassifier
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, MaxPooling1D, SimpleRNN, GRU, RepeatVector
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from pickle import load
from Excel_Output import Excel_Output
import os

tf.get_logger().setLevel('ERROR')

st.set_page_config(page_title="Forecast Time Series",page_icon=":bar_chart:",layout="centered")

st.write("# DỰ ĐOÁN BẰNG CÁC MÔ HÌNH HỌC SÂU - LAI GHÉP CÁC MÔ HÌNH SÂU")
st.divider()
st.sidebar.write("# Thiết lập mô hình")


# ------------------------- Biến toàn cục ------------------------- #
df = None
df_target = None
df_scaled = None
model = None
d = None
t = None
train_size = None
degree = None
unit = None
epoch= None
batch_size = None
m = None
scaler = None
test_size = None
train = None
test = None
train_time = None
test_time = None
predict = None
actual = None
info_table = None
result_table = None
metric_table = None
metrics = None
learning_rate = None
generator = None
discriminator = None
model_training = None

# ------------------------- Function ------------------------- #
# load data.csv
@st.cache_data
def LoadData(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

#Tính CV_RMSE
@st.cache_data
def CV_RMSE(predict, actual):
    # Số lượng fold (chẳng hạn, 5-fold cross-validation)
    num_folds = 5

    # Khởi tạo K-fold cross-validation
    kf = KFold(n_splits=num_folds)
    # Tạo danh sách để lưu kết quả RMSE từ từng fold
    rmse_scores = []

    for train_index, test_index in kf.split(actual):
        predicted_test, actual_test = predict[test_index], actual[test_index]
        
        mse = mean_squared_error(actual_test, predicted_test)
        rmse = math.sqrt(mse)
        
        rmse_scores.append(rmse)

    # Tính tổng RMSE từ các fold và tính RMSE trung bình
    average_rmse = np.mean(rmse_scores)
    return average_rmse

# Hàm đánh giá
@st.cache_data
def Score(predict, actual):
    mae = mean_absolute_error(actual, predict)
    mse = mean_squared_error(actual, predict)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predict) / predict)) # bỏ
    cv_rmse = CV_RMSE(predict,actual)
    return mae, mse, rmse ,mape ,cv_rmse



# Xóa dữ liệu lưu trong streamlit
def ClearCache():
    st.session_state.clear()

def dfs_tabs(df_list, sheet_list):

    output = io.BytesIO()

    writer = pd.ExcelWriter(output,engine='xlsxwriter')   
    for dataframe, sheet in zip(df_list, sheet_list):
        dataframe.to_excel(writer, sheet_name=sheet, startrow=0 , startcol=0)   
    writer.close()

    processed_data = output.getvalue()
    return processed_data

def dfs_tabs_all(file_name_test, df_test, input_dim, output_dim, selected_predict_column_name_test, scaler, table_info_test):

    output_data = Excel_Output()
    
    error_list, result_test_list, para_train_list = output_data.LoadTable(df_test, input_dim, output_dim, selected_predict_column_name_test, scaler)

    output = io.BytesIO()

    writer = pd.ExcelWriter(output,engine='xlsxwriter')
    error_list.to_excel(writer, sheet_name=file_name_test , startrow=0 , startcol=0)
    result_test_list.to_excel(writer, sheet_name=file_name_test , startrow=1 , startcol=8)
    para_train_list.to_excel(writer, sheet_name=file_name_test , startrow=50 , startcol=0)
    table_info_test.to_excel(writer, sheet_name=file_name_test , startrow=95 , startcol=0)
    writer.close()   

    processed_data = output.getvalue()
    return processed_data

def PrintWeight(cp):
    a = cp.get_weights()
    # Tạo DataFrame để lưu trọng số của từng lớp   
    weight_df = pd.DataFrame()  
    # Xác định số lượng trọng số tối đa trong các cột
    max_weight_count = max(len(w.ravel()) for w in a)
    # Duyệt qua các trọng số và thêm vào DataFrame
    for i, layer_weights in enumerate(a):
        layer_name = f'Layer_{i+1}'  # Đặt tên cho mỗi lớp
        # Làm phẳng trọng số thành một mảng 1D
        flattened_weights = layer_weights.ravel()
        # Nếu số lượng trọng số ít hơn số lượng trọng số tối đa
        if len(flattened_weights) < max_weight_count:
            # Điền vào các giá trị null cho các hàng còn thiếu
            missing_count = max_weight_count - len(flattened_weights)
            flattened_weights = np.concatenate([flattened_weights, [np.nan] * missing_count])
        # Thêm vào DataFrame
        weight_df[layer_name] = flattened_weights

    # In ra một số hàng đầu tiên của DataFrame
    st.write(weight_df)

def save_file_to_output_folder(file_content, file_name):
    output_dir = "./output/"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "wb") as f:
        f.write(file_content)
    return file_path

if 'clicked_train' not in st.session_state:
    st.session_state.clicked_train = False

def click_button_train():
    st.session_state.clicked_train = True

if 'clicked_save' not in st.session_state:
    st.session_state.clicked_save = False
if 'display_info' not in st.session_state:
        st.session_state.display_info = {}
def click_button_save():
    st.session_state.clicked_save = True
    

#--------------------------------------
# Sidebar
# Chọn mô hình



option = st.sidebar.radio(
    "Chọn loại mô hình:",
    ("Đơn", "Tuần tự", "Tuần tự nhân", "Tuần tự cộng", "Song song")
)

# Hiển thị kết quả dựa trên lựa chọn
if option == "Đơn":
    mod = st.sidebar.selectbox(
    "Chọn mô hình:",
    ["CNN", "LSTM","RNN", "GRU"],
    on_change=ClearCache).lstrip('*').rstrip('*')
elif option == "Tuần tự":
    mod = st.sidebar.selectbox(
    "Chọn mô hình:",
    ["CNN-LSTM", "LSTM-CNN", "CNN-RNN", "RNN-CNN", "CNN-GRU", "GRU-CNN","LSTM-RNN","RNN-LSTM","LSTM-GRU","GRU-LSTM","RNN-GRU","GRU-RNN"],
    on_change=ClearCache).lstrip('*').rstrip('*')
else:
    mod = st.sidebar.selectbox(
    "Chọn mô hình:",
    ["CNN", "LSTM","RNN", "GRU"],
    on_change=ClearCache).lstrip('*').rstrip('*')
    if option == "Song song":
        match mod:
            case "CNN":
                mod1 = st.sidebar.selectbox(
                    "Chọn mô hình kết hợp:",
                    ["LSTM", "RNN", "GRU"],
                    on_change=ClearCache).lstrip('*').rstrip('*')
            case "LSTM":
                mod1 = st.sidebar.selectbox(
                    "Chọn mô hình kết hợp:",
                    ["RNN", "GRU"],
                    on_change=ClearCache).lstrip('*').rstrip('*')
            case "RNN":
                mod1 = st.sidebar.selectbox(
                    "Chọn mô hình kết hợp:",
                    ["GRU"],
                    on_change=ClearCache).lstrip('*').rstrip('*')
            case "GRU":
                mod1 = st.sidebar.selectbox(
                    "Chọn mô hình kết hợp:",
                    [""],
                    on_change=ClearCache).lstrip('*').rstrip('*')
    else: 
        mod1 = st.sidebar.selectbox(
        "Chọn mô hình kết hợp:",
        ["LSTM", "RNN", "GRU"],
        on_change=ClearCache).lstrip('*').rstrip('*')


# Chọn ngày để dự đoán
col1, col2 = st.sidebar.columns(2)
with col1:
    input_dim = st.number_input('**Số ngày dùng để dự đoán:**',
                            value=7, step=1, min_value=1, on_change=ClearCache)

with col2:
    output_dim = st.number_input('**Số ngày muốn dự đoán:**', value=1,
                            step=1, min_value=1, on_change=ClearCache)

# Chọn tỉ lệ chia tập train/test
train_size = st.sidebar.slider('**Tỉ lệ training**', 10, 70, 70, step=10)
valid_size = st.sidebar.slider('**Tỉ lệ Validation**', 10, 90 - train_size, 20, step=10)
train_ratio = train_size/100
valid_ratio = valid_size/100

activation = st.sidebar.selectbox(
    '**Chọn Activation funcion**', ('ReLU', 'LeakyReLU', 'tanh'), on_change=ClearCache)


scaler = st.sidebar.selectbox(
    '**Chọn phương pháp chuẩn hóa dữ liệu**', ('Min-Max', 'Zero-Mean', 'Dữ liệu gốc'), on_change=ClearCache)


#Cai dat mo hinh   
#Mo hinh don
def LSTM_Model(input_dim=10, output_dim=1, units =32, learning_rate=0.0001, activation = 'relu') -> tf.keras.models.Model:
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(input_dim, 1), activation=activation))
    model.add(LSTM(units=units, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(units=output_dim))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def GRU_Model(input_dim=10, output_dim=1, units =32, learning_rate=0.0001, activation = 'relu') -> tf.keras.models.Model:
    model = Sequential()
    model.add(GRU(units=units, return_sequences=True, input_shape=(input_dim, 1), activation=activation))
    model.add(GRU(units=units, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(units=output_dim))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def RNN_Model(input_dim=10, output_dim=1, units =32, learning_rate=0.0001, activation = 'relu') -> tf.keras.models.Model:
    model = Sequential()
    model.add(SimpleRNN(units=units, return_sequences=True, \
                        input_shape=(input_dim, 1), activation=activation))
    model.add(SimpleRNN(units=units, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(units=output_dim))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def CNN_Model(input_dim=10, output_dim=1, units = 32, learning_rate = 0.0001, activation = 'relu'):
    model = Sequential()
    # Thêm lớp Convolutional 1D đầu tiên
    model.add(Conv1D(units, input_shape=(input_dim, 1), \
                     kernel_size=3, strides=1, padding='same', activation=activation)) 
    model.add(Conv1D(units, kernel_size=3, strides=1, padding='same', activation=activation))
    model.add(MaxPooling1D(pool_size=2,strides=2, padding='same'))
    # Hoàn thiện mô hình
    model.add(Flatten())

    model.add(Dense(32, activation=activation))
    model.add(Dense(units=output_dim))
    # Thiết lập cấu hình cho mô hình để sẵn sàng cho quá trình huấn luyện.
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model


#Seq
def CNN_LSTM_Model(input_dim=10, output_dim=1, units = 32, learning_rate = 0.0001, activation = 'relu') -> tf.keras.models.Model:
    model = Sequential()
    #CNN
    model.add(Conv1D(units, input_shape=(input_dim, 1), \
                     kernel_size=3, strides=1, padding='same', activation=activation)) 
    model.add(Conv1D(units, kernel_size=3, strides=1, padding='same', activation=activation))
    model.add(MaxPooling1D(pool_size=2,strides=2, padding='same'))
    model.add(Flatten())
    #tạo ra một tensor mới có hình dạng (None, out, units/filter)
    model.add(RepeatVector(output_dim))
    #LSTM
    model.add(LSTM(units=units, return_sequences=True, activation=activation))
    model.add(LSTM(units=units, activation=activation))
    # fully connection
    model.add(Dense(32, activation=activation))
    model.add(Dense(units=output_dim))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def LSTM_CNN_Model(input_dim=10, output_dim=1, units = 32, learning_rate = 0.0001, activation = 'relu') -> tf.keras.models.Model:
    model = Sequential()

    #LSTM
    model.add(LSTM(units=units, return_sequences=True, input_shape=(input_dim, 1), activation=activation))
    model.add(LSTM(units=units, activation=activation))
    
    #tạo ra một tensor mới có hình dạng (None, out, units/filter)
    model.add(RepeatVector(output_dim))

    #CNN
    model.add(Conv1D(units, kernel_size=3, strides=1, padding='same', activation=activation))
    model.add(MaxPooling1D(pool_size=2,strides=2, padding='same')) 
    model.add(Flatten())
    # fully connection
    model.add(Dense(32, activation=activation))
    model.add(Dense(units=output_dim))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def CNN_RNN_Model(input_dim=10, output_dim=1, units = 32, learning_rate = 0.0001, activation = 'relu') -> tf.keras.models.Model:
    model = Sequential()
    #CNN
    model.add(Conv1D(units, input_shape=(input_dim, 1), \
                     kernel_size=3, strides=1, padding='same', activation=activation)) 
    model.add(Conv1D(units, kernel_size=3, strides=1, padding='same', activation=activation))
    model.add(MaxPooling1D(pool_size=2,strides=2, padding='same'))
    model.add(Flatten())
    #tạo ra một tensor mới có hình dạng (None, out, units/filter)
    model.add(RepeatVector(output_dim))
    #RNN
    model.add(SimpleRNN(units=units, return_sequences=True, activation=activation))
    model.add(SimpleRNN(units=units, activation=activation))
    # fully connection
    model.add(Dense(32, activation=activation))
    model.add(Dense(units=output_dim))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def RNN_CNN_Model(input_dim=10, output_dim=1, units = 32, learning_rate = 0.0001, activation = 'relu') -> tf.keras.models.Model:
    model = Sequential()
    #RNN
    model.add(SimpleRNN(units=units, return_sequences=True,\
                    input_shape=(input_dim, 1), activation=activation))
    model.add(SimpleRNN(units=units, activation=activation))
    #tạo ra một tensor mới có hình dạng (None, out, units/filter)
    model.add(RepeatVector(output_dim))
    #CNN
    model.add(Conv1D(units, kernel_size=3, strides=1, padding='same', activation=activation))
    model.add(MaxPooling1D(pool_size=2,strides=2, padding='same')) 
    model.add(Flatten())
    # fully connection
    model.add(Dense(32, activation=activation))
    model.add(Dense(units=output_dim))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def CNN_GRU_Model(input_dim=10, output_dim=1, units = 32, learning_rate = 0.0001, activation = 'relu') -> tf.keras.models.Model:
    model = Sequential()
    #CNN
    model.add(Conv1D(units, input_shape=(input_dim, 1), \
                     kernel_size=3, strides=1, padding='same', activation=activation)) 
    model.add(Conv1D(units, kernel_size=3, strides=1, padding='same', activation=activation))
    model.add(MaxPooling1D(pool_size=2,strides=2, padding='same'))
    model.add(Flatten())
    #tạo ra một tensor mới có hình dạng (None, out, units/filter)
    model.add(RepeatVector(output_dim))
    #GRU
    model.add(GRU(units=units, return_sequences=True, activation=activation))
    model.add(GRU(units=units, activation=activation))
    # fully connection
    model.add(Dense(32, activation=activation))
    model.add(Dense(units=output_dim))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def GRU_CNN_Model(input_dim=10, output_dim=1, units = 32, learning_rate = 0.0001, activation = 'relu') -> tf.keras.models.Model:
    model = Sequential()
    #GRU
    model.add(GRU(units=units, return_sequences=True,\
                    input_shape=(input_dim, 1), activation=activation))
    model.add(GRU(units=units, activation=activation))
    #tạo ra một tensor mới có hình dạng (None, out, units/filter)
    model.add(RepeatVector(output_dim))
    #CNN
    model.add(Conv1D(units, kernel_size=3, strides=1, padding='same', activation=activation))
    model.add(MaxPooling1D(pool_size=2,strides=2, padding='same')) 
    model.add(Flatten())
    # fully connection
    model.add(Dense(32, activation=activation))
    model.add(Dense(units=output_dim))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def LSTM_RNN_Model(input_dim=10, output_dim=1, units = 32, learning_rate = 0.0001, activation = 'relu') -> tf.keras.models.Model:
    model = Sequential()

    #LSTM
    model.add(LSTM(units=units, return_sequences=True, input_shape=(input_dim, 1), activation=activation))
    model.add(LSTM(units=units, activation=activation))
    
    #tạo ra một tensor mới có hình dạng (None, out, units/filter)
    model.add(RepeatVector(output_dim))

    #RNN
    model.add(SimpleRNN(units=units, return_sequences=True, activation=activation))
    model.add(SimpleRNN(units=units, activation=activation))
    # fully connection
    model.add(Dense(32, activation=activation))
    model.add(Dense(units=output_dim))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def RNN_LSTM_Model(input_dim=10, output_dim=1, units = 32, learning_rate = 0.0001, activation = 'relu') -> tf.keras.models.Model:
    model = Sequential()

    #RNN
    model.add(SimpleRNN(units=units, return_sequences=True, input_shape=(input_dim, 1), activation=activation))
    model.add(SimpleRNN(units=units, activation=activation))
    
    #tạo ra một tensor mới có hình dạng (None, out, units/filter)
    model.add(RepeatVector(output_dim))

    #LSTM
    model.add(LSTM(units=units, return_sequences=True, activation=activation))
    model.add(LSTM(units=units, activation=activation))
    # fully connection
    model.add(Dense(32, activation=activation))
    model.add(Dense(units=output_dim))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def LSTM_GRU_Model(input_dim=10, output_dim=1, units = 32, learning_rate = 0.0001, activation = 'relu') -> tf.keras.models.Model:
    model = Sequential()

    #LSTM
    model.add(LSTM(units=units, return_sequences=True, input_shape=(input_dim, 1), activation=activation))
    model.add(LSTM(units=units, activation=activation))
    
    #tạo ra một tensor mới có hình dạng (None, out, units/filter)
    model.add(RepeatVector(output_dim))

    #GRU
    model.add(GRU(units=units, return_sequences=True, activation=activation))
    model.add(GRU(units=units, activation=activation))
    # fully connection
    model.add(Dense(32, activation=activation))
    model.add(Dense(units=output_dim))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def GRU_LSTM_Model(input_dim=10, output_dim=1, units = 32, learning_rate = 0.0001, activation = 'relu') -> tf.keras.models.Model:
    model = Sequential()

    #GRU
    model.add(GRU(units=units, return_sequences=True, input_shape=(input_dim, 1), activation=activation))
    model.add(GRU(units=units, activation=activation))
    
    #tạo ra một tensor mới có hình dạng (None, out, units/filter)
    model.add(RepeatVector(output_dim))

    #LSTM
    model.add(LSTM(units=units, return_sequences=True, activation=activation))
    model.add(LSTM(units=units, activation=activation))
    # fully connection
    model.add(Dense(32, activation=activation))
    model.add(Dense(units=output_dim))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def RNN_GRU_Model(input_dim=10, output_dim=1, units = 32, learning_rate = 0.0001, activation = 'relu') -> tf.keras.models.Model:
    model = Sequential()

    #LSTM
    model.add(SimpleRNN(units=units, return_sequences=True, input_shape=(input_dim, 1), activation=activation))
    model.add(SimpleRNN(units=units, activation=activation))
    
    #tạo ra một tensor mới có hình dạng (None, out, units/filter)
    model.add(RepeatVector(output_dim))

    #GRU
    model.add(GRU(units=units, return_sequences=True, activation=activation))
    model.add(GRU(units=units, activation=activation))
    # fully connection
    model.add(Dense(32, activation=activation))
    model.add(Dense(units=output_dim))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def GRU_RNN_Model(input_dim=10, output_dim=1, units = 32, learning_rate = 0.0001, activation = 'relu') -> tf.keras.models.Model:
    model = Sequential()

    #GRU
    model.add(GRU(units=units, return_sequences=True, input_shape=(input_dim, 1), activation=activation))
    model.add(GRU(units=units, activation=activation))
    
    #tạo ra một tensor mới có hình dạng (None, out, units/filter)
    model.add(RepeatVector(output_dim))

    #LSTM
    model.add(SimpleRNN(units=units, return_sequences=True, activation=activation))
    model.add(SimpleRNN(units=units, activation=activation))
    # fully connection
    model.add(Dense(32, activation=activation))
    model.add(Dense(units=output_dim))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

#Tinh Omega
def calculate_omega(y, y1, y2):
    numerator = np.sum((y2 - y1) * (y - y1))
    denominator = np.sum((y2 - y1) ** 2)
    omega = numerator / denominator if denominator != 0 else 0
    # Ensure w_parallel is between 0 and 1
    omega = max(0, min(omega, 1))
    return omega


# Chọn tập dữ liệu
st.header("Chọn tập dữ liệu tiến hành huấn luyện")
uploaded_file = st.file_uploader(
    "Chọn tệp dữ liệu", type=["csv"], on_change=ClearCache)

if uploaded_file is not None:
    file_name = uploaded_file.name
    df = LoadData(uploaded_file)

    # nếu tập dữ liệu không phải tập weather

    selected_predict_column_name = st.sidebar.selectbox(
        '**Chọn cột để tiến hành training:**', tuple(df.drop(df.columns[0],axis = 1).columns.values), on_change=ClearCache)
    
    # Tạo đối tượng EDA
    eda = EDA(df = df, n_steps_in = input_dim, n_steps_out = output_dim, feature=selected_predict_column_name, train_ratio = train_ratio, valid_ratio = valid_ratio, scaler = scaler)

    # Thông tin tập dữ liệu
    st.subheader('Tập dữ liệu ' + file_name)
    st.write(df)

    # Vẽ biểu đồ đường cho tập dữ liệu
    st.subheader('Trực quan hóa tập dữ liệu ' + file_name)

    column_names = eda.data_old.columns.tolist()
    selected_column_name = st.selectbox("**Chọn cột:**", column_names)
    fig = MultipleLines.OneLine(eda, selected_column_name)
    st.plotly_chart(fig)

    df_target = df[selected_column_name]
    if option != "Song song":
        # Cho người dùng có thể tự chọn các tham số mình muốn 
        st.session_state.use_custom_params = st.sidebar.checkbox('Nhập các tham số huấn luyện tự chọn:', on_change=ClearCache)
        # Kiểm tra trạng thái của checkbox trong session_state
        if st.session_state.use_custom_params:
            units_cus = st.sidebar.selectbox('Unit:',[16, 32, 64, 128, 256],on_change=ClearCache)
            epoch_cus = st.sidebar.slider('Epoch:', 1, 100, 1, step=1)
            BS_cus = st.sidebar.selectbox('Batch size:',[16, 32, 64, 128, 256],on_change=ClearCache)
            LR_cus = st.sidebar.selectbox('Learning rate:',[0.001,0.0001],on_change=ClearCache)
        # nếu không thì người dùng sẽ optimize để nhận các tham số
        else:
            # nếu mô hình được chọn không phải song song thì hiện nút optimize
            # Optimize Model
            opti = st.sidebar.selectbox('***Optimizer:***',('RandomSearchCV', 'GridSearchCV','BayesSearchcv'),on_change=ClearCache)
            if st.sidebar.button('Optimize Model', type="primary"):
                st.divider()
                st.header("Optimize Mô Hình")
                with st.spinner('Đang tiến hành Optimize...'):
                    start_time = time.time()
                    # Khởi tạo các giá trị Optimize
                    if option == 'Đơn':
                        match mod:
                            case 'CNN':
                                m = KerasRegressor(model = CNN_Model, input_dim=input_dim, output_dim=output_dim, units =32, learning_rate = 0.0001, activation= activation)
                            case 'LSTM':
                                m =  KerasRegressor(model=LSTM_Model, input_dim=input_dim, output_dim=output_dim, units =32, learning_rate = 0.0001, activation= activation)
                            case 'RNN':
                                m =  KerasRegressor(model=RNN_Model, input_dim=input_dim, output_dim=output_dim, units =32, learning_rate = 0.0001, activation= activation)
                            case 'GRU':
                                m =  KerasRegressor(model=GRU_Model, input_dim=input_dim, output_dim=output_dim, units =32, learning_rate = 0.0001, activation= activation) 
                    elif option == 'Tuần tự':
                        match mod:
                            case 'CNN-LSTM':
                                m =  KerasRegressor(model=CNN_LSTM_Model, input_dim=input_dim, output_dim=output_dim, units =32, learning_rate = 0.0001, activation= activation) 
                            case 'LSTM-CNN':
                                m =  KerasRegressor(model=LSTM_CNN_Model, input_dim=input_dim, output_dim=output_dim, units =32, learning_rate = 0.0001, activation= activation)
                            case 'CNN-GRU':
                                m =  KerasRegressor(model=CNN_GRU_Model, input_dim=input_dim, output_dim=output_dim, units =32, learning_rate = 0.0001, activation= activation)
                            case 'GRU-CNN':
                                m =  KerasRegressor(model=GRU_CNN_Model, input_dim=input_dim, output_dim=output_dim, units =32, learning_rate = 0.0001, activation= activation)
                            case 'CNN-RNN':
                                m =  KerasRegressor(model=CNN_RNN_Model, input_dim=input_dim, output_dim=output_dim, units =32, learning_rate = 0.0001, activation= activation)
                            case 'RNN-CNN':
                                m =  KerasRegressor(model=RNN_CNN_Model, input_dim=input_dim, output_dim=output_dim, units =32, learning_rate = 0.0001, activation= activation)
                            case 'LSTM-RNN':
                                m =  KerasRegressor(model=LSTM_RNN_Model, input_dim=input_dim, output_dim=output_dim, units =32, learning_rate = 0.0001, activation= activation)
                            case 'RNN-LSTM':
                                m =  KerasRegressor(model=RNN_LSTM_Model, input_dim=input_dim, output_dim=output_dim, units =32, learning_rate = 0.0001, activation= activation)
                            case 'LSTM-GRU':
                                m =  KerasRegressor(model=LSTM_GRU_Model, input_dim=input_dim, output_dim=output_dim, units =32, learning_rate = 0.0001, activation= activation)
                            case 'GRU-LSTM':
                                m =  KerasRegressor(model=GRU_LSTM_Model, input_dim=input_dim, output_dim=output_dim, units =32, learning_rate = 0.0001, activation= activation)
                            case 'RNN-GRU':
                                m =  KerasRegressor(model=RNN_GRU_Model, input_dim=input_dim, output_dim=output_dim, units =32, learning_rate = 0.0001, activation= activation)
                            case 'GRU-RNN':
                                m =  KerasRegressor(model=GRU_RNN_Model, input_dim=input_dim, output_dim=output_dim, units =32, learning_rate = 0.0001, activation= activation)
                    else:
                        match mod:
                            case 'CNN':
                                m = KerasRegressor(model = CNN_Model, input_dim=input_dim, output_dim=output_dim, units =32, learning_rate = 0.0001, activation= activation)
                            case 'LSTM':
                                m =  KerasRegressor(model=LSTM_Model, input_dim=input_dim, output_dim=output_dim, units =32, learning_rate = 0.0001, activation= activation)
                            case 'RNN':
                                m =  KerasRegressor(model=RNN_Model, input_dim=input_dim, output_dim=output_dim, units =32, learning_rate = 0.0001, activation= activation)
                            case 'GRU':
                                m =  KerasRegressor(model=GRU_Model, input_dim=input_dim, output_dim=output_dim, units =32, learning_rate = 0.0001, activation= activation) 
                    
                    match opti:
                        case 'RandomSearchCV':     
                            # Các model được lựa chọn bởi người dùng
                            # Thực hiện quá trình Optimizer bằng phương pháp RandomizedSearchCV
                            param_ran = {
                                'units': [16, 32, 64, 128, 256],
                                'epochs': range(1, 101),
                                'batch_size': [16, 32, 64, 128, 256],
                                'learning_rate': [0.0001, 0.001]
                                }
                            opti_search = RandomizedSearchCV(m, param_distributions=param_ran, cv=3, n_iter=10, n_jobs=-1, scoring='neg_mean_squared_error')
                        
                        case 'GridSearchCV':
                            param_grid = {
                                'units': [32, 64, 128, 256],
                                'epochs': [20, 40, 60, 80, 100],
                                'batch_size': [16, 32, 64, 128],
                                'learning_rate': [0.0001, 0.001]
                                }
                            opti_search = GridSearchCV(m, param_grid=param_grid, n_jobs=-1, cv=3 ,scoring='neg_mean_squared_error', error_score='raise')           
                            
                        case 'BayesSearchcv':
                            param_bay = {
                                'units': Categorical([16, 32, 64, 128, 256]),
                                'epochs': Integer(1, 100),
                                'batch_size': Categorical([16, 32, 64, 128, 256]),
                                'learning_rate': Categorical([0.0001, 0.001])
                                }
                            opti_search =BayesSearchCV(m, search_spaces=param_bay, cv=3, n_iter=10, n_jobs=-1, scoring='neg_mean_squared_error')
 
                    opti_search.fit(eda.X_valid, eda.y_valid)
                    cv_results = opti_search.cv_results_

                    # Lấy lỗi tối ưu của từng bộ tham số
                    mean_test_scores = -1 * cv_results['mean_test_score']
                    params_used = cv_results['params']
                    
                    # Khởi tạo danh sách để lưu trữ giá trị 
                    units_values = []
                    epoch_values = []
                    batch_size_values = []
                    LR_values = []

                    # Trích xuất giá trị 'units' từ mỗi bộ tham số
                    for params in params_used:
                        units_values.append(params['units'])
                        epoch_values.append(params['epochs'])
                        batch_size_values.append(params['batch_size'])
                        LR_values.append(params['learning_rate'])

                    # In các bộ tham số đã dùng để tối ưu
                    st.write("Các thông số đã dùng tối ưu hóa:")
                    assert len(mean_test_scores) == len(params_used)
                    results_df = pd.DataFrame({'units': units_values,\
                                            'epochs': epoch_values,\
                                            'batch_size': batch_size_values,\
                                            'learning_rate': LR_values,\
                                            'Mean Test Score': mean_test_scores})
                    min_index = results_df['Mean Test Score'].idxmin()

                    # Tạo một DataFrame mới để đánh dấu dòng có giá trị nhỏ nhất
                    highligt_df = results_df.style.apply(lambda x: ['background-color: yellow' \
                                                                if x.name == min_index else '' for _ in x], axis=1)
                    st.write(highligt_df)
                    
                    #Lưu tham số sau khi optimize
                    torch.save({
                    'model': opti_search,
                    'best_params':opti_search.best_params_
                    }, "./model/Optimize_Model.pth")

                    #In thời gian optimize
                    optimize_time = "{:.4f}".format((time.time()) - (start_time))
                    st.write(f"Thời gian Optimize {optimize_time}s")
                    st.session_state.optimize_time = optimize_time
                    st.session_state.display_info['best_params'] = opti_search.best_params_
                    st.write("Optimize Complete!")
    #Traing Model        
    if st.sidebar.button('Train Model'):
        st.divider()
        start_time_train = time.time()
        # Kiểm tra tồn tại của checkbox trong session_state
        # Huấn luyện với các tham số tự chọn bởi người dùng
        if option != 'Song song':
            if st.session_state.use_custom_params:
                st.header("Huấn luyện Mô Hình với các tham số tự chọn")
                with st.spinner("Đang huấn luyện mô hình với bộ tham số tự chọn..."):
                    if option == 'Đơn':
                        match mod:
                            case 'CNN':
                                m1 = CNN_Model(input_dim=input_dim, output_dim=output_dim, units = units_cus, learning_rate = LR_cus, activation= activation)
                            case 'LSTM':
                                m1 = LSTM_Model(input_dim=input_dim, output_dim=output_dim, units = units_cus, learning_rate = LR_cus, activation= activation)
                            case 'RNN':
                                m1 = RNN_Model(input_dim=input_dim, output_dim=output_dim, units = units_cus, learning_rate = LR_cus, activation= activation)
                            case 'GRU':
                                m1 = GRU_Model(input_dim=input_dim, output_dim=output_dim, units = units_cus, learning_rate = LR_cus, activation= activation)
                    elif option == 'Tuần tự':
                        match mod:
                            case 'CNN-LSTM':
                                m1 = CNN_LSTM_Model(input_dim=input_dim, output_dim=output_dim, units = units_cus, learning_rate = LR_cus, activation= activation)
                            case 'LSTM-CNN':
                                m1 = LSTM_CNN_Model(input_dim=input_dim, output_dim=output_dim, units = units_cus, learning_rate = LR_cus, activation= activation)
                            case 'CNN-GRU':
                                m1 = CNN_GRU_Model(input_dim=input_dim, output_dim=output_dim, units = units_cus, learning_rate = LR_cus, activation= activation)
                            case 'GRU-CNN':
                                m1 = GRU_CNN_Model(input_dim=input_dim, output_dim=output_dim, units = units_cus, learning_rate = LR_cus, activation= activation)
                            case 'CNN-RNN':
                                m1 = CNN_RNN_Model(input_dim=input_dim, output_dim=output_dim, units = units_cus, learning_rate = LR_cus, activation= activation)
                            case 'RNN-CNN':
                                m1 = RNN_CNN_Model(input_dim=input_dim, output_dim=output_dim, units = units_cus, learning_rate = LR_cus, activation= activation)
                            case 'LSTM-RNN':
                                m1 = LSTM_RNN_Model(input_dim=input_dim, output_dim=output_dim, units = units_cus, learning_rate = LR_cus, activation= activation)
                            case 'RNN-LSTM':
                                m1 = RNN_LSTM_Model(input_dim=input_dim, output_dim=output_dim, units = units_cus, learning_rate = LR_cus, activation= activation)
                            case 'LSTM-GRU':
                                m1 = LSTM_GRU_Model(input_dim=input_dim, output_dim=output_dim, units = units_cus, learning_rate = LR_cus, activation= activation)
                            case 'GRU-LSTM':
                                m1 = GRU_LSTM_Model(input_dim=input_dim, output_dim=output_dim, units = units_cus, learning_rate = LR_cus, activation= activation)
                            case 'RNN-GRU':
                                m1 = RNN_GRU_Model(input_dim=input_dim, output_dim=output_dim, units = units_cus, learning_rate = LR_cus, activation= activation)
                            case 'GRU-RNN':
                                m1 = GRU_RNN_Model(input_dim=input_dim, output_dim=output_dim, units = units_cus, learning_rate = LR_cus, activation= activation)
                    else:
                        match mod:
                            case 'CNN':
                                m1 = CNN_Model(input_dim=input_dim, output_dim=output_dim, units = units_cus, learning_rate = LR_cus, activation= activation)
                            case 'RNN':
                                m1 = RNN_Model(input_dim=input_dim, output_dim=output_dim, units = units_cus, learning_rate = LR_cus, activation= activation)
                            case 'GRU':
                                m1 = GRU_Model(input_dim=input_dim, output_dim=output_dim, units = units_cus, learning_rate = LR_cus, activation= activation)
                            case 'LSTM':
                                m1 = LSTM_Model(input_dim=input_dim, output_dim=output_dim, units = units_cus, learning_rate = LR_cus, activation= activation)          
                        match mod1:
                            case 'CNN':
                                m2 = CNN_Model(input_dim=input_dim, output_dim=output_dim, units = units_cus, learning_rate = LR_cus, activation= activation)
                            case 'RNN':
                                m2 = RNN_Model(input_dim=input_dim, output_dim=output_dim, units = units_cus, learning_rate = LR_cus, activation= activation)
                            case 'GRU':
                                m2 = GRU_Model(input_dim=input_dim, output_dim=output_dim, units = units_cus, learning_rate = LR_cus, activation= activation)
                            case 'LSTM':
                                m2 = LSTM_Model(input_dim=input_dim, output_dim=output_dim, units = units_cus, learning_rate = LR_cus, activation= activation)
            # Nếu không tồn tại trong session_state thì thực hiện huấn luyện với bộ tham số được lưu trong file Optimize_Model.pth
            else:
                st.header("Huấn luyện Mô Hình")
                st.subheader('Mô hình đã optimize')
                #Load siêu tham số sau khi optimize
                model_op = torch.load("./model/Optimize_Model.pth")
                st.session_state.display_info = model_op['best_params']
                st.write(st.session_state.display_info)
                
                with st.spinner("Đang huấn luyện mô hình với bộ siêu tham số..."):
                    # Lấy bộ tham số tốt nhất từ quá trình optimize
                    best_params =  model_op['best_params']
                    if option == 'Đơn':
                        match mod:
                            case 'CNN':
                                m1 = CNN_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)
                            case 'LSTM':
                                m1 = LSTM_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)
                            case 'RNN':
                                m1 = RNN_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)
                            case 'GRU':
                                m1 = GRU_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)
                    elif option == 'Tuần tự':
                        match mod:
                            case 'CNN-LSTM':
                                m1 = CNN_LSTM_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)
                            case 'LSTM-CNN':
                                m1 = LSTM_CNN_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)
                            case 'CNN-GRU':
                                m1 = CNN_GRU_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)
                            case 'GRU-CNN':
                                m1 = GRU_CNN_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)
                            case 'CNN-RNN':
                                m1 = CNN_RNN_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)
                            case 'RNN-CNN':
                                m1 = RNN_CNN_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)
                            case 'LSTM-RNN':
                                m1 = LSTM_RNN_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)
                            case 'RNN-LSTM':
                                m1 = RNN_LSTM_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)
                            case 'LSTM-GRU':
                                m1 = LSTM_GRU_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)
                            case 'GRU-LSTM':
                                m1 = GRU_LSTM_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)
                            case 'RNN-GRU':
                                m1 = RNN_GRU_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)
                            case 'GRU-RNN':
                                m1 = GRU_RNN_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)
                    else:
                        match mod:
                            case 'CNN':
                                m1 = CNN_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)
                            case 'RNN':
                                m1 = RNN_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)
                            case 'GRU':
                                m1 = GRU_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)
                            case 'LSTM':
                                m1 = LSTM_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)          
                        match mod1:
                            case 'CNN':
                                m2 = CNN_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)
                            case 'RNN':
                                m2 = RNN_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)
                            case 'GRU':
                                m2 = GRU_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)
                            case 'LSTM':
                                m2 = LSTM_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)

        #Tiến hành training

        if option == 'Song song':
            checkpoint = torch.load("./model/"+ mod +".pth")
            model_training = checkpoint["model"]
            unit_training = checkpoint["units"]
            epochs_traning = checkpoint['epochs']
            batch_size_training = checkpoint['batch_size']
            LR_training = checkpoint['learning_rate']
            time_traing = checkpoint['time_train']
            checkpoint_2 = torch.load("./model/"+ mod1 +".pth")
            model_training_2 = checkpoint_2["model"]
            unit_training_2 = checkpoint_2["units"]
            epochs_traning_2 = checkpoint_2['epochs']
            batch_size_training_2 = checkpoint_2['batch_size']
            LR_training_2 = checkpoint_2['learning_rate']
            time_traing_2 = checkpoint_2['time_train']
        else:
            if st.session_state.use_custom_params:
                if option == 'Tuần tự nhân':
                    model_training = eda.train_model(m1,epochs=epoch_cus, batch_size=BS_cus)
                    predict_train, actual_train, index_train, predict_scale_1_train, actua_scale_train = eda.TestingModel(model_training)
                    model_training_2 = eda.train_model_seq(m2, (actual_train/predict_train),epochs=epoch_cus, batch_size=BS_cus)
                elif option == 'Tuần tự cộng':
                    model_training = eda.train_model(m1,epochs=epoch_cus, batch_size=BS_cus)
                    predict_train, actual_train, index_train, predict_scale_1_train, actua_scale_train = eda.TestingModel(model_training)
                    model_training_2 = eda.train_model_seq(m2, (actual_train-predict_train),epochs=epoch_cus, batch_size=BS_cus)
                else:
                    model_training = eda.train_model(m1,epochs=epoch_cus, batch_size=BS_cus)
            else:
                if option == 'Tuần tự nhân':
                    model_training = eda.train_model(m1,epochs=best_params['epochs'], batch_size=best_params['batch_size'])
                    predict_train, actual_train, index_train, predict_scale_1_train, actua_scale_train = eda.TestingModel(model_training)
                    model_training_2 = eda.train_model_seq(m2, (actual_train/predict_train),epochs=best_params['epochs'], batch_size=best_params['batch_size'])
                elif option == 'Tuần tự cộng':
                    model_training = eda.train_model(m1,epochs=best_params['epochs'], batch_size=best_params['batch_size'])
                    predict_train, actual_train, index_train, predict_scale_1_train, actua_scale_train = eda.TestingModel(model_training)
                    model_training_2 = eda.train_model_seq(m2, (actual_train-predict_train),epochs=best_params['epochs'], batch_size=best_params['batch_size'])
                else:
                    model_training = eda.train_model(m1,epochs=best_params['epochs'], batch_size=best_params['batch_size'])
        

        train_time = "{:.4f}".format((time.time()) - (start_time_train))
        st.write(f"Thời gian Training {train_time}s")
        st.session_state.train_time = train_time
        st.write("Training Complete!")

        if option != "Song song":
            if st.session_state.use_custom_params:
                unit_train = units_cus
                BS_train = BS_cus
                epoch_train = epoch_cus
                LR_train = LR_cus
            else:
                    unit_train = best_params['units']
                    BS_train = best_params['batch_size']
                    epoch_train = best_params['epochs']
                    LR_train = best_params['learning_rate']

        if option == 'Đơn' or option == 'Tuần tự':
            torch.save({
            'model': model_training,
            'time_train': train_time,
            'units': unit_train,
            'epochs': epoch_train,
            'batch_size': BS_train,
            'learning_rate': LR_train,
            }, "./model/"+ mod +".pth")
        elif option == 'Song song':
            torch.save({
            'model': model_training,
            'model_2': model_training_2,
            'time_train': train_time + time_traing + time_traing_2,
            'units_1': unit_training,
            'epochs_1': epochs_traning,
            'batch_size_1': batch_size_training,
            'learning_rate_1': LR_training,
            'time_train_1': time_traing,
            'units_2': unit_training_2,
            'epochs_2': epochs_traning_2,
            'batch_size_2': batch_size_training_2,
            'learning_rate_2': LR_training_2,
            'time_train_2': time_traing_2,
            }, "./model/"+ mod + "-"+ mod1 + "_para"+".pth")
        elif option == 'Tuần tự cộng':
            torch.save({
            'model': model_training,
            'model_2': model_training_2,
            'time_train': train_time,
            'units': unit_train,
            'epochs': epoch_train,
            'batch_size': BS_train,
            'learning_rate': LR_train,
            }, "./model/"+ mod + "-"+ mod1 + "_add"+".pth")
        elif option == 'Tuần tự nhân':
            torch.save({
            'model': model_training,
            'model_2': model_training_2,
            'time_train': train_time,
            'units': unit_train,
            'epochs': epoch_train,
            'batch_size': BS_train,
            'learning_rate': LR_train,
            }, "./model/"+ mod + "-"+ mod1 + "_mul"+".pth")


        #In trọng số từng lớp
        if model_training is not None:
            st.write("Trọng số các lớp mô hình " + mod)
            PrintWeight(model_training)
        if option == "Tuần tự nhân" or option == "Song song" or option == "Tuần tự cộng":
            if model_training_2 is not None:
                st.write("Trọng số các lớp mô hình " + mod1)
                PrintWeight(model_training_2)

    if st.sidebar.button('Re-Train Model'):
        st.divider()
        start_time_retrain = time.time()
        st.header("Tái huấn luyện Mô Hình")
        if option != 'Song song':
            if option == 'Đơn' or option == 'Tuần tự':
                # Load các paramter được lưu 
                checkpoint = torch.load("./model/"+ mod +".pth")
            elif option == 'Tuần tự cộng':
                checkpoint = torch.load("./model/"+ mod + "-"+ mod1 + "_add"+".pth")
            elif option == 'Tuần tự nhân':
                checkpoint = torch.load("./model/"+ mod + "-"+ mod1 + "_mul"+".pth")

            unit_retrain = checkpoint["units"]
            epoch_retrain = checkpoint["epochs"]
            batch_size_retrain = checkpoint["batch_size"]
            LR_retrain = checkpoint["learning_rate"]

            st.subheader('Bộ tham số đã dùng:')
            
            re_info_table = pd.DataFrame(
                {"units": [unit_retrain],"epochs": [epoch_retrain], "batch_zize": [batch_size_retrain], "learning_rate": [LR_retrain]})
            st.table(re_info_table[:10])

        with st.spinner("Đang tái huấn luyện mô hình với bộ siêu tham số đã lưu..."):
            if option == 'Đơn':
                match mod:
                    case 'CNN':
                        m1 = CNN_Model(input_dim=input_dim, output_dim=output_dim, units = unit_retrain, learning_rate = LR_retrain, activation= activation)
                    case 'LSTM':
                        m1 = LSTM_Model(input_dim=input_dim, output_dim=output_dim, units = unit_retrain, learning_rate = LR_retrain, activation= activation)
                    case 'RNN':
                        m1 = RNN_Model(input_dim=input_dim, output_dim=output_dim, units = unit_retrain, learning_rate = LR_retrain, activation= activation)
                    case 'GRU':
                        m1 = GRU_Model(input_dim=input_dim, output_dim=output_dim, units = unit_retrain, learning_rate = LR_retrain, activation= activation)
            elif option == 'Tuần tự':
                match mod:
                    case 'CNN-LSTM':
                        m1 = CNN_LSTM_Model(input_dim=input_dim, output_dim=output_dim, units = unit_retrain, learning_rate = LR_retrain, activation= activation)
                    case 'LSTM-CNN':
                        m1 = LSTM_CNN_Model(input_dim=input_dim, output_dim=output_dim, units = unit_retrain, learning_rate = LR_retrain, activation= activation)
                    case 'CNN-GRU':
                        m1 = CNN_GRU_Model(input_dim=input_dim, output_dim=output_dim, units = unit_retrain, learning_rate = LR_retrain, activation= activation)
                    case 'GRU-CNN':
                        m1 = GRU_CNN_Model(input_dim=input_dim, output_dim=output_dim, units = unit_retrain, learning_rate = LR_retrain, activation= activation)
                    case 'CNN-RNN':
                        m1 = CNN_RNN_Model(input_dim=input_dim, output_dim=output_dim, units = unit_retrain, learning_rate = LR_retrain, activation= activation)
                    case 'RNN-CNN':
                        m1 = RNN_CNN_Model(input_dim=input_dim, output_dim=output_dim, units = unit_retrain, learning_rate = LR_retrain, activation= activation)
                    case 'LSTM-RNN':
                        m1 = LSTM_RNN_Model(input_dim=input_dim, output_dim=output_dim, units = unit_retrain, learning_rate = LR_retrain, activation= activation)
                    case 'RNN-LSTM':
                        m1 = RNN_LSTM_Model(input_dim=input_dim, output_dim=output_dim, units = unit_retrain, learning_rate = LR_retrain, activation= activation)
                    case 'LSTM-GRU':
                        m1 = LSTM_GRU_Model(input_dim=input_dim, output_dim=output_dim,units = unit_retrain, learning_rate = LR_retrain, activation= activation)
                    case 'GRU-LSTM':
                        m1 = GRU_LSTM_Model(input_dim=input_dim, output_dim=output_dim, units = unit_retrain, learning_rate = LR_retrain, activation= activation)
                    case 'RNN-GRU':
                        m1 = RNN_GRU_Model(input_dim=input_dim, output_dim=output_dim, units = unit_retrain, learning_rate = LR_retrain, activation= activation)
                    case 'GRU-RNN':
                        m1 = GRU_RNN_Model(input_dim=input_dim, output_dim=output_dim, units = unit_retrain, learning_rate = LR_retrain, activation= activation)
            elif option == 'Tuần tự cộng' or option == 'Tuần tự nhân':
                match mod:
                    case 'CNN':
                        m1 = CNN_Model(input_dim=input_dim, output_dim=output_dim, units = unit_retrain, learning_rate = LR_retrain, activation= activation)
                    case 'RNN':
                        m1 = RNN_Model(input_dim=input_dim, output_dim=output_dim, units = unit_retrain, learning_rate = LR_retrain, activation= activation)
                    case 'GRU':
                        m1 = GRU_Model(input_dim=input_dim, output_dim=output_dim, units = unit_retrain, learning_rate = LR_retrain, activation= activation)
                    case 'LSTM':
                        m1 = LSTM_Model(input_dim=input_dim, output_dim=output_dim, units = unit_retrain, learning_rate = LR_retrain, activation= activation)          
                match mod1:
                    case 'CNN':
                        m2 = CNN_Model(input_dim=input_dim, output_dim=output_dim, units = unit_retrain, learning_rate = LR_retrain, activation= activation)
                    case 'RNN':
                        m2 = RNN_Model(input_dim=input_dim, output_dim=output_dim, units = unit_retrain, learning_rate = LR_retrain, activation= activation)
                    case 'GRU':
                        m2 = GRU_Model(input_dim=input_dim, output_dim=output_dim, units = unit_retrain, learning_rate = LR_retrain, activation= activation)
                    case 'LSTM':
                        m2 = LSTM_Model(input_dim=input_dim, output_dim=output_dim, units = unit_retrain, learning_rate = LR_retrain, activation= activation)
                                
        #Tiến hành training
            if option == 'Song song':
                checkpoint = torch.load("./model/"+ mod +".pth")
                model_training = checkpoint["model"]
                unit_training = checkpoint["units"]
                epochs_traning = checkpoint['epochs']
                batch_size_training = checkpoint['batch_size']
                LR_training = checkpoint['learning_rate']
                time_traing = checkpoint['time_train']
                checkpoint_2 = torch.load("./model/"+ mod1 +".pth")
                model_training_2 = checkpoint_2["model"]
                unit_training_2 = checkpoint_2["units"]
                epochs_traning_2 = checkpoint_2['epochs']
                batch_size_training_2 = checkpoint_2['batch_size']
                LR_training_2 = checkpoint_2['learning_rate']
                time_traing_2 = checkpoint_2['time_train']
            elif option == 'Tuần tự nhân':
                model_training = eda.train_model(m1,epochs=epoch_retrain, batch_size=batch_size_retrain)
                predict_train, actual_train, index_train, predict_scale_1_train, actua_scale_train = eda.TestingModel(model_training)
                model_training_2 = eda.train_model_seq(m2, (actual_train/predict_train),epochs=epoch_retrain, batch_size=batch_size_retrain)
            elif option == 'Tuần tự cộng':
                model_training = eda.train_model(m1,epochs=epoch_retrain, batch_size=batch_size_retrain)
                predict_train, actual_train, index_train, predict_scale_1_train, actua_scale_train = eda.TestingModel(model_training)
                model_training_2 = eda.train_model_seq(m2, (actual_train-predict_train),epochs=epoch_retrain, batch_size=batch_size_retrain)
            else:
                model_training = eda.train_model(m1,epochs=epoch_retrain, batch_size=batch_size_retrain)


            Rtrain_time = "{:.4f}".format((time.time()) - (start_time_retrain))
            st.write(f"Thời gian Re-Training {Rtrain_time}s")
            st.session_state.Rtrain_time = Rtrain_time
            st.write("Re-Train Complete!")

            if option == 'Đơn' or option == 'Tuần tự':
                torch.save({
                'model': model_training,
                'time_train': Rtrain_time,
                'units': unit_retrain,
                'epochs': epoch_retrain,
                'batch_size': batch_size_retrain,
                'learning_rate': LR_retrain,
                }, "./model/"+ mod +".pth")
            elif option == 'Song song':
                torch.save({
                'model': model_training,
                'model_2': model_training_2,
                'time_train': Rtrain_time+time_traing_2+time_traing,
                'units_1': unit_training,
                'epochs_1': epochs_traning,
                'batch_size_1': batch_size_training,
                'learning_rate_1': LR_training,
                'time_train_1': time_traing,
                'units_2': unit_training_2,
                'epochs_2': epochs_traning_2,
                'batch_size_2': batch_size_training_2,
                'learning_rate_2': LR_training_2,
                'time_train_2': time_traing_2
                }, "./model/"+ mod + "-"+ mod1 + "_para"+".pth")
            elif option == 'Tuần tự cộng':
                torch.save({
                'model': model_training,
                'model_2': model_training_2,
                'time_train': Rtrain_time,
                'units': unit_retrain,
                'epochs': epoch_retrain,
                'batch_size': batch_size_retrain,
                'learning_rate': LR_retrain,
                }, "./model/"+ mod + "-"+ mod1 + "_add"+".pth")
            elif option == 'Tuần tự nhân':
                torch.save({
                'model': model_training,
                'model_2': model_training_2,
                'time_train': Rtrain_time,
                'units': unit_retrain,
                'epochs': epoch_retrain,
                'batch_size': batch_size_retrain,
                'learning_rate': LR_retrain,
                }, "./model/"+ mod + "-"+ mod1 + "_mul"+".pth")


            #In trọng số từng lớp
            if model_training is not None:
                st.write("Trọng số các lớp mô hình " + mod)
                PrintWeight(model_training)
            if option == "Tuần tự nhân" or option == "Song song" or option == "Tuần tự cộng":
                if model_training_2 is not None:
                    st.write("Trọng số các lớp mô hình " + mod1)
                    PrintWeight(model_training_2)


#Load tập dữ liệu test
st.header("Chọn tập dữ liệu tiến hành dự đoán")
uploaded_file1 = st.file_uploader(
"Chọn tệp dữ liệu test", type=["csv"],on_change=ClearCache)

# Nếu đã upload file
if uploaded_file1 is not None:
    file_name_test = uploaded_file1.name
    df_test = LoadData(uploaded_file1)
    
    # #Chọn cột để dự đoán
    selected_predict_column_name_test = st.sidebar.selectbox(
        '**Chọn cột để dự đoán:**', tuple(df_test.drop(df_test.columns[0],axis = 1).columns.values), on_change=ClearCache)
    

    # Tạo đối tượng EDA
    eda = EDA(df = df_test, n_steps_in = input_dim, n_steps_out = output_dim, feature=selected_predict_column_name_test, train_ratio = 0, valid_ratio = 0, scaler = scaler)
    # Thông tin tập dữ liệu
    st.subheader('Tập dữ liệu test ' + file_name_test)
    st.write(df_test)

    # Vẽ biểu đồ đường cho tập dữ liệu
    st.subheader('Trực quan hóa tập dữ liệu ' + file_name_test)

    column_names_test = eda.data_old.columns.tolist()
    selected_column_name_test = st.selectbox("**Chọn cột vẽ biểu đồ:**", column_names_test)
    fig_test = MultipleLines.OneLine(eda, selected_column_name_test)
    st.plotly_chart(fig_test)

    #tên cột và kiểu dữ liệu của tập test
    column_type_test = eda.data_old.dtypes.tolist()
    table_info_test = pd.DataFrame({"Tên cột":column_names_test,"Kiểu dữ liệu":column_type_test})

    if st.sidebar.button('Xuất tất cả model train trên cột muốn test', type="primary"):
        with st.spinner('Vui lòng chờ trong giây lát...'):
            try:
                file_name = 'Result-all-test.xlsx'

                # Save file to output directory
                file_path = save_file_to_output_folder(dfs_tabs_all(file_name_test,df_test, input_dim, output_dim, selected_predict_column_name_test, scaler, table_info_test), file_name)
                st.write("Xuất file thành công!!")
            except:
                st.subheader('Vui lòng huấn luyện đủ mô hình để có thể xuất file!!')
                model_im = pd.DataFrame({
                    "model_1":["CNN","CNN-LSTM","CNN-RNN","CNN-GRU","CNN-LSTM_add","CNN-RNN_add","CNN-GRU_add","CNN-LSTM_mul","CNN-RNN_mul","CNN-GRU_mul", "CNN-RNN_para", "CNN-LSTM_para"]\
                        ,"model_2":["RNN","RNN-LSTM","RNN-CNN","RNN-GRU","RNN-LSTM_add","RNN-RNN_add","RNN-GRU_add", "RNN-LSTM_mul","RNN-RNN_mul","RNN-GRU_mul","CNN-GRU_para", "LSTM_RNN_para"]\
                        ,"model_3":["LSTM", "LSTM-CNN","LSTM-RNN","LSTM-GRU", "LSTM-LSTM_add","LSTM-RNN_add","LSTM-GRU_add","LSTM-LSTM_mul","LSTM-RNN_mul","LSTM-GRU_mul","LSTM-GRU_para",""]\
                        ,"model_4":["GRU","GRU-CNN","GRU-RNN","GRU-LSTM", "GRU-GRU_add","GRU-RNN_add","GRU-LSTM_add", "GRU-GRU_mul","GRU-RNN_mul","GRU-LSTM_mul","RNN-GRU_para",""]
                })
                st.write(model_im)
 

    #Thực hiện nút test model
    st.sidebar.button('Test Model', type="primary", on_click= click_button_train)   
    if st.session_state.clicked_train:
        #try:
            start_time_test = time.time()
            if option == 'Đơn' or option == 'Tuần tự':
                # Load các paramter được lưu trong CNN_Model.pth
                checkpoint = torch.load("./model/"+ mod +".pth")
                model_train = checkpoint["model"]
                unit_train = checkpoint["units"]
                epoch_train = checkpoint["epochs"]
                batch_size_train = checkpoint["batch_size"]
                LR_train = checkpoint["learning_rate"]
                time_train = checkpoint["time_train"]  

            elif option == 'Song song':
                checkpoint = torch.load("./model/"+ mod + "-"+ mod1 + "_para"+".pth")
                model_train = checkpoint["model"]
                model_train_2 = checkpoint["model_2"]
                unit_train = checkpoint["units_1"]
                epoch_train = checkpoint["epochs_1"]
                batch_size_train = checkpoint["batch_size_1"]
                LR_train = checkpoint["learning_rate_1"]
                unit_train_2 = checkpoint["units_2"]
                epoch_train_2 = checkpoint["epochs_2"]
                batch_size_train_2 = checkpoint["batch_size_2"]
                LR_train_2 = checkpoint["learning_rate_2"]
                time_train = checkpoint["time_train"]
                time_train_1 = checkpoint['time_train_1']
                time_train_2 = checkpoint['time_train_2']

            elif option == 'Tuần tự cộng':
                checkpoint = torch.load("./model/"+ mod + "-"+ mod1 + "_add"+".pth")
                model_train = checkpoint["model"]
                model_train_2 = checkpoint["model_2"]
                unit_train = checkpoint["units"]
                epoch_train = checkpoint["epochs"]
                batch_size_train = checkpoint["batch_size"]
                LR_train = checkpoint["learning_rate"]
                time_train = checkpoint["time_train"]

            elif option == 'Tuần tự nhân':
                checkpoint = torch.load("./model/"+ mod + "-"+ mod1 + "_mul"+".pth")
                model_train = checkpoint["model"]
                model_train_2 = checkpoint["model_2"]
                unit_train = checkpoint["units"]
                epoch_train = checkpoint["epochs"]
                batch_size_train = checkpoint["batch_size"]
                LR_train = checkpoint["learning_rate"]
                time_train = checkpoint["time_train"]

            if model_train is not None:
                st.write("Trọng số các lớp mô hình " + mod)
                PrintWeight(model_train)
            if option == "Tuần tự nhân" or option =="Song song" or option == "Tuần tự cộng":
                if model_train_2 is not None:
                    st.write("Trọng số các lớp mô hình " + mod1)
                    PrintWeight(model_train_2)

            # Thể hiện các giá trị đã train lên bảng và dùng để test
            st.write("****Các siêu tham số được dùng để dự đoán:****")
            if option != 'Song song':
                train_table = pd.DataFrame(
                    {"units": [unit_train],"epochs": [epoch_train], "batch_zize": [batch_size_train], "learning_rate": [LR_train],"time train (s)": [time_train]})
                st.table(train_table[:10])
            else:
                train_table = pd.DataFrame(
                    {"units": [unit_train,unit_train_2],"epochs": [epoch_train, epoch_train_2], "batch_zize": [batch_size_train,batch_size_train_2], "learning_rate": [LR_train,LR_train_2],"time train (s)": [time_train_1,time_train_2]})
                st.table(train_table[:10])

            # Thực hiện test
            if option == 'Song song':
                predict_1, actual_1, index_1, predict_scale_1, actua_scale_1 = eda.TestingModel(model_train)
                predict_2, actual_2, index_2, predict_scale_2, actua_scale_2 = eda.TestingModel(model_train_2)
                actual, index, actua_scale = actual_1, index_1, actua_scale_1
                omega = calculate_omega(actual, predict_1, predict_2)
                predict = omega * predict_1 + (1 - omega) * predict_2
                predict_scale = omega * predict_scale_1 + (1 - omega) * predict_scale_2
            elif option == 'Tuần tự cộng':    
                predict_1, actual, index, predict_scale_1, actua_scale = eda.TestingModel(model_train)
                test_model = (actual-predict_1)
                predict_2 = eda.TestingModelSeq(model_train_2, test_model)
                predict = predict_2 + predict_1
                y_scaler = load(open('./static/y_scaler.pkl', 'rb'))
                predict_scale = y_scaler.fit_transform(predict)
            elif option == 'Tuần tự nhân':
                predict_1, actual, index, predict_scale_1, actua_scale = eda.TestingModel(model_train)
                test_model = (actual/predict_1)
                predict_2  = eda.TestingModelSeq(model_train_2, test_model)
                predict = predict_2 * predict_1
                y_scaler = load(open('./static/y_scaler.pkl', 'rb'))
                predict_scale = y_scaler.fit_transform(predict)
            else:
                predict, actual, index, predict_scale, actua_scale = eda.TestingModel(model_train)

            #In thời gian 
            test_time = "{:.4f}".format((time.time()) - (start_time_test))
            st.write(f"Thời gian thực thi {test_time}s")

            # Tính lỗi của tập dữ liệu và in ra màn hình 
            mae, mse, rmse, mape, cv_rmse = Score(predict_scale,actua_scale)
            metrics = pd.DataFrame({
                "MAE": [mae],
                "MSE": [mse],
                "RMSE": [rmse],
                "CV_RMSE": [cv_rmse]})
            st.write("****Thông số lỗi sau khi dự đoán:****")
            st.table(metrics)
            st.write("****So sánh kết quả dự đoán và thực tế:****")
            #Tính lỗi trên từng datapoint để xuất ra exel 
            mse_test = (predict_scale-actua_scale)**2
            rmse_test=np.sqrt(mse_test)
            mae_test = np.abs(predict_scale-actua_scale)
            cvrmse_test = rmse_test/np.mean(predict_scale)

            # Kiểm tra kết quả dự đoán và thực tế 
            if scaler != "Dữ liệu gốc":
                result_test_table = pd.DataFrame(
                    {"Ngày" : index.tolist(),"Giá trị dự đoán": predict.tolist(), "Giá trị thực": actual.tolist(), "MSE": mse_test.tolist(),"RMSE": rmse_test.tolist(), "MAE": mae_test.tolist(), "CVRMSE": cvrmse_test.tolist()})
            else:
                result_test_table = pd.DataFrame(
                    {"Ngày" : index.tolist(),"Giá trị dự đoán": predict_scale.tolist(), "Giá trị thực": actua_scale.tolist(), "MSE": mse_test.tolist(),"RMSE": rmse_test.tolist(), "MAE": mae_test.tolist(), "CVRMSE": cvrmse_test.tolist()})
            st.session_state.result_test_table = result_test_table
            st.write(result_test_table)  


            # Biểu đồ so sánh
            compare_date = st.selectbox("****Chọn ngày để so sánh kết quả dự đoán****",list(range(1,output_dim+1)))
            mline = MultipleLines.MultipLines(predict[:,compare_date-1], actual[:,compare_date-1], index)
            st.plotly_chart(mline)


            if option == 'Song song' or option == 'Tuần tự cộng' or option == 'Tuần tự nhân':
                result_test_table_2 = pd.DataFrame(
                    {"Ngày" : index.tolist(),"Giá trị dự đoán": predict.tolist(),\
                     "Giá trị dự đoán model 1": predict_1.tolist(), "Giá trị dự đoán model 2": predict_2.tolist(),\
                          "Giá trị thực": actual.tolist(), \
                            "MSE": mse_test.tolist(),"RMSE": rmse_test.tolist(), "MAE": mae_test.tolist(), "CVRMSE": cvrmse_test.tolist()})

                csv_output = [result_test_table_2, table_info_test, metrics, train_table]

                
            else:
                csv_output = [result_test_table, table_info_test, metrics, train_table]

            # list of sheet names
            sheets = ['Result test', 'table info test', 'metrics', 'train parameters']  

            #Download kết quả về file excel
            file_name = 'Result-test.xlsx'

            if st.button('Xuất kết quả', type="primary"):
                # Save file to output directory
                file_path = save_file_to_output_folder(dfs_tabs(csv_output, sheets), file_name)
                st.write("Xuất kết quả thành công!!")
                      
        # except:
        #     st.error("****Hiện tại chưa có Model!****")