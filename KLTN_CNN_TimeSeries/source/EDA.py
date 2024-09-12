import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pickle import dump,load



class EDA:

    def __init__(self, df ,n_steps_in, n_steps_out, feature, train_ratio, valid_ratio , scaler):
          self.data, self.data_old, self.X_train, self.X_test, self.X_valid,\
            self.y_train, self.y_test, self.y_valid,\
              self.index_train, self.index_test, self.index_valid = self.CleanData(df, n_steps_in, n_steps_out, feature, train_ratio, valid_ratio, scaler)

    def CleanData(self, data, n_steps_in = 10, n_steps_out = 1, feature = 'Price',\
                   train_ratio = 0.7, valid_ratio = 0.2, scaler = 'Min-Max'):
        
        # loai bo cac ki tu va dau tren cac bien du lieu
        data = data.dropna()
        column_names = tuple(data.drop(data.columns[0], axis=1).columns.values)
        for column in column_names:
            data[column] = data[column].fillna('0').astype(str).str.replace(',', '')\
                .str.replace('K', 'e3').str.replace('M', 'e6')\
                    .str.replace('%', 'e-2').map(lambda x: pd.eval(x) if x != 'nan' else np.nan).astype(float)

        # Dinh dang cot dau tien theo dang ngay gio va sap xep du lieu
        data[data.columns[0]] = pd.to_datetime(data[data.columns[0]])
        data = data.sort_values(by=data.columns[0])
        data.set_index(data.columns[0], inplace=True)

        # sao luu data truoc khi chia tap du lieu
        data_old = data
        X_value = data[[feature]]
        y_value = data[[feature]]

        # Chuan hoa du lieu
        X_scale_dataset, y_scale_dataset = self.normalize_data(X_value, y_value, scaler)

        # Chia tap X-y
        X, y = self.get_X_y(X_scale_dataset, y_scale_dataset, n_steps_in, n_steps_out)

        # Chia tap train/validation/test
        X_train, X_valid, X_test, = self.split_train_test(X, train_ratio, valid_ratio)
        y_train, y_valid, y_test, = self.split_train_test(y, train_ratio, valid_ratio)

        # Chia index(ngay) train/validation/test
        index_train, index_valid, index_test, = self.predict_index(data, X_train, X_valid, n_steps_in, n_steps_out)

        return data, data_old, X_train, X_test, X_valid, y_train, y_test, \
            y_valid, index_train, index_test, index_valid


    #Chuan hoa du lieu
    def normalize_data(self, X_value, y_value, scaler = 'Min-Max'):
        #chuan hoa min-max
        if scaler == 'Min-Max':
            X_scaler = MinMaxScaler()
            y_scaler = MinMaxScaler()
        # Chuan hoa zero-mean
        elif scaler == 'Zero-Mean':
            X_scaler = StandardScaler()
            y_scaler = StandardScaler()
        # du lieu goc
        else:          
            return X_value.to_numpy(), y_value.to_numpy()

        X_scaler.fit(X_value)
        y_scaler.fit(y_value)

        
        dump(X_scaler, open('./static/X_scaler.pkl', 'wb'))
        dump(y_scaler, open('./static/y_scaler.pkl', 'wb'))
        X_scale_dataset = X_scaler.fit_transform(X_value)
        y_scale_dataset = y_scaler.fit_transform(y_value)

        return X_scale_dataset, y_scale_dataset

    def get_X_y(self, X_data, y_data, n_steps_in, n_steps_out):

        X = list()
        y = list()       
        length = len(X_data)
        # chia dữ liệu vào các mẫu dữ liệu có độ dài nhất định cho việc huấn luyện mô hình
        for i in range(0, length, 1):
            X_value = X_data[i: i + n_steps_in][:, :]
            y_value = y_data[i + n_steps_in: i + (n_steps_in + n_steps_out)][:, 0]
            if len(X_value) == n_steps_in and len(y_value) == n_steps_out:
                X.append(X_value)
                y.append(y_value)

        return np.array(X), np.array(y)


    #Chia tap du lieu train/validation/test
    def split_train_test(self, data, train_ratio = 0.7, valid_ratio = 0.2):

        # tính Độ dài của tập train/valid/test
        total_size = len(data)
        train_size = int(total_size * train_ratio)
        valid_size = int(total_size * valid_ratio)
        test_size = total_size - train_size - valid_size

        # Chia tap du lieu dua tren do dai cua tung tap
        data_train, data_valid, data_test = (
        data[:train_size, :],
        data[train_size:train_size + valid_size, :],
        data[train_size + valid_size: train_size + valid_size + test_size, :])

        return data_train, data_valid, data_test

    #Chia index(ngay) tuong ung voi gia tri cua train/valid/test
    def predict_index(self, dataset, X_train, X_valid, n_steps_in, n_steps_out):

        #Gia tri index train
        train_predict_index = dataset.iloc[n_steps_in: X_train.shape[0] + n_steps_in + n_steps_out - 1, :].index

        #Gia tri index valid
        valid_start_index = X_train.shape[0] + n_steps_in + n_steps_out # Define the starting index for the validation set
        valid_end_index = valid_start_index + X_valid.shape[0] -1  # Define the ending index for the validation set
        valid_predict_index = dataset.iloc[valid_start_index:valid_end_index, :].index
        
        #Gia tri index test
        test_start_index = valid_end_index  # Define the starting index for the test set
        test_predict_index = dataset.iloc[test_start_index:, :].index

        return train_predict_index, valid_predict_index, test_predict_index

    #Test model
    def TestingModelSeq(self, model, mse_test_model): 

        # Kiểm thử trên tập giá trị test và trả về kết quả dự đoán - thực tế
        predictions = model.predict(mse_test_model)
        # dự đoán đã được chuyển đổi về đơn vị ban đầu, dữ liệu thực tế đã được chuyển đổi về đơn vị ban đầu, 
        # các chỉ mục của dữ liệu kiểm tra, kết quả dự đoán trước khi chuyển đổi, dữ liệu thực tế trước khi chuyển đổi.
        return predictions

    def TestingModel(self, model): 

        # Kiểm thử trên tập giá trị test và trả về kết quả dự đoán - thực tế
        predictions = model.predict(self.X_test, verbose=0)

        y_scaler = load(open('./static/y_scaler.pkl', 'rb'))
        rescaled_real_y = y_scaler.inverse_transform(self.y_test)
        rescaled_predicted_y = y_scaler.inverse_transform(predictions)

        # dự đoán đã được chuyển đổi về đơn vị ban đầu, dữ liệu thực tế đã được chuyển đổi về đơn vị ban đầu, 
        # các chỉ mục của dữ liệu kiểm tra, kết quả dự đoán trước khi chuyển đổi, dữ liệu thực tế trước khi chuyển đổi.
        return rescaled_predicted_y, rescaled_real_y, self.index_test, predictions, self.y_test
    
    #Train model
    def train_model(self, model , epochs, batch_size):
        model.fit(self.X_train, self.y_train, epochs=epochs, \
                  batch_size=batch_size, validation_data=(self.X_valid, self.y_valid),
                verbose=2, shuffle=False)
        return model
    
    def train_model_seq(self, model, predict_1, epochs, batch_size):
        model.fit(predict_1, predict_1, epochs=epochs, \
                  batch_size=batch_size, verbose=2, shuffle=False)
        return model
    