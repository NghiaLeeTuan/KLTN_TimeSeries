import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
import torch
from EDA import EDA
import time
from pickle import load

class Excel_Output:

    def calculate_omega(self, y, y1, y2):
        numerator = np.sum((y2 - y1) * (y - y1))
        denominator = np.sum((y2 - y1) ** 2)
        if denominator != 0:
            omega = numerator / denominator  
        else: 0
        # Ensure w_parallel is between 0 and 1
        omega = max(0, min(omega, 1))
        return omega

    def CV_RMSE(self, predict, actual):
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
    def Score(self, predict, actual):
        mae = mean_absolute_error(actual, predict)
        mse = mean_squared_error(actual, predict)
        rmse = np.sqrt(mse)
        cv_rmse = self.CV_RMSE(predict,actual)
        return mae, mse, rmse ,cv_rmse

    def LoadModel(self, df_test, input_dim, output_dim, selected_predict_column_name_test, scaler):
        eda = EDA(df = df_test, n_steps_in = input_dim, n_steps_out = output_dim, \
                  feature=selected_predict_column_name_test, train_ratio = 0, valid_ratio = 0, scaler = scaler)

        mod1 = ['CNN','LSTM','RNN','GRU']
        mod2 = ['LSTM','RNN','GRU']

        name_model = ['CNN','LSTM','RNN','GRU']        
        error_list = ({
                "MAE": [],
                "MSE": [],
                "RMSE": [],
                "CV_RMSE": [],
                "train_time (s)": [],
                "test_time (s)": []})
        
        para_list = ({
            "units": [],
            "epochs": [],
            "batch_size": [],
            "learning_rate": []
        })
        result_test_table = pd.DataFrame({"Ngày": eda.index_test.tolist()})

        ##Load model đơn
        for model in mod1:
            start_time_test = time.time()
            checkpoint = torch.load("./model/"+ model+".pth")
            model_train = checkpoint["model"]
            unit_train = checkpoint["units"]
            epoch_train = checkpoint["epochs"]
            batch_size_train = checkpoint["batch_size"]
            LR_train = checkpoint["learning_rate"]
            time_train = checkpoint["time_train"]
            predict, actual, index, predict_scale, actua_scale = eda.TestingModel(model_train)
            mae, mse, rmse, cv_rmse = self.Score(predict_scale,actua_scale)
            mse_test = (predict_scale-actua_scale)**2
            rmse_test=np.sqrt(mse_test)
            mae_test = np.abs(predict_scale-actua_scale)
            cvrmse_test = rmse_test/np.mean(predict_scale)

            predict_test_model = pd.DataFrame(
                    {"Giá trị dự đoán": predict.tolist(), "Giá trị thực": actual.tolist(), \
                     "MSE": mse_test.tolist(),"RMSE": rmse_test.tolist(), "MAE": mae_test.tolist(), "CVRMSE": cvrmse_test.tolist()})
            result_test_table = pd.concat([result_test_table, predict_test_model], axis=1)
            test_time = "{:.2f}".format((time.time()) - (start_time_test))

            new_error = {"MAE": [mae],
                    "MSE": [mse],
                    "RMSE": [rmse],
                    "CV_RMSE": [cv_rmse],
                    "train_time (s)": [time_train],
                    "test_time (s)": [test_time]}

            for key, value in new_error.items():
                error_list[key].extend(value)

            new_para = {"units": [unit_train],
                    "epochs": [epoch_train],
                    "batch_size": [batch_size_train],
                    "learning_rate": [LR_train]}

            for key, value in new_para.items():
                para_list[key].extend(value)
            
        ## Load model tuan tu
        for model_1 in mod1:
            for model_2 in mod1:
                if model_1 != model_2:
                    start_time_test = time.time()
                    checkpoint = torch.load("./model/"+ model_1 + "-"+ model_2 +".pth")
                    time_train = checkpoint["time_train"]
                    name_model.append(''+model_1+'-'+model_2)
                    model_train = checkpoint["model"]
                    unit_train = checkpoint["units"]
                    epoch_train = checkpoint["epochs"]
                    batch_size_train = checkpoint["batch_size"]
                    LR_train = checkpoint["learning_rate"]
                    predict, actual, index, predict_scale, actua_scale = eda.TestingModel(model_train)
                    mae, mse, rmse, cv_rmse = self.Score(predict_scale,actua_scale)
                    test_time = "{:.2f}".format((time.time()) - (start_time_test))

                    mse_test = (predict_scale-actua_scale)**2
                    rmse_test=np.sqrt(mse_test)
                    mae_test = np.abs(predict_scale-actua_scale)
                    cvrmse_test = rmse_test/np.mean(predict_scale)

                    predict_test_model = pd.DataFrame(
                            {"Giá trị dự đoán": predict.tolist(),\
                            "Giá trị thực": actual.tolist(), \
                                "MSE": mse_test.tolist(),"RMSE": rmse_test.tolist(), "MAE": mae_test.tolist(), "CVRMSE": cvrmse_test.tolist()})
                    result_test_table = pd.concat([result_test_table, predict_test_model], axis=1)

                    new_error = {"MAE": [mae],
                            "MSE": [mse],
                            "RMSE": [rmse],
                            "CV_RMSE": [cv_rmse],
                            "train_time (s)": [time_train],
                            "test_time (s)": [test_time]}

                    new_para = {"units": [unit_train],
                        "epochs": [epoch_train],
                        "batch_size": [batch_size_train],
                        "learning_rate": [LR_train]}


                    for key, value in new_error.items():
                        error_list[key].extend(value)

                    for key, value in new_para.items():
                        para_list[key].extend(value)

        ##Load model tuần tự cộng
        for model_1 in mod1:
            for model_2 in mod2:
                start_time_test = time.time()
                checkpoint = torch.load("./model/"+ model_1 + "-"+ model_2 + "_add"+".pth")
                time_train = checkpoint["time_train"]
                name_model.append(''+model_1+'-'+model_2+'_add')
                model_train_1 = checkpoint["model"]
                model_train_2 = checkpoint["model_2"]
                unit_train = checkpoint["units"]
                epoch_train = checkpoint["epochs"]
                batch_size_train = checkpoint["batch_size"]
                LR_train = checkpoint["learning_rate"]
                predict_1, actual, index, predict_scale_1, actua_scale = eda.TestingModel(model_train_1)
                test_model = (actual-predict_1)
                predict_2 = eda.TestingModelSeq(model_train_2, test_model)
                predict = predict_2 + predict_1
                y_scaler = load(open('./static/y_scaler.pkl', 'rb'))
                predict_scale = y_scaler.fit_transform(predict)
                
                mae, mse, rmse, cv_rmse = self.Score(predict_scale,actua_scale)
                test_time = "{:.2f}".format((time.time()) - (start_time_test))

                mse_test = (predict_scale-actua_scale)**2
                rmse_test=np.sqrt(mse_test)
                mae_test = np.abs(predict_scale-actua_scale)
                cvrmse_test = rmse_test/np.mean(predict_scale)

                predict_test_model = pd.DataFrame(
                        {"Giá trị dự đoán": predict.tolist(),\
                    "Giá trị dự đoán model 1": predict_1.tolist(), "Giá trị dự đoán model 2": predict_2.tolist(),\
                        "Giá trị thực": actual.tolist(), \
                                "MSE": mse_test.tolist(),"RMSE": rmse_test.tolist(), "MAE": mae_test.tolist(), "CVRMSE": cvrmse_test.tolist()})
                result_test_table = pd.concat([result_test_table, predict_test_model], axis=1)

                new_error = {"MAE": [mae],
                        "MSE": [mse],
                        "RMSE": [rmse],
                        "CV_RMSE": [cv_rmse],
                        "train_time (s)": [time_train],
                        "test_time (s)": [test_time]}

                new_para = {"units": [unit_train],
                    "epochs": [epoch_train],
                    "batch_size": [batch_size_train],
                    "learning_rate": [LR_train]}

                for key, value in new_error.items():
                    error_list[key].extend(value)

                for key, value in new_para.items():
                    para_list[key].extend(value)

        ## Load model tuần tự nhân
        for model_1 in mod1:
            for model_2 in mod2:
                start_time_test = time.time()
                checkpoint = torch.load("./model/"+ model_1 + "-"+ model_2 + "_mul"+".pth")
                time_train = checkpoint["time_train"]
                name_model.append(''+model_1+'-'+model_2+'_mul')
                model_train_1 = checkpoint["model"]
                model_train_2 = checkpoint["model_2"]
                unit_train = checkpoint["units"]
                epoch_train = checkpoint["epochs"]
                batch_size_train = checkpoint["batch_size"]
                LR_train = checkpoint["learning_rate"]
                predict_1, actual, index, predict_scale_1, actua_scale = eda.TestingModel(model_train_1)
                test_model = (actual/predict_1)
                predict_2 = eda.TestingModelSeq(model_train_2, test_model)
                predict = predict_2 * predict_1
                y_scaler = load(open('./static/y_scaler.pkl', 'rb'))
                predict_scale = y_scaler.fit_transform(predict)
                
                mae, mse, rmse, cv_rmse = self.Score(predict_scale,actua_scale)
                test_time = "{:.2f}".format((time.time()) - (start_time_test))

                mse_test = (predict_scale-actua_scale)**2
                rmse_test=np.sqrt(mse_test)
                mae_test = np.abs(predict_scale-actua_scale)
                cvrmse_test = rmse_test/np.mean(predict_scale)

                predict_test_model = pd.DataFrame(
                        {"Giá trị dự đoán": predict.tolist(),\
                    "Giá trị dự đoán model 1": predict_1.tolist(), "Giá trị dự đoán model 2": predict_2.tolist(),\
                        "Giá trị thực": actual.tolist(), \
                                "MSE": mse_test.tolist(),"RMSE": rmse_test.tolist(), "MAE": mae_test.tolist(), "CVRMSE": cvrmse_test.tolist()})
                result_test_table = pd.concat([result_test_table, predict_test_model], axis=1)

                new_error = {"MAE": [mae],
                        "MSE": [mse],
                        "RMSE": [rmse],
                        "CV_RMSE": [cv_rmse],
                        "train_time (s)": [time_train],
                        "test_time (s)": [test_time]}

                new_para = {"units": [unit_train],
                    "epochs": [epoch_train],
                    "batch_size": [batch_size_train],
                    "learning_rate": [LR_train]}

                for key, value in new_error.items():
                    error_list[key].extend(value)

                for key, value in new_para.items():
                    para_list[key].extend(value)

        ## Load model song song
        for model_1 in mod1:
            for model_2 in mod1:
                name_temp = ''+model_2+'-'+model_1+'_para'
                if name_temp in name_model:
                    continue
                else:
                    if model_1 != model_2:
                        start_time_test = time.time()
                        checkpoint = torch.load("./model/"+ model_1 + "-"+ model_2 + "_para"+".pth")
                        time_train = checkpoint["time_train"]
                        name_model.append(''+model_1+'-'+model_2+'_para')
                        model_train_1 = checkpoint["model"]
                        model_train_2 = checkpoint["model_2"]

                        predict_1, actual, index, predict_scale_1, actua_scale = eda.TestingModel(model_train_1)
                        predict_2, actual_2, index_2, predict_scale_2, actua_scale_2 = eda.TestingModel(model_train_2)
                        omega = self.calculate_omega(actual, predict_1, predict_2)
                        predict = omega * predict_1 + (1 - omega) * predict_2
                        predict_scale = omega * predict_scale_1 + (1 - omega) * predict_scale_2

                        mae, mse, rmse, cv_rmse = self.Score(predict_scale,actua_scale)
                        test_time = "{:.2f}".format((time.time()) - (start_time_test))


                        mse_test = (predict_scale-actua_scale)**2
                        rmse_test=np.sqrt(mse_test)
                        mae_test = np.abs(predict_scale-actua_scale)
                        cvrmse_test = rmse_test/np.mean(predict_scale)

                        predict_test_model = pd.DataFrame(
                                {"Giá trị dự đoán": predict.tolist(),\
                            "Giá trị dự đoán model 1": predict_1.tolist(), "Giá trị dự đoán model 2": predict_2.tolist(),\
                                "Giá trị thực": actual.tolist(), \
                                      "MSE": mse_test.tolist(),"RMSE": rmse_test.tolist(), "MAE": mae_test.tolist(), "CVRMSE": cvrmse_test.tolist()})
                        result_test_table = pd.concat([result_test_table, predict_test_model], axis=1)

                        new_error = {"MAE": [mae],
                                "MSE": [mse],
                                "RMSE": [rmse],
                                "CV_RMSE": [cv_rmse],
                                "train_time (s)": [time_train],
                                "test_time (s)": [test_time]}

                        for key, value in new_error.items():
                            error_list[key].extend(value)

        return error_list, name_model, result_test_table, para_list 


    def LoadTable(self, df_test, input_dim, output_dim, selected_predict_column_name_test, scaler):
        error_list, name_model, result_test_table, para_list = self.LoadModel(df_test, input_dim, output_dim, selected_predict_column_name_test, scaler)
        err_table = pd.DataFrame(error_list)
        para_table = pd.DataFrame(para_list)
        err_table.index = name_model
        return err_table, result_test_table, para_table
    