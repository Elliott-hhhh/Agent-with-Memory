import os
import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import requests

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['part_time_job'] = df['part_time_job'].map({'Yes': 1, 'No': 0})
    df['extracurricular_participation'] = df['extracurricular_participation'].map({'Yes': 1, 'No': 0})
    df['diet_quality'] = df['diet_quality'].map({'Poor': 0, 'Fair': 1, 'Good': 2})
    df['internet_quality'] = df['internet_quality'].map({'Poor': 0, 'Average': 1, 'Good': 2})
    df['parental_education_level'] = df['parental_education_level'].map({
        'uneducated': 0, 'High School': 1, 'Bachelor': 2, 'Master': 3})
    return df

def student_performance_prediction(file_path: str) -> str:
    data = pd.read_csv(file_path)  # 替换为你实际的文件名
    data.drop(["student_id", "age", "gender"], axis=1, inplace=True)  # 删除无用的列
    data.fillna({"parental_education_level": "uneducated"}, inplace=True)

    X = data.drop(columns="exam_score")
    y = data["exam_score"]

    # 随机森林
    X = preprocess_input(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    rf = RandomForestRegressor(n_estimators=100, random_state=0)
    param_dist = {
        "n_estimators": randint(10, 200),
        "max_depth": [None] + list(np.arange(1, 20)),  # 控制树深度，避免过拟合
        "min_samples_split": randint(2, 15),  # 剪枝参数，较大值有助于减少复杂度
        "min_samples_leaf": randint(1, 20),  # 同样是剪枝参数
        "max_features": randint(1, 10)  # 控制特征使用量，影响模型复杂度
    }
    # 设置 RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=30,  # 迭代次数，根据需求可增减
        scoring='neg_mean_squared_error',
        cv=5,  # 5折交叉验证
        random_state=999,
        n_jobs=-1,
    )

    random_search.fit(X_train, y_train)
    # 输出最优参数和得分
    best_params = random_search.best_params_

    # 使用最优参数创建新的 RandomForestRegressor 模型
    rf_best = RandomForestRegressor(**best_params, random_state=0)

    # 训练模型
    rf_best.fit(X_train, y_train)
    return rf_best

def predict_exam_score(input_data):
    if isinstance(input_data, pd.DataFrame):
        df = input_data
    elif isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        raise ValueError("输入必须是 dict 或 pd.DataFrame")

    # 预处理数据
    df_processed = preprocess_input(df)
    rf_best = student_performance_prediction("student_habits_performance.csv")
    # 执行预测
    predictions = rf_best.predict(df_processed)
    return predictions

def get_weather(city: str) -> str:
    API_KEY = os.getenv("WEATHER_API_KEY")
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&lang=zh"
    response = requests.get(url)
    if response.status_code != 200:
        return "天气服务不可用，请稍后再试。"
    data = response.json()
    temp = data['current']['temp_c']
    condition = data['current']['condition']['text']
    return f"{city} 当前温度 {temp}°C，天气状况：{condition}"

def calculate(expression: str) -> str:
    try:
        return str(eval(expression))
    except:
        return "无法解析表达式"

def translate_to_chinese(english_text: str) -> str:
    return f"{english_text} 的中文翻译是："

def emotion_detection(text: str) -> str:

    return 0


def give_advices(student_habits: dict) -> str:
    """
        根据传入的数据对学生成绩做预测。
        把传入的数据变为dict格式。
        要求dict中必须包含study_hours_per_day，social_media_hours，netflix_hours，part_time_job，attendance_percentage，sleep_hours，diet_quality，exercise_frequency，parental_education_level，internet_quality，mental_health_rating，extracurricular_participation这几列。
        返回学生预测分数及提升建议。
        """

    # if not os.path.exists(file_path):
    #     return "文件路径不存在，请检查文件是否正确上传。"
    # student = {"study_hours_per_day": 4.6,"social_media_hours": 2, "netflix_hours": 3.6, "part_time_job": "Yes",
    #                     "attendance_percentage": 81.1, "sleep_hours": 6.8, "diet_quality": "Fair", "exercise_frequency": 5,
    #                     "parental_education_level": "High School", "internet_quality": "Average",
    #                     "mental_health_rating": 4,
    #                     "extracurricular_participation": "No"}

    # # 清理列（防止含有 student_id, age, gender）
    # for col in ["student_id", "age", "gender"]:
    #     if col in df.columns:
    #         df.drop(col, axis=1, inplace=True)

    # df.fillna({"parental_education_level": "uneducated"}, inplace=True)

    required_columns = [
        "study_hours_per_day", "social_media_hours", "netflix_hours",
        "part_time_job", "attendance_percentage", "sleep_hours",
        "diet_quality", "exercise_frequency", "parental_education_level",
        "internet_quality", "mental_health_rating", "extracurricular_participation"
    ]
    student_habits = json.loads(student_habits)
    df = pd.DataFrame([student_habits])

    # 确保所有列都存在
    for col in required_columns:
        if col not in df.columns:
            print(df,type(df))
            raise ValueError(f"缺少必要字段：{col}")

    df = preprocess_input(df)
    rf_best = student_performance_prediction("student_habits_performance.csv")
    predictions = rf_best.predict(df)

    results = f"该学生预测成绩：{predictions[0]:.2f}"
    return results

