import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# 'train.csv' 파일과 'submission.csv' 파일의 전체 경로 지정
train_file_path = r'C:\Users\user\Desktop\train.csv'
submission_file_path = r'C:\Users\user\Desktop\submission.csv'

# 파일들 읽어오기
df_train = pd.read_csv(train_file_path, encoding='latin1', low_memory=False)
df_test = pd.read_csv(submission_file_path)

df_train.head()

# 데이터 전처리

# 레이블 인코딩

def label_encoding(series: pd.Series, one_hot_encode_cols: list = None) -> pd.Series:
    # 범주형 데이터를 시리즈 형태로 받아 숫자형 데이터로 변환합니다.
    my_dict = {}

    # 모든 요소를 문자열로 변환
    series = series.astype(str)

    for idx, value in enumerate(sorted(series.unique())):
        my_dict[value] = idx
    series = series.map(my_dict)

    return series


label_columns = [
    "customer_country",
    "business_subarea",
    "business_area",
    "business_unit",
    "customer_type",
    "enterprise",
    "customer_job",
    "inquiry_type",
    "product_category",
    "product_subcategory",
    "product_modelname",
    "customer_country.1",
    "customer_position",
    "response_corporate",
    "expected_timeline",
]

# df_all은 df_train과 df_test에서 공통으로 존재하는 열들 중에서 label_columns에 명시된 열들을 포함한 데이터프레임
# pd.concat 함수는 Pandas에서 데이터프레임을 합치는데 사용되는 함수
df_all = pd.concat([df_train[label_columns], df_test[label_columns]])

for col in label_columns:
    df_all[col] = label_encoding(df_all[col])

# 학습 데이터와 테스트 데이터로 다시 나누기
for col in label_columns:
    df_train[col] = df_all.iloc[: len(df_train)][col]
    df_test[col] = df_all.iloc[len(df_train):][col]

# train_test_split 함수가 학습 데이터를 훈련용과 검증용 데이터로 무작위로 분리하는 데 사용.
x_train, x_val, y_train, y_val = train_test_split(
    df_train.drop("is_converted", axis=1),  # is_converted 열 제외 나머지 열을 특성(X)로
    df_train["is_converted"],  # is_converted 열을 타겟(Y)로.
    test_size=0.2,
    shuffle=True,
    random_state=400,
)

# decision tree 모델 훈련
model = DecisionTreeClassifier()  # Decision Tree 모델을 생성
#>> model = RUSBoostClassifier(estimator = DecisionTreeClassifier()) 부스팅 모델.
model.fit(x_train.fillna(0), y_train)


# 학습 데이터의 결측값을 0으로 대체한 데이터를 사용, 학습 데이터의 타겟 변수(레이블)
# fit 메서드를 사용하여 모델을 학습


# 평가 함수
def get_clf_eval(y_test, y_pred=None):  # 실제값(타겟 변수), 모델 예측값
    confusion = confusion_matrix(y_test, y_pred, labels=[True, False])
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, labels=[True, False])
    recall = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred, labels=[True, False])

    print("오차행렬:\n", confusion)
    print("\n정확도: {:.4f}".format(accuracy))
    print("정밀도: {:.4f}".format(precision))
    print("재현율: {:.4f}".format(recall))
    print("F1: {:.4f}".format(F1))


# 학습된 Decision Tree 모델을 검증 데이터 x_val.fillna(0)에 대해 예측을 수행
pred = model.predict(x_val.fillna(0))
get_clf_eval(y_val, pred)  # 모델 성능 출력

# 예측에 필요한 데이터 분리
x_test = df_test.drop(["is_converted", "id"], axis=1)
test_pred = model.predict(x_test.fillna(0))
sum(test_pred)  # True로 예측된 개수

# 제출 데이터 읽어오기 (df_test는 전처리된 데이터가 저장됨)
df_sub = pd.read_csv("submission.csv")
df_sub["is_converted"] = test_pred

# 제출 파일 저장
df_sub.to_csv("submission.csv", index=False)