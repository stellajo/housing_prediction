# housing_prediction
HOW TO USE

1. HousePrice Class Description
    HousePrice는 이 프로젝트의 목표인 집값예측을 위한 클래스이다.
    HousePrice 클래스 object를 생성할 때, 아래 정보를 같이 초기값으로 넣어줘야한다.
    - filename: raw 데이터가 있는 file (csv 또는 excel파일)
    - column_list: raw 데이터의 column값
    - y_name: raw 데이터에서 예측하려고 하는 값의 column이름
    - feature_importance_list: 예측모델에 사용하려는 feature의 이름 리스트
    - train_ratio: data에서 train dataset의 비율

2. Functions in HousePrice
    - grid_search(cv, **params): 알고리즘의 최적 인자를 찾기 위해 사용 되는 함수, cv는 cross validation에 사용될 folds 수를 뜻함.
    params는 dictionary type으로 gridsearch할 인자 리스트를 넣어줌
    - run_model(**params)
    model을 훈련시키는 함수로 params에 모델의 인자들을 넣어줌
    - predict(clf) : run_model의 return값인 예측모델을 이용하여 예측을 하는 함수
    - fit_func(**params): 모델을 staged_predict을 이용, n_estimator를 증가시키며 훈련을 하여 결과를 보여주는 함수