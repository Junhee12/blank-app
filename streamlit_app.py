#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

#######################
# Page configuration
st.set_page_config(
    page_title="US Population Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("default")

#######################
# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #f3f3f3;
    color: black !important;
    text-align: center;
    padding: 15px 0;
    border-radius: 8px;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)


#######################
# Load data
df_reshaped = pd.read_csv('titanic.csv') ## 분석 데이터 넣기


#######################
# Sidebar
with st.sidebar:

    st.markdown(
        "승객 특성에 따라 **생존 패턴**과 **군집 구조**를 탐색할 수 있는 대시보드입니다."
    )

    st.markdown("---")

    # 탑승 클래스 필터
    pclass_options = sorted(df_reshaped["Pclass"].dropna().unique())
    selected_pclass = st.multiselect(
        "탑승 클래스 (Pclass)",
        options=pclass_options,
        default=pclass_options,
    )

    # 성별 필터
    sex_options = sorted(df_reshaped["Sex"].dropna().unique())
    selected_sex = st.multiselect(
        "성별 (Sex)",
        options=sex_options,
        default=sex_options,
    )

    # 출발 항구 필터
    embarked_options = sorted(df_reshaped["Embarked"].dropna().unique())
    selected_embarked = st.multiselect(
        "출발 항구 (Embarked)",
        options=embarked_options,
        default=embarked_options,
    )

    # 나이 범위 슬라이더 (결측치는 추후 전처리 단계에서 별도 처리)
    age_min = int(df_reshaped["Age"].min())
    age_max = int(df_reshaped["Age"].max())
    selected_age_range = st.slider(
        "나이 범위 (Age)",
        min_value=age_min,
        max_value=age_max,
        value=(age_min, age_max),
        step=1,
    )

    st.markdown("---")

    # 머신러닝 분석 모드 선택
    ml_mode = st.radio(
        "머신러닝 분석 선택",
        options=["생존 예측 (분류)", "승객 군집화 (군집)", "둘 다 보기"],
        index=0,
    )

    st.markdown(
        "<small>사이드바의 필터는 아래 모든 시각화와 모델 학습에 공통으로 적용됩니다.</small>",
        unsafe_allow_html=True,
    )

# 사이드바 필터를 적용한 데이터프레임 생성
df_filtered = df_reshaped.copy()

if selected_pclass:
    df_filtered = df_filtered[df_filtered["Pclass"].isin(selected_pclass)]

if selected_sex:
    df_filtered = df_filtered[df_filtered["Sex"].isin(selected_sex)]

if selected_embarked:
    df_filtered = df_filtered[df_filtered["Embarked"].isin(selected_embarked)]

# Age 결측치는 우선 제외하고 필터 (모델/시각화 단계에서 별도 전략 적용 가능)
df_filtered = df_filtered[df_filtered["Age"].between(selected_age_range[0], selected_age_range[1])]

# 이후 다른 영역에서 사용하기 쉽게 세션에 저장 (선택 사항)
st.session_state["df_filtered"] = df_filtered
st.session_state["ml_mode"] = ml_mode

#######################
# Plots



#######################
# Dashboard Main Panel
col = st.columns((1.5, 4.5, 2), gap='medium')

with col[0]:
    st.markdown("### 📊 승객 요약 정보")

    # 사이드바에서 필터링된 데이터 사용 (없으면 원본 사용)
    df_filtered = st.session_state.get("df_filtered", df_reshaped)

    # 필터 결과가 없을 때 처리
    if df_filtered.empty:
        st.info("선택한 조건을 만족하는 승객 데이터가 없습니다. 사이드바 필터를 조정해 주세요.")
    else:
        # 기본 요약 통계
        total_passengers = len(df_filtered)
        survival_rate = df_filtered["Survived"].mean() * 100

        avg_age = df_filtered["Age"].mean()
        avg_fare = df_filtered["Fare"].mean()
        avg_family = (df_filtered["SibSp"] + df_filtered["Parch"]).mean()

        # 상단 메트릭 카드
        m1, m2 = st.columns(2)
        m1.metric("총 승객 수", f"{total_passengers}")
        m2.metric("생존률", f"{survival_rate:.1f}%")

        m3, m4, m5 = st.columns(3)
        m3.metric("평균 나이", f"{avg_age:.1f}")
        m4.metric("평균 요금 (Fare)", f"{avg_fare:.1f}")
        m5.metric("평균 동반 가족 수", f"{avg_family:.2f}")

        st.markdown("---")

        # 성별 분포 바 차트
        st.markdown("#### 성별 분포")
        sex_counts = (
            df_filtered["Sex"]
            .value_counts()
            .reset_index(name="Count")
            .rename(columns={"index": "Sex"})
        )

        sex_chart = (
            alt.Chart(sex_counts)
            .mark_bar()
            .encode(
                x=alt.X("Sex:N", title="성별"),
                y=alt.Y("Count:Q", title="승객 수"),
                tooltip=["Sex", "Count"],
            )
        )
        st.altair_chart(sex_chart, use_container_width=True)

        # 클래스별 승객 수 바 차트
        st.markdown("#### 탑승 클래스 분포 (Pclass)")
        class_counts = (
            df_filtered["Pclass"]
            .value_counts()
            .sort_index()
            .reset_index(name="Count")
            .rename(columns={"index": "Pclass"})
        )

        class_chart = (
            alt.Chart(class_counts)
            .mark_bar()
            .encode(
                x=alt.X("Pclass:O", title="탑승 클래스"),
                y=alt.Y("Count:Q", title="승객 수"),
                tooltip=["Pclass", "Count"],
            )
        )
        st.altair_chart(class_chart, use_container_width=True)

with col[1]:
    st.markdown("### 🧭 생존 패턴 분석 & 분류 모델")

    # 사이드바에서 필터링된 데이터 사용
    df_filtered = st.session_state.get("df_filtered", df_reshaped)
    ml_mode = st.session_state.get("ml_mode", "생존 예측 (분류)")

    if df_filtered.empty:
        st.info("선택한 조건을 만족하는 승객 데이터가 없습니다. 사이드바 필터를 조정해 주세요.")
    else:
        ############################
        # 1) Pclass × Sex 생존률 히트맵
        ############################
        st.markdown("#### 🔥 탑승 클래스 × 성별 생존률 히트맵")

        survival_pivot = (
            df_filtered
            .groupby(["Pclass", "Sex"])["Survived"]
            .mean()
            .reset_index()
        )
        survival_pivot["SurvivalRate"] = survival_pivot["Survived"] * 100

        heatmap = (
            alt.Chart(survival_pivot)
            .mark_rect()
            .encode(
                x=alt.X("Pclass:O", title="탑승 클래스 (Pclass)"),
                y=alt.Y("Sex:N", title="성별 (Sex)"),
                color=alt.Color(
                    "SurvivalRate:Q",
                    title="생존률 (%)",
                    scale=alt.Scale(scheme="blues"),
                ),
                tooltip=[
                    alt.Tooltip("Pclass:O", title="Pclass"),
                    alt.Tooltip("Sex:N", title="Sex"),
                    alt.Tooltip("SurvivalRate:Q", title="생존률", format=".1f")
                ],
            )
        )

        st.altair_chart(heatmap, use_container_width=True)

        st.markdown("---")

        ############################
        # 2) 나이 × 요금 산점도 (생존 여부 색상)
        ############################
        # st.markdown("#### 🎯 나이 vs 요금 (생존 여부)")

        # scatter_df = df_filtered[["Age", "Fare", "Survived", "Pclass"]].dropna()

        # if scatter_df.empty:
        #     st.info("나이(Age)와 요금(Fare)에 결측치가 많아 산점도를 그릴 수 없습니다.")
        # else:
        #     scatter = (
        #         alt.Chart(scatter_df)
        #         .mark_circle(size=60, opacity=0.7)
        #         .encode(
        #             x=alt.X("Age:Q", title="나이 (Age)"),
        #             y=alt.Y("Fare:Q", title="요금 (Fare)"),
        #             color=alt.Color(
        #                 "Survived:N",
        #                 title="생존 여부",
        #                 scale=alt.Scale(domain=["0", "1"], range=["#d62728", "#1f77b4"]),
        #             ),
        #             shape=alt.Shape("Pclass:O", title="Pclass"),
        #             tooltip=[
        #                 alt.Tooltip("Age:Q", title="나이", format=".1f"),
        #                 alt.Tooltip("Fare:Q", title="요금", format=".1f"),
        #                 alt.Tooltip("Pclass:O", title="Pclass"),
        #                 alt.Tooltip("Survived:N", title="생존 여부"),
        #             ],
        #         )
        #     )

        #     st.altair_chart(scatter, use_container_width=True)
        st.markdown("#### 🎯 나이 vs 요금 (생존 여부)")

        # Age, Fare만 결측 제거 (굳이 Survived, Pclass까지 모두 dropna 하지 않음)
        scatter_df = df_filtered[["Age", "Fare", "Survived", "Pclass"]].dropna(subset=["Age", "Fare"])

        if scatter_df.empty:
            st.info("나이(Age)와 요금(Fare)에 해당하는 데이터가 없습니다. 사이드바 필터를 조정해 주세요.")
        else:
            # Survived를 범주형(문자열)으로 변환하면 레전드가 더 명확해짐
            scatter_df = scatter_df.copy()
            scatter_df["Survived_str"] = scatter_df["Survived"].map({0: "0", 1: "1"})

            scatter = (
                alt.Chart(scatter_df)
                .mark_circle(size=60, opacity=0.7)
                .encode(
                    x=alt.X("Age:Q", title="나이 (Age)"),
                    y=alt.Y("Fare:Q", title="요금 (Fare)"),
                    # ⚠ 도메인 강제 지정 제거, Altair가 자동으로 도메인 추론하게 둠
                    color=alt.Color(
                        "Survived_str:N",
                        title="생존 여부",
                    ),
                    shape=alt.Shape("Pclass:O", title="Pclass"),
                    tooltip=[
                        alt.Tooltip("Age:Q", title="나이", format=".1f"),
                        alt.Tooltip("Fare:Q", title="요금", format=".1f"),
                        alt.Tooltip("Pclass:O", title="Pclass"),
                        alt.Tooltip("Survived_str:N", title="생존 여부"),
                    ],
                )
            )

            st.altair_chart(scatter, use_container_width=True)

        st.markdown("---")

        ########################################
        # 3) 머신러닝 – 생존 예측 분류 모델
        ########################################
        if ml_mode in ["생존 예측 (분류)", "둘 다 보기"]:
            st.markdown("#### 🤖 생존 예측 모델 (분류)")

            # 필요 라이브러리 (파일 상단에 두어도 됨)
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            from sklearn.metrics import accuracy_score, f1_score
            from sklearn.ensemble import RandomForestClassifier
            import numpy as np

            # 분석에 사용할 컬럼
            feature_cols = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]
            target_col = "Survived"

            df_ml = df_filtered[feature_cols + [target_col]].dropna()

            if len(df_ml) < 50:
                st.info("필터 조건으로 인해 학습 가능한 데이터가 충분하지 않습니다. 필터 범위를 넓혀주세요.")
            else:
                X = df_ml[feature_cols]
                y = df_ml[target_col]

                categorical_features = ["Sex", "Embarked"]
                numeric_features = ["Pclass", "Age", "Fare", "SibSp", "Parch"]

                # 전처리 파이프라인
                categorical_transformer = OneHotEncoder(handle_unknown="ignore")
                numeric_transformer = "passthrough"

                preprocessor = ColumnTransformer(
                    transformers=[
                        ("cat", categorical_transformer, categorical_features),
                        ("num", numeric_transformer, numeric_features),
                    ]
                )

                # 분류 모델 정의
                clf = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=5,
                    random_state=42,
                )

                model = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        ("classifier", clf),
                    ]
                )

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                m1, m2 = st.columns(2)
                m1.metric("Accuracy", f"{acc:.3f}")
                m2.metric("F1-score", f"{f1:.3f}")

                # 특성 중요도 시각화
                try:
                    rf = model.named_steps["classifier"]
                    ohe = model.named_steps["preprocessor"].named_transformers_["cat"]

                    cat_feature_names = list(
                        ohe.get_feature_names_out(categorical_features)
                    )
                    all_feature_names = cat_feature_names + numeric_features

                    importances = rf.feature_importances_
                    fi_df = (
                        pd.DataFrame(
                            {
                                "feature": all_feature_names,
                                "importance": importances,
                            }
                        )
                        .sort_values("importance", ascending=False)
                        .head(10)
                    )

                    st.markdown("##### 🔍 주요 특징 중요도 (Top 10)")

                    fi_chart = (
                        alt.Chart(fi_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("importance:Q", title="중요도"),
                            y=alt.Y("feature:N", sort="-x", title="특징"),
                            tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")],
                        )
                    )
                    st.altair_chart(fi_chart, use_container_width=True)
                except Exception as e:
                    # 중요도 계산 실패 시 메시지만 출력
                    st.caption("특징 중요도 계산 중 문제가 발생하여 지표만 표시합니다.")

with col[2]:
    # ───────────────────────────────
    # 0. 기본 정보
    # ───────────────────────────────
    st.markdown("### 🧠 승객 군집화 (K-Means)")

    df_filtered = st.session_state.get("df_filtered", df_reshaped)
    ml_mode = st.session_state.get("ml_mode", "생존 예측 (분류)")

    st.caption(f"현재 선택된 머신러닝 모드: **{ml_mode}**")

    if df_filtered.empty:
        st.info("선택한 조건을 만족하는 승객 데이터가 없습니다. 사이드바 필터를 조정해 주세요.")
        st.stop()

    # 군집 모드가 아니면 안내만 보여주고 종료
    if ml_mode not in ["승객 군집화 (군집)", "둘 다 보기"]:
        st.info("사이드바에서 **'승객 군집화 (군집)'** 또는 **'둘 다 보기'** 를 선택하면 군집 결과가 표시됩니다.")
        st.stop()

    # ───────────────────────────────
    # 1. 라이브러리 확인
    # ───────────────────────────────
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
    except ModuleNotFoundError:
        st.error("⚠ scikit-learn 라이브러리가 설치되어 있지 않아 군집화를 실행할 수 없습니다.")
        st.code("pip install scikit-learn")
        st.stop()

    # ───────────────────────────────
    # 2. 군집에 사용할 데이터 준비
    # ───────────────────────────────
    # 필요한 컬럼만 선택
    df_clust = df_filtered[["Age", "Fare", "SibSp", "Parch", "Pclass", "Survived"]].copy()
    # Age, Fare 결측 제거
    df_clust = df_clust.dropna(subset=["Age", "Fare"])

    st.caption(f"군집 분석용 데이터 행 수: **{len(df_clust)}**")

    if len(df_clust) < 10:
        st.info("군집화를 수행하기에 데이터가 충분하지 않습니다. 필터 범위를 넓혀주세요.")
        st.stop()

    # 가족 수 변수 추가
    df_clust["FamilySize"] = df_clust["SibSp"] + df_clust["Parch"]

    feature_cols = ["Age", "Fare", "SibSp", "Parch", "Pclass", "FamilySize"]

    # ───────────────────────────────
    # 3. 하이퍼파라미터 입력 (k)
    # ───────────────────────────────
    k = st.selectbox(
        "군집 개수 (k)",
        options=[2, 3, 4, 5],
        index=1,
    )

    # ───────────────────────────────
    # 4. 스케일링 + KMeans 학습
    # ───────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clust[feature_cols])

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    df_clust["cluster"] = clusters.astype(str)

    # ───────────────────────────────
    # 5. 군집별 요약 테이블
    # ───────────────────────────────
    profile = (
        df_clust
        .groupby("cluster")
        .agg(
            count=("cluster", "size"),
            avg_age=("Age", "mean"),
            avg_fare=("Fare", "mean"),
            avg_family=("FamilySize", "mean"),
            survival_rate=("Survived", lambda s: s.mean() * 100),
        )
        .reset_index()
    )

    profile["avg_age"] = profile["avg_age"].round(1)
    profile["avg_fare"] = profile["avg_fare"].round(1)
    profile["avg_family"] = profile["avg_family"].round(2)
    profile["survival_rate"] = profile["survival_rate"].round(1)

    st.markdown("#### 📋 군집별 평균 특성 요약")
    st.dataframe(profile, use_container_width=True)

    # ───────────────────────────────
    # 6. 군집 시각화 (Age vs Fare)
    # ───────────────────────────────
    st.markdown("#### 📍 군집 시각화 (Age vs Fare)")

    scatter_cluster = (
        alt.Chart(df_clust)
        .mark_circle(size=60, opacity=0.75)
        .encode(
            x=alt.X("Age:Q", title="나이 (Age)"),
            y=alt.Y("Fare:Q", title="요금 (Fare)"),
            color=alt.Color("cluster:N", title="Cluster"),
            tooltip=[
                alt.Tooltip("Age:Q", title="나이", format=".1f"),
                alt.Tooltip("Fare:Q", title="요금", format=".1f"),
                alt.Tooltip("cluster:N", title="Cluster"),
                alt.Tooltip("Survived:N", title="생존 여부"),
                alt.Tooltip("SibSp:Q", title="형제/배우자 수"),
                alt.Tooltip("Parch:Q", title="부모/자녀 수"),
            ],
        )
    )
    st.altair_chart(scatter_cluster, use_container_width=True)


    st.markdown("---")

    ############################
    # 3) About 패널
    ############################
    with st.expander("ℹ️ About this dashboard / 데이터 설명"):
        st.markdown(
            """
            - **데이터셋**: Titanic 생존 데이터  
            - **분석 목적**: 필터 조건에 따른 생존 패턴 및 승객 군집 구조 탐색  
            - **모델**: K-Means 군집화 (나이, 요금, 가족 수, Pclass 등 특징 사용)
            """
        )