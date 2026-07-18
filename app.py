import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Vaccine Discourse Monitor", layout="wide")

# ----------------------------------------------------------------------
# МАППИНГ ТЕМ (из твоего диплома, Table 1)
# ----------------------------------------------------------------------
TOPIC_LABELS = {
    1: "Личный опыт и здравоохранение",
    2: "Политика и государство",
    3: "Лечение и медицина",
    4: "Иммунология и наука",
    5: "Вакцинация животных",
}
# ВАЖНО: проверь соответствие номеров! В LDA нумерация тем может не совпадать
# с порядком в дипломе. Открой посты по каждому topic и сверь ключевые слова.

SENT_COLORS = {
    "negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c",
    "негатив": "#d62728", "нейтрально": "#7f7f7f", "позитив": "#2ca02c",
}

# ----------------------------------------------------------------------
# ЗАГРУЗКА
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    posts = pd.read_parquet("posts.parquet")
    comments = pd.read_parquet("comments.parquet")

    for df in (posts, comments):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "year" not in df.columns and "date" in df.columns:
            df["year"] = df["date"].dt.year
        # текстовое имя темы
        if "topic" in df.columns:
            # приводим 1.0 -> 1 (int), аккуратно с NaN
            df["topic_int"] = df["topic"].astype("Int64")  # nullable int
            df["topic_label"] = df["topic_int"].map(TOPIC_LABELS)
            df["topic_label"] = df["topic_label"].fillna(
                "Тема " + df["topic_int"].astype(str)
            )
    return posts, comments


posts, comments = load_data()

# ----------------------------------------------------------------------
# САЙДБАР
# ----------------------------------------------------------------------
st.sidebar.title("Фильтры")

# исключаем 2026 из годового анализа (как в дипломе) — но оставляем возможность
years = sorted([int(y) for y in comments["year"].dropna().unique()])
year_range = st.sidebar.select_slider(
    "Период", options=years, value=(min(years), max(years))
)

all_topics = sorted(comments["topic_label"].dropna().unique())
selected_topics = st.sidebar.multiselect("Темы", all_topics, default=all_topics)

# переключатель модели сентимента
sent_source = "Трансформер (rubert)"
if "rusentilex_sentiment" in comments.columns:
    sent_source = st.sidebar.radio(
        "Модель сентимента",
        ["Трансформер (rubert)", "Словарь (RuSentiLex)"],
    )

# выбираем рабочие колонки в зависимости от модели
if sent_source.startswith("Словарь"):
    SENT_CAT = "rusentilex_sentiment"
    SENT_W = "rusentilex_score"
else:
    SENT_CAT = "sentiment"
    SENT_W = "sentiment_weighted"

# ----------------------------------------------------------------------
# ФИЛЬТРАЦИЯ
# ----------------------------------------------------------------------
c = comments[
    comments["year"].between(year_range[0], year_range[1])
    & comments["topic_label"].isin(selected_topics)
].copy()

p = posts[
    posts["year"].between(year_range[0], year_range[1])
    & posts["topic_label"].isin(selected_topics)
].copy()

# ----------------------------------------------------------------------
# ЗАГОЛОВОК + KPI
# ----------------------------------------------------------------------
st.title("Мониторинг дискуссии о вакцинации (Pikabu)")
st.caption("Прототип системы мониторинга общественного мнения по вакцинам и препаратам")

# категории негатив/позитив (учитываем оба варианта названий)
neg_vals = ["negative", "негатив"]
pos_vals = ["positive", "позитив"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Постов", f"{len(p):,}")
col2.metric("Комментариев", f"{len(c):,}")
neg_share = c[SENT_CAT].isin(neg_vals).mean() * 100 if len(c) else 0
pos_share = c[SENT_CAT].isin(pos_vals).mean() * 100 if len(c) else 0
col3.metric("Доля негатива", f"{neg_share:.1f}%")
col4.metric("Доля позитива", f"{pos_share:.1f}%")

st.divider()

# ----------------------------------------------------------------------
# ОБЩИЙ ТОН + ТЕМЫ
# ----------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Общий тон дискуссии")
    sc = (c[SENT_CAT].value_counts(normalize=True) * 100).reset_index()
    sc.columns = ["sentiment", "percent"]
    fig = px.bar(sc, x="sentiment", y="percent", color="sentiment",
                 color_discrete_map=SENT_COLORS,
                 text=sc["percent"].round(1))
    fig.update_layout(showlegend=False, yaxis_title="%", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Распределение постов по темам")
    tc = p["topic_label"].value_counts().reset_index()
    tc.columns = ["topic_label", "count"]
    fig = px.bar(tc, x="count", y="topic_label", orientation="h")
    fig.update_layout(yaxis_title="", xaxis_title="Число постов")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# СЕНТИМЕНТ ПО ТЕМАМ
# ----------------------------------------------------------------------
st.subheader("Сентимент внутри тем")
ts = c.groupby(["topic_label", SENT_CAT]).size().reset_index(name="count")
ts["percent"] = ts.groupby("topic_label")["count"].transform(
    lambda x: x / x.sum() * 100
)
fig = px.bar(ts, x="topic_label", y="percent", color=SENT_CAT,
             color_discrete_map=SENT_COLORS, barmode="stack")
fig.update_layout(xaxis_title="", yaxis_title="%")
st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# ДИНАМИКА ПО ГОДАМ
# ----------------------------------------------------------------------
if SENT_W in c.columns:
    st.subheader("Средний тон по годам")
    yr = c.groupby("year")[SENT_W].mean().reset_index()
    fig = px.line(yr, x="year", y=SENT_W, markers=True)
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(xaxis_title="Год", yaxis_title="Средний сентимент")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# ТОП ОБСУЖДАЕМЫХ ПОСТОВ
# ----------------------------------------------------------------------
st.subheader("Самые обсуждаемые посты")
sort_col = "controversy" if "controversy" in p.columns else "comments_count"
cols = [x for x in ["title", "topic_label", "comments_count",
                    "upvotes", "downvotes", "controversy", "url"]
        if x in p.columns]
st.dataframe(
    p.sort_values(sort_col, ascending=False).head(15)[cols],
    use_container_width=True, hide_index=True
)

# ----------------------------------------------------------------------
# ПРИМЕРЫ КОММЕНТАРИЕВ (негативные)
# ----------------------------------------------------------------------
if "text_for_sentiment" in c.columns:
    with st.expander("Примеры негативных комментариев"):
        neg = c[c[SENT_CAT].isin(neg_vals)]
        sample = neg[["topic_label", "text_for_sentiment"]].dropna().head(20)
        st.dataframe(sample, use_container_width=True, hide_index=True)
