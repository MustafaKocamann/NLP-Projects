import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# Veriyi Yüklemek
df = pd.read_csv("Hotel_Reviews.csv")

# Label oluşturma (train.py'deki gibi)
def extract_review_and_label(row):
    neg = row["Negative_Review"].strip()
    pos = row["Positive_Review"].strip()
    
    if neg != "No Negative":
        return neg, 0
    if pos != "No Positive":
        return pos, 1
    return pos, 1

df["review"], df["label"] = zip(*df.apply(extract_review_and_label, axis=1))

# =============================================================================
# 1. COĞRAFİ DUYGU ANALİZİ (Geospatial Analysis)
# =============================================================================

# Şehir bilgisini Hotel_Address'ten çıkarmak
def extract_city(address):
    # Adresin son kısmı genellikle şehir adını içerir
    parts = address.split()
    # Bilinen şehirler
    cities = ["Amsterdam", "London", "Paris", "Barcelona", "Milan", "Vienna", "Berlin"]
    for city in cities:
        if city in address:
            return city
    return "Other"

df["city"] = df["Hotel_Address"].apply(extract_city)

print("Şehir Dağılımı:")
print(df["city"].value_counts())

# -----------------------------------------------------------------------------
# 1.1 Dinamik Isı Haritası - Şehir Bazında Ortalama Puanlar
# -----------------------------------------------------------------------------
city_scores = df.groupby("city").agg({
    "Reviewer_Score": "mean",
    "lat": "mean",
    "lng": "mean",
    "Hotel_Name": "count"
}).reset_index()
city_scores.columns = ["city", "avg_score", "lat", "lng", "review_count"]

fig_heatmap = px.scatter_mapbox(
    city_scores,
    lat="lat",
    lon="lng",
    size="review_count",
    color="avg_score",
    color_continuous_scale=["red", "yellow", "green"],
    range_color=[6, 9],
    hover_name="city",
    hover_data={"avg_score": ":.2f", "review_count": True},
    title="🌍 Şehir Bazında Ortalama Otel Puanları",
    zoom=3,
    mapbox_style="carto-positron"
)
fig_heatmap.update_layout(
    height=600,
    margin={"r": 0, "t": 50, "l": 0, "b": 0}
)
fig_heatmap.show()

# -----------------------------------------------------------------------------
# 1.2 Otel Bazlı Detaylı Isı Haritası
# -----------------------------------------------------------------------------
hotel_scores = df.groupby(["Hotel_Name", "city"]).agg({
    "Reviewer_Score": "mean",
    "lat": "first",
    "lng": "first",
    "review": "count"
}).reset_index()
hotel_scores.columns = ["hotel_name", "city", "avg_score", "lat", "lng", "review_count"]

fig_hotel_heatmap = px.scatter_mapbox(
    hotel_scores,
    lat="lat",
    lon="lng",
    size="review_count",
    color="avg_score",
    color_continuous_scale=["red", "yellow", "green"],
    range_color=[6, 9],
    hover_name="hotel_name",
    hover_data={"city": True, "avg_score": ":.2f", "review_count": True},
    title="🏨 Otel Bazında Puan Dağılımı Haritası",
    zoom=3,
    mapbox_style="carto-positron"
)
fig_hotel_heatmap.update_layout(
    height=700,
    margin={"r": 0, "t": 50, "l": 0, "b": 0}
)
fig_hotel_heatmap.show()

# -----------------------------------------------------------------------------
# 1.3 Milliyet Bazlı Beklenti Analizi - Şehir ve Milliyet Karşılaştırması
# -----------------------------------------------------------------------------
# En çok yorum yapan 10 milliyet
top_nationalities = df["Reviewer_Nationality"].value_counts().head(10).index.tolist()
df_top_nat = df[df["Reviewer_Nationality"].isin(top_nationalities)]

nationality_city_scores = df_top_nat.groupby(["Reviewer_Nationality", "city"]).agg({
    "Reviewer_Score": "mean"
}).reset_index()

fig_nat_city = px.bar(
    nationality_city_scores,
    x="city",
    y="Reviewer_Score",
    color="Reviewer_Nationality",
    barmode="group",
    title="🌐 Milliyet Bazında Şehirlere Verilen Ortalama Puanlar",
    labels={"Reviewer_Score": "Ortalama Puan", "city": "Şehir", "Reviewer_Nationality": "Milliyet"},
    color_discrete_sequence=px.colors.qualitative.Set3
)
fig_nat_city.update_layout(
    height=500,
    xaxis_tickangle=-45,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig_nat_city.show()

# Heatmap versiyonu - Milliyet vs Şehir
pivot_nat_city = nationality_city_scores.pivot(
    index="Reviewer_Nationality", 
    columns="city", 
    values="Reviewer_Score"
)

fig_heatmap_nat = px.imshow(
    pivot_nat_city,
    color_continuous_scale="RdYlGn",
    aspect="auto",
    title="🗺️ Milliyet vs Şehir - Puan Isı Haritası",
    labels=dict(x="Şehir", y="Milliyet", color="Ortalama Puan")
)
fig_heatmap_nat.update_layout(height=500)
fig_heatmap_nat.show()

# =============================================================================
# 2. ETİKET MADENCİLİĞİ (Tag Analysis & Segmentation)
# =============================================================================

# Tags sütununu parse etmek
def parse_tags(tags_str):
    """Tags stringini listeye çevirir"""
    try:
        # "[' Trip ', ' Couple ', ' Stayed 2 nights ']" formatını parse et
        tags = eval(tags_str)
        return [tag.strip() for tag in tags]
    except:
        return []

df["parsed_tags"] = df["Tags"].apply(parse_tags)

# Trip type çıkarma
def extract_trip_type(tags):
    tags_lower = [t.lower() for t in tags]
    for tag in tags_lower:
        if "business" in tag:
            return "Business Trip"
        elif "leisure" in tag:
            return "Leisure Trip"
    return "Other"

# Traveler type çıkarma
def extract_traveler_type(tags):
    tags_lower = [t.lower() for t in tags]
    for tag in tags_lower:
        if "couple" in tag:
            return "Couple"
        elif "family" in tag:
            return "Family"
        elif "solo" in tag:
            return "Solo Traveler"
        elif "group" in tag:
            return "Group"
    return "Other"

df["trip_type"] = df["parsed_tags"].apply(extract_trip_type)
df["traveler_type"] = df["parsed_tags"].apply(extract_traveler_type)

print("\nTrip Type Dağılımı:")
print(df["trip_type"].value_counts())

print("\nTraveler Type Dağılımı:")
print(df["traveler_type"].value_counts())

# -----------------------------------------------------------------------------
# 2.1 Segmentasyon Analizi - İş vs Tatil Şikayetleri
# -----------------------------------------------------------------------------

df["sentiment"] = df["label"].map({0: "Negative", 1: "Positive"})

# Trip type bazında sentiment dağılımı
trip_sentiment = df.groupby(["trip_type", "sentiment"]).size().reset_index(name="count")

fig_trip_sentiment = px.bar(
    trip_sentiment,
    x="trip_type",
    y="count",
    color="sentiment",
    barmode="group",
    title="🧳 Seyahat Tipi Bazında Duygu Dağılımı",
    labels={"trip_type": "Seyahat Tipi", "count": "Yorum Sayısı", "sentiment": "Duygu"},
    color_discrete_map={"Negative": "#e74c3c", "Positive": "#2ecc71"}
)
fig_trip_sentiment.update_layout(height=450)
fig_trip_sentiment.show()

# Trip type + Traveler type kombinasyonu
combined_analysis = df.groupby(["trip_type", "traveler_type", "sentiment"]).size().reset_index(name="count")

fig_combined = px.sunburst(
    combined_analysis,
    path=["trip_type", "traveler_type", "sentiment"],
    values="count",
    title="☀️ Seyahat Tipi → Yolcu Tipi → Duygu Sunburst Grafiği",
    color="sentiment",
    color_discrete_map={"Negative": "#e74c3c", "Positive": "#2ecc71", "(?)": "#95a5a6"}
)
fig_combined.update_layout(height=600)
fig_combined.show()

# -----------------------------------------------------------------------------
# 2.2 Sunburst Chart - Trip Type ve Sentiment
# -----------------------------------------------------------------------------
sunburst_data = df.groupby(["trip_type", "sentiment"]).size().reset_index(name="count")

fig_sunburst = px.sunburst(
    sunburst_data,
    path=["trip_type", "sentiment"],
    values="count",
    title="🎯 Seyahat Tipi ve Duygu Dağılımı - Sunburst Chart",
    color="sentiment",
    color_discrete_map={"Negative": "#e74c3c", "Positive": "#2ecc71"}
)
fig_sunburst.update_layout(height=550)
fig_sunburst.show()

# -----------------------------------------------------------------------------
# 2.3 Şehir Bazında Trip Type Analizi
# -----------------------------------------------------------------------------
city_trip_analysis = df.groupby(["city", "trip_type"]).agg({
    "Reviewer_Score": "mean",
    "review": "count"
}).reset_index()
city_trip_analysis.columns = ["city", "trip_type", "avg_score", "review_count"]

fig_city_trip = px.bar(
    city_trip_analysis,
    x="city",
    y="avg_score",
    color="trip_type",
    barmode="group",
    title="🏙️ Şehir Bazında Seyahat Tipine Göre Ortalama Puanlar",
    labels={"city": "Şehir", "avg_score": "Ortalama Puan", "trip_type": "Seyahat Tipi"},
    color_discrete_sequence=px.colors.qualitative.Pastel
)
fig_city_trip.update_layout(height=450, xaxis_tickangle=-45)
fig_city_trip.show()

# -----------------------------------------------------------------------------
# 2.4 En Sık Kullanılan Etiketler - Word Cloud Alternatifi (Bar Chart)
# -----------------------------------------------------------------------------
from collections import Counter

all_tags = [tag for tags in df["parsed_tags"] for tag in tags]
tag_counts = Counter(all_tags)
top_tags = pd.DataFrame(tag_counts.most_common(20), columns=["tag", "count"])

fig_tags = px.bar(
    top_tags,
    x="count",
    y="tag",
    orientation="h",
    title="🏷️ En Sık Kullanılan Etiketler (Top 20)",
    labels={"count": "Kullanım Sayısı", "tag": "Etiket"},
    color="count",
    color_continuous_scale="Viridis"
)
fig_tags.update_layout(height=600, yaxis=dict(autorange="reversed"))
fig_tags.show()

# =============================================================================
# 3. ÖZET İSTATİSTİKLER
# =============================================================================
print("\n" + "="*60)
print("📊 ÖZET İSTATİSTİKLER")
print("="*60)
print(f"Toplam Yorum Sayısı: {len(df):,}")
print(f"Toplam Otel Sayısı: {df['Hotel_Name'].nunique():,}")
print(f"Genel Ortalama Puan: {df['Reviewer_Score'].mean():.2f}")
print(f"Negatif Yorum Oranı: {(df['label']==0).mean()*100:.1f}%")
print(f"Pozitif Yorum Oranı: {(df['label']==1).mean()*100:.1f}%")
print("="*60)

# =============================================================================
# 4. ZAMAN SERİSİ VE TREND ANALİZİ (Time-Series Insights)
# =============================================================================

# Review_Date'i datetime formatına çevirme
df["Review_Date"] = pd.to_datetime(df["Review_Date"])
df["year"] = df["Review_Date"].dt.year
df["month"] = df["Review_Date"].dt.month
df["month_name"] = df["Review_Date"].dt.month_name()
df["year_month"] = df["Review_Date"].dt.to_period("M").astype(str)

print("\n📅 Tarih Aralığı:")
print(f"En eski yorum: {df['Review_Date'].min()}")
print(f"En yeni yorum: {df['Review_Date'].max()}")

# -----------------------------------------------------------------------------
# 4.1 Mevsimsel Duygu Dalgalanması - Stacked Area Chart
# -----------------------------------------------------------------------------
monthly_sentiment = df.groupby(["year_month", "sentiment"]).size().reset_index(name="count")
monthly_pivot = monthly_sentiment.pivot(index="year_month", columns="sentiment", values="count").fillna(0)
monthly_pivot = monthly_pivot.reset_index()

fig_seasonal = go.Figure()
fig_seasonal.add_trace(go.Scatter(
    x=monthly_pivot["year_month"],
    y=monthly_pivot["Negative"],
    mode="lines",
    name="Negatif Yorumlar",
    fill="tozeroy",
    line=dict(color="#e74c3c"),
    fillcolor="rgba(231, 76, 60, 0.5)"
))
fig_seasonal.add_trace(go.Scatter(
    x=monthly_pivot["year_month"],
    y=monthly_pivot["Positive"],
    mode="lines",
    name="Pozitif Yorumlar",
    fill="tozeroy",
    line=dict(color="#2ecc71"),
    fillcolor="rgba(46, 204, 113, 0.5)"
))
fig_seasonal.update_layout(
    title="📈 Mevsimsel Duygu Dalgalanması (Aylık Trend)",
    xaxis_title="Ay",
    yaxis_title="Yorum Sayısı",
    height=500,
    hovermode="x unified",
    xaxis=dict(tickangle=-45, nticks=20)
)
fig_seasonal.show()

# Ay bazında ortalama puan
monthly_scores = df.groupby("month_name")["Reviewer_Score"].mean().reindex([
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
])

fig_monthly_score = px.bar(
    x=monthly_scores.index,
    y=monthly_scores.values,
    title="🌡️ Aylara Göre Ortalama Puan - Mevsimsel Kalite Değişimi",
    labels={"x": "Ay", "y": "Ortalama Puan"},
    color=monthly_scores.values,
    color_continuous_scale="RdYlGn"
)
fig_monthly_score.update_layout(height=450)
fig_monthly_score.show()

# -----------------------------------------------------------------------------
# 4.2 Days Since Review vs Score - Hareketli Ortalama
# -----------------------------------------------------------------------------

df["days_since_review"] = pd.to_numeric(df["days_since_review"].str.replace(" days", "").str.replace(" day", ""), errors="coerce")


df_sorted = df.sort_values("days_since_review")
df_sorted["rolling_avg_score"] = df_sorted["Reviewer_Score"].rolling(window=5000, min_periods=100).mean()

# Sample for visualization (too many points)
df_sampled = df_sorted[::100].copy()

fig_rolling = px.line(
    df_sampled,
    x="days_since_review",
    y="rolling_avg_score",
    title="📊 Zaman İçinde Otel Kalitesi Değişimi (Hareketli Ortalama)",
    labels={"days_since_review": "Yorum Yaşı (Gün)", "rolling_avg_score": "Hareketli Ortalama Puan"}
)
fig_rolling.update_traces(line_color="#3498db", line_width=2)
fig_rolling.update_layout(height=450)
fig_rolling.show()

# =============================================================================
# 5. METİN VE SKOR İLİŞKİSİ (Word Count & Sentiment Gap)
# =============================================================================

# -----------------------------------------------------------------------------
# 5.1 The "Complaining" Correlation - Kelime Sayısı vs Puan
# -----------------------------------------------------------------------------
print("\n📝 Kelime Sayısı Korelasyonu:")
corr_neg = df["Review_Total_Negative_Word_Counts"].corr(df["Reviewer_Score"])
corr_pos = df["Review_Total_Positive_Word_Counts"].corr(df["Reviewer_Score"])
print(f"Negatif Kelime Sayısı - Puan Korelasyonu: {corr_neg:.4f}")
print(f"Pozitif Kelime Sayısı - Puan Korelasyonu: {corr_pos:.4f}")

# Joint Plot benzeri scatter with marginal histograms
fig_joint = px.scatter(
    df.sample(10000),  
    x="Review_Total_Negative_Word_Counts",
    y="Reviewer_Score",
    color="sentiment",
    color_discrete_map={"Negative": "#e74c3c", "Positive": "#2ecc71"},
    opacity=0.4,
    title="😤 Şikayet Korelasyonu: Negatif Kelime Sayısı vs Puan",
    labels={
        "Review_Total_Negative_Word_Counts": "Negatif Yorum Kelime Sayısı",
        "Reviewer_Score": "Verilen Puan"
    },
    marginal_x="histogram",
    marginal_y="histogram"
)
fig_joint.update_layout(height=600)
fig_joint.show()

# Ortalama kelime sayısı karşılaştırması
word_count_comparison = df.groupby("sentiment").agg({
    "Review_Total_Negative_Word_Counts": "mean",
    "Review_Total_Positive_Word_Counts": "mean"
}).reset_index()

fig_word_comp = go.Figure()
fig_word_comp.add_trace(go.Bar(
    x=word_count_comparison["sentiment"],
    y=word_count_comparison["Review_Total_Negative_Word_Counts"],
    name="Negatif Yorum Uzunluğu",
    marker_color="#e74c3c"
))
fig_word_comp.add_trace(go.Bar(
    x=word_count_comparison["sentiment"],
    y=word_count_comparison["Review_Total_Positive_Word_Counts"],
    name="Pozitif Yorum Uzunluğu",
    marker_color="#2ecc71"
))
fig_word_comp.update_layout(
    title="📏 Duygu Kategorisine Göre Ortalama Yorum Uzunluğu",
    xaxis_title="Duygu Kategorisi",
    yaxis_title="Ortalama Kelime Sayısı",
    barmode="group",
    height=450
)
fig_word_comp.show()

# -----------------------------------------------------------------------------
# 5.2 Gap Analysis - Beklenti vs Gerçek Puan
# -----------------------------------------------------------------------------
df["score_gap"] = df["Reviewer_Score"] - df["Average_Score"]

# En çok hayal kırıklığı yaratan oteller
hotel_gaps = df.groupby("Hotel_Name").agg({
    "score_gap": "mean",
    "Average_Score": "first",
    "Reviewer_Score": "mean",
    "review": "count"
}).reset_index()
hotel_gaps.columns = ["hotel_name", "avg_gap", "hotel_avg_score", "reviewer_avg_score", "review_count"]

# Minimum 100 yorum alan oteller
hotel_gaps_filtered = hotel_gaps[hotel_gaps["review_count"] >= 100]

# En kötü 15 otel (en büyük negatif gap)
worst_hotels = hotel_gaps_filtered.nsmallest(15, "avg_gap")

fig_gap = px.bar(
    worst_hotels,
    x="avg_gap",
    y="hotel_name",
    orientation="h",
    title="😞 Hayal Kırıklığı Yaratan Oteller (Beklenti Altında Kalanlar)",
    labels={"avg_gap": "Ortalamadan Sapma (Puan Farkı)", "hotel_name": "Otel"},
    color="avg_gap",
    color_continuous_scale="Reds_r",
    hover_data={"hotel_avg_score": ":.2f", "reviewer_avg_score": ":.2f", "review_count": True}
)
fig_gap.update_layout(height=550, yaxis=dict(autorange="reversed"))
fig_gap.show()

# En iyi 15 otel (beklentinin üzerinde)
best_hotels = hotel_gaps_filtered.nlargest(15, "avg_gap")

fig_gap_best = px.bar(
    best_hotels,
    x="avg_gap",
    y="hotel_name",
    orientation="h",
    title="🌟 Beklentileri Aşan Oteller (Ortalamanın Üzerinde)",
    labels={"avg_gap": "Ortalamadan Sapma (Puan Farkı)", "hotel_name": "Otel"},
    color="avg_gap",
    color_continuous_scale="Greens",
    hover_data={"hotel_avg_score": ":.2f", "reviewer_avg_score": ":.2f", "review_count": True}
)
fig_gap_best.update_layout(height=550, yaxis=dict(autorange="reversed"))
fig_gap_best.show()

# Gap dağılımı histogram
fig_gap_dist = px.histogram(
    df,
    x="score_gap",
    nbins=50,
    title="📊 Puan Farkı Dağılımı (Reviewer Score - Average Score)",
    labels={"score_gap": "Puan Farkı"},
    color_discrete_sequence=["#3498db"]
)
fig_gap_dist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Beklenti = Gerçek")
fig_gap_dist.update_layout(height=400)
fig_gap_dist.show()



# -----------------------------------------------------------------------------
# Topic Modeling (LDA) - Negatif Yorumlardaki Konular
# -----------------------------------------------------------------------------
print("\n🔍 Topic Modeling başlatılıyor...")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Sadece negatif yorumlar
negative_reviews = df[df["label"] == 0]["review"].sample(min(20000, len(df[df["label"] == 0])), random_state=42)

# Metin temizleme
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return text

negative_reviews_clean = negative_reviews.apply(clean_text)

# Vectorize
vectorizer = CountVectorizer(max_features=1000, stop_words="english", max_df=0.95, min_df=5)
doc_term_matrix = vectorizer.fit_transform(negative_reviews_clean)

# LDA Model
n_topics = 6
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20)
lda.fit(doc_term_matrix)

# Topic kelimelerini çıkarmak
feature_names = vectorizer.get_feature_names_out()
topics = []
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
    topics.append({
        "topic": f"Konu {topic_idx + 1}",
        "words": ", ".join(top_words),
        "weight": topic.sum()
    })

topics_df = pd.DataFrame(topics)

print("\n📌 Negatif Yorumlardaki Ana Konular (LDA):")
for i, row in topics_df.iterrows():
    print(f"  Konu {i+1}: {row['words']}")

# Topic dağılımı bubble chart
fig_topics = px.scatter(
    topics_df,
    x=[1, 2, 3, 1, 2, 3],
    y=[1, 1, 1, 2, 2, 2],
    size="weight",
    text="topic",
    hover_data={"words": True},
    title="💭 Negatif Yorumlardaki Ana Konular (LDA Topic Modeling)",
    color=topics_df.index,
    color_continuous_scale="Viridis"
)
fig_topics.update_traces(textposition="middle center")
fig_topics.update_layout(
    height=500,
    xaxis=dict(showticklabels=False, title=""),
    yaxis=dict(showticklabels=False, title=""),
    showlegend=False
)
fig_topics.show()

# Topic kelimeleri bar chart
for i in range(n_topics):
    topic_words = [(feature_names[j], lda.components_[i][j]) for j in lda.components_[i].argsort()[:-11:-1]]
    topic_df = pd.DataFrame(topic_words, columns=["word", "weight"])
    
    fig_topic = px.bar(
        topic_df,
        x="weight",
        y="word",
        orientation="h",
        title=f"📝 Konu {i+1} - En Önemli Kelimeler",
        color="weight",
        color_continuous_scale="Blues"
    )
    fig_topic.update_layout(height=350, yaxis=dict(autorange="reversed"))
    fig_topic.show()

# =============================================================================
# 7. NATIONALITY-SENTIMENT CHORD DIAGRAM (Showstopper)
# =============================================================================
print("\n🎯 Chord Diagram oluşturuluyor...")

# Top 10 milliyet ve 6 şehir
top_10_nationalities = df["Reviewer_Nationality"].value_counts().head(10).index.tolist()
top_cities = ["London", "Barcelona", "Paris", "Amsterdam", "Vienna", "Milan"]

# Filter data
chord_data = df[
    (df["Reviewer_Nationality"].isin(top_10_nationalities)) & 
    (df["city"].isin(top_cities))
]

# Aggregate
chord_agg = chord_data.groupby(["Reviewer_Nationality", "city"]).agg({
    "review": "count",
    "Reviewer_Score": "mean",
    "label": "mean"  # sentiment ratio
}).reset_index()
chord_agg.columns = ["nationality", "city", "review_count", "avg_score", "positive_ratio"]


all_nodes = top_10_nationalities + top_cities
node_indices = {node: i for i, node in enumerate(all_nodes)}

sources = [node_indices[nat] for nat in chord_agg["nationality"]]
targets = [node_indices[city] for city in chord_agg["city"]]
values = chord_agg["review_count"].tolist()

# Renk: ortalama skora göre
colors = []
for score in chord_agg["avg_score"]:
    if score < 7.5:
        colors.append("rgba(231, 76, 60, 0.6)")  # Kırmızı
    elif score < 8.5:
        colors.append("rgba(241, 196, 15, 0.6)")  # Sarı
    else:
        colors.append("rgba(46, 204, 113, 0.6)")  # Yeşil

fig_chord = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=all_nodes,
        color=["#3498db"] * len(top_10_nationalities) + ["#9b59b6"] * len(top_cities)
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color=colors,
        hovertemplate="<b>%{source.label}</b> → <b>%{target.label}</b><br>" +
                      "Yorum Sayısı: %{value:,}<extra></extra>"
    )
))

fig_chord.update_layout(
    title="🎭 Milliyet-Şehir-Memnuniyet Akış Diyagramı (Chord/Sankey)",
    font_size=12,
    height=700
)
fig_chord.show()

# Alternatif: Heatmap versiyonu
pivot_chord = chord_agg.pivot(index="nationality", columns="city", values="avg_score")

fig_chord_heat = px.imshow(
    pivot_chord,
    color_continuous_scale="RdYlGn",
    aspect="auto",
    title="🗺️ Milliyet vs Şehir - Detaylı Memnuniyet Haritası",
    labels=dict(x="Şehir", y="Milliyet", color="Ortalama Puan")
)
fig_chord_heat.update_layout(height=500)
fig_chord_heat.show()

# Yorum sayısı ile ağırlıklı
pivot_count = chord_agg.pivot(index="nationality", columns="city", values="review_count")

fig_chord_count = px.imshow(
    pivot_count,
    color_continuous_scale="Blues",
    aspect="auto",
    title="📊 Milliyet vs Şehir - Yorum Yoğunluğu Haritası",
    labels=dict(x="Şehir", y="Milliyet", color="Yorum Sayısı")
)
fig_chord_count.update_layout(height=500)
fig_chord_count.show()

# =============================================================================
# 8. FINAL ÖZET
# =============================================================================
print("\n" + "="*60)
print("📊 ÖZET İSTATİSTİKLER")
print("="*60)
print(f"Toplam Yorum Sayısı: {len(df):,}")
print(f"Toplam Otel Sayısı: {df['Hotel_Name'].nunique():,}")
print(f"Genel Ortalama Puan: {df['Reviewer_Score'].mean():.2f}")
print(f"Negatif Yorum Oranı: {(df['label']==0).mean()*100:.1f}%")
print(f"Pozitif Yorum Oranı: {(df['label']==1).mean()*100:.1f}%")
print("="*60)

print("\n" + "="*60)
print("📈 GRU MODEL SONUÇLARI (Referans)")
print("="*60)
print("GRU Accuracy: 0.9669 (96.69%)")
print("GRU F1 Score: 0.9331 (93.31%)")
print("Confusion Matrix: [[18980, 412], [442, 5953]]")
print("="*60)

print("\n✅ Tüm görselleştirmeler oluşturuldu!")
