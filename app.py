"""
Interactive Streamlit Dashboard
Socio-Economic Inequality Mining & Analysis (UN SDG 10)
Nişantaşı University - Data Mining Course Project Exhibition 2024
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, silhouette_score

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Socio-Economic Inequality Mining | SDG 10",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .badge-sdg {
        background-color: #e11d48;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------
@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "data", "inequality_data.csv")
    if not os.path.exists(csv_path):
        # Fallback to current directory
        csv_path = "data/inequality_data.csv"
    df = pd.read_csv(csv_path)
    return df


df_raw = load_data()

FEATURE_NAMES = {
    "gini_coefficient": "Gini Coefficient",
    "poverty_rate_pct": "Poverty Rate (%)",
    "income_disparity_ratio": "Income Disparity Ratio (Top 10% / Bottom 10%)",
    "gender_wage_gap_pct": "Gender Wage Gap (%)",
    "education_quality_index": "Education Quality Index",
    "urban_unemployment_rate": "Urban Unemployment Rate (%)",
    "rural_unemployment_rate": "Rural Unemployment Rate (%)",
    "unemployment_disparity": "Urban-Rural Unemployment Disparity (%)",
}

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown('<span class="badge-sdg">UN SDG 10</span>', unsafe_allow_html=True)
    st.title("Reduced Inequality")
    st.markdown("**Data Mining & ML Research**")
    st.caption("Nişantaşı University • Data Mining Exhibition 2024")

    st.divider()

    st.subheader("🔍 Filters")
    all_countries = sorted(df_raw["country"].unique().tolist())
    selected_countries = st.multiselect(
        "Select Countries", options=all_countries, default=all_countries
    )

    min_yr = int(df_raw["year"].min())
    max_yr = int(df_raw["year"].max())
    selected_years = st.slider(
        "Year Range", min_value=min_yr, max_value=max_yr, value=(min_yr, max_yr)
    )

    # Filtered dataframe
    filtered_df = df_raw[
        (df_raw["country"].isin(selected_countries))
        & (df_raw["year"] >= selected_years[0])
        & (df_raw["year"] <= selected_years[1])
    ].copy()

    st.divider()

    st.subheader("👤 Project Author")
    st.markdown(
        """
        - **Irmak Kartalkayası**
        """
    )

    st.divider()
    csv_bytes = df_raw.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Clean Dataset (CSV)",
        data=csv_bytes,
        file_name="inequality_dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ---------------------------------------------------------
# Main Header
# ---------------------------------------------------------
st.markdown('<div class="main-title">🌐 Socio-Economic Inequality Mining & Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Empirical study of socio-economic disparity indicators across Turkey & OECD nations using Supervised Regression and Unsupervised Clustering</div>',
    unsafe_allow_html=True,
)

if filtered_df.empty:
    st.warning("No records match the selected filters. Please adjust the sidebar options.")
    st.stop()

# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tabs = st.tabs([
    "📊 Executive Dashboard",
    "📈 Exploratory Data Analysis",
    "🤖 Linear Regression Modeling",
    "🧩 K-Means Clustering",
    "📜 Academic Research & Poster",
])

# =========================================================
# TAB 1: Executive Dashboard
# =========================================================
with tabs[0]:
    st.subheader("Key Performance Indicators (KPIs)")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    with kpi1:
        st.metric("Total Observations", len(filtered_df), f"{len(selected_countries)} Countries")
    with kpi2:
        avg_gini = filtered_df["gini_coefficient"].mean()
        st.metric("Avg Gini Coefficient", f"{avg_gini:.3f}", f"Range: {filtered_df['gini_coefficient'].min():.2f} - {filtered_df['gini_coefficient'].max():.2f}")
    with kpi3:
        avg_pov = filtered_df["poverty_rate_pct"].mean()
        st.metric("Avg Poverty Rate", f"{avg_pov:.1f}%")
    with kpi4:
        avg_wage = filtered_df["gender_wage_gap_pct"].mean()
        st.metric("Avg Gender Wage Gap", f"{avg_wage:.1f}%")
    with kpi5:
        avg_disp = filtered_df["income_disparity_ratio"].mean()
        st.metric("Avg Income Disparity", f"{avg_disp:.1f}:1")

    st.write("")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        fig_bar = px.bar(
            filtered_df.groupby("country")["gini_coefficient"].mean().reset_index(),
            x="country",
            y="gini_coefficient",
            color="country",
            title="Mean Gini Index by Country (Higher = More Unequal)",
            labels={"gini_coefficient": "Average Gini", "country": "Country"},
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        fig_bar.update_layout(showlegend=False, template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_chart2:
        fig_scatter_kpi = px.scatter(
            filtered_df,
            x="poverty_rate_pct",
            y="gender_wage_gap_pct",
            color="country",
            size="income_disparity_ratio",
            hover_data=["year", "gini_coefficient"],
            title="Poverty Rate vs. Gender Wage Gap (Bubble Size: Income Disparity)",
            labels={
                "poverty_rate_pct": "Poverty Rate (%)",
                "gender_wage_gap_pct": "Gender Wage Gap (%)",
            },
            template="plotly_white",
        )
        st.plotly_chart(fig_scatter_kpi, use_container_width=True)

    with st.expander("📋 View Clean Data Table", expanded=False):
        st.dataframe(filtered_df, use_container_width=True)


# =========================================================
# TAB 2: Exploratory Data Analysis
# =========================================================
with tabs[1]:
    st.subheader("Historical Trends & Comparative Dynamics")

    indicator_choice = st.selectbox(
        "Select Metric to Track Over Time",
        options=list(FEATURE_NAMES.keys()),
        format_func=lambda x: FEATURE_NAMES[x],
        index=0,
    )

    fig_trend = px.line(
        filtered_df.sort_values(by="year"),
        x="year",
        y=indicator_choice,
        color="country",
        markers=True,
        title=f"Evolution of {FEATURE_NAMES[indicator_choice]} Across Years",
        labels={"year": "Year", indicator_choice: FEATURE_NAMES[indicator_choice]},
        template="plotly_white",
    )
    fig_trend.update_layout(hovermode="x unified")
    st.plotly_chart(fig_trend, use_container_width=True)

    st.divider()
    st.subheader("Correlation Matrix Heatmap")
    num_cols = list(FEATURE_NAMES.keys())
    corr_matrix = filtered_df[num_cols].corr()

    fig_corr = px.imshow(
        corr_matrix,
        x=[FEATURE_NAMES[c] for c in num_cols],
        y=[FEATURE_NAMES[c] for c in num_cols],
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Pearson Correlation Between Socio-Economic Indicators",
        text_auto=".2f",
        aspect="auto",
    )
    fig_corr.update_layout(template="plotly_white")
    st.plotly_chart(fig_corr, use_container_width=True)


# =========================================================
# TAB 3: Linear Regression Modeling
# =========================================================
with tabs[2]:
    st.subheader("Supervised Learning: Linear Regression Model")
    st.markdown(
        "Analyze the linear relationship between socio-economic indicators as demonstrated in the research progress reports."
    )

    reg_col1, reg_col2 = st.columns([1, 2])

    with reg_col1:
        st.markdown("#### Model Features")
        x_feature = st.selectbox(
            "Independent Variable (X)",
            options=list(FEATURE_NAMES.keys()),
            format_func=lambda x: FEATURE_NAMES[x],
            index=1,  # poverty_rate_pct
        )
        y_feature = st.selectbox(
            "Dependent Variable (Y)",
            options=[k for k in FEATURE_NAMES.keys() if k != x_feature],
            format_func=lambda x: FEATURE_NAMES[x],
            index=2,  # gender_wage_gap_pct
        )

        # Train model
        X_vals = filtered_df[[x_feature]].values
        y_vals = filtered_df[y_feature].values

        lr = LinearRegression()
        lr.fit(X_vals, y_vals)
        y_pred = lr.predict(X_vals)

        r2 = r2_score(y_vals, y_pred)
        mse = mean_squared_error(y_vals, y_pred)
        rmse = np.sqrt(mse)
        slope = lr.coef_[0]
        intercept = lr.intercept_

        st.markdown("#### Regression Results")
        st.latex(f"Y = {slope:.4f} \\cdot X + {intercept:.4f}")
        st.metric("R² Score (Variance Explained)", f"{r2:.4f}")
        st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")

        st.divider()
        st.markdown("#### 🔮 Live Prediction Simulator")
        min_x = float(filtered_df[x_feature].min())
        max_x = float(filtered_df[x_feature].max())
        mean_x = float(filtered_df[x_feature].mean())
        sim_val = st.slider(
            f"Input value for {FEATURE_NAMES[x_feature]}",
            min_value=round(min_x, 2),
            max_value=round(max_x, 2),
            value=round(mean_x, 2),
            step=round((max_x - min_x) / 50.0, 2) or 0.1,
        )
        pred_out = lr.predict([[sim_val]])[0]
        st.success(f"**Predicted {FEATURE_NAMES[y_feature]}:** `{pred_out:.3f}`")

    with reg_col2:
        # Plot regression line
        x_smooth = np.linspace(X_vals.min(), X_vals.max(), 100).reshape(-1, 1)
        y_smooth = lr.predict(x_smooth)

        fig_reg = go.Figure()
        fig_reg.add_trace(
            go.Scatter(
                x=filtered_df[x_feature],
                y=filtered_df[y_feature],
                mode="markers",
                marker=dict(size=10, color="#2563eb", opacity=0.8),
                text=[f"{c} ({y})" for c, y in zip(filtered_df["country"], filtered_df["year"])],
                name="Data Points",
            )
        )
        fig_reg.add_trace(
            go.Scatter(
                x=x_smooth.flatten(),
                y=y_smooth,
                mode="lines",
                line=dict(color="#dc2626", width=3),
                name=f"Fit Line: R²={r2:.2f}",
            )
        )
        fig_reg.update_layout(
            title=f"OLS Linear Regression: {FEATURE_NAMES[y_feature]} vs {FEATURE_NAMES[x_feature]}",
            xaxis_title=FEATURE_NAMES[x_feature],
            yaxis_title=FEATURE_NAMES[y_feature],
            template="plotly_white",
            height=500,
        )
        st.plotly_chart(fig_reg, use_container_width=True)


# =========================================================
# TAB 4: K-Means Clustering
# =========================================================
with tabs[3]:
    st.subheader("Unsupervised Learning: K-Means Clustering")
    st.markdown(
        "Segmenting socio-economic profiles using multi-attribute clustering to detect systemic inequality groups."
    )

    cluster_col1, cluster_col2 = st.columns([1, 2])

    with cluster_col1:
        st.markdown("#### Clustering Configuration")
        k_clusters = st.slider("Number of Clusters (K)", min_value=2, max_value=5, value=3)

        default_cluster_features = [
            "gini_coefficient",
            "poverty_rate_pct",
            "income_disparity_ratio",
            "gender_wage_gap_pct",
            "education_quality_index",
        ]
        selected_cluster_features = st.multiselect(
            "Clustering Features",
            options=list(FEATURE_NAMES.keys()),
            default=default_cluster_features,
            format_func=lambda x: FEATURE_NAMES[x],
        )

        if len(selected_cluster_features) < 2:
            st.error("Please select at least 2 features for clustering.")
            st.stop()

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(filtered_df[selected_cluster_features])

        # Fit KMeans
        km = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        c_labels = km.fit_predict(X_scaled)

        cluster_df = filtered_df.copy()
        cluster_df["Cluster"] = [f"Group {c}" for c in c_labels]

        sil = silhouette_score(X_scaled, c_labels) if len(filtered_df) > k_clusters else 0.0

        st.metric("Silhouette Score (Clustering Quality)", f"{sil:.3f}")
        st.caption(
            "Score ranges from -1 to 1. Values > 0.4 indicate well-separated and cohesive clusters."
        )

        # Elbow visualization
        st.markdown("##### Elbow Curve Preview")
        inertias = []
        for k in range(2, 6):
            m_temp = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
            inertias.append(m_temp.inertia_)

        fig_elbow = px.line(
            x=list(range(2, 6)),
            y=inertias,
            markers=True,
            title="Elbow Method (Inertia vs K)",
            labels={"x": "K", "y": "Inertia"},
            template="plotly_white",
        )
        fig_elbow.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_elbow, use_container_width=True)

    with cluster_col2:
        st.markdown("#### Cluster Visualizer")
        plot_x = st.selectbox(
            "Scatter X-Axis",
            options=selected_cluster_features,
            format_func=lambda x: FEATURE_NAMES[x],
            index=0,
            key="cl_x",
        )
        plot_y = st.selectbox(
            "Scatter Y-Axis",
            options=selected_cluster_features,
            format_func=lambda x: FEATURE_NAMES[x],
            index=1 if len(selected_cluster_features) > 1 else 0,
            key="cl_y",
        )

        fig_cluster = px.scatter(
            cluster_df,
            x=plot_x,
            y=plot_y,
            color="Cluster",
            hover_name="country",
            hover_data=["year", "gini_coefficient", "poverty_rate_pct"],
            title=f"Cluster Segments: {FEATURE_NAMES[plot_x]} vs {FEATURE_NAMES[plot_y]}",
            labels={plot_x: FEATURE_NAMES[plot_x], plot_y: FEATURE_NAMES[plot_y]},
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Bold,
            height=500,
        )
        fig_cluster.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
        st.plotly_chart(fig_cluster, use_container_width=True)

    st.markdown("#### 📊 Cluster Profiles (Group Averages)")
    cluster_means = cluster_df.groupby("Cluster")[selected_cluster_features].mean().round(3)
    cluster_means.rename(columns=FEATURE_NAMES, inplace=True)
    st.dataframe(cluster_means, use_container_width=True)


# =========================================================
# TAB 5: Academic Research & Poster
# =========================================================
with tabs[4]:
    st.subheader("Academic Research Exhibition (Nişantaşı University 2024)")

    poster_path = os.path.join(os.path.dirname(__file__), "assets", "academic_poster_preview.png")
    if os.path.exists(poster_path):
        st.image(
            poster_path,
            caption="Academic Research Poster: Reduced Inequality (SDG 10) - Nişantaşı University Project Exhibition 2024",
            use_container_width=True,
        )
    else:
        st.info("Poster preview image available in repository docs/ directory.")

    st.divider()

    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown("### 🎯 Problem Statement & Objectives")
        st.write(
            """
            Societal inequalities across income distribution, education access, and healthcare systems
            represent major structural challenges for modern economies.
            
            This research applies **data mining methodologies** to quantify and predict multi-dimensional 
            inequality disparities in line with **United Nations Sustainable Development Goal 10 (SDG 10)**.
            
            **Core Objectives:**
            - Measure the empirical relationship between gender pay gaps and poverty thresholds.
            - Segment countries and years into distinct socio-economic clusters.
            - Provide data-driven recommendations for targeted economic and social interventions.
            """
        )

    with col_a2:
        st.markdown("### 📚 Academic References & Bibliography")
        st.markdown(
            """
            1. **Chen, L., Wang, H., & Liu, M. (2018).** *Cluster Analysis for Identifying Socio-Economic Factors Contributing to Inequality.* Journal of Public Policy and Administration, 20(2), 78-92.
            2. **Smith, J., & Johnson, R. (2020).** *Addressing Income Inequality through Machine Learning Techniques.* Journal of Social Sciences, 12(3), 45-58.
            3. **Acemoglu, D., Johnson, S., & Robinson, J. A. (2005).** *Institutions as a fundamental cause of long-run growth.* NBER Working Paper Series, No. 11511.
            4. **Pogge, T. (2015).** *Eliminating Extreme Inequality: A Sustainable Development Goal, 2015–2030.* Ethics & International Affairs, 29(1), 89-98.
            5. **Anand, S., & Segal, P. (2008).** *What do we know about global income inequality?* Journal of Economic Literature, 46(1), 57–94.
            """
        )

# ---------------------------------------------------------
# Vercel Entrypoint Exports (app, application, handler)
# ---------------------------------------------------------
def app(environ, start_response):
    """Fallback WSGI entrypoint for Vercel deployment."""
    html_file = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_file):
        with open(html_file, "rb") as f:
            body = f.read()
    else:
        body = b"<h1>Socio-Economic Inequality Mining</h1>"
    start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
    return [body]

application = app
handler = app

