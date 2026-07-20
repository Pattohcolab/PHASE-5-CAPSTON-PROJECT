# ==========================================================
# Predicting Acute Food Insecurity Risk in Kenyan Counties
# Streamlit Application
# ==========================================================

# -------------------------
# Import Libraries
# -------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import plotly.express as px
import shap

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Food Insecurity Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Load Saved Artifacts
# -------------------------

MODEL_PATH = "models/best_model.pkl"
IMPUTER_PATH = "models/imputer.pkl"
FEATURES_PATH = "models/feature_columns.pkl"
METADATA_PATH = "models/model_metadata.pkl"

# -------------------------
# Load Evaluation Files
# -------------------------

metrics_df = pd.read_csv("metrics.csv")
cm_df = pd.read_csv("confusion_matrix.csv", index_col=0)
roc_df = pd.read_csv("roc_curve.csv")
feature_importance = pd.read_csv("feature_importance.csv")

metadata = joblib.load(METADATA_PATH)

model = joblib.load(MODEL_PATH)
imputer = joblib.load(IMPUTER_PATH)
feature_cols = joblib.load(FEATURES_PATH)
metadata = joblib.load(METADATA_PATH)

# -------------------------
# Load Prediction Dataset
# -------------------------

model_df = pd.read_csv("model_df.csv")

model_df["reporting_date"] = pd.to_datetime(
    model_df["reporting_date"]
)

# -------------------------
# Load Kenya GeoJSON
# -------------------------
with open("assets/kenya_counties.geojson", "r", encoding="utf-8") as f:
    kenya_geojson = json.load(f)

# -------------------------
# Sidebar
# -------------------------
st.sidebar.image(
    "assets/logo.png",
    width=120
)

st.sidebar.title("Food Insecurity Dashboard")

st.sidebar.success("Random Forest Classifier")

st.sidebar.write("🇰🇪 47 Counties")

st.sidebar.write("📊 13 Features")

st.sidebar.write("🗓 Updated: July 2026")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📈 Make Prediction",
        "📊 County Risk Bulletin",
        "📉 Model Performance",
        "ℹ️ About"
    ]
)

st.sidebar.divider()

st.sidebar.info("""
                ### Project
                
                **Acute Food Insecurity Prediction**
                Machine Learning Capstone Project
                Random Forest Classifier
                Kenya (47 Counties)
                """)

# -------------------------
# HOME PAGE
# -------------------------

if page == "🏠 Home":

    st.image(
        "assets/kenya_banner.jpg",
        use_container_width=True
        )
    

    st.markdown("""
                # 🌍 Predicting Acute Food Insecurity Risk in Kenyan Counties
                 
                ### Machine Learning Early Warning System
                ---
                """)

    st.markdown("""
    ### Machine Learning Early Warning System

    This application predicts the likelihood that a Kenyan county
    will experience **Acute Food Insecurity (IPC Phase 3+)**
    using a trained **Random Forest Classifier**.

    The objective is to support humanitarian agencies and decision-makers
    by providing an early warning tool for identifying counties
    at greatest risk.
    """)

    st.divider()

    # ====================================
    # Dashboard Cards
    # ====================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "🇰🇪 Counties",
        "47"
    )

    col2.metric(
        "📊 Features",
        metadata["number_of_features"]
    )

    col3.metric(
        "🤖 Model",
        metadata["algorithm"]
    )

    col4.metric(
        "📅 Training",
        metadata["training_date"]
    )

    st.divider()

    # ====================================
    # Project Overview
    # ====================================

    left, right = st.columns([2,1])

    with left:

        st.subheader("🎯 Project Objective")

        st.write("""
The goal of this project is to predict food insecurity before a crisis occurs.

The model combines:

- 🌧 Rainfall indicators
- 🌽 Food prices
- 📈 Historical food insecurity
- 🌾 Seasonal patterns

to estimate the probability that a county will experience
**IPC Phase 3 or above**.

The predictions can support:

- Early warning systems
- Resource allocation
- Humanitarian response planning
- Food security monitoring
""")

    with right:

        st.subheader("📚 Data Sources")

        st.success("✅ FEWS NET IPC")

        st.success("✅ CHIRPS Rainfall")

        st.success("✅ WFP Market Prices")

    st.divider()

    # ====================================
    # CRISP-DM Workflow
    # ====================================

    st.subheader("🔄 Project Workflow")

    workflow = pd.DataFrame({

        "Phase":[
            "Business Understanding",
            "Data Understanding",
            "Data Preparation",
            "Modeling",
            "Evaluation",
            "Deployment"
        ],

        "Status":[
            "✅",
            "✅",
            "✅",
            "✅",
            "✅",
            "✅"
        ]

    })

    st.dataframe(
        workflow,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ====================================
    # Quick Navigation
    # ====================================

    st.subheader("🚀 Quick Navigation")

    c1, c2 = st.columns(2)

    with c1:

        st.info("""
### 📈 Make Prediction

Predict the food insecurity risk
for a county using rainfall,
food prices and historical indicators.
""")

        st.info("""
### 📊 County Risk Bulletin

View monthly county risk maps,
risk summaries and download reports.
""")

    with c2:

        st.info("""
### 📉 Model Performance

Explore evaluation metrics,
ROC Curve,
Confusion Matrix
and Feature Importance.
""")

        st.info("""
### ℹ️ About

Learn more about the datasets,
CRISP-DM methodology,
and technologies used.
""")

    st.divider()

    st.success(
        "🏆 Best Model Selected: Random Forest"
        
        "Best Algorithm deployed: Random Forest Classifier"
    )

    st.divider()

    st.caption(
        """
        Predicting Acute Food Insecurity Risk in Kenyan Counties
        Machine Learning Capstone Project
        Flatiron School • Data Alchemist • 2026
        """
        )


# -------------------------
# PREDICTION PAGE
# -------------------------
elif page == "📈 Make Prediction":

    st.title("📈 Food Insecurity Prediction")

# ---------------------------------------
# Load historical dataset
# ---------------------------------------

    model_df = pd.read_csv("model_df.csv")

    model_df["reporting_date"] = pd.to_datetime(
        model_df["reporting_date"]
        )

    # ---------------------------------------
    # County selector
    # ---------------------------------------
    
    st.subheader("Select County and Reporting Month")
    selected_county = st.selectbox(
        "County",
        sorted(model_df["county"].unique())
        )
    
    county_months = (
        model_df[
            model_df["county"] == selected_county
            ]["reporting_date"]
            .sort_values()
            .unique()
        )
    
    selected_month = st.selectbox(
        "Reporting Month",
        county_months,
        format_func=lambda x: pd.to_datetime(x).strftime("%B %Y")
        )
    
    selected_record = model_df[
        (model_df["county"] == selected_county)
        &
        (model_df["reporting_date"] == pd.to_datetime(selected_month))
        ]


    if selected_record.empty:
        st.warning(
            "No data available for this county and month."
            )
        st.stop()

    record = selected_record.iloc[0]

    st.write("Enter the predictor values below and click **Predict**.")


    # -------------------------
    # Rainfall
    # -------------------------
    st.markdown("## 🌧 Rainfall Indicators")

    record = selected_record.iloc[0]

    rfq_mean = st.number_input(
        "Quarterly Rainfall (RFQ Mean)",
        value=float(record["rfq_mean"])
    )

    r3q_mean = st.number_input(
        "3-Month Rainfall Mean",
        value=float(record["r3q_mean"])
    )

    r1q_mean = st.number_input(
        "1-Month Rainfall Mean",
        value=float(record["r1q_mean"])
    )

    # -------------------------
    # Food Prices
    # -------------------------
    st.markdown("## 🌽 Food Prices")

    staple_price_kes = st.number_input(
        "Staple Price (KES)",
        value=float(record["staple_price_kes"])
    )

    staple_price_usd = st.number_input(
        "Staple Price (USD)",
        value=float(record["staple_price_usd"])
    )

    has_price_data = st.selectbox(
        "Price Data Available",
        [0, 1],
        index=int(record["has_price_data"])
    )

    price_mom_change = st.number_input(
        "Monthly Price Change",
        value=float(record["price_mom_change"])
    )

    # -------------------------
    # Historical Indicators
    # -------------------------
    st.subheader("📈 Historical Indicators")

    prev_crisis_value = (
        int(record["prev_crisis"])
        if pd.notna(record["prev_crisis"])
        else 0
        )

    crisis_streak_value = (
        int(record["crisis_streak"])
        if pd.notna(record["crisis_streak"])
        else 0
        )

    drought_streak_value = (
        int(record["drought_streak"])
        if pd.notna(record["drought_streak"])
        else 0
        )

    prev_crisis = st.selectbox(
        "Previous Crisis",
        [0, 1],
        index=prev_crisis_value
        )
    
    crisis_streak = st.number_input(
        "Crisis Streak",
        min_value=0,
        value=crisis_streak_value
        )
    
    drought_streak = st.number_input(
        "Drought Streak",
        min_value=0,
        value=drought_streak_value
        )

    # -------------------------
    # County Characteristics
    # -------------------------
    st.subheader("🏜 County Characteristics")

    is_asal = st.selectbox(
        "ASAL County",
        [0, 1],
        index=int(record["is_asal"])
        )

    season_long_rains = st.selectbox(
        "Long Rains Season",
        [0, 1],
        index=int(record["season_long_rains"])
        )

    season_short_rains = st.selectbox(
        "Short Rains Season",
        [0, 1],
        index=int(record["season_short_rains"])
        )

    if st.button("Predict"):

    # Create input dataframe
       input_df = pd.DataFrame({
           'rfq_mean': [rfq_mean],
           'r3q_mean': [r3q_mean],
           'r1q_mean': [r1q_mean],
           'staple_price_kes': [staple_price_kes],
           'staple_price_usd': [staple_price_usd],
           'has_price_data': [has_price_data],
           'price_mom_change': [price_mom_change],
           'prev_crisis': [prev_crisis],
           'crisis_streak': [crisis_streak],
           'drought_streak': [drought_streak],
           'is_asal': [is_asal],
           'season_long_rains': [season_long_rains],
           'season_short_rains': [season_short_rains]
        })

       # Ensure feature order matches training
       input_df = input_df[feature_cols]

       # Apply the trained imputer
       input_imputed = imputer.transform(input_df)

       # Predict
       prediction = model.predict(input_imputed)[0]

       probability = model.predict_proba(input_imputed)[0][1]

       # -------------------------
       # Display Results
       # -------------------------
       st.divider()

       st.header("📊 Prediction Results")

       probability_percent = probability * 100

       col1, col2 = st.columns(2)

       with col1:
           st.metric(
               "Predicted Probability",
               f"{probability_percent:.1f}%"
               )

           st.progress(float(probability))

       with col2:
           st.metric(
               "Predicted Class",
               "Crisis" if prediction == 1 else "No Crisis"
               )
           
           st.progress(float(probability))

       # Risk Level
       if probability < 0.30:
           st.success("🟢 LOW RISK")

           recommendation = """
           Current indicators suggest a relatively low likelihood of acute food insecurity.
           Continue routine monitoring of rainfall and food prices.
           """

       elif probability < 0.60:
           st.warning("🟡 MODERATE RISK")

           recommendation = """
           Conditions indicate moderate risk.
           Authorities should increase monitoring and prepare contingency measures.
           """

       else:
           st.error("🔴 HIGH RISK")

           recommendation = """
           High probability of Acute Food Insecurity (IPC Phase 3+).
           Humanitarian agencies should prioritize detailed assessments,
           food assistance planning, and early intervention.
           """

       st.progress(float(probability))

       st.subheader("📋 Recommended Actions")
       st.info(recommendation)

       st.divider()

       st.subheader("🧠 Why did the model make this prediction?")

# Create SHAP explainer
       explainer = shap.TreeExplainer(model)

# Calculate SHAP values for this prediction
       shap_values = explainer.shap_values(input_imputed)

# Binary classification:
# class 1 = Acute Food Insecurity
       values = shap_values[1][0]

       shap_df = pd.DataFrame({
           "Feature": feature_cols,
           "SHAP Value": values
           })

# Largest impacts first
       shap_df["Impact"] = shap_df["SHAP Value"].abs()

       shap_df = (
           shap_df
           .sort_values("Impact", ascending=False)
           .drop(columns="Impact")
           )
       
       st.write(
           "Top factors influencing this prediction:"
           )

       def explain(value):
        if value > 0:
            return "⬆ Increased Risk"
        
        elif value < 0:
            return "⬇ Reduced Risk"
        return "No Effect"

       shap_df["Direction"] = shap_df["SHAP Value"].apply(explain)
       
# ---------------------------------
# SHAP Contribution Chart
# ---------------------------------

       plot_df = (
           shap_df.head(10)
           .sort_values("SHAP Value")
           )

       fig = px.bar(
           plot_df,
           x="SHAP Value",
           y="Feature",
           orientation="h",
           color="SHAP Value",
           color_continuous_scale="RdYlGn_r",
           title="Top Factors Influencing this Prediction"
           )

       fig.update_layout(
           yaxis_title="",
           xaxis_title="SHAP Contribution",
           height=500
           )

       st.plotly_chart(
           fig,
           use_container_width=True
           )

       st.dataframe(
           shap_df[
               [
                   "Feature",
                   "Direction",
                   "SHAP Value"
                   ]
                   ].head(10),
                   use_container_width=True
                   )

       st.dataframe(
           shap_df.head(10),
           use_container_width=True
           )

       st.write(f"### Probability: **{probability_percent:.1f}%**")

       confidence = max(probability, 1 - probability)

       st.write(f"### Prediction Confidence: **{confidence:.1%}**")

       st.subheader("💡 Recommendation")

       st.info(recommendation)

       st.divider()

       st.subheader("📈 County Risk Trend")

# Get all records for the selected county
       county_history = model_df[
           model_df["county"] == selected_county
       ].copy()

# Prepare features
       X_history = county_history[feature_cols]

       X_history = pd.DataFrame(
           imputer.transform(X_history),
           columns=feature_cols
           )

# Predict probabilities
       county_history["crisis_probability"] = model.predict_proba(X_history)[:, 1]

# Plot over time
       fig = px.line(
           county_history,
           x="reporting_date",
           y="crisis_probability",
           markers=True,
           title=f"Food Insecurity Risk Trend for {selected_county}"
           )

       fig.update_layout(
           yaxis_title="Predicted Probability",
           xaxis_title="Reporting Date"
           )

       st.plotly_chart(
           fig,
           use_container_width=True
           )

       st.subheader("📋 Historical Predictions")

       history_table = county_history[
           ["reporting_date", "crisis_probability"]
           ].copy()

       history_table["crisis_probability"] = (
       history_table["crisis_probability"] * 100
       ).round(1)
       history_table.columns = [
           "Reporting Date",
           "Predicted Probability (%)"
           ]

       st.dataframe(
           history_table,
           use_container_width=True
           )


       st.divider()

       st.caption(
        """
        Predicting Acute Food Insecurity Risk in Kenyan Counties
        Machine Learning Capstone Project
        Flatiron School • Data Alchemist • 2026
        """
        )


# -------------------------
# COUNTY RISK BULLETIN
# -------------------------
elif page == "📊 County Risk Bulletin":

    st.title("📊 County Risk Bulletin")

    # Load prediction data
    bulletin_df = pd.read_csv("model_df.csv")

    bulletin_df["reporting_date"] = pd.to_datetime(
        bulletin_df["reporting_date"]
    )

    # Select month
    available_dates = (
        bulletin_df["reporting_date"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    selected_date = st.selectbox(
        "Select Reporting Month",
        available_dates,
        format_func=lambda x: x.strftime("%B %Y")
    )

    bulletin_df = bulletin_df[
        bulletin_df["reporting_date"] == selected_date
    ].copy()

    # Prepare features

    X = pd.DataFrame(
        imputer.transform(
            bulletin_df[feature_cols]
        ),
        columns=feature_cols
    )

    # Predictions
    bulletin_df["crisis_probability"] = model.predict_proba(X)[:, 1]

    bulletin_df["prediction"] = model.predict(X)

    # Threshold saved during training
    threshold = metadata["threshold"]
    

    # Moderate-risk threshold (half of the model threshold)
    moderate_threshold = threshold / 2

    bulletin_df["risk_level"] = np.select(
        [
            bulletin_df["crisis_probability"] >= threshold,
            bulletin_df["crisis_probability"] >= moderate_threshold
        ],
        [
            "High Risk",
            "Moderate Risk"
        ],
        default="Low Risk"
    )

    bulletin_df = bulletin_df.sort_values(
        "crisis_probability",
        ascending=False
    )


    high = (bulletin_df["risk_level"]=="High Risk").sum()
    moderate = (bulletin_df["risk_level"]=="Moderate Risk").sum()
    low = (bulletin_df["risk_level"]=="Low Risk").sum()

    st.subheader("Summary")

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "🔴 High Risk Counties",
        f"{high} Counties"
        )
    
    c2.metric(
        "🟠 Moderate Risk Counties",
        f"{moderate} Counties"
        )
    
    c3.metric(
        "🟢 Low Risk Counties",
        f"{low} Counties"
        )
    
    c4.metric(
        "📈 Average Probability",
        f"{bulletin_df['crisis_probability'].mean():.1%}"
        )

    st.divider()

    # -------------------------
    # Kenya County Risk Map
    # -------------------------

    st.subheader("🗺 Kenya County Risk Map")

    # Sort by probability (highest first)
    bulletin_df = bulletin_df.sort_values(
        "crisis_probability",
        ascending=False
    )

    fig = px.choropleth(
        bulletin_df,
        geojson=kenya_geojson,
        featureidkey="properties.COUNTY",
        locations="county",color="crisis_probability",

        hover_name="county",

        hover_data={
            "crisis_probability":":.1%",
            "crisis_probability":":.1%",
            "prediction":False
        },

        color_continuous_scale="Reds",
        range_color=(0, 1),

        labels={
            "crisis_probability":"Crisis Probability"
        }
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        showcountries=False,
        showcoastlines=False,
        showframe=False
    )

    fig.update_layout(
        height=700,
        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0
        ),

        coloraxis_colorbar=dict(
            title="Probability",
            tickformat=".0%"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # Bar chart

    st.subheader("County Risk Distribution")

    risk_counts = (
        bulletin_df["risk_level"]
        .value_counts()
        .reset_index()
    )

    risk_counts.columns = [
        "Risk Level",
        "Count"
    ]

    fig = px.bar(
        risk_counts,
        x="Risk Level",
        y="Count",
        text="Count",
        color="Risk Level"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    fig_pie = px.pie(
        risk_counts,
        names="Risk Level",
        values="Count",
        title="County Risk Distribution"
        )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
        )
    
    st.divider()

    # Top 10 Counties

    st.subheader("🚨 Top 10 Counties Requiring Attention")

    top10 = bulletin_df.sort_values(
        "crisis_probability",
        ascending=False
        ).head(10)

    def color_risk(v):

        if v=="High Risk":
            return "background-color:#ffcccc"
        
        elif v=="Moderate Risk":
            return "background-color:#fff2cc"
        
        return "background-color:#ccffcc"
    
    st.dataframe(
        top10[
            [
                "county",
                "crisis_probability",
                "risk_level"
            ]
        
        ].style
           .format(
               {"crisis_probability":"{:.1%}"}
           )
           .applymap(
               color_risk,
               subset=["risk_level"]
           ),

           use_container_width=True
    )

    st.divider()

    # Complete Bulletin

    st.subheader("All County Predictions")

    st.dataframe(
        bulletin_df[
            [
                "county",
                "crisis_probability",
                "risk_level"
            ]
        ].style.format(
            {"crisis_probability":"{:.1%}"}
        ),

        use_container_width=True
    )

    # Download button

    csv = bulletin_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download County Bulletin",

        csv,

        "county_risk_bulletin.csv",

        "text/csv"
    )


    st.divider()

    st.caption(
        """
        Predicting Acute Food Insecurity Risk in Kenyan Counties
        Machine Learning Capstone Project
        Flatiron School • Data Alchemist • 2026
        """
        )
    


# -------------------------
# MODEL PERFORMANCE
# -------------------------
elif page == "📉 Model Performance":

    st.title("📉 Model Performance Dashboard")

    st.markdown(
        "Evaluation results for the trained **Random Forest Classifier**."
    )

    st.divider()

    # ============================
    # Performance Metrics
    # ============================

    st.subheader("📊 Model Metrics")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Accuracy",
        f"{metrics_df.loc[0,'Value']:.3f}"
    )

    c2.metric(
        "Precision",
        f"{metrics_df.loc[1,'Value']:.3f}"
    )

    c3.metric(
        "Recall",
        f"{metrics_df.loc[2,'Value']:.3f}"
    )

    c4.metric(
        "F1 Score",
        f"{metrics_df.loc[3,'Value']:.3f}"
    )

    c5.metric(
        "ROC AUC",
        f"{metrics_df.loc[4,'Value']:.3f}"
    )

    st.divider()

# ============================
# Confusion Matrix
# ============================

    st.subheader("🎯 Confusion Matrix")
    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        labels=dict(
            x="Predicted",
            y="Actual",
            color="Count"
            )
            )
    
    fig_cm.update_layout(
        height=500
        )
    
    st.plotly_chart(
        fig_cm,
        use_container_width=True
        )
    
# ============================
# ROC Curve
# ============================

    st.divider()
    
    st.subheader("📈 ROC Curve")
    
    fig_roc = px.line(
        roc_df,
        x="FPR",
        y="TPR",
        title="Receiver Operating Characteristic (ROC) Curve"
        )

    # Add diagonal reference line
    fig_roc.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(
            dash="dash",
            color="gray"
            )
        )

    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500
        )

    st.plotly_chart(
        fig_roc,
        use_container_width=True
        )
    
   # ============================
   # Feature Importance
   # ============================

    st.divider()

    st.subheader("🌳 Feature Importance")
    
    fig_importance = px.bar(
        feature_importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Feature Importance"
        )

    fig_importance.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=600
        )

    st.plotly_chart(
        fig_importance,
        use_container_width=True
        )
    
    st.divider()

    st.header("🧠 Model Explainability")

    st.write("""
             SHAP (SHapley Additive exPlanations) is an explainable AI technique that
             shows how each feature influences the model's predictions.
             
             Positive SHAP values increase the likelihood of Acute Food Insecurity,
             while negative values decrease it.
             
             These plots help interpret why the Random Forest model makes its predictions.
             """)
    
    st.subheader("Global SHAP Feature Importance")

    st.image(
        "assets/shap_bar.png",
        use_container_width=True
        )
    
    st.subheader("SHAP Summary Plot")

    st.image(
        "assets/shap_summary.png",
        use_container_width=True
        )
    

    

    st.divider()

    st.subheader("📥 Download Evaluation Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            "Download Metrics CSV",
            metrics_df.to_csv(index=False),
            file_name="metrics.csv",
            mime="text/csv"
            )
    
    with col2:
        st.download_button(
            "Download Feature Importance CSV",
            feature_importance.to_csv(index=False),
            file_name="feature_importance.csv",
            mime="text/csv"
            )

    st.divider()

    st.caption(
        """
        Predicting Acute Food Insecurity Risk in Kenyan Counties
        Machine Learning Capstone Project
        Flatiron School • Data Alchemist • 2026
        """
        )

# -------------------------
# ABOUT
# -------------------------


elif page == "ℹ️ About":

    st.title("ℹ️ About the Project")

    st.markdown("""
    ## Predicting Acute Food Insecurity Risk in Kenyan Counties
                
    This application was developed as part of a Machine Learning Capstone Project.
    It uses historical rainfall, food prices, seasonal patterns and previous crisis
    information to estimate the probability that a Kenyan county will experience
    **Acute Food Insecurity (IPC Phase 3 or above).**
    """)

    st.divider()

    # --------------------------------------------------
    # Project Information
    # --------------------------------------------------

    st.subheader("🎯 Project Objective")

    st.write("""
    The objective of this project is to support early warning systems by
    predicting food insecurity risk before a crisis occurs.

    The model enables humanitarian organizations and decision-makers to
    identify counties at greatest risk and prioritize interventions.
    """)

    st.divider()

    # --------------------------------------------------
    # Datasets
    # --------------------------------------------------

    st.subheader("📂 Datasets Used")

    datasets = pd.DataFrame({
        "Dataset":[
            "FEWS NET IPC",
            "CHIRPS Rainfall",
            "WFP Market Prices"
        ],
        "Purpose":[
            "Food insecurity classifications",
            "Rainfall indicators",
            "Food price monitoring"
        ]
    })

    st.dataframe(
        datasets,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------
    # Features
    # --------------------------------------------------

    st.subheader("📊 Model Features")

    feature_table = pd.DataFrame({
        "Feature": feature_cols
    })

    st.dataframe(
        feature_table,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------
    # Algorithms
    # --------------------------------------------------

    st.subheader("🤖 Machine Learning Algorithms Evaluated")

    algorithms = pd.DataFrame({
        "Algorithm":[
            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ],
        "Status":[
            "Evaluated",
            "Selected",
            "Evaluated"
        ]
    })

    st.dataframe(
        algorithms,
        use_container_width=True
    )

    st.success("🏆 Random Forest achieved the best predictive performance and was selected for deployment.")

    st.divider()

    # --------------------------------------------------
    # CRISP-DM
    # --------------------------------------------------

    st.subheader("🔄 CRISP-DM Methodology")

    crisp = pd.DataFrame({
        "Phase":[
            "1. Business Understanding",
            "2. Data Understanding",
            "3. Data Preparation",
            "4. Modeling",
            "5. Evaluation",
            "6. Deployment"
        ],
        "Description":[
            "Define prediction objective",
            "Explore IPC, rainfall and market data",
            "Clean, merge and engineer features",
            "Train multiple machine learning models",
            "Compare model performance",
            "Deploy using Streamlit"
        ]
    })

    st.dataframe(
        crisp,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------
    # Technologies
    # --------------------------------------------------

    st.subheader("🛠 Technologies Used")

    tech1, tech2 = st.columns(2)

    with tech1:
        st.write("""
        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        """)

    with tech2:
        st.write("""
        - Plotly
        - Streamlit
        - Joblib
        - Git & GitHub
        """)

    st.divider()

    # --------------------------------------------------
    # Model Information
    # --------------------------------------------------

    st.subheader("📈 Deployed Model")

    st.write(f"**Model Name:** {metadata['model_name']}")
    st.write(f"**Algorithm:** {metadata['algorithm']}")
    st.write(f"**Features:** {metadata['number_of_features']}")
    st.write(f"**Training Date:** {metadata['training_date']}")
    st.write(f"**Classification Threshold:** {metadata['threshold']:.2f}")

    st.divider()

    # --------------------------------------------------
    # Author
    # --------------------------------------------------

    st.subheader("👨‍💻 Project Information")

    st.info("""
**Project:** Predicting Acute Food Insecurity Risk in Kenyan Counties

**Author:** Patrick Njoroge, Chelimo Ivy, Rebecca Monchari, Joel Andrew, Alex Njihia, Halima Ahmed 

**Programme:** Flatiron School – Machine Learning Capstone

**Year:** 2026
""")
                

    st.divider()

    st.caption(
        """
        Predicting Acute Food Insecurity Risk in Kenyan Counties
        Machine Learning Capstone Project
        Flatiron School • Data Alchemist • 2026
        """
        )

# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.markdown(
    """
    <div style='text-align:center; color:gray;'>

    <b>Predicting Acute Food Insecurity Risk in Kenyan Counties</b><br>

    Machine Learning Capstone Project<br>

    Developed by Data Alchemist • 2026

    </div>
    """,
    unsafe_allow_html=True
)