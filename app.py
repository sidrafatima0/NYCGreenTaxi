import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
@st.cache_data
def load_data():
    # Replace with the correct file path or method to load your dataset
    data = pd.read_parquet('green_tripdata_2023-05.parquet')
    data['trip_duration'] = (data['lpep_dropoff_datetime'] - data['lpep_pickup_datetime']).dt.total_seconds() / 60
    data['weekday'] = data['lpep_dropoff_datetime'].dt.day_name()
    data['hourofday'] = data['lpep_dropoff_datetime'].dt.hour
    return data

data = load_data()

# Set up the Streamlit app layout
st.title("Taxi Trip Data Analysis")
st.sidebar.header("Navigation")

# Sidebar options
options = st.sidebar.radio(
    "Select an option:",
    ["Overview", "Visualizations", "Statistics", "Modeling"]
)

if options == "Overview":
    st.header("Dataset Overview")
    st.write("This dataset contains information about taxi trips, including pickup and dropoff times, payment types, trip distances, and more.")
    
    # Display dataset info and preview
    st.subheader("Data Preview")
    st.dataframe(data.head())

    st.subheader("Basic Statistics")
    st.write(data.describe())

elif options == "Visualizations":
    st.header("Data Visualizations")

    # Time-based filtering
    time_filter = st.radio(
        "Select time aggregation:",
        ["Daily", "Weekly", "Monthly"],
        horizontal=True
    )
    
    # Create time-aggregated data
    if time_filter == "Daily":
        time_data = data.copy()
        time_data['date'] = pd.to_datetime(data['lpep_dropoff_datetime']).dt.date
        agg_data = time_data.groupby('date').agg({
            'total_amount': 'mean',
            'trip_distance': 'mean',
            'tip_amount': 'mean'
        }).reset_index()
        x_label = 'Date'
    elif time_filter == "Weekly":
        time_data = data.copy()
        time_data['week'] = pd.to_datetime(data['lpep_dropoff_datetime']).dt.isocalendar().week
        agg_data = time_data.groupby('week').agg({
            'total_amount': 'mean',
            'trip_distance': 'mean',
            'tip_amount': 'mean'
        }).reset_index()
        x_label = 'Week of Year'
    else:  # Monthly
        time_data = data.copy()
        time_data['month'] = pd.to_datetime(data['lpep_dropoff_datetime']).dt.month
        agg_data = time_data.groupby('month').agg({
            'total_amount': 'mean',
            'trip_distance': 'mean',
            'tip_amount': 'mean'
        }).reset_index()
        x_label = 'Month'
    
    # Display time-aggregated charts
    st.subheader(f"{time_filter} Average Trip Metrics")
    
    # Create tabs for different metrics
    tab1, tab2, tab3 = st.tabs(["Total Amount", "Trip Distance", "Tip Amount"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(agg_data.iloc[:, 0], agg_data['total_amount'], color='#66c2a5')
        ax.set_xlabel(x_label)
        ax.set_ylabel('Average Total Amount')
        ax.set_title(f'{time_filter} Average Total Amount')
        st.pyplot(fig)
    
    with tab2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(agg_data.iloc[:, 0], agg_data['trip_distance'], color='#fc8d62')
        ax.set_xlabel(x_label)
        ax.set_ylabel('Average Trip Distance')
        ax.set_title(f'{time_filter} Average Trip Distance')
        st.pyplot(fig)
    
    with tab3:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(agg_data.iloc[:, 0], agg_data['tip_amount'], color='#8da0cb')
        ax.set_xlabel(x_label)
        ax.set_ylabel('Average Tip Amount')
        ax.set_title(f'{time_filter} Average Tip Amount')
        st.pyplot(fig)

    # Payment type distribution pie chart
    st.subheader("Payment Type Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        payment_type_counts = data['payment_type'].value_counts()
        fig, ax = plt.subplots()
        payment_type_counts.plot.pie(autopct='%1.1f%%', startangle=90, ax=ax, colors=['#66c2a5', '#fc8d62', '#8da0cb'])
        ax.set_ylabel('')
        ax.set_title('Payment Type Distribution')
        st.pyplot(fig)
    
    with col2:
        trip_type_counts = data['trip_type'].value_counts()
        fig, ax = plt.subplots()
        trip_type_counts.plot.pie(autopct='%1.1f%%', startangle=90, ax=ax, colors=['#e78ac3', '#a6d854'])
        ax.set_ylabel('')
        ax.set_title('Trip Type Distribution')
        st.pyplot(fig)

    # Correlation heatmap - fixed with numeric_only=True
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix of Numeric Variables')
    st.pyplot(fig)

elif options == "Statistics":
    st.header("Statistical Analysis")

    # Groupby analysis for average total_amount by weekday
    avg_total_by_weekday = data.groupby('weekday')['total_amount'].mean().sort_values()
    
    st.subheader("Average Total Amount by Weekday")
    fig, ax = plt.subplots()
    avg_total_by_weekday.plot(kind='bar', color='#66c2a5', ax=ax)
    ax.set_ylabel('Average Total Amount')
    ax.set_xlabel('Weekday')
    ax.set_title('Average Total Amount by Weekday')
    st.pyplot(fig)
    
    # Average total by payment type
    avg_total_by_payment_type = data.groupby('payment_type')['total_amount'].mean().sort_values()
    
    st.subheader("Average Total Amount by Payment Type")
    fig, ax = plt.subplots()
    avg_total_by_payment_type.plot(kind='bar', color='#fc8d62', ax=ax)
    ax.set_ylabel('Average Total Amount')
    ax.set_xlabel('Payment Type')
    ax.set_title('Average Total Amount by Payment Type')
    st.pyplot(fig)
    
    # Average tip by weekday
    avg_tip_by_weekday = data.groupby('weekday')['tip_amount'].mean().sort_values()
    
    st.subheader("Average Tip Amount by Weekday")
    fig, ax = plt.subplots()
    avg_tip_by_weekday.plot(kind='bar', color='#8da0cb', ax=ax)
    ax.set_ylabel('Average Tip Amount')
    ax.set_xlabel('Weekday')
    ax.set_title('Average Tip Amount by Weekday')
    st.pyplot(fig)

elif options == "Modeling":
    st.header("Linear Regression Model")
    
    # Select only numeric columns for modeling
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Remove the target variable from features
    if 'total_amount' in numeric_cols:
        numeric_cols.remove('total_amount')
    
    # Feature selection
    features = st.multiselect(
        "Select features for the model:",
        options=numeric_cols,
        default=numeric_cols[:3]  # Default to first 3 numeric columns
    )
    
    if features and st.button("Train Model"):
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Prepare data
        X = data[features]
        y = data['total_amount']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Display model coefficients
        st.subheader("Model Coefficients")
        coef_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': model.coef_
        })
        st.dataframe(coef_df)
        
        # Display metrics
        col1, col2 = st.columns(2)
        col1.metric("Mean Squared Error", f"{mse:.2f}")
        col2.metric("R² Score", f"{r2:.2f}")
        
        # Plot actual vs predicted
        st.subheader("Actual vs Predicted Values")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        st.pyplot(fig)
