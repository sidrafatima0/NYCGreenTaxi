import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

    # Pie chart for payment_type distribution
    st.subheader("Payment Type Distribution")
    payment_type_counts = data['payment_type'].value_counts()
    fig, ax = plt.subplots()
    payment_type_counts.plot.pie(autopct='%1.1f%%', startangle=90, ax=ax, colors=['#66c2a5', '#fc8d62', '#8da0cb'])
    ax.set_ylabel('')
    ax.set_title('Payment Type Distribution')
    st.pyplot(fig)

    # Pie chart for trip_type distribution
    st.subheader("Trip Type Distribution")
    trip_type_counts = data['trip_type'].value_counts()
    fig, ax = plt.subplots()
    trip_type_counts.plot.pie(autopct='%1.1f%%', startangle=90, ax=ax, colors=['#e78ac3', '#a6d854'])
    ax.set_ylabel('')
    ax.set_title('Trip Type Distribution')
    st.pyplot(fig)

    # Heatmap for correlation matrix
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
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

elif options == "Modeling":
    st.header("Modeling Section")
    
    # Placeholder for model results or predictions
    st.write("This section will include model training and predictions.")
    
# Footer
st.markdown("---")
st.markdown("👨‍💻 Created by *Sidra Fatima*")