import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

st.set_option('deprecation.showPyplotGlobalUse', False)

# Set page title and favicon
st.set_page_config(
    page_title="Exploratory Data Analysis App",
    page_icon=":bar_chart:",
    layout="wide"
)

# Define custom CSS styles
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")  # Load custom CSS file

# Function to generate a graph based on user selection
def graph_generator(graph_index):
    chart_type = st.selectbox(
        "Select chart type:",
        ("scatter_chart", "line_chart", "bar_chart", "map", "area_chart", "histogram", "heatmap"),
        key=f'chart_type_{graph_index}'
    )
    
    if chart_type not in ["histogram", "heatmap"]:
        x_input = st.selectbox(
            "Select X column:",
            df.columns,
            key=f'x_col_{graph_index}'
        )
        y_input = st.selectbox(
            "Select Y column:",
            df.columns,
            key=f'y_col_{graph_index}',
            index=min(1, len(df.columns) - 1)
        )
    
    if chart_type == "histogram":
        column = st.selectbox("Select column for histogram", df.columns, key=f'hist_col_{graph_index}')
        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, ax=ax)
        st.pyplot(fig)
    elif chart_type == "heatmap":
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if numeric_df.shape[1] < 2:
            st.warning("Need at least two numeric columns for heatmap.")
        else:
            fig, ax = plt.subplots()
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
    elif chart_type == "scatter_chart":
        st.write(alt.Chart(df).mark_circle().encode(
            x=x_input,
            y=y_input
        ))
    elif chart_type == "line_chart":
        st.line_chart(df.set_index(x_input)[[y_input]])
    elif chart_type == "bar_chart":
        st.bar_chart(df.set_index(x_input)[[y_input]])
    elif chart_type == "area_chart":
        st.area_chart(df.set_index(x_input)[[y_input]])
    elif chart_type == "map":
        if "latitude" in df.columns and "longitude" in df.columns:
            st.map(df)
        else:
            st.error("The dataframe must contain 'latitude' and 'longitude' columns for map plotting.")

# Function to add a new graph to the list
def add_graph():
    st.session_state.graph_count += 1
    st.session_state.graphs.append(st.session_state.graph_count)


st.title("Exploratory Data Analysis App")

file = st.file_uploader("Pick a csv file")

if file is not None:
    df = pd.read_csv(file)

    # Set up initial session state
    if 'graph_count' not in st.session_state:
        st.session_state.graph_count = 0
    if 'graphs' not in st.session_state:
        st.session_state.graphs = []

    st.write("## Preview")
    st.write(df.head(1))

    # Display raw data
    if st.checkbox("Show raw data"):
        st.write(df)

    # Display basic statistics
    if st.checkbox("Show basic statistics"):
        st.write(df.describe())

        # Missing Values Count
    if st.checkbox("Missing Values Count"):
        # Count missing values for each column
        missing_counts = df.isnull().sum()
        st.write(missing_counts)

    # Visualize missing values
    if st.checkbox("Show missing values heatmap"):
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cmap='viridis', cbar=False)
        st.pyplot()

    # Distribution plots
    if st.checkbox("Show distribution plots"):
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            st.write(f"### Distribution of {col}")
            sns.histplot(df[col], kde=True)
            st.pyplot()

    # Correlation matrix
    if st.checkbox("Show correlation matrix"):
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if numeric_df.shape[1] < 2:
            st.warning("Need at least two numeric columns for correlation matrix.")
        else:
            corr_matrix = numeric_df.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
            st.pyplot()

    # Value counts for categorical variables
    if st.checkbox("Show value counts for categorical variables"):
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            st.write(f"### Value counts for {col}")
            st.write(df[col].value_counts())

    # Outlier Detection
    if st.checkbox("Outlier Detection"):
        from sklearn.ensemble import IsolationForest
        
        # Fit Isolation Forest
        clf = IsolationForest(contamination=0.1)
        outlier_labels = clf.fit_predict(df.select_dtypes(include=['float64', 'int64']))
        
        # Create a new DataFrame with the outlier labels added as a column
        df_with_outliers = df.copy()
        df_with_outliers['Outlier'] = outlier_labels
        
        # Visualize DataFrame with outliers
        st.write(df_with_outliers)

    # Button to add a graph
    if st.button("Add graph"):
        add_graph()

    # Button to reset the graphs
    if st.button("Reset", type="primary"):
        st.session_state.graph_count = 0
        st.session_state.graphs = []

    # Slider to select rows
    count = len(df)
    number = st.slider("Row inclusion", 0, count)
    df = df.iloc[:number]

    # Generate graphs
    for graph_index in st.session_state.graphs:
        graph_generator(graph_index)
