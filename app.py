import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

# Define the sidebar menu
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Select a Page", ("Convert Data", "Train & Predict"))

# Page 1: Convert Categorical Data to Numeric
if menu == "Convert Data":
    st.title("George Auto ML App")
    st.subheader("Convert Categorical Data into Numeric")
    st.image("image.jpeg", width=700)

    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Dataset", df.head())

        # Encoding method selection
        encoding_method = st.radio(
            "Select Encoding Method", 
            ("Label Encoding", "One-Hot Encoding")
        )

        if encoding_method == "Label Encoding":
            label_encoders = {}
            for column in df.select_dtypes(include=["object"]).columns:
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])
                label_encoders[column] = le
            st.write("### Label Encoded Data", df.head())

        elif encoding_method == "One-Hot Encoding":
            df = pd.get_dummies(df)
            st.write("### One-Hot Encoded Data", df.head())

        # Download the encoded data
        st.download_button(
            label="Download Encoded CSV",
            data=df.to_csv(index=False),
            file_name="encoded_data.csv",
            mime="text/csv"
        )

# Page 2: Train & Predict Model





elif menu == "Train & Predict":
    st.title("George Auto ML App")
    st.subheader("Train Model And Make Prediction")
    st.image("images2.jpeg", width=700)

    # Upload training data
    train_file = st.file_uploader("Upload Training Data CSV", type=["csv"])

    if train_file:
        # Load training data
        train_data = pd.read_csv(train_file)
        st.write("### Training Dataset", train_data.head())

        # Select target column
        target_column = st.selectbox("Select Target Column", train_data.columns)

        if st.button("Train Model"):
            # Encode categorical columns
            label_encoders = {}
            for col in train_data.select_dtypes(include=["object"]).columns:
                if col != target_column:
                    le = LabelEncoder()
                    train_data[col] = le.fit_transform(train_data[col])
                    label_encoders[col] = le

            # Split data into training and testing sets
            X = train_data.drop(target_column, axis=1)
            y = train_data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Train a RandomForest model
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            # Evaluate the model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)*100
            st.write(f"### Model Accuracy  : {accuracy:.2f}%")
            predictions_array = np.array(y_pred)
            st.write(predictions_array)

            # # Prediction on new data
            pred_file = st.file_uploader("Upload Data for Prediction", type=["csv"])

            if pred_file:
                new_data = pd.read_csv(pred_file)

                # Apply label encoding if necessary
                for col, le in label_encoders.items():
                    if col in new_data.columns:
                        new_data[col] = le.transform(new_data[col])

                # Make predictions
                predictions = model.predict(new_data)
                st.write("### Predictions", predictions)

                #Download predictions
                result = new_data.copy()
                result["Prediction"] = predictions
                st.download_button(
                    label="Download Predictions as CSV",
                    data=result.to_csv(index=False),
                    file_name="predictions.csv",
                    mime="text/csv"
                )



    