
Content is user-generated and unverified.
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Set page config
st.set_page_config(
    page_title="Sales Prediction ML App",
    page_icon="📊",
    layout="wide"
)

# Define PyTorch Neural Network
class SalesPredictor(nn.Module):
    def __init__(self, input_size=3):
        super(SalesPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def validate_csv_headers(df):
    """Validate that CSV has required headers"""
    required_headers = ['Date', 'Product_Category', 'Region', 'Sales_Rep', 
                       'Sales_Amount', 'Quantity_Sold', 'Marketing_Spend', 'target_sales']
    return all(header in df.columns for header in required_headers)

def train_pytorch_model(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, epochs=100):
    """Train PyTorch model and return training history"""
    model = SalesPredictor(input_size=3)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    test_losses = []
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)
            test_losses.append(test_loss.item())
        
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        if epoch % 10 == 0:
            status_text.text(f'Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.4f} - Test Loss: {test_loss.item():.4f}')
    
    status_text.text(f'Training Complete! Final Train Loss: {train_losses[-1]:.4f} - Test Loss: {test_losses[-1]:.4f}')
    
    return model, train_losses, test_losses

def main():
    st.title("📊 Sales Prediction ML App")
    st.markdown("Upload your sales CSV data and train a PyTorch neural network for sales prediction!")
    
    # Sidebar with app info
    st.sidebar.header("App Information")
    st.sidebar.markdown("""
    **Required CSV Headers:**
    - Date
    - Product_Category  
    - Region
    - Sales_Rep
    - Sales_Amount
    - Quantity_Sold
    - Marketing_Spend
    - target_sales
    
    **Features (X):** Sales_Amount, Quantity_Sold, Marketing_Spend
    **Target (y):** target_sales
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.subheader("📋 Data Preview")
            st.dataframe(df.head())
            
            # Validate headers
            if not validate_csv_headers(df):
                st.error("❌ CSV file doesn't have the required headers. Please check the sidebar for required format.")
                return
            
            st.success("✅ CSV file format validated successfully!")
            
            # Display basic statistics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", len(df))
                st.metric("Features", "3 (Sales_Amount, Quantity_Sold, Marketing_Spend)")
            with col2:
                st.metric("Target Variable", "target_sales")
                st.metric("Train/Test Split", "80% / 20%")
            
            # Prepare data
            st.subheader("🔄 Data Processing")
            
            # Extract features and target
            feature_columns = ['Sales_Amount', 'Quantity_Sold', 'Marketing_Spend']
            X = df[feature_columns].values
            y = df['target_sales'].values.reshape(-1, 1)
            
            # Check for missing values
            if pd.DataFrame(X).isnull().sum().sum() > 0 or pd.Series(y.flatten()).isnull().sum() > 0:
                st.warning("⚠️ Missing values detected. Removing rows with missing data...")
                # Remove rows with missing values
                mask = ~(pd.DataFrame(X).isnull().any(axis=1) | pd.Series(y.flatten()).isnull())
                X = X[mask]
                y = y[mask]
                st.info(f"Cleaned dataset size: {len(X)} records")
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            st.write(f"**Training Set:** {len(X_train)} samples")
            st.write(f"**Test Set:** {len(X_test)} samples")
            
            # Feature scaling
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            y_train_scaled = scaler_y.fit_transform(y_train)
            y_test_scaled = scaler_y.transform(y_test)
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train_scaled)
            X_test_tensor = torch.FloatTensor(X_test_scaled)
            y_train_tensor = torch.FloatTensor(y_train_scaled)
            y_test_tensor = torch.FloatTensor(y_test_scaled)
            
            st.success("✅ Data successfully converted to PyTorch tensors!")
            
            # Train model
            st.subheader("🤖 PyTorch Model Training")
            
            if st.button("Start Training", type="primary"):
                with st.spinner("Training PyTorch neural network..."):
                    model, train_losses, test_losses = train_pytorch_model(
                        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, epochs=200
                    )
                
                # Make predictions
                model.eval()
                with torch.no_grad():
                    train_pred_scaled = model(X_train_tensor).numpy()
                    test_pred_scaled = model(X_test_tensor).numpy()
                    
                    # Inverse transform predictions
                    train_pred = scaler_y.inverse_transform(train_pred_scaled)
                    test_pred = scaler_y.inverse_transform(test_pred_scaled)
                
                # Calculate metrics
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                train_mae = mean_absolute_error(y_train, train_pred)
                test_mae = mean_absolute_error(y_test, test_pred)
                
                # Display results
                st.subheader("📈 Model Performance")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Train R²", f"{train_r2:.4f}")
                    st.metric("Test R²", f"{test_r2:.4f}")
                with col2:
                    st.metric("Train MSE", f"{train_mse:.2f}")
                    st.metric("Test MSE", f"{test_mse:.2f}")
                with col3:
                    st.metric("Train MAE", f"{train_mae:.2f}")
                    st.metric("Test MAE", f"{test_mae:.2f}")
                
                # Visualizations
                st.subheader("📊 Results Visualization")
                
                # Training history
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(train_losses, label='Training Loss', alpha=0.8)
                    ax.plot(test_losses, label='Validation Loss', alpha=0.8)
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss (MSE)')
                    ax.set_title('Training History')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                with col2:
                    # Predictions vs Actual
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(y_test, test_pred, alpha=0.6, label=f'Test (R²={test_r2:.3f})')
                    ax.scatter(y_train, train_pred, alpha=0.4, label=f'Train (R²={train_r2:.3f})')
                    
                    # Perfect prediction line
                    min_val = min(y_train.min(), y_test.min())
                    max_val = max(y_train.max(), y_test.max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
                    
                    ax.set_xlabel('Actual Sales')
                    ax.set_ylabel('Predicted Sales')
                    ax.set_title('Predictions vs Actual')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                # Feature importance (approximate using model weights)
                st.subheader("🎯 Feature Analysis")
                feature_names = ['Sales_Amount', 'Quantity_Sold', 'Marketing_Spend']
                
                # Get first layer weights as proxy for feature importance
                first_layer_weights = model.fc1.weight.data.abs().mean(dim=0).numpy()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(feature_names, first_layer_weights)
                ax.set_ylabel('Average Weight Magnitude')
                ax.set_title('Feature Importance (Approximate)')
                ax.grid(True, alpha=0.3)
                
                # Color bars
                colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                
                st.pyplot(fig)
                
                # Sample predictions table
                st.subheader("🔍 Sample Predictions")
                sample_size = min(10, len(X_test))
                sample_df = pd.DataFrame({
                    'Sales_Amount': X_test[:sample_size, 0],
                    'Quantity_Sold': X_test[:sample_size, 1],
                    'Marketing_Spend': X_test[:sample_size, 2],
                    'Actual_Sales': y_test[:sample_size, 0],
                    'Predicted_Sales': test_pred[:sample_size, 0],
                    'Error': np.abs(y_test[:sample_size, 0] - test_pred[:sample_size, 0])
                })
                st.dataframe(sample_df)
                
                # Model summary
                st.subheader("🏗️ Model Architecture")
                st.code(f"""
Neural Network Architecture:
- Input Layer: {X_train.shape[1]} features
- Hidden Layer 1: 64 neurons (ReLU + Dropout 0.2)
- Hidden Layer 2: 32 neurons (ReLU + Dropout 0.2)
- Hidden Layer 3: 16 neurons (ReLU)
- Output Layer: 1 neuron (Linear)

Training Parameters:
- Optimizer: Adam (lr=0.001)
- Loss Function: MSE
- Epochs: 200
- Batch Size: Full dataset
                """)
        
        except Exception as e:
            st.error(f"❌ An error occurred while processing the file: {str(e)}")
            st.info("Please make sure your CSV file has the correct format and contains numeric data for the feature columns.")
    
    else:
        st.info("👆 Please upload a CSV file to get started!")
        
        # Show example data format
        st.subheader("📝 Example CSV Format")
        example_data = {
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Product_Category': ['Electronics', 'Clothing', 'Electronics'],
            'Region': ['North', 'South', 'East'],
            'Sales_Rep': ['John', 'Jane', 'Bob'],
            'Sales_Amount': [1000, 750, 1200],
            'Quantity_Sold': [10, 5, 12],
            'Marketing_Spend': [200, 150, 250],
            'target_sales': [1100, 800, 1300]
        }
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df)

if __name__ == "__main__":
    main()
