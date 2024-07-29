import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import math

def get_correlation_heatmap(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')
    
    return graphic

def get_feature_distributions(df, numeric_cols):
    n_cols = len(numeric_cols)
    n_rows = math.ceil(n_cols / 3)
    fig, axes = plt.subplots(n_rows, 3, figsize=(20, 5*n_rows))
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easier indexing
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            sns.histplot(data=df, x=col, hue='Churn', multiple='stack', ax=axes[i])
            axes[i].set_title(col)
    
    # Remove any unused subplots
    for i in range(n_cols, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')
    
    return graphic

def get_churn_by_category(df, cat_cols):
    n_cols = len(cat_cols)
    n_rows = math.ceil(n_cols / 3)
    fig, axes = plt.subplots(n_rows, 3, figsize=(20, 5*n_rows))
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easier indexing
    
    for i, col in enumerate(cat_cols):
        if i < len(axes):
            sns.barplot(x=col, y='Churn', data=df, ax=axes[i])
            axes[i].set_title(col)
    
    # Remove any unused subplots
    for i in range(n_cols, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')
    
    return graphic