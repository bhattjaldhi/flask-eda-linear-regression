from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'  # Set a secret key for flash messages

def plot_linear_regression(X, y, model):
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.regplot(x=X, y=y, ci=None, line_kws={'color': 'red'})
    plt.title('Linear Regression')
    plt.xlabel(X.name)
    plt.ylabel('Sales')
    plt.tight_layout()

    # Save plot to a BytesIO object
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)

    # Encode the plot image to base64 for HTML display
    encoded_image = base64.b64encode(image_stream.read()).decode('utf-8')

    plt.close()
    return encoded_image

def perform_eda(df):
    # Summary statistics
    eda_summary = df.describe().to_html()

    # Distribution plots for each feature
    distribution_plots = []
    for column in df.columns:
        if column != 'Sales':  # Exclude the target variable from distribution plots
            plt.figure(figsize=(8, 6))
            sns.histplot(df[column], kde=True)
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            img_stream = BytesIO()
            plt.savefig(img_stream, format='png')
            img_stream.seek(0)
            encoded_img = base64.b64encode(img_stream.read()).decode('utf-8')
            plt.close()
            distribution_plots.append({'column': column, 'plot': encoded_img})

    # Pairplot
    plt.figure(figsize=(12, 8))
    sns.pairplot(df, diag_kind='kde')
    pairplot_image_stream = BytesIO()
    plt.savefig(pairplot_image_stream, format='png')
    pairplot_image_stream.seek(0)
    encoded_pairplot = base64.b64encode(pairplot_image_stream.read()).decode('utf-8')
    plt.close()

    # Correlation matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix Heatmap')
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)
    encoded_corr_matrix = base64.b64encode(img_stream.read()).decode('utf-8')
    plt.close()

    return eda_summary, distribution_plots, encoded_pairplot, encoded_corr_matrix

@app.route('/')
def index():
    return render_template('index.html')

def validate_csv(file):
    try:
        df = pd.read_csv(file)

        # Check if required columns are present and numeric
        required_columns = ['TV', 'Radio', 'Newspaper', 'Sales']
        if set(required_columns).issubset(df.columns) and df[required_columns].applymap(lambda x: isinstance(x, (int, float))).all().all():
            return df
        else:
            flash('Error: CSV should contain columns - TV, Radio, Newspaper, and Sales with numeric values.')
            return None
    except Exception as e:
        flash(f'Error: {str(e)}')
        return None

@app.route('/analyze', methods=['POST', 'GET'])
def analyze():
    if request.method == 'POST':
        # Get the uploaded file or data
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            df = validate_csv(file)

            if df is not None:
                # Perform linear regression
                X = df['TV'].values.reshape(-1, 1)
                y = df['Sales']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                # Perform EDA
                eda_summary, distribution_plots, encoded_pairplot, encoded_corr_matrix = perform_eda(df)

                # Display the regression line plot
                regression_plot = plot_linear_regression(df['TV'], df['Sales'], model)

                return render_template('results.html', predictions=predictions, regression_plot=regression_plot, eda_summary=eda_summary, distribution_plots=distribution_plots, encoded_pairplot=encoded_pairplot, encoded_corr_matrix=encoded_corr_matrix)
        else:
            flash('Error: Please upload a valid CSV file.')
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

