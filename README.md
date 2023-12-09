# Machine Learning Application

This Flask-based web application allows users to upload a CSV file, perform linear regression analysis, and visualize the results.

## Features

1. **Upload Data:** Users can upload a CSV file with columns 'TV', 'Radio', 'Newspaper', and 'Sales'.

2. **Linear Regression Analysis:** The application performs linear regression on the 'TV' and 'Sales' columns.

3. **Exploratory Data Analysis (EDA):** Visualizations and summary statistics for EDA.

4. **Dockerized:** The application can be containerized using Docker for easy deployment.

## Getting Started

### Prerequisites

- Python 3.x
- Docker (optional)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository
   ```

2. Install dependencies:

    ```bash
        pip install -r requirements.txt
    ```

### Usage

1. Run the application:

   ```bash
   python app.py
   ```
   Visit http://127.0.0.1:5000 in your web browser.

2. Upload a CSV file with the specified columns and analyze the data.

### Docker
Optionally, use Docker for containerization:

1. Build the Docker image:

    ```bash
        docker build -t your-image-name:latest .
    ```

2. Run the Docker container:

    ```bash
    docker run -p 5000:5000 --name your-container-name your-image-name:latest
    ```
    Visit http://127.0.0.1:5000 in your web browser.

