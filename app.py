import matplotlib  #for creating visualizations
matplotlib.use('Agg')  # Set the backend to Agg (non-interactive)
import matplotlib.pyplot as plt # Imports the pyplot module from matplotlib, used for creating plots and graphs.
import seaborn as sns  #Imports the seaborn library, a high-level data visualization library based on matplotlib.
from flask import Flask, render_template, request 
#Imports flask module, used to create a Flask application.
#render_template: Renders HTML templates.
#request: Handles incoming HTTP requests.
import pandas as pd# statistics and data manipulation library
import os
import numpy as np# numerical computing library

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['STATIC_FOLDER'] = './static'

# Ensure upload and static folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

# Index Route
@app.route('/')
def index():
    return render_template('index.html')


# File Upload and Processing Route
@app.route('/process', methods=['POST'])
def process():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return "No file part in the request", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    grading_choice = request.form.get('grading_choice')
    desired_distribution = request.form.get('desired_distribution')  # For relative grading
    
    # For absolute grading, get the thresholds
    if grading_choice == 'absolute':
        A_threshold = float(request.form.get('A_threshold', 90))
        B_threshold = float(request.form.get('B_threshold', 80))
        C_threshold = float(request.form.get('C_threshold', 70))
        D_threshold = float(request.form.get('D_threshold', 60))
    else:
        A_threshold = B_threshold = C_threshold = D_threshold = None

    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        # Process the file and compute results
        results, original_plot, adjusted_plot, grade_freq_table, scatter_plot, line_graph, box_plot, grade_percentages, summary_stats = process_file(file_path, grading_choice, desired_distribution, A_threshold, B_threshold, C_threshold, D_threshold)
    except Exception as e:
        return str(e), 500

    # Render the results page
    return render_template(
        'results.html',
        table=results.to_html(classes="table table-striped", index=False),
        original_plot=original_plot,
        adjusted_plot=adjusted_plot,
        grade_freq_table=grade_freq_table.to_html(classes="table table-striped", index=False),
        scatter_plot=scatter_plot,
        line_graph=line_graph,
        box_plot=box_plot,
        grade_percentages=grade_percentages,
        summary_stats=summary_stats
    )


# File Processing Logic
def process_file(file_path, grading_choice, desired_distribution, A_threshold, B_threshold, C_threshold, D_threshold):
    # Load the file
    try:
        data = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")

    # Ensure necessary columns exist
    if 'Student' not in data.columns or 'Score' not in data.columns:
        raise ValueError("File must contain 'Student' and 'Score' columns")

    # Ensure scores are within the valid range
    if (data['Score'] > 100).any() or (data['Score'] < 0).any():
        raise ValueError("Scores must be between 0 and 100")

    # Store original grades for comparison
    original_grades = data['Score'].copy()

    # Grading Functions
    def absolute_grading(score):
        if score >= A_threshold:
            return 'A'
        elif score >= B_threshold:
            return 'B'
        elif score >= C_threshold:
            return 'C'
        elif score >= D_threshold:
            return 'D'
        else:
            return 'F'

    def relative_grading_zscore(score, mean, std_dev):
        z_score = (score - mean) / std_dev
        if z_score >= 1.5:
            return 'A'
        elif z_score >= 0.5:
            return 'B'
        elif z_score >= -0.5:
            return 'C'
        elif z_score >= -1.5:
            return 'D'
        else:
            return 'F'

    def relative_grading_percentile(score, percentiles):
        if score >= percentiles[-2]:  # 90th percentile corresponds to the second-to-last index
            return 'A'
        elif score >= percentiles[-3]:  # 75th percentile corresponds to the third-to-last index
            return 'B'
        elif score >= percentiles[-6]:  # 50th percentile corresponds to the sixth-to-last index
            return 'C'
        elif score >= percentiles[-8]:  # 25th percentile corresponds to the eighth-to-last index
            return 'D'
        else:
            return 'F'

    def custom_percentile_grading(score, custom_thresholds):
        if score >= custom_thresholds['A']:
            return 'A'
        elif score >= custom_thresholds['B']:
            return 'B'
        elif score >= custom_thresholds['C']:
            return 'C'
        elif score >= custom_thresholds['D']:
            return 'D'
        else:
            return 'F'

    # Apply the grading function based on the user's choice
    if grading_choice == 'absolute':
        data['Grade'] = data['Score'].apply(absolute_grading)
    elif grading_choice == 'relative':
        if desired_distribution == 'z-score':
            mean = data['Score'].mean()
            std_dev = data['Score'].std()
            data['Grade'] = data['Score'].apply(lambda x: relative_grading_zscore(x, mean, std_dev))
        elif desired_distribution == 'percentile':
            # Generate percentiles
            percentiles = np.percentile(data['Score'], np.arange(0, 101, 10))  # 0th to 100th percentiles at intervals of 10
            # Assign grades based on percentiles
            data['Grade'] = data['Score'].apply(lambda x: relative_grading_percentile(x, percentiles))
        elif desired_distribution == 'custom':
            custom_thresholds = {
                'A': float(request.form.get('custom_A_threshold', 90)),
                'B': float(request.form.get('custom_Bthreshold', 80)),
                'C': float(request.form.get('custom_C_threshold', 70)),
                'D': float(request.form.get('custom_D_threshold', 60)),
            }
            data['Grade'] = data['Score'].apply(lambda x: custom_percentile_grading(x, custom_thresholds))

    # Calculate percentages of students in each grade
    grade_counts = data['Grade'].value_counts(normalize=True) * 100
    grade_percentages = grade_counts.to_dict()

    # Summary statistics
    total_students = len(data)
    mean_score = data['Score'].mean()
    std_dev = data['Score'].std()
    min_score = data['Score'].min()
    max_score = data['Score'].max()
    median_score = data['Score'].median()
    percentile_25 = data['Score'].quantile(0.25)
    percentile_75 = data['Score'].quantile(0.75)

    summary_stats = {
        'TOTAL STUDENTS': total_students,
        'MEAN SCORE': mean_score,
        'STADARD DEVIATION': std_dev,
        'MINIMUM SCORE': min_score,
        'MAXIMUM SCORE': max_score,
        'MEDIAN SCORE': median_score,
        '25th PERCENTILE': percentile_25,
        '75th PERCENTILE': percentile_75,
    }

    # Create plots
    original_plot, adjusted_plot = create_grade_plots(data)
    scatter_plot = create_scatter_plot(data)
    line_graph = create_line_graph(data)
    box_plot = create_box_plot(data)

    # Create grade frequency table
    grade_freq_table = data['Grade'].value_counts().reset_index()
    grade_freq_table.columns = ['Grade', 'Frequency']

    return data, original_plot, adjusted_plot, grade_freq_table, scatter_plot, line_graph, box_plot, grade_percentages, summary_stats


def create_grade_plots(data):
    original_grades = data['Score'].copy()
    adjusted_grades = data['Grade']
    plt.figure(figsize=(10, 6))

    # Original Grade Distribution
    plt.subplot(1, 2, 1)
    sns.histplot(original_grades, bins=10, kde=True, color='blue')
    plt.title('Original Grade Distribution')

    # Adjusted Grade Distribution
    plt.subplot(1, 2, 2)
    sns.histplot(adjusted_grades, bins=10, kde=True, color='green')
    plt.title('Adjusted Grade Distribution')

    # Save the plots as images
    original_plot = '/static/grade_plot_original.png'
    adjusted_plot = '/static/grade_plot_adjusted.png'
    plt.savefig(os.path.join(app.config['STATIC_FOLDER'], 'grade_plot_original.png'))
    plt.savefig(os.path.join(app.config['STATIC_FOLDER'], 'grade_plot_adjusted.png'))

    return original_plot, adjusted_plot


def create_scatter_plot(data):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=data['Student'], y=data['Score'], hue=data['Grade'], palette='coolwarm')
    plt.title('Grade vs Score')
    scatter_plot = '/static/scatter_plot.png'
    plt.savefig(os.path.join(app.config['STATIC_FOLDER'], 'scatter_plot.png'))
    plt.close()
    return scatter_plot


def create_line_graph(data):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=data['Student'], y=data['Score'])
    plt.title('Student Score Trend')
    line_graph = '/static/line_graph.png'
    plt.savefig(os.path.join(app.config['STATIC_FOLDER'], 'line_graph.png'))
    plt.close()
    return line_graph


def create_box_plot(data):
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=data['Grade'], y=data['Score'], palette='coolwarm')
    plt.title('Score Distribution by Grade')
    box_plot = '/static/box_plot.png'
    plt.savefig(os.path.join(app.config['STATIC_FOLDER'], 'box_plot.png'))
    plt.close()
    return box_plot


if __name__ == '__main__':
    app.run(debug=True)
