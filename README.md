# AI-Landslide-Mitigation-
**Here's a summary of the code for landslide detection using Gradient Boosting Machines (GBM) along with explanations for each section of the code:

1. **Import Libraries**: Import necessary libraries such as NumPy, pandas, scikit-learn, Matplotlib, Seaborn, and others for data analysis and machine learning.

2. **Define File Path**: Set the file path to the location of your dataset, which appears to be a Jupyter Notebook file (`iit-mandi.ipynb`) on your desktop.

3. **List Files in Directory**: Use the `os.walk` function to list all the files in the specified directory (in your case, the Jupyter Notebook file).

4. **Import Additional Libraries**: Import scitime for measuring the time complexity of machine learning algorithms.

5. **Read Data**: Load the dataset from the specified path into a Pandas DataFrame (`df`).

6. **Exploratory Data Analysis (EDA)**:
   - Check the basic information about the DataFrame using `df.info()`.
   - Rearrange columns to put the target variable (`'Landslide'`) at the end of the DataFrame.
   - Display summary statistics for the DataFrame.
   - Check the number of unique labels in each column.
   - Visualize the frequency distribution of the `'Landslide'` column using a bar plot.
   - Save the bar plot as an EPS file.
   - Repeat the above steps for other columns like `'Precipitation'`, `'Lithology'`, and categorical columns using pie charts.
   - Visualize relationships between `'Landslide'` and `'NDWI'` using a swarm plot.
   - Create a heatmap to visualize the correlation between features.

7. **Feature Importance using Mutual Information Classification**:
   - Split the data into training and validation sets.
   - Calculate mutual information scores between features and the target variable `'Landslide'`.
   - Visualize the mutual information scores as a bar plot.

8. **Feature Engineering**:
   - Create frequency-encoded features for selected categorical columns such as `'Aspect'`, `'Curvature'`, etc.
   - Group these frequency-encoded features and the target variable into separate DataFrames.
   - Create a heatmap to visualize the correlation between the frequency-encoded features.

9. **Feature Selection using PCA (Principal Component Analysis)**:
   - Import PCA from the `pca` library.
   - Prepare the data for PCA by removing the target variable.
   - Initialize the PCA model and fit it to the data.
   - Get the top features contributing to variance.
   - Plot the PCA results in 2D and 3D space.

10. **Save Visualizations**: Save the generated visualizations as EPS files.

This code appears to be a comprehensive data analysis and preprocessing script for landslide detection, including EDA, feature engineering, feature selection using PCA, and visualization of data and results. It also uses various machine learning libraries for analysis and modeling, with a focus on Gradient Boosting for classification.
