![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

# Predicting New Covid-19 Cases in Malaysia by using TensorFlow

## Summary
<p>Humanity experienced a catastrophe in the year 2020. The first case of pneumonia with an unknown cause was recorded in December 2019. Since then, COVID-19 has spread around the world and has become a pandemic. The epidemic hit more than 200 countries, and many of them implemented travel restrictions, quarantines, social distances, event postponements, and lockdowns in an effort to preserve the lives of their citizens. However, the governments' efforts were compromised by their slack behaviour, which made the virus more likely to spread widely and result in fatalities. The lack of an AI-assisted automated tracking and prediction system, according to scientists, is what has led to the COVID-19 pandemic's rapid spread.</p>
<p>Therefore, the scientist suggested using a deep learning algorithm to forecast the daily COVID cases to decide whether or not travel prohibitions should be implemented.</p>
<p>This project is to predict the number of Covid-19 cases in Malaysia by using past 30 days of Covid-19 cases.</p>
<p>There are 5 steps need to be done to complete this project which are:</p>
<p>1. Data Loading</p>
  <ol>- Upload the dataset using pandas</ol>
  <ol>- Use pd.read_csv( <strong>your_path.csv</strong> )</ol>
  
<p>2. Data Inspection</p>
   <ol>- Inspect the dataset to check whether the dataset contains NULL or any other unwanted things.</ol>
   <ol>- I used <strong>df.info()</strong> to explore the data. The datatype for <em>new_cases</em> is object. It means that, there is something besides numerical value in the dataset.</ol>
   <ol>- Then, I used <strong>df.isnull().sum()</strong> to find the amount of NaN in the data. As for now, there is no NaN value in the data. However, we can still check if there is NaN value in the dataset, again.</ol>

<p>3. Data Cleaning</p>
   <ol>- Data cleaning need to be done to increase overall productivity and allow for the highest quality information in your decision-making.</ol>
   <ol>- I used interpolate to convert any values that is not in numerical to numericals. Then, the data can be visualized clearly by using graphical method</ol>
   <p align="center"><img src="model/before_cleaning_graph.png" alt="graph" width="500"/></p>
   <div align="center"><ol> As we can see in the graph, there is missing value in between 400 to 500. Thus, the missing value can be filled by using interpolate.</ol></div>
   <p align="center"><img src="model/after_cleaning_graph.png" alt="graph" width="500"/></p>
   <div align="center"><ol> Based on the graph above, the missing value already been filled. </ol></div>

<p>4. Features Selection</p>
   <ol>- In this data, I selected <em>new_cases</em> to do the predictions.</ol>
          
<p>5. Data Pre-processing</p>
   <ol>- <strong>MinMaxScaler</strong> is being used in this part to convert the data into 0 until 1.</ol>
   <ol>- I did train-test-split to split the <em>X_train</em> and <em>y_train</em></ol>
   
<p>Then only we can do <strong>Model Development</strong>.</p>
 <p> a) In Model Development, I used Input as an input layer.</p>
 <p> b) For hidden layers, I used 3 LSTM layers and 2 dropouts.</p>
 <p align="center"><img src="model/model.png" alt="model layers" width="200"/></p>
  
  <p> c) The graph can be visualized by using TensorBoard. The graphs below shows the training data and validation data of my model.</p>
 <p align="center"><img src="model/epoch_loss.png" alt="loss" width="500"/></p>
 <div align="center"><ol> The graph above shows the loss data.</ol></div>
 
 <p>After that, we can proceed to do predictions for testing data.</p>
 <p>1. Data Loading</p>
 <p>- I start uploading the Testing Dataset by using pandas as well.</p>
 
 <p>2. Data Inspection</p>
 <p>- I did <strong>df_test.info()</strong> to look at the datatypes. The datatype for <em>new_cases</em> is float.</p>
 
 <p>3. Data Cleaning</p>
 <p>- I converted the dataype of <em>new_cases</em> to <strong>integer</strong></p>
 
 <p>4. Features Selection</p>
  <p>- In this process, we also choose <em>new_cases</em> only.</p>
  
  <p>5. Data Pre-processing</p>
  <p>- In this part, I combine the training and testing data by using concatenation.</p>
  <p>- Then, I proceed to find the predicted_case.</p>
  
 <p align="center"><img src="model/epoch_mape.png" alt="mape" width="500"/></p>
 <div align="center"><ol> This is the MAPE graph. As we can see in the graph, the training data which is in orange colour starts to overfitting at 12-axis. However, it then went down. It might be due to <em>Dropout layer</em> in the model and also, the node value in LSTM layer before the output layer.</ol></div>
  
 <p> Then, the project is being compiled. This is my MAPE result which is below than 1% and also, the MAPE graphs:</p>
 <p align="center"><img src="model/prediction_graph.png" alt="mape graph" width="500"/></p>
 


## Acknowledgement
Special thanks to ([https://github.com/MoH-Malaysia/covid19-public](https://github.com/MoH-Malaysia/covid19-public.git))) :smile:

