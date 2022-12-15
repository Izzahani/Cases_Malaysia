![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

# Predicting The New Covid-19 Cases in Malaysia by using TensorFlow

## Summary
<p>This project is to predict the number of Covid-19 cases in Malaysia by using past 30 days of Covid-19 cases.</p>
<p>There are 5 steps need to be done to complete this project which are:</p>
<p>1. Data Loading</p>
  <ol>- Upload the dataset using pandas</ol>
  <ol>- Use pd.read_csv( <strong>your_path.csv</strong> )</ol>
  
<p>2. Data Inspection</p>
   <ol>- Inspect the dataset to check whether the dataset contains NULL, duplicated data or any other unwanted things.</ol>
   <ol>- I used <strong>df.info()</strong> to find the amount of NULL in the data. </ol>
   <ol>- Then, I used <strong>df.duplicated().sum()</strong> to find duplicated data.</ol>
   <ol>- I also checked for the amount of <em>'bit.ly'</em> in the text to remove it since I don't want it to be classified in this data. </ol>

<p>3. Data Cleaning</p>
   <ol>- Data cleaning need to be done to increase overall productivity and allow for the highest quality information in your decision-making.</ol>
   <ol>- I used Regex to remove unwanted words which then leave only the words with alphabets A-Z</ol>
   <ol>- I also make all of the alphabets in lower form.</ol>
   <ol>- All of the duplicated data has been removed in this part as well</ol>

<p>4. Features Selection</p>
   <ol>- Select the text data as feature.</ol>
   <ol>- Select the subject data as target.</ol>
          
<p>5. Data Pre-processing</p>
   <ol> <strong>For feature:</strong></ol>
   <ol>- Tokenizers is being used in this part to convert the text into numerical.</ol>
   <ol>- I used train sequences to convert the text to horizontal.</ol>
   <ol> <strong>For target:</strong></ol>
   <ol>- One Hot Encoder is being used to convert the outputs which are politicsNews and worldnews to 1.0 and 0.1 respectively.</ol>
   
 <p>Finally, <strong>Model Development</strong> can be done if all of the steps above has already finished.</p>
 <p> In Model Development, I did train-test split. Then, i used Embedding as an input layer. For hidden layers, I used LSTM.</p>
 <p> Then, the project is being compiled. This is my result:</p>
 <img src="https://github.com/Izzahani/Text_Classification/blob/main/Predictions1.png" alt="descriptive text">
 <p>These are the graph I got from TensorBoard</p>
 <p>For Epoch Acc</p>
 <img src="https://github.com/Izzahani/Text_Classification/blob/main/epoch_acc.png" alt="descriptive text">

 <p>For Epoch Loss</p>
 <img src="https://github.com/Izzahani/Text_Classification/blob/main/epoch%20loss.png" alt="descriptive text">
 
## Acknowledgement
Special thanks to ([https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://github.com/MoH-Malaysia/covid19-public)) :smile:

