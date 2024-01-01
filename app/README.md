This repository contains my full-stack data science project for the JEDHA bootcamp and validation of Bloc 6 of the RNCP certificate. 
Here is the description of the project:

# Every Second Matters: 🚨an app for timely disaster response🚨

Using Turkish twitter data from the earthquake that hits Syria and Turkey on 6 February, this app performs three task:

- Detection of rescue calls and emergency needs with a text classifier
-  Extraction of person names, city names and addresses in emergency tweets with a Named Entity Recognition (NER) model
-  Geoplotting of those addresses on an interactive map simulating a real-time plot

This is a Turkish language NLP project.

## Datasets

The original dataset I used for this project was available on [kaggle]("https://www.kaggle.com/datasets/swaptr/turkey-earthquake-tweets") as csv file. It contains tweets on earthquake that hit Turkey and Syria on 6 February. The dataset includes the text of each tweet, the user profile information, the time and location of each tweet, and the number of likes, retweets, and replies for each tweet. The dataset also includes any hashtags, mentions, and links used in the tweets.

Since it contains over 400,000 tweets in more than 60 languages, I first filtered down the corpus to the Turkish language tweets. You can also find it in this repository inside _tweets.csv_

Then, I worked on 2 different train set which I manually annotated myself (since there was no dependent variable readily-available and I had to build them in accordance with the task I have at hand):
- sample_list_10K.json consists of 10,000 tweets I annotated using doccano based on whether they are an emergency call or not. I saved the results of annotation in _earthquake10K.json_
- 'JSONL_sample_NER.jsonl' consists of 2304 tweets which I classified as 'emergency call' in the dataset above. After some standardization of common abbreviations, I annotated this dataset on doccano once again, this time assigning each word to a corresponded NER-tag I designed for this project: PER (person), CITY, ADDR (address) or none(OTHER). I saved the results from this annotation in _admin2.jsonl_ file.

Since the original dataset is too big and the models I use are GPU-intensive, for the final version of the app, I used a small sample of the original dataset which contains 10% of the data, and which was stratified on the date column, keeping the ratio of tweets per day intact. You can see that data in stratified_sample_0.1.csv.



## Doccano
Doccano is an open source text annotation tool. It can be used to create labeled datasets for: Text classification, Entity extraction,  Sequence to sequence translation.
For more information you can check its [tutorial]('https://doccano.github.io/doccano/tutorial/')

You can also have look at the snapshots in Doccano_task1_annotation.png and Doccano_task2_NER_annotation.png files to see how I annotated the tweets for Task 1 and Task 2 respectively.


Below you will see a general description of each step of the project.

### Task 1: Text Classification
- Annotation with Doccano
- Text Preprocessing (cleaning tags and hashtags, adjusting spaeces, removing emojis, treating Turkish special characters properly, etc)
- Machine Learning: training different models on _earthquake10K.json_ dataset to predict the labels I manually annotated. At this step, I trained 4 different models:
    - Logistic Regression
    - Naive Bayes Classifier
    - SVM
    - LSTM
    - Bidirectional LSTM
and I finetuned one pre-trained transformers model (BERTÜRK).
- Model comparison and performance evaluation

### Task 2: NER
- Annotation with Doccano
- Creating Token-Tag Pairs, and converting the tags to IOB format. (Realignment needed)
- Machine Learning: In this step, I decided to use a pre-trained model on Turkish NER tasks and fine-tune it. This entailed a second (this time subword) tokenization and hence, a second realignment of the tags.
- Performance Evaluation

### Task 3: Geocoding and Visualization
- Collection of all entities classified as address and cities together
- Using googlemaps library and Google Maps API-key to extract latitudes and longtitudes for those addresses
- Visualization of all emergency calls that was shared with a location on a plotly map.

#### Note: 
Since there is no real-time data feeding currently going on, the app is designed in way to show you a snapshot of how the map would have looked like on a selected date.

## Deployment
- Because of the GPU requirements, the app is deployed on HuggingFace Spaces. [Click here]('https://huggingface.co/spaces/yhaslan/every_second_matters') to access to the app.
- Check out the Demo video of the app also in this repository: Demo_EverySecondMatters_App
- You can also see the app's main repository [here]('https://huggingface.co/spaces/yhaslan/every_second_matters/tree/main')
- You can also find my fine-tuned [tweet classification]('https://huggingface.co/yhaslan/berturk-earthquake-tweets-classification') and [NER]('https://huggingface.co/yhaslan/turkish-earthquake-tweets-ner') models on HuggingFace.
