import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from PIL import Image
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from text_preprocessor import CustomPreprocessor

import numpy as np
import torch
import datetime as dt
import googlemaps


import __main__
__main__.CustomPreprocessor = CustomPreprocessor


### Config
st.set_page_config(
    page_title="Emergency Call Tweets Location Extractor",
    page_icon="🚨",
    layout="wide"
)



st.title("Every Second Matters: 🚨an app for timely disaster response🚨")
st.header("Dashboard")
st.markdown("""
    Using Turkish twitter data from the earthquake that hits Syria and Turkey on 6 February,
    this app performs three task:

    * Detection of rescue calls and emergency needs with a text classifier
    * Extraction of person names, city names and addresses in emergency tweets with a Named Entity Recognition (NER) model
    * Geoplotting of those addresses on an interactive map simulating a real-time plot

    This dashboard provides some EDA figures, allows users to test the Task 1 and Task 2 models I trained, and offers a demo simulation of the emergency call map.


        """)

DATA_URL = ("https://myjedhabucket-yagmur.s3.eu-west-3.amazonaws.com/stratified_sample_0.1.csv")


@st.cache_data
def load_data(url):
    d = pd.read_csv(url)
    return d



data_load_state = st.text('Loading data...')

@st.cache_resource  # 👈 Add the caching decorator
def load_classifier():
    return pipeline("text-classification", model="yhaslan/berturk-earthquake-tweets-classification",
                    device = 0
                     )

clf = load_classifier()

@st.cache_resource  # 👈 Add the caching decorator
def load_ner_model():
    return pipeline("token-classification",
              model="yhaslan/turkish-earthquake-tweets-ner",
              aggregation_strategy = 'first',
              
              device =0
              )

ner = load_ner_model()

df = load_data(DATA_URL)
df['date'] = pd.to_datetime(df['date'])
df['day'] = df['date'].dt.date
data_load_state.text("")

def random_data(nrows, data=df):
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=nrows, random_state =42)
    train_indices, test_indices = next(stratified_split.split(data, data['day']))
    data = data.iloc[test_indices].reset_index()
    return data


data = random_data(7000)
## Run the below code if the check is checked ✅
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data.head(100))

st.header("Explore the Data")
stratified_sample_multiplied = pd.concat([df] * 10, ignore_index=True)
fig = px.histogram(stratified_sample_multiplied, x=stratified_sample_multiplied['date'].dt.date,
                       title="Number of Tweets per Day", width=400, height=400)

fig.update_layout(bargap=0.1)
st.plotly_chart(fig, use_container_width=False)

#### Create two columns
col1, col2 = st.columns(2)


with col1:
    image = Image.open('WordCloud_Other.png')
    col1.image(image, caption='WordCloud for non-emergency tweets')
    st.markdown("""
    The image on the left shows the most frequent words in tweets I annotated as 'Other' before trainin the model.
    It contains mostly the common stop words in Turkish such as "ve" (and), "bu" (this), "bir" (one/a/an), "çok" (much) as one can expect.
    Other most frequent words are "Allah" (God), "geçmiş olsun" (my condolences), "lütfen" (please), which also makes sense as
    these are mostly the tweets to share their condolences and pray for the people.
    """)
with col2:
    image_2 = Image.open('WordCloud_emergency_calls.png')
    col2.image(image_2, caption='WordCloud for emergency tweets')
    st.markdown("""
    The image on the right, on the other hand, shows the frequent words in the tweets I annotated as "emergency call".
    The frequent words are "lütfen" (please)," yardım / yardım edin" (help), "enkaz altında"
    (under the rubbles), "hatay / antakya" (one of the most affected cities), "mahallesi" (neighborhood),
    "sokak" / "caddesi" (street), "apartmanı" (building) which also makes sense since people
    who make emergency calls often share locations. Interestingly, in these tweets, these words were
    even more frequently used than typical stopwords.
    """)


st.header("Task 1: Binary Text Classification")
st.markdown("Click the button below to print a random tweet from the dataset")


if st.button('Print a random tweet'):
    random_tweet = data['content'].sample(n=1).iloc[0]
    st.subheader('Here is an example tweet:')
    st.write(random_tweet)


    st.subheader('Below you will see the AI-based classification of the tweet')
    #clf = pipeline("text-classification", model="yhaslan/berturk-earthquake-tweets-classification")
    result = clf(random_tweet)
    for i in result:
        if i['label'] == 'LABEL_1':
            st.write("✅This is an emergency call")
        else:
            st.write("❌This is not an emergency call")


st.markdown("Feel free to try out with a custom tweet as well")
#clf = pipeline("text-classification", model="yhaslan/berturk-earthquake-tweets-classification", device = 1)
text_input = st.text_input( "Enter some text 👇")
result = clf(text_input)
for i in result:
    if i['label'] == 'LABEL_1':
        st.write("✅This is an emergency call")
    else:
        st.write("❌This is not an emergency call")

st.markdown("""
     For more information,
     checkout the [huggingface page of my classification model](https://huggingface.co/yhaslan/berturk-earthquake-tweets-classification).

""")

tag_to_color = {'CITY': 'blue', 'ADDR': 'yellow', 'PER': 'red', 'OTHER': 'green'}



st.header("Task 2: NER")
st.markdown("Click the button below to print a random tweet from the dataset")
if st.button('Print another random tweet'):
    random_tweet = data['content'].sample(n=1).iloc[0]
    st.subheader('Here is an example tweet:')
    st.write(random_tweet)
    st.subheader('Below you will see the NER entities of the tweet')
    result = ner(random_tweet)
    st.markdown('Here is the color legend for each tag:')
    tag_to_background = {'CITY': 'blue', 'ADDR': 'purple', 'PER': 'red', 'OTHER': 'green'}
    legend = ""
    for tag, background_color in tag_to_background.items():
        legend += f"<span style='background-color: {background_color}; color: white; padding: 4px; margin-right: 10px;'>{tag}</span> &nbsp;"
    st.write(legend, unsafe_allow_html=True)


    sentence = []
    for item in result:
        background_color = tag_to_background[item['entity_group']]
        sentence.append(f"<span style='background-color: {background_color}; color: white'>{item['word']}</span> ")
        sentence.append(f"<span style='background-color: {background_color}; color: white'>[{item['entity_group']}]</span>  ")
    final_sentence = "".join(sentence)
    st.write(final_sentence, unsafe_allow_html=True)

st.markdown("""
     For more information, checkout the
     [huggingface page of my NER model](https://huggingface.co/yhaslan/turkish-earthquake-tweets-ner).

""")

st.header("Task 3: Geocoding and Visualization")
st.markdown("""Move the slider to pick a date for which you wish to visuzalize how the emergency call map would have looked
    like.
            """)

data = random_data(2000)

data['date'] = data['date'].astype(str)
data['date'] = data['date'].apply(lambda x: dt.datetime.fromisoformat(x[:-6]))


# Create a slider for selecting a date
selected_date = st.slider("Select a date", 
                                         min_value=data['date'].min().to_pydatetime(), 
                                         max_value=data['date'].max().to_pydatetime(),
                           format = "MMM-DD, HH:mm:ss")



# Display the selected date
st.write("Selected DateTime:", selected_date)

filtered_data = data[data['date'] <= selected_date]


with open('text_preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

filtered_data.rename(columns={'content': 'text'}, inplace=True)
filtered_data = preprocessor.transform(filtered_data)

clf_preds = clf(filtered_data['text'].to_list())
emergencies = []
for i in range(len(clf_preds)):
  if clf_preds[i]['label'] == 'LABEL_1':
    emergencies.append(i)
emergency_tweets = filtered_data.iloc[emergencies].reset_index(drop=True)

ner_preds = []
for tweet in emergency_tweets['text'].tolist():
    ner_preds.append(ner(tweet))
ner_preds = pd.Series(ner_preds)



def create_entity_column(pred,entity_group):
    word=''
    for i in range(len(pred)):
        if pred[i]['entity_group'] == entity_group:
            word+= pred[i]['word'] + ' '
    return word

emergency_tweets['PERSON'] = ner_preds.apply(lambda x :create_entity_column(x,'PER') )
emergency_tweets['ADDRESS'] = ner_preds.apply(lambda x :create_entity_column(x,'ADDR') )
emergency_tweets['CITY'] = ner_preds.apply(lambda x :create_entity_column(x,'CITY') )

emergency_tweets['FULL_ADDRESS'] = emergency_tweets['ADDRESS'].astype(str) + ',' + \
                emergency_tweets['CITY'] + ',' + \
                'Turkey'
for i in range(len(emergency_tweets)):
  if emergency_tweets['ADDRESS'][i] == '':
    emergency_tweets['FULL_ADDRESS'][i] = None

emergency_tweets['Lat'] = np.zeros(len(emergency_tweets))
emergency_tweets['Lon'] = np.zeros(len(emergency_tweets))


### Retrieveing latitudes and longtitudes

gmaps = googlemaps.Client(key= st.secrets["my_api_key"])

for i in range(len(emergency_tweets)):
    if not pd.isnull(emergency_tweets.loc[i, 'FULL_ADDRESS']):
        geocode_result = gmaps.geocode(emergency_tweets.loc[i, 'FULL_ADDRESS'])
        if geocode_result:  # Check if the result is not empty
            emergency_tweets.loc[i, 'Lat'] = geocode_result[0]['geometry']['location']['lat']
            emergency_tweets.loc[i, 'Lon'] = geocode_result[0]['geometry']['location']['lng']
        else:
          emergency_tweets.loc[i, 'Lat'] = None
          emergency_tweets.loc[i, 'Lon'] = None



# Calculate the numerical representation of each date (days since the reference date)

emergency_tweets['how much time has passed (in hours)'] = (selected_date - emergency_tweets['date']).dt.total_seconds() / 3600

emergency_tweets['tweet'] = emergency_tweets['text'].str.wrap(30)
emergency_tweets['tweet'] = emergency_tweets['tweet'].apply(lambda x: x.replace('\n', '<br>'))

fig = px.scatter_mapbox(emergency_tweets, lat="Lat", lon="Lon",
                        color_continuous_scale="hot",
                         zoom=6, height=600,
                         hover_data = ["date", "tweet"],
                        hover_name="PERSON", 
                        color = 'how much time has passed (in hours)',  
                        center = {'lat': 37 , 'lon':37},
                        )  
fig.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
            
st.markdown("""
        If you are interested in learning more on this project, check out my [Github](https://github.com/yhaslan) account.
    """)
