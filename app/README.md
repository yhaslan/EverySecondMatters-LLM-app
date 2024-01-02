
# Every Second Matters: 🚨an app for timely disaster response🚨

Using Turkish twitter data from the earthquake that hits Syria and Turkey on 6 February, this app performs three task:

- Detection of rescue calls and emergency needs with a text classifier
-  Extraction of person names, city names and addresses in emergency tweets with a Named Entity Recognition (NER) model
-  Geoplotting of those addresses on an interactive map simulating a real-time plot

This is a Turkish language NLP project.

### In this folder you will find:
- _requirements.txt_ that contains necessary libraries to install to run application,
- _app.py_ script which contains the code for streamlit app that merges all three tasks,
- text_preprocessor.py script which consists of the custom preprocessor I designed for this task: it normalizes the abbreviations and removes all entities with tags(@) and hashtags(#).
- text_preprocessor.pkl file that contains the trained preprocessor
- WordClouds that show the most frequent words in emergency tweets and other tweets.