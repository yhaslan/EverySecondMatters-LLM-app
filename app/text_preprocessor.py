import re
from sklearn.base import BaseEstimator, TransformerMixin

class CustomPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.abbreviations = [
            ('apt', 'Apartmanı'),
      ('Apt', 'Apartmanı'),
      ('APT', 'Apartmanı'),
      ('apart', 'Apartmanı'),
      ('Apart', 'Apartmanı'),
      ('APART', 'Apartmanı'),
      ('sok', 'Sokak'),
      ('sk', 'Sokak'),
      ('Sok', 'Sokak'),
      ('Sk', 'Sokak'),
      ('SOK', 'Sokak'),
      ('SK', 'Sokak'),
      ('cad', 'Caddesi'),
      ('Cad', 'Caddesi'),
      ('CAD', 'Caddesi'),
      ('cd', 'Caddesi'),
      ('Cd', 'Caddesi'),
      ('CD', 'Caddesi'),
      ('bşk', 'başkanlığı'),
      ('bul', 'Bulvarı'),
      ('blv', 'Bulvarı'),
      ('Blv', 'Bulvarı'),
      ('BLV', 'Bulvarı'),
      ('bulv', 'Bulvarı'),
      ('Bulv', 'Bulvarı'),
      ('BULV', 'Bulvarı'),
      ('mey', 'meydanı'),
      ('meyd', 'meydanı'),
      ('ecz', 'Eczanesi'),
      ('Ecz', 'Eczanesi'),
      ('ECZ', 'Eczanesi'),
      ('mh', 'Mahallesi'),
      ('mah', 'Mahallesi'),
      ('Mh', 'Mahallesi'),
      ('Mah', 'Mahallesi'),
      ('MH', 'Mahallesi'),
      ('MAH', 'Mahallesi'),
      ('şb', 'şube'),
      ('maraş', 'Kahramanmaraş'),
      ('maras', 'Kahramanmaraş'),
      ('Maraş', 'Kahramanmaraş'),
      ('Maras', 'Kahramanmaraş'),
      ('MARAŞ', 'Kahramanmaraş'),
      ('MARAS', 'Kahramanmaraş'),
      ('kmaraş', 'Kahramanmaraş'),
      ('kmaras', 'Kahramanmaraş'),
      ('KMaraş', 'Kahramanmaraş'),
      ('KMaras', 'Kahramanmaraş'),
      ('KMARAŞ', 'Kahramanmaraş'),
      ('KMARAS', 'Kahramanmaraş'),
      ('antep', 'Gaziantep'),
      ('Antep', 'Gaziantep'),
      ('ANTEP', 'Gaziantep'),
      ('anteb', 'Gaziantep'),
      ('Anteb', 'Gaziantep'),
      ('ANTEB', 'Gaziantep'),
      ('Urfa', 'Şanlıuarfa'),
      ('urfa', 'Şanlıuarfa'),
      ('URFA', 'Şanlıuarfa'),

        ]

    def fit(self, X, y=None):
        return self

    def cleaned_tags_and_hashtags(self,text):
          i = 0
          while i < (len(text)):
              if text[i] == '@' or text[i]== '#':
                  a=0
                  while i+a < len(text):
                      if text[i+a] == ' ':
                          break
                      else:
                          a += 1
                  text = text[:i] + ' '*(a) + text[i+a:]
              else:
                  i+=1
          return text



    def normalize_abbreviations(self, text):
        text  = text.replace('k.maraş', 'Kahramanmaraş')
        text  = text.replace('K.maraş', 'Kahramanmaraş')
        text  = text.replace('K.Maraş', 'Kahramanmaraş')
        text  = text.replace('k.maras', 'kahramanmaraş')
        text  = text.replace('K.maras', 'Kahramanmaraş')
        text  = text.replace('K.Maras', 'kahramanmaraş')
        for regex, replacement in self.abbreviations:
            text = re.sub(rf'\b{re.escape(regex)}\b', replacement, text)
            text  = re.sub(r'\s\s+', ' ',text )
          
        return text



    def transform(self, X, y=None):
        X['text'] = X['text'].apply(lambda x: self.cleaned_tags_and_hashtags(x))
        X['text'] = X['text'].apply(lambda x: self.normalize_abbreviations(x))
        return X
        