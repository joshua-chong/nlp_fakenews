import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


#the actual pipelines reducing noise how filtering slightly
def build_lr_pipeline(
    stop_words = "english",
    ngram_range = (1,2),
    min_df = 5,
    max_df = 0.9,
    class_weight = "balanced",
    max_iter = 2000
):
  return Pipeline([
      ("tfidf", TfidfVectorizer(stop_words=stop_words,
                                lowercase = True,
                                ngram_range=ngram_range,
                                min_df=min_df,
                                max_df=max_df)),
      
      ("clf", LogisticRegression(class_weight=class_weight,
                                 max_iter=max_iter))
      
  ])

#extract the top terms that declare if real or fake
def top_terms (pipe, k=20):
  vec = pipe.named_steps["tfidf"]
  clf = pipe.named_steps["clf"]
  feature_names = np.array(vec.get_feature_names_out())
  coefs = clf.coef_[0]
  top_real_idx = np.argsort(coefs)[-k:][::-1]
  top_fake_idx = np.argsort(coefs)[:k]
  top_real = list(zip(feature_names[top_real_idx], coefs[top_real_idx]))
  top_fake = list(zip(feature_names[top_fake_idx], coefs[top_fake_idx]))
  return top_real, top_fake
