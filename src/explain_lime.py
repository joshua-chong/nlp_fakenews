# src/explain_lime.py
from lime.lime_text import LimeTextExplainer

def lime_explain_text(pipe, text, class_names=("fake", "real"), num_features=10, out_html=None):
    explainer = LimeTextExplainer(class_names=list(class_names))
    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=lambda texts: pipe.predict_proba(texts),
        num_features=num_features
    )
    if out_html:
        exp.save_to_file(out_html)
    return exp   # return Explanation, not string
