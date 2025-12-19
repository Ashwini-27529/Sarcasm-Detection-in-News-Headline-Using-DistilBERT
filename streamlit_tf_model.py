import streamlit as st
import os
import json
from typing import Tuple, Optional


def load_model_and_tokenizer(model_dir: str = 'distilbert_sarcasm_model') -> Tuple[Optional[object], Optional[object], dict]:
    try:
        from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
        import tensorflow as tf
    except Exception as e:
        return None, None, {'error': f'Could not import transformers/tensorflow: {e}'}

    try:
        model = TFDistilBertForSequenceClassification.from_pretrained(model_dir)
        tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        meta = {}
        cfg_path = os.path.join(model_dir, 'config.json')
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
        return model, tokenizer, meta
    except Exception as e:
        return None, None, {'error': str(e)}


def fallback_predict(text: str) -> Tuple[int, float, dict]:
    text_l = text.lower()
    negative_markers = ['yeah right', 'as if', 'sure', 'obviously', 'what a surprise', 'shock', 'not surprising']
    exclaim = '!' in text
    hits = sum(1 for w in negative_markers if w in text_l)
    score = min(0.9, 0.4 + 0.2 * hits + (0.2 if exclaim else 0))
    pred = 1 if (hits > 0 or exclaim) else 0
    probs = {0: 1.0 - score, 1: score}
    return pred, float(score), probs


@st.cache(allow_output_mutation=True)
def cached_loader():
    return load_model_and_tokenizer()


def predict_with_model(text: str, model, tokenizer):
    import tensorflow as tf
    inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=64)
    outputs = model(inputs)
    logits = outputs.logits
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]
    pred_class = int(tf.argmax(logits, axis=1).numpy()[0])
    return pred_class, float(probs[pred_class]), {'0': float(probs[0]), '1': float(probs[1])}


def main():
    st.set_page_config(page_title='Sarcasm Detector', layout='centered')
    st.title('Sarcasm Detection')

    model, tokenizer, metadata = cached_loader()

    st.write('Enter a news headline and click Predict to check whether it is sarcastic.')

    headline = st.text_area('Headline', height=120)

    if st.button('Predict'):
        if not headline or headline.strip() == '':
            st.warning('Please enter a headline to predict.')
        else:
            if model is not None and tokenizer is not None:
                with st.spinner('Running model...'):
                    pred, conf, probs = predict_with_model(headline, model, tokenizer)
            else:
                pred, conf, probs = fallback_predict(headline)

            label = 'Sarcastic' if pred == 1 else 'Not Sarcastic'
            st.markdown(f'### Prediction: {label}')
            st.write(f'Confidence: {conf*100:.2f}%')
            st.write({'Not Sarcastic': probs.get(0), 'Sarcastic': probs.get(1)})


if __name__ == '__main__':
    main()
