from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained("saved_model")
tokenizer = DistilBertTokenizer.from_pretrained("saved_model")

model.push_to_hub("aeesh1/arxiv-abstract-classifier")
tokenizer.push_to_hub("aeesh1/arxiv-abstract-classifier")