from transformers import pipeline

# Create a text classification pipeline
pipe = pipeline("text-classification")

# Classify the given text
output1 = pipe("This restaurant is awesome")
output2 = pipe("This cupboard is aweful")
print(output1, output2)


#Output Response --
#[{'label': 'POSITIVE', 'score': 0.9998743534088135}] [{'label': 'POSITIVE', 'score': 0.9984404444694519}]