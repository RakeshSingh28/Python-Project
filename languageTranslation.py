from transformers import T5Tokenizer, TFT5ForConditionalGeneration

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = TFT5ForConditionalGeneration.from_pretrained('t5-small')

# Define the input text
input_text = "Hello, how are you?"

# Tokenize the input text
prefix = 'translate English to French:'
input_text = prefix + input_text
input_ids = tokenizer.encode(input_text, return_tensors='tf')

# Generate the translation
translated_ids = model.generate(input_ids=input_ids, max_length=100, num_beams=4, early_stopping=True)
translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)

# Print the translated text
print("Translated text:", translated_text)


#Output Response --
# Translated text: Oui, comment êtes-vous?