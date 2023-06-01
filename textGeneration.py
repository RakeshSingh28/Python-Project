from transformers import pipeline, set_seed

# Create a text generation pipeline using the GPT-2 medium model
generator = pipeline('text-generation', model='gpt2-medium')

# Set the random seed to ensure reproducibility
set_seed(42)

# Generate text based on the given prompt
output = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
print(output)

#Output Response --
#[{'generated_text': "Hello, I'm a language model, a linguist. I love models.\n\nSo first – let's focus on what we learn. What"}, {'generated_text': "Hello, I'm a language model, which means I think about the language and its syntax so that I can express the semantics needed to describe a given"}, {'generated_text': "Hello, I'm a language model, a programming language.\n\nMy job is to build great systems and languages capable of building more systems and languages"}, {'generated_text': "Hello, I'm a language model, and I just wrote a model library for Angular which should be really useful in your Angular projects. How can I"}, {'generated_text': "Hello, I'm a language model, I'll be getting the results of a lot of experimentation, in a short amount of time I'd like to"}]

