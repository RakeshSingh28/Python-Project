from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2-medium')
set_seed(42)
print(generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5))


#Output Response --
#[{'generated_text': "Hello, I'm a language model, a linguist. I love models.\n\nSo first – let's focus on what we learn. What"}, {'generated_text': "Hello, I'm a language model, which means I think about the language and its syntax so that I can express the semantics needed to describe a given"}, {'generated_text': "Hello, I'm a language model, a programming language.\n\nMy job is to build great systems and languages capable of building more systems and languages"}, {'generated_text': "Hello, I'm a language model, and I just wrote a model library for Angular which should be really useful in your Angular projects. How can I"}, {'generated_text': "Hello, I'm a language model, I'll be getting the results of a lot of experimentation, in a short amount of time I'd like to"}]

