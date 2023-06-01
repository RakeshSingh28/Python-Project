from transformers import pipeline

# Create a text filling pipeline using the BERT base model
unmasker = pipeline('fill-mask', model='bert-base-uncased')

# Fill in the masked token in the given text
output = unmasker("Hello I'm a [MASK] player.")
print(output)


#Output Response --
#[{'score': 0.1403701901435852, 'token': 2374, 'token_str': 'football', 'sequence': "hello i'm a football player."}, {'score': 0.0888328030705452, 'token': 2502, 'token_str': 'big', 'sequence': "hello i'm a big player."}, {'score': 0.07617121189832687, 'token': 2204, 'token_str': 'good', 'sequence': "hello i'm a good player."}, {'score': 0.06574778258800507, 'token': 2307, 'token_str': 'great', 'sequence': "hello i'm a great player."}, {'score': 0.02408873289823532, 'token': 2858, 'token_str': 'guitar', 'sequence': "hello i'm a guitar player."}]