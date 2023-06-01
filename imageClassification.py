from transformers import pipeline

classifier = pipeline("image-classification")

output = classifier("./tree.png")
print(output)


#Output Response --
#[{'score': 0.023298686370253563, 'label': 'safety pin'}, {'score': 0.016361841931939125, 'label': 'broom'}, {'score': 0.015550976619124413, 'label': 'quill, quill pen'}, {'score': 0.015069527551531792, 'label': 'sarong'}, {'score': 0.013910593464970589, 'label': 'seashore, coast, seacoast, sea-coast'}]