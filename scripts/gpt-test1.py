import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("/om/user/ericjm/omnigrok/mistral/arwen-gpt2-medium-x21").to(device)

prompt = "Solve the following arithmetic problems to the best of your ability. Your answer should be an integer: 1 + 1 ="
# prompt = "1 + 1 ="
# input_ids = tokenizer.encode(
#     prompt, return_tensors="pt"
# ).to(device)

inputs = tokenizer(
    prompt, return_tensors="pt"
).to(device)

# sample_output = model.generate(input_ids, do_sample=False, max_length=1)
# print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

outputs = model(**inputs)
top_preds = [tokenizer.decode(i) for i in torch.topk(outputs.logits[0][-1], k=10).indices]
probs = torch.nn.functional.softmax(torch.topk(outputs.logits[0][-1], k=10).values, dim=0)
for i in range(len(top_preds)):
    print("{}: {:.3f}".format(top_preds[i], probs[i]))
loss = torch.nn.functional.cross_entropy(outputs.logits[0, -1], inputs['input_ids'][0, -1], reduction='none')
print(loss)

# print(outputs.logits[-1])

# print("Output:\n" + 100 * "-")
# print(sample_output)




