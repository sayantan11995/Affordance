from transformers import Blip2ForConditionalGeneration, AutoProcessor
from peft import PeftModel, PeftConfig
import torch
from PIL import Image

peft_model_id = "fine-tuned-blip2"
config = PeftConfig.from_pretrained(peft_model_id)

model = Blip2ForConditionalGeneration.from_pretrained(config.base_model_name_or_path, load_in_8bit=True, device_map="auto")
model = PeftModel.from_pretrained(model, peft_model_id)

print(config.base_model_name_or_path)
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")


model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"

image = Image.open("0.png")

# prepare image for the model
# inputs = processor(images=image, return_tensors="pt").to(device)
# pixel_values = inputs.pixel_values

# text_inputs = processor.tokenizer("Can Helicopter be used for Ride by Human? Answer Yes or No: \n\n", padding=True, return_tensors="pt")
# input_ids = text_inputs["input_ids"].to(device)

# model.eval()

inputs = processor(image, text="Context: I saw a flying helicopter.\n\nQuestion: Can Helicopter be used for Feed by Human?\n\nAnswer: ", return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs, max_new_tokens=10)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)

# print(pixel_values)
# print(input_ids)
# generate caption
# model.generate(pixel_values=pixel_values, tomax_length=50)
# generated_ids = model.generate(pixel_values=pixel_values, text="Question: Can Helicopter be used for Ride by Human? Answer Yes or No:", max_length=50)
# generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
# print(generated_ids)
# print(generated_caption)
# print(processor.tokenizer.decode([1437]))