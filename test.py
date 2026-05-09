import google.generativeai as genai

genai.configure(api_key="")

model = genai.GenerativeModel("gemini-2.5-flash")
response = model.generate_content("Your prompt here")
print(response.text)