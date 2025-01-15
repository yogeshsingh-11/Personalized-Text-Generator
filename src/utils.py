from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_model(model_path='./models/marketing_model'):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    return model, tokenizer

def save_model(model, tokenizer, model_path='./models/marketing_model'):
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
