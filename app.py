import torch
import numpy as np
from transformers import AutoProcessor, TextStreamer
from peft import PeftModel
from unsloth import FastVisionModel
from PIL import Image
import gradio as gr

## Load the fine-tuned model and tokenizer with LoRA adapter
base_model_name = "unsloth/Llama-3.2-11B-Vision-Instruct"
fine_tuned_model_name = "Murasajo/Llama-3.2-VL-Finetuned-on-HandwrittenText"

# Load base model with 4-bit to save memory
base_model, tokenizer = FastVisionModel.from_pretrained(
    base_model_name,
    load_in_4bit=True,  # Efficient for small GPUs
    use_gradient_checkpointing="unsloth",
)


# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, fine_tuned_model_name)
processor = AutoProcessor.from_pretrained(fine_tuned_model_name)

# Model in evaluation mode
model.eval()


def process_image(image, prompt):
    if image is None:
        return "Please upload an image first."
    
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        inputs = processor(
            images=image,
            text=input_text,
            return_tensors="pt",
            add_special_tokens=False
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        text_streamer = TextStreamer(processor, skip_prompt=True)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                streamer=text_streamer,
                max_new_tokens=128,
                use_cache=True,
                temperature=1.5,
                min_p=0.1
            )
        
        return processor.decode(outputs[0], skip_special_tokens=True)
    
    except Exception as e:
        return f"Error processing image: {str(e)}"

def extract_text(image, prompt):
    if not prompt.strip():
        prompt = "Extract all the handwritten text in the image, ensure precision and clarity. The text is handwritten and in English. Display it in a paragraph form as it is written in the image."
    
    return process_image(image, prompt)

# Create the Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# Handwritten Text Recognition System")
    
    with gr.Row():
        with gr.Column():
            # Input components
            image_input = gr.Image(type="pil", label="Upload Handwritten Text Image")
            prompt_input = gr.Textbox(
                label="Custom Prompt (optional)",
                placeholder="Enter your prompt here or leave empty for default prompt",
                lines=2
            )
            submit_btn = gr.Button("Extract Text")
        
        with gr.Column():
            # Output component - simple textbox instead of chat history
            text_output = gr.Textbox(
                label="Extracted Text",
                lines=10,
                interactive=False
            )
    
    # Event handler
    submit_btn.click(
        fn=extract_text,
        inputs=[image_input, prompt_input],
        outputs=[text_output]
    )
    
    gr.Markdown("""
    ## Instructions:
    1. Upload an image containing handwritten text
    2. (Optional) Customize the prompt for specific requirements
    3. Click 'Extract Text' to get the results
    4. The extracted text will appear in the output box
    """)

# Launch the interface
iface.launch()



