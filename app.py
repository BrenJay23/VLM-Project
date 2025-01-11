import gradio as gr
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

def model_inference(image, prompt):
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceTB/SmolVLM-Instruct",
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    ).to(DEVICE)

    # Create input messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = inputs.to(DEVICE)

    # Generate outputs
    generated_ids = model.generate(**inputs, max_new_tokens=200)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return generated_texts[0].split("Assistant:")[1].strip()

# Create a Gradio interface
with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row():
            # Create an image output to display the captured image
            captured_image = gr.Image(label="Captured Image", type="pil")
    
            # Create a text input for the prompt
            prompt_input = gr.Textbox(label="Prompt", submit_btn=True, placeholder="Enter your prompt here")
        with gr.Row():
            # Create a text output to display the description
            description_output = gr.Textbox(label="Output")
    
    # When the prompt is submitted, describe the captured image
    prompt_input.submit(model_inference, inputs=[captured_image, prompt_input], outputs=description_output)

# Launch the Gradio app
demo.launch(show_error=True)