
import torch
import os
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler
from transformers import T5EncoderModel, CLIPTextModel, CLIPTextModelWithProjection

def main():
    print("Testing SD3.5 Loading...")
    
    model_id = "stabilityai/stable-diffusion-3.5-medium"
    
    # Check if authorization token is available if needed
    # (Assuming environment is configured or token is cached)

    try:
        print(f"Loading Pipeline from {model_id}...")
        # Load pipeline with CPU offload initially to save VRAM
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            text_encoder_3=None, # Optional T5 - load it separately if needed or let pipeline handle it
            tokenizer_3=None
        )
        # Note: By default from_pretrained might try to load T5. T5 is huge. 
        # SD3.5 Medium uses T5. 
        
        # Explicitly enabling CPU offload
        pipe.enable_model_cpu_offload()
        print("Model loaded successfully with CPU offload enabled.")
        
        # Checking Components
        print(f"Transformer Config: {pipe.transformer.config}")
        
        # Simple inference test
        print("Running inference test...")
        prompt = "A close-up photo of a woman with beautiful hair, studio lighting, 8k"
        
        # If running on CPU only machine without CUDA, this might be slow or fail if float16 is not supported on CPU properly for some ops
        # But user has CUDA based on previous context (though warning said "CUDA not available" in previous step).
        # Wait, previous step said: "User provided device_type of 'cuda', but CUDA is not available."
        # This is CRITICAL. If CUDA is not available, I cannot really run SD3.5 efficiently or at all for training.
        # But maybe it was just that specific command session? 
        # The user's metadata says WSL. Usually WSL has GPU access if configured. 
        # I will assume CUDA is available or we fallback to CPU (which will be super slow).
        
        if torch.cuda.is_available():
            print("CUDA is available.")
            pipe.to("cuda") # enable_model_cpu_offload handles this actually
        else:
            print("WARNING: CUDA is NOT available. Running on CPU (expect slowness).")
            # For CPU we might need float32
            # pipe = pipe.to(dtype=torch.float32) 
        
        image = pipe(
            prompt, 
            num_inference_steps=20, 
            guidance_scale=4.5,
            height=512, # SD3 usually generates 1024, but keeping small for test
            width=512
        ).images[0]
        
        os.makedirs("output", exist_ok=True)
        image.save("output/test_sd3.png")
        print("Test image saved to output/test_sd3.png")
        
    except Exception as e:
        print(f"FAILED to load SD3.5: {e}")
        # Print detailed auth info if likely
        if "401" in str(e) or "gated" in str(e).lower():
            print("This seems to be an authentication error. Please ensure you have access to the gated model and are logged in.")

if __name__ == "__main__":
    main()
