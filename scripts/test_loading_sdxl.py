import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

def main():
    print("Initializing SDXL Loading Test...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float16
    
    print(f"Device: {device}")
    
    # 1. Load VAE (FP16 Fix)
    print("Loading VAE (madebyollin/sdxl-vae-fp16-fix)...")
    try:
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=weight_dtype
        ).to(device)
        print("[SUCCESS] VAE loaded successfully.")
    except Exception as e:
        print(f"[FAILED] VAE load failed: {e}")
        return

    # 2. Load UNet
    print("Loading UNet (stabilityai/stable-diffusion-xl-base-1.0)...")
    try:
        unet = UNet2DConditionModel.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            subfolder="unet", 
            torch_dtype=weight_dtype
        ).to(device)
        print("[SUCCESS] UNet loaded successfully.")
    except Exception as e:
        print(f"[FAILED] UNet load failed: {e}")
        return

    # 3. Load Text Encoders
    print("Loading Text Encoders...")
    try:
        text_encoder_1 = CLIPTextModel.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            subfolder="text_encoder", 
            torch_dtype=weight_dtype
        ).to(device)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            subfolder="text_encoder_2", 
            torch_dtype=weight_dtype
        ).to(device)
        print("[SUCCESS] Text Encoders loaded successfully.")
    except Exception as e:
        print(f"[FAILED] Text Encoders load failed: {e}")
        return

    # 4. Initialize ControlNets (Dummy)
    print("Initializing Dummy ControlNets...")
    try:
        # Geometry (1ch -> converted via conv_in, standard SDXL CtrlNet expects 3ch usually but we can init from UNet)
        # We will test initializing from UNet for 1 channel
        controlnet_a = ControlNetModel.from_unet(unet)
        # Manually adjust first conv for 1 channel if needed, but for now just load standard
        print("[SUCCESS] ControlNet A (from UNet) initialized.")
    except Exception as e:
        print(f"[FAILED] ControlNet Init failed: {e}")

    print("Checking VRAM usage...")
    print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    
    print("Test Complete. SDXL Environment looks good! (Rocket)")

if __name__ == "__main__":
    main()
