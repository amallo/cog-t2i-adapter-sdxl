import replicate

image = open("./city.jpeg", "rb")
output = replicate.run(
    "amallo/post-apocalyptic-t2i-adapter-sdxl:663adf0feb01e88993c62b4652e54aa5d204923967c1f52555238cb9a232221d",
    input={
        "image": image,
        "prompt": "A photo of TOK, 4k photo, highly detailed",
        "scheduler": "K_EULER_ANCESTRAL",
        "num_samples": 1,
        "guidance_scale": 7.5,
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
        "num_inference_steps": 30,
        "adapter_conditioning_scale": 1,
        "adapter_conditioning_factor": 1
    }
)
print(output)