from diffusers import ( 
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
import os
import subprocess
import tarfile
import requests

SDXL_URL_MAP= {
    "sdxl-1.0": "https://weights.replicate.delivery/default/sdxl/sdxl-1.0.tar",
    "scheduler": "https://weights.replicate.delivery/default/T2I-Adapter-SDXL/scheduler.tar",
    "vae-fp16-fix": "https://weights.replicate.delivery/default/T2I-Adapter-SDXL/sdxl-vae-fp16-fix.tar",
    "lora-train-global": "https://replicate.delivery/pbxt/dwlcMNj38xJ1CxrneqrFjT64NtQ0N38G7LrcOf87dPBWqRbTA/trained_model.tar",
}

ADAPTER_URL_MAP = {
    "openpose": "https://weights.replicate.delivery/default/T2I-Adapter-SDXL/t2i-adapter-openpose-sdxl-1.0.tar",
    "lineart": "https://weights.replicate.delivery/default/T2I-Adapter-SDXL/t2i-adapter-lineart-sdxl-1.0.tar",
    "canny": "https://weights.replicate.delivery/default/T2I-Adapter-SDXL/t2i-adapter-canny-sdxl-1.0.tar",
    "sketch": "https://weights.replicate.delivery/default/T2I-Adapter-SDXL/t2i-adapter-sketch-sdxl-1.0.tar",
    "depth-midas": "https://weights.replicate.delivery/default/T2I-Adapter-SDXL/t2i-adapter-depth-midas-sdxl-1.0.tar",
}

ANNOTATOR_URL_MAP = {
    "openpose": "https://weights.replicate.delivery/default/T2I-Adapter-SDXL/t2i-openpose-annotator.tar",
    "lineart": "https://weights.replicate.delivery/default/T2I-Adapter-SDXL/t2i-lineart-annotator.tar",
    "sketch": "https://weights.replicate.delivery/default/T2I-Adapter-SDXL/t2i-sketch-annotator.tar",
    "depth-midas": "https://weights.replicate.delivery/default/T2I-Adapter-SDXL/t2i-depth-midas-annotator.tar",
}

class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "LMSDiscrete": LMSDiscreteScheduler,
}

def download_and_extract(url: str, dest: str):
    download_file(url, "/src/tmp.tar")
    extract_tar_file("/src/tmp.tar", dest)

def download_file(url: str, destination_path: str):
    # Vérifier si le fichier existe déjà et le supprimer
    if os.path.exists(destination_path):
        os.remove(destination_path)
        print(f"File removed: {destination_path}")
    
    # Télécharger le fichier
    print(f"Téléchargement du fichier depuis {url}...")
    response = requests.get(url, stream=True)
    
    # Vérifier si la requête est réussie
    if response.status_code == 200:
        # Créer le répertoire si besoin
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        # Écrire le contenu dans le fichier
        with open(destination_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"Téléchargement terminé et sauvegardé à {destination_path}")
    else:
        print(f"Erreur lors du téléchargement: {response.status_code}")
        return False

    return True

def extract_tar_file(file_path: str, extract_to: str):
    # Extraire le fichier .tar
    print(f"Extraction du fichier {file_path} dans {extract_to}...")

    if tarfile.is_tarfile(file_path):
        with tarfile.open(file_path, 'r') as tar:
            tar.extractall(path=extract_to)
            tar.close()
            os.remove("/src/tmp.tar")
        print(f"Extraction terminée.")
    else:
        print("Le fichier n'est pas un fichier .tar valide.")

def install_t2i_adapter_cache(
        model_type:str,
        model_base_cache:str,
        model_scheduler_cache:str,
        model_vae_cache:str,
        model_adapter_cache:str,
        model_annotator_cache:str,
        model_lora_cache:str
):
    # Base Model
    if not os.path.exists(model_base_cache):
        os.makedirs(model_base_cache)
        download_and_extract(SDXL_URL_MAP["sdxl-1.0"], model_base_cache)
    # Scheduler
    if not os.path.exists(model_scheduler_cache):
        os.makedirs(model_scheduler_cache)
        download_and_extract(SDXL_URL_MAP["scheduler"], model_scheduler_cache)
    # VAE
    if not os.path.exists(model_vae_cache):
        os.makedirs(model_vae_cache)
        download_and_extract(SDXL_URL_MAP["vae-fp16-fix"], model_vae_cache)
    # Adapter
    if not os.path.exists(model_adapter_cache):
        os.makedirs(model_adapter_cache)
        download_and_extract(ADAPTER_URL_MAP[model_type], model_adapter_cache)
    # Annotator
    if not os.path.exists(model_annotator_cache) and model_type != "canny":
        os.makedirs(model_annotator_cache)
        download_and_extract(ANNOTATOR_URL_MAP[model_type], model_annotator_cache)
    # Lora
    if not os.path.exists(model_lora_cache):
        os.makedirs(model_lora_cache)
        download_and_extract(SDXL_URL_MAP['lora-train-global'], model_lora_cache)
