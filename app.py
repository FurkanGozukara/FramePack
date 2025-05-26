from diffusers_helper .hf_login import login 

import os 
import subprocess 
import platform 
import random 
import time 
import shutil 
import glob 
import re 
import math 
from typing import Optional ,Dict ,Any 
import sys 
import cv2 
import json 
from natsort import natsorted 

os .environ ['HF_HOME']=os .path .abspath (os .path .realpath (os .path .join (os .path .dirname (__file__ ),'./hf_download')))

import gradio as gr 
import torch 
import traceback 
import einops 
import safetensors .torch as sf 
import numpy as np 
import argparse 

from PIL import Image 
from diffusers import AutoencoderKLHunyuanVideo 
from transformers import LlamaModel ,CLIPTextModel ,LlamaTokenizerFast ,CLIPTokenizer 
from diffusers_helper .hunyuan import encode_prompt_conds ,vae_decode ,vae_encode ,vae_decode_fake 
from diffusers_helper .utils import save_bcthw_as_mp4 ,crop_or_pad_yield_mask ,soft_append_bcthw ,resize_and_center_crop ,state_dict_weighted_merge ,state_dict_offset_merge ,generate_timestamp ,save_bcthw_as_gif ,save_bcthw_as_apng ,save_bcthw_as_webp ,generate_new_timestamp ,save_individual_frames 
from diffusers_helper .models .hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked 
from diffusers_helper .pipelines .k_diffusion_hunyuan import sample_hunyuan 
from diffusers_helper .memory import cpu ,gpu ,get_cuda_free_memory_gb ,move_model_to_device_with_memory_preservation ,offload_model_from_device_for_memory_preservation ,fake_diffusers_current_device ,DynamicSwapInstaller ,unload_complete_models ,load_model_as_complete 
from diffusers_helper .thread_utils import AsyncStream ,async_run 
from diffusers_helper .gradio .progress_bar import make_progress_bar_css ,make_progress_bar_html 
from transformers import SiglipImageProcessor ,SiglipVisionModel 
from diffusers_helper .clip_vision import hf_clip_vision_encode 
from diffusers_helper .load_lora import load_lora ,set_adapters, unload_all_loras, apply_lora_scaling
from pathlib import PurePath 
from diffusers_helper .bucket_tools import find_nearest_bucket 
from PIL import ImageOps

parser =argparse .ArgumentParser ()
parser .add_argument ('--share',action ='store_true')
args =parser .parse_args ()

print (args )

MODEL_NAME_ORIGINAL ='lllyasviel/FramePackI2V_HY'
MODEL_NAME_F1 ='lllyasviel/FramePack_F1_I2V_HY_20250503'
MODEL_DISPLAY_NAME_ORIGINAL ="Original FramePack"
MODEL_DISPLAY_NAME_F1 ="FramePack F1"
DEFAULT_MODEL_NAME =MODEL_DISPLAY_NAME_ORIGINAL 

free_mem_gb =get_cuda_free_memory_gb (gpu )
high_vram =False 

print (f'Free VRAM {free_mem_gb} GB')
print (f'High-VRAM Mode: {high_vram}')

text_encoder =LlamaModel .from_pretrained ("hunyuanvideo-community/HunyuanVideo",subfolder ='text_encoder',torch_dtype =torch .float16 ).cpu ()
text_encoder_2 =CLIPTextModel .from_pretrained ("hunyuanvideo-community/HunyuanVideo",subfolder ='text_encoder_2',torch_dtype =torch .float16 ).cpu ()
tokenizer =LlamaTokenizerFast .from_pretrained ("hunyuanvideo-community/HunyuanVideo",subfolder ='tokenizer')
tokenizer_2 =CLIPTokenizer .from_pretrained ("hunyuanvideo-community/HunyuanVideo",subfolder ='tokenizer_2')
vae =AutoencoderKLHunyuanVideo .from_pretrained ("hunyuanvideo-community/HunyuanVideo",subfolder ='vae',torch_dtype =torch .float16 ).cpu ()

feature_extractor =SiglipImageProcessor .from_pretrained ("lllyasviel/flux_redux_bfl",subfolder ='feature_extractor')
image_encoder =SiglipVisionModel .from_pretrained ("lllyasviel/flux_redux_bfl",subfolder ='image_encoder',torch_dtype =torch .float16 ).cpu ()

print (f"Loading initial transformer model: {DEFAULT_MODEL_NAME}")
transformer :HunyuanVideoTransformer3DModelPacked 
if DEFAULT_MODEL_NAME ==MODEL_DISPLAY_NAME_ORIGINAL :
    transformer =HunyuanVideoTransformer3DModelPacked .from_pretrained (MODEL_NAME_ORIGINAL ,torch_dtype =torch .bfloat16 ).cpu ()
    active_model_name =MODEL_DISPLAY_NAME_ORIGINAL 
elif DEFAULT_MODEL_NAME ==MODEL_DISPLAY_NAME_F1 :
    transformer =HunyuanVideoTransformer3DModelPacked .from_pretrained (MODEL_NAME_F1 ,torch_dtype =torch .bfloat16 ).cpu ()
    active_model_name =MODEL_DISPLAY_NAME_F1 
else :
    raise ValueError (f"Unknown default model name: {DEFAULT_MODEL_NAME}")

print (f"Initial model '{active_model_name}' loaded to CPU.")

vae .eval ()
text_encoder .eval ()
text_encoder_2 .eval ()
image_encoder .eval ()
transformer .eval ()

if not high_vram :
    vae .enable_slicing ()
    vae .enable_tiling ()

transformer .high_quality_fp32_output_for_inference =True 
print ('transformer.high_quality_fp32_output_for_inference = True')

transformer .to (dtype =torch .bfloat16 )
vae .to (dtype =torch .float16 )
image_encoder .to (dtype =torch .float16 )
text_encoder .to (dtype =torch .float16 )
text_encoder_2 .to (dtype =torch .float16 )

vae .requires_grad_ (False )
text_encoder .requires_grad_ (False )
text_encoder_2 .requires_grad_ (False )
image_encoder .requires_grad_ (False )
transformer .requires_grad_ (False )

if not high_vram :

    DynamicSwapInstaller .install_model (transformer ,device =gpu )
    DynamicSwapInstaller .install_model (text_encoder ,device =gpu )

else :

    pass 

stream =AsyncStream ()

current_dir =os .path .dirname (os .path .abspath (__file__ ))
outputs_folder =os .path .join (current_dir ,'outputs')
used_images_folder =os .path .join (outputs_folder ,'used_images')
intermediate_videos_folder =os .path .join (outputs_folder ,'intermediate_videos')
gif_videos_folder =os .path .join (outputs_folder ,'gif_videos')
apng_videos_folder =os .path .join (outputs_folder ,'apng_videos')
webp_videos_folder =os .path .join (outputs_folder ,'webp_videos')
webm_videos_folder =os .path .join (outputs_folder ,'webm_videos')
intermediate_gif_videos_folder =os .path .join (outputs_folder ,'intermediate_gif_videos')
intermediate_apng_videos_folder =os .path .join (outputs_folder ,'intermediate_apng_videos')
intermediate_webp_videos_folder =os .path .join (outputs_folder ,'intermediate_webp_videos')
intermediate_webm_videos_folder =os .path .join (outputs_folder ,'intermediate_webm_videos')
individual_frames_folder =os .path .join (outputs_folder ,'individual_frames')
intermediate_individual_frames_folder =os .path .join (individual_frames_folder ,'intermediate_videos')
last_frames_folder =os .path .join (outputs_folder ,'last_frames')
intermediate_last_frames_folder =os .path .join (last_frames_folder ,'intermediate_videos')
loras_folder =os .path .join (current_dir ,'loras')
presets_folder =os .path .join (current_dir ,'presets')
last_used_preset_file =os .path .join (presets_folder ,'_lastused.txt')

TEMP_METADATA_FOLDER = os.path.join(outputs_folder, 'outputs')

batch_stop_requested =False 

N_LORA = 4  # Number of supported LoRA slots
currently_loaded_lora_info = [
    {"adapter_name": None, "lora_path": None} for _ in range(N_LORA)
]

for directory in [
outputs_folder ,
used_images_folder ,
intermediate_videos_folder ,
gif_videos_folder ,
apng_videos_folder ,
webp_videos_folder ,
webm_videos_folder ,
intermediate_gif_videos_folder ,
intermediate_apng_videos_folder ,
intermediate_webp_videos_folder ,
intermediate_webm_videos_folder ,
individual_frames_folder ,
intermediate_individual_frames_folder ,
last_frames_folder ,
intermediate_last_frames_folder ,
loras_folder ,
presets_folder,
TEMP_METADATA_FOLDER
]:
    try :
        os .makedirs (directory ,exist_ok =True )

    except Exception as e :
        print (f"Error creating directory {directory}: {str(e)}")

outputs_batch_folder =os .path .join (outputs_folder ,'batch_outputs')
try :
    os .makedirs (outputs_batch_folder ,exist_ok =True )
    print (f"Created batch outputs directory: {outputs_batch_folder}")
except Exception as e :
    print (f"Error creating batch outputs directory: {str(e)}")

def open_folder (folder_path ):

    try :
        folder_path =os .path .abspath (folder_path )
        if not os .path .exists (folder_path ):
             return f"Folder does not exist: {folder_path}"
        if platform .system ()=="Windows":
            os .startfile (folder_path )
        elif platform .system ()=="Darwin":
            subprocess .run (["open",folder_path ])
        else :
            subprocess .run (["xdg-open",folder_path ])
        return f"Opened {os.path.basename(folder_path)} folder"
    except Exception as e :
        return f"Error opening folder: {str(e)}"

def print_supported_image_formats ():

    extensions =[
    '.png','.jpg','.jpeg','.bmp','.webp',
    '.tif','.tiff','.gif','.eps','.ico',
    '.ppm','.pgm','.pbm','.tga','.exr','.dib'
    ]

    supported_formats =[]
    unsupported_formats =[]

    for ext in extensions :
        format_name =ext [1 :].upper ()
        if format_name =='JPG':
            format_name ='JPEG'

        try :
            Image .init ()
            if format_name in Image .ID or format_name in Image .MIME :
                supported_formats .append (ext )
            else :
                unsupported_formats .append (ext )
        except :
            unsupported_formats .append (ext )

    return supported_formats 

def get_images_from_folder (folder_path ):

    if not folder_path or not os .path .exists (folder_path ):
        return []

    image_extensions =print_supported_image_formats ()
    images =[]

    for file in os .listdir (folder_path ):
        file_path =os .path .join (folder_path ,file )
        if os .path .isfile (file_path )and os .path .splitext (file )[1 ].lower ()in image_extensions :
            images .append (file_path )

    print (f"Found {len(images)} images. Sorting naturally...")
    sorted_images =natsorted (images )

    return sorted_images 

def get_prompt_from_txt_file (image_path ):

    txt_path =os .path .splitext (image_path )[0 ]+'.txt'
    if os .path .exists (txt_path ):
        try :
            with open (txt_path ,'r',encoding ='utf-8')as f :

                content =f .read ()
                return content .rstrip ()
        except Exception as e :
            print (f"Error reading prompt file {txt_path}: {str(e)}")
    return None 

def format_time_human_readable (seconds ):

    hours ,remainder =divmod (int (seconds ),3600 )
    minutes ,seconds =divmod (remainder ,60 )

    if hours >0 :
        return f"{hours} hour{'s' if hours > 1 else ''} {minutes} min {seconds} seconds"
    elif minutes >0 :
        return f"{minutes} min {seconds} seconds"
    else :
        return f"{seconds} seconds"

def save_processing_metadata (output_path_for_final_file: Optional[str],
                              metadata_dict: Dict[str, Any],
                              is_temporary: bool = False,
                              job_id_for_temp: Optional[str] = None,
                              base_folder_for_temp_metadata: Optional[str] = None
                             ):
    metadata_path = ""
    if is_temporary:
        if not job_id_for_temp:
            print("ERROR: job_id_for_temp must be provided for temporary metadata.")
            return False
        if not base_folder_for_temp_metadata:
            print("ERROR: base_folder_for_temp_metadata must be provided for temporary metadata.")
            return False
        os.makedirs(base_folder_for_temp_metadata, exist_ok=True) # Ensure base folder exists
        metadata_path = get_temp_metadata_filepath_for_job(job_id_for_temp, base_folder_for_temp_metadata)
    elif output_path_for_final_file:
        metadata_path = os.path.splitext(output_path_for_final_file)[0] + '_metadata.txt'
    else:
        print("ERROR: output_path_for_final_file must be provided for final metadata, or is_temporary must be True with job_id_for_temp and base_folder_for_temp_metadata.")
        return False

    try :
        os .makedirs (os .path .dirname (metadata_path ),exist_ok =True )
        with open (metadata_path ,'w',encoding ='utf-8')as f :
            # Ensure Model is always first if present
            if "Model" in metadata_dict:
                 f .write (f"Model: {metadata_dict.pop('Model', 'Unknown')}\n") # Use .pop with default
            # Write other items
            for key ,value in metadata_dict .items (): 
                f .write (f"{key}: {value}\n")
        print(f"Successfully saved metadata to {metadata_path}")
        return True 
    except Exception as e :
        print (f"Error saving metadata to {metadata_path}: {str(e)}")
        return False 

def move_and_rename_output_file (original_file ,target_folder ,original_image_filename ):

    if not original_file or not os .path .exists (original_file ):
        return None 

    ext =os .path .splitext (original_file )[1 ]

    new_filename =os .path .splitext (original_image_filename )[0 ]+ext 
    new_filepath =os .path .join (target_folder ,new_filename )

    try :

        os .makedirs (os .path .dirname (new_filepath ),exist_ok =True )

        import shutil 
        shutil .copy2 (original_file ,new_filepath )
        print (f"Saved output to {new_filepath}")
        return new_filepath 
    except Exception as e :
        print (f"Error moving/renaming file to {new_filepath}: {str(e)}")
        return None 

def scan_lora_files ():

    try :
        safetensors_files =glob .glob (os .path .join (loras_folder ,"**/*.safetensors"),recursive =True )
        pt_files =glob .glob (os .path .join (loras_folder ,"**/*.pt"),recursive =True )
        all_lora_files =safetensors_files +pt_files 

        lora_options =[("None","none")]
        for lora_file in all_lora_files :
            display_name =os .path .splitext (os .path .basename (lora_file ))[0 ]
            lora_options .append ((display_name ,lora_file ))

        return lora_options 
    except Exception as e :
        print (f"Error scanning LoRA files: {str(e)}")
        return [("None","none")]

def get_lora_path_from_name (display_name ):

    if display_name =="None":
        return "none"

    lora_options =scan_lora_files ()
    for name ,path in lora_options :
        if name ==display_name :
            return path 

    print (f"Warning: LoRA '{display_name}' not found in options, using None")
    return "none"

def refresh_loras ():

    lora_options =scan_lora_files ()
    return gr .update (choices =[name for name ,_ in lora_options ],value ="None")

def safe_unload_lora (model ,device =None ):

    if device is not None :
        model .to (device )

    is_dynamic_swap ='DynamicSwap'in model .__class__ .__name__ 

    try :
        # First try the new unload_all_loras function
        if hasattr(model, 'peft_config') and model.peft_config:
            print("Using new unload_all_loras method")
            unload_all_loras(model)
            return True

        # Handle DynamicSwap models
        elif is_dynamic_swap :
            print ("DynamicSwap model detected, attempting to reset internal model state")

            internal_model =model 
            if hasattr (model ,"model"):
                internal_model =model .model 

            print (f"Attempting unload on model type: {type(internal_model).__name__}")

            # Try new method on internal model
            if hasattr(internal_model, 'peft_config') and internal_model.peft_config:
                print("Using new unload_all_loras method on internal model")
                unload_all_loras(internal_model)
                
                # Clear wrapper state
                if hasattr (model ,"peft_config"):model .peft_config ={}
                if hasattr (model ,"active_adapters"):model .active_adapters =[]
                print ("Cleared LoRA state on DynamicSwap wrapper.")
                return True

            # Fallback to old method for DynamicSwap
            unloaded_internal =False 
            if hasattr (internal_model ,"unload_lora_weights"):
                print ("Unloading LoRA from internal model using unload_lora_weights")
                internal_model .unload_lora_weights ()
                unloaded_internal =True 
            if hasattr (internal_model ,"peft_config")and internal_model .peft_config :
                if hasattr (internal_model ,"disable_adapters"):
                    print ("Disabling adapters on internal model")
                    internal_model .disable_adapters ()
                if hasattr (internal_model ,"active_adapters"):
                     internal_model .active_adapters =[]
                internal_model .peft_config ={}
                print ("Cleared peft_config on internal model.")
                unloaded_internal =True 

            if unloaded_internal :
                if hasattr (model ,"peft_config"):model .peft_config ={}
                if hasattr (model ,"active_adapters"):model .active_adapters =[]
                print ("Cleared LoRA state on DynamicSwap wrapper.")
                return True 

            print ("Attempting direct LoRA module removal as fallback")
            return force_remove_lora_modules (model )

        # Fallback to old methods
        elif hasattr (model ,"unload_lora_weights"):
            print ("Unloading LoRA using unload_lora_weights method")
            model .unload_lora_weights ()

            if hasattr (model ,"disable_adapters"):
                model .disable_adapters ()
                print ("Additionally called disable_adapters.")
            if hasattr (model ,"peft_config"):
                model .peft_config ={}
                print ("Cleared peft_config.")
            return True 

        else :
            print ("No LoRA adapter found to unload")
            return True 
    except Exception as e :
        print (f"Error during LoRA unloading: {str(e)}")
        traceback .print_exc ()

        print ("Attempting direct LoRA module removal after error")
        return force_remove_lora_modules (model )

    return False 

def force_remove_lora_modules (model ):

    try :

        lora_removed =False 

        model_to_check =model .model if hasattr (model ,"model")else model 
        print (f"Force removing modules from: {type(model_to_check).__name__}")

        for name ,module in list (model_to_check .named_modules ()):

            is_lora_layer =hasattr (module ,'lora_A')or hasattr (module ,'lora_B')or 'lora'in name .lower ()
            is_peft_layer ='lora'in getattr (module ,'__class__',type (None )).__module__ .lower ()

            if is_lora_layer or is_peft_layer :
                print (f"Found potential LoRA module: {name}")
                lora_removed =True 

                parent_name ,_ ,attr_name =name .rpartition ('.')
                if parent_name :
                    try :
                        parent =model_to_check .get_submodule (parent_name )
                        if hasattr (parent ,attr_name ):

                            original_module =getattr (module ,'base_layer',getattr (module ,'model',getattr (module ,'base_model',None )))
                            if original_module is not None :
                                setattr (parent ,attr_name ,original_module )
                                print (f"Restored original module for {name}")
                            else :
                                print (f"Could not find original module for {name} to restore.")
                        else :
                             print (f"Parent module {parent_name} does not have attribute {attr_name}")
                    except Exception as e :
                        print (f"Error accessing parent module {parent_name}: {str(e)}")
                else :

                    if hasattr (model_to_check ,name ):
                         original_module =getattr (module ,'base_layer',getattr (module ,'model',getattr (module ,'base_model',None )))
                         if original_module is not None :
                            setattr (model_to_check ,name ,original_module )
                            print (f"Restored original top-level module for {name}")

        for m in [model ,model_to_check ]:
             if hasattr (m ,"peft_config"):
                 m .peft_config ={}
                 print (f"Cleared peft_config on {type(m).__name__}")
                 lora_removed =True 
             if hasattr (m ,"active_adapters"):
                 m .active_adapters =[]
                 print (f"Cleared active_adapters on {type(m).__name__}")
                 lora_removed =True 

        return lora_removed 
    except Exception as e :
        print (f"Error during force LoRA removal: {str(e)}")
        traceback .print_exc ()
        return False 

print_supported_image_formats ()

def get_temp_metadata_filepath_for_job(job_id: str, base_folder: str) -> str:
    return os.path.join(base_folder, f"{job_id}_temp_metadata.txt")

def delete_temporary_metadata_for_job(job_id: str, base_folder: str):
    temp_meta_path = get_temp_metadata_filepath_for_job(job_id, base_folder)
    try:
        if os.path.exists(temp_meta_path):
            os.remove(temp_meta_path)
            print(f"Successfully deleted temporary metadata: {temp_meta_path}")
    except Exception as e:
        print(f"Error deleting temporary metadata {temp_meta_path}: {str(e)}")

def get_preset_path (name :str )->str :

    safe_name ="".join (c for c in name if c .isalnum ()or c in (' ','_','-')).rstrip ()
    if not safe_name :
        safe_name ="Unnamed_Preset"
    return os .path .join (presets_folder ,f"{safe_name}.json")

def scan_presets ()->list [str ]:

    presets =["Default"]
    try :
        os .makedirs (presets_folder ,exist_ok =True )
        for filename in os .listdir (presets_folder ):
            if filename .endswith (".json")and filename !="Default.json":
                preset_name =os .path .splitext (filename )[0 ]
                if preset_name !="_lastused":
                    presets .append (preset_name )
    except Exception as e :
        print (f"Error scanning presets folder: {e}")
    return sorted (list (set (presets )))

def save_last_used_preset_name (name :str ):

    try :
        with open (last_used_preset_file ,'w',encoding ='utf-8')as f :
            f .write (name )
    except Exception as e :
        print (f"Error saving last used preset name: {e}")

def load_last_used_preset_name ()->Optional [str ]:

    if os .path .exists (last_used_preset_file ):
        try :
            with open (last_used_preset_file ,'r',encoding ='utf-8')as f :
                return f .read ().strip ()
        except Exception as e :
            print (f"Error loading last used preset name: {e}")
    return None 

def create_default_preset_if_needed (components_values :dict ):

    default_path =get_preset_path ("Default")
    if not os .path .exists (default_path ):
        print ("Default preset not found, creating...")
        try :

            valid_values ={k :v for k ,v in components_values .items ()if v is not None }
            if valid_values :
                with open (default_path ,'w',encoding ='utf-8')as f :
                    json .dump (valid_values ,f ,indent =4 )
                print ("Created Default.json")
            else :
                print ("Warning: Could not create Default.json - no valid component values provided.")
        except Exception as e :
            print (f"Error creating default preset: {e}")

def load_preset_data (name :str )->Optional [dict ]:

    if not name :
        return None 
    preset_path =get_preset_path (name )
    if os .path .exists (preset_path ):
        try :
            with open (preset_path ,'r',encoding ='utf-8')as f :
                return json .load (f )
        except Exception as e :
            print (f"Error loading preset '{name}': {e}")
            return None 
    else :
        print (f"Preset file not found: {preset_path}")
        return None 

def parse_metadata_text_content (text_content :str )->Dict [str ,str ]:

    parsed ={}
    for line in text_content .splitlines ():
        line =line .strip ()
        if ": "in line :
            key ,value =line .split (": ",1 )
            parsed [key .strip ()]=value .strip ()
        elif ":"in line :
            key ,value =line .split (":",1 )
            parsed [key .strip ()]=value .strip ()
    return parsed 

def convert_metadata_to_preset_dict (metadata :Dict [str ,str ])->Dict [str ,Any ]:

    preset_dict ={}

    if "Model"in metadata :
        preset_dict ["model_selector"]=metadata ["Model"]
    if "Prompt"in metadata :
        preset_dict ["prompt"]=metadata ["Prompt"]
    if "Negative Prompt"in metadata :
        preset_dict ["n_prompt"]=metadata ["Negative Prompt"]
    if "Seed"in metadata :
        try :preset_dict ["seed"]=int (metadata ["Seed"])
        except ValueError :print (f"Warning: Could not parse Seed '{metadata['Seed']}' as int in metadata.")

    if "TeaCache"in metadata :
        tc_val =metadata ["TeaCache"]
        if "Disabled"in tc_val :
            preset_dict ["teacache_threshold"]=0.0 
        elif "Enabled (Threshold: "in tc_val :
            try :
                threshold_str =tc_val .split ("Enabled (Threshold: ")[1 ].split (")")[0 ]
                preset_dict ["teacache_threshold"]=float (threshold_str )
            except Exception as e :print (f"Warning: Could not parse TeaCache threshold from '{tc_val}': {e}")
        else :
            print (f"Warning: Unrecognized TeaCache format: '{tc_val}'")

    if "Video Length (seconds)"in metadata :
        try :preset_dict ["total_second_length"]=float (metadata ["Video Length (seconds)"])
        except ValueError :print (f"Warning: Could not parse Video Length '{metadata['Video Length (seconds)']}' as float in metadata.")
    if "FPS"in metadata :
        try :preset_dict ["fps"]=int (metadata ["FPS"])
        except ValueError :print (f"Warning: Could not parse FPS '{metadata['FPS']}' as int in metadata.")
    if "Latent Window Size"in metadata :
        try :preset_dict ["latent_window_size"]=int (metadata ["Latent Window Size"])
        except ValueError :print (f"Warning: Could not parse Latent Window Size '{metadata['Latent Window Size']}' as int in metadata.")
    if "Steps"in metadata :
        try :preset_dict ["steps"]=int (metadata ["Steps"])
        except ValueError :print (f"Warning: Could not parse Steps '{metadata['Steps']}' as int in metadata.")
    if "CFG Scale"in metadata :
        try :preset_dict ["cfg"]=float (metadata ["CFG Scale"])
        except ValueError :print (f"Warning: Could not parse CFG Scale '{metadata['CFG Scale']}' as float in metadata.")
    if "Distilled CFG Scale"in metadata :
        try :preset_dict ["gs"]=float (metadata ["Distilled CFG Scale"])
        except ValueError :print (f"Warning: Could not parse Distilled CFG Scale '{metadata['Distilled CFG Scale']}' as float in metadata.")
    if "Guidance Rescale"in metadata :
        try :preset_dict ["rs"]=float (metadata ["Guidance Rescale"])
        except ValueError :print (f"Warning: Could not parse Guidance Rescale '{metadata['Guidance Rescale']}' as float in metadata.")
    if "Resolution"in metadata :
        preset_dict ["resolution"]=metadata ["Resolution"]

    if "Final Width"in metadata :
        try :preset_dict ["target_width"]=int (metadata ["Final Width"])
        except ValueError :print (f"Warning: Could not parse Final Width '{metadata['Final Width']}' as int in metadata.")
    if "Final Height"in metadata :
        try :preset_dict ["target_height"]=int (metadata ["Final Height"])
        except ValueError :print (f"Warning: Could not parse Final Height '{metadata['Final Height']}' as int in metadata.")

    # Multi-LoRA: parse all 4 LoRA names/scales from metadata
    for i in range(N_LORA):
        if i == 0:
            lora_name = metadata.get("LoRA", "None")
            lora_scale = metadata.get("LoRA Scale", 1.0)
            preset_dict["selected_lora_1"] = lora_name
            try:
                preset_dict["lora_scale_1"] = float(lora_scale)
            except Exception:
                preset_dict["lora_scale_1"] = 1.0
        else:
            lora_name = metadata.get(f"LoRA {i+1} Name", "None")
            lora_scale = metadata.get(f"LoRA {i+1} Scale", 1.0)
            preset_dict[f"selected_lora_{i+1}"] = lora_name
            try:
                preset_dict[f"lora_scale_{i+1}"] = float(lora_scale)
            except Exception:
                preset_dict[f"lora_scale_{i+1}"] = 1.0

    if "LoRA" not in metadata and "selected_lora_1" not in preset_dict:
        preset_dict["selected_lora_1"] = "None"
        preset_dict["lora_scale_1"] = 1.0
    for i in range(2, N_LORA+1):
        if f"selected_lora_{i}" not in preset_dict:
            preset_dict[f"selected_lora_{i}"] = "None"
            preset_dict[f"lora_scale_{i}"] = 1.0

    if "LoRA Scale" in metadata and "lora_scale_1" not in preset_dict:
        try:
            preset_dict["lora_scale_1"] = float(metadata["LoRA Scale"])
        except Exception:
            preset_dict["lora_scale_1"] = 1.0

    if "Timestamped Prompts Used"in metadata :
        if metadata ["Timestamped Prompts Used"].lower ()=="true":
            preset_dict ["use_multiline_prompts"]=False 
        elif metadata ["Timestamped Prompts Used"].lower ()=="false":
            preset_dict ["use_multiline_prompts"]=True 

    bool_map ={
    "Save Processing Metadata":"save_metadata",
    "Save Individual Frames":"save_individual_frames",
    "Save Intermediate Frames":"save_intermediate_frames",
    "Save Last Frame Of Generations (MP4 Only)":"save_last_frame",
    "Skip Existing Files":"batch_skip_existing",
    "Random Seed":"use_random_seed",
    "Enable RIFE (2x/4x FPS)":"rife_enabled",
    "Export as GIF":"export_gif",
    "Export as APNG":"export_apng",
    "Export as WebP":"export_webp",
    }
    for meta_key ,preset_key in bool_map .items ():
        if meta_key in metadata :
            preset_dict [preset_key ]=metadata [meta_key ].lower ()=="true"

    if "Number of Generations"in metadata :
        try :preset_dict ["num_generations"]=int (metadata ["Number of Generations"])
        except ValueError :print (f"Warning: Could not parse Number of Generations '{metadata['Number of Generations']}' as int.")

    if "Video Quality"in metadata :
        preset_dict ["video_quality"]=metadata ["Video Quality"]

    if "RIFE FPS Multiplier"in metadata :
        preset_dict ["rife_multiplier"]=metadata ["RIFE FPS Multiplier"]

    if "GPU Inference Preserved Memory (GB)"in metadata :
        try :preset_dict ["gpu_memory_preservation"]=float (metadata ["GPU Inference Preserved Memory (GB)"])
        except ValueError :print (f"Warning: Could not parse GPU Memory '{metadata['GPU Inference Preserved Memory (GB)']}' as float.")

    if "Save Temporary Metadata Setting" in metadata:
        preset_dict["save_temp_metadata"] = metadata["Save Temporary Metadata Setting"].lower() == "true"

    return preset_dict 

def load_settings_from_metadata_file (metadata_file_obj ,progress =gr .Progress ()):

    if metadata_file_obj is None :
        no_file_updates =[gr .update ()for _ in preset_components_list ]+[gr .update (value ="<p style='color:red;'>No metadata file uploaded.</p>"),gr .update (),gr .update ()]
        return no_file_updates 

    try :

        metadata_content =metadata_file_obj .decode ('utf-8')
    except Exception as e :
        read_error_updates =[gr .update ()for _ in preset_components_list ]+[gr .update (value =f"<p style='color:red;'>Error reading file: {e}</p>"),gr .update (),gr .update ()]
        return read_error_updates 

    parsed_metadata =parse_metadata_text_content (metadata_content )
    if not parsed_metadata :
        parse_fail_updates =[gr .update ()for _ in preset_components_list ]+[gr .update (value ="<p style='color:red;'>Could not parse metadata from file or file is empty.</p>"),gr .update (),gr .update ()]
        return parse_fail_updates 

    preset_data_from_metadata =convert_metadata_to_preset_dict (parsed_metadata )
    if not preset_data_from_metadata :
        convert_fail_updates =[gr .update ()for _ in preset_components_list ]+[gr .update (value ="<p style='color:red;'>Failed to convert parsed metadata to preset format (no settings found).</p>"),gr .update (),gr .update ()]
        return convert_fail_updates 

    temp_preset_name ="_temp_metadata_upload"
    temp_preset_path =get_preset_path (temp_preset_name )

    try :
        with open (temp_preset_path ,'w',encoding ='utf-8')as f :
            json .dump (preset_data_from_metadata ,f ,indent =4 )
        print (f"Temporarily saved metadata to preset file: {temp_preset_path}")
    except Exception as e :
        save_error_msg =f"Error saving temporary metadata preset: {e}"
        print (save_error_msg )
        save_error_updates =[gr .update ()for _ in preset_components_list ]+[gr .update (value =f"<p style='color:red;'>{save_error_msg}</p>"),gr .update (),gr .update ()]
        return save_error_updates 

    load_results =load_preset_action (temp_preset_name ,progress =progress )

    try :
        if os .path .exists (temp_preset_path ):
            os .remove (temp_preset_path )
            print (f"Removed temporary preset file: {temp_preset_path}")
    except Exception as e :
        print (f"Warning: Could not remove temporary preset file {temp_preset_path}: {e}")

    component_updates =load_results [:-3 ]
    original_preset_status_gr_obj =load_results [-3 ]
    model_status_gr_obj =load_results [-2 ]
    iter_info_gr_obj =load_results [-1 ]

    original_preset_status_val =getattr (original_preset_status_gr_obj ,'value','')

    final_status_message =f"<p style='color:green;'>Settings successfully loaded from metadata file.</p>"
    if "Error"in original_preset_status_val or "Failed"in original_preset_status_val or "Warning"in original_preset_status_val :
        final_status_message =f"<p style='color:orange;'>Settings loaded from metadata, but with issues from preset system: {original_preset_status_val}</p>"
    elif not original_preset_status_val :
         final_status_message =f"<p style='color:green;'>Settings loaded from metadata file. (Preset system had no specific message)</p>"

    return component_updates +[gr .update (value =final_status_message ),model_status_gr_obj ,iter_info_gr_obj ]

def save_last_frame_to_file (frames ,output_dir ,filename_base ):

    try :

        os .makedirs (output_dir ,exist_ok =True )

        if frames is None :
            print ("Error: frames tensor is None")
            return None 

        if not (isinstance (frames ,torch .Tensor )and len (frames .shape )==5 ):
            print (f"Error: Invalid frames tensor shape: {frames.shape if hasattr(frames, 'shape') else 'unknown'}")
            return None 

        try :
            last_frame_tensor =frames [:,:,-1 :,:,:]
        except Exception as slicing_error :
            print (f"Error slicing last frame: {str(slicing_error)}")
            print (f"Frames tensor shape: {frames.shape}")
            return None 

        try :
            from diffusers_helper .utils import save_individual_frames 

            frame_paths =save_individual_frames (last_frame_tensor ,output_dir ,filename_base ,return_frame_paths =True )
        except ImportError :
            print ("Error importing save_individual_frames, trying to import at global scope")

            import sys 
            sys .path .append (os .path .join (os .path .dirname (os .path .abspath (__file__ )),'diffusers_helper'))
            from utils import save_individual_frames 
            frame_paths =save_individual_frames (last_frame_tensor ,output_dir ,filename_base ,return_frame_paths =True )

        if frame_paths and len (frame_paths )>0 :
            print (f"Saved last frame to {frame_paths[0]}")
            return frame_paths [0 ]
        else :
            print ("No frames were saved by save_individual_frames")
            return None 
    except Exception as e :
        print (f"Error saving last frame: {str(e)}")
        traceback .print_exc ()
        return None 

def parse_simple_timestamped_prompt (prompt_text :str ,total_duration :float ,latent_window_size :int ,fps :int )->Optional [list [tuple [float ,str ]]]:

    lines =prompt_text .strip ().split ('\n')
    sections =[]
    has_timestamps =False 
    default_prompt =None 

    pattern =r'^\s*\[(\d+(?:\.\d+)?(?:s)?)\]\s*(.*)'

    for i ,line in enumerate (lines ):
        line =line .strip ()
        if not line :
            continue 

        match =re .match (pattern ,line )
        if match :
            has_timestamps =True 
            try :
                time_str =match .group (1 )

                if time_str .endswith ('s'):
                    time_str =time_str [:-1 ]
                time_sec =float (time_str )
                text =match .group (2 ).strip ()
                if text :
                    sections .append ({"original_time":time_sec ,"prompt":text })
            except ValueError :
                print (f"Warning: Invalid time format in line: {line}")
                return None 
        elif i ==0 and not has_timestamps :

             default_prompt =line 
        elif has_timestamps :

             print (f"Warning: Ignoring line without timestamp after timestamped lines detected: {line}")

    if not has_timestamps :

        return None 

    if not sections and default_prompt :

         return None 
    elif not sections :

         return None 

    sections .sort (key =lambda x :x ["original_time"])

    if not any (s ['original_time']==0.0 for s in sections ):
        first_prompt =sections [0 ]['prompt']if sections else "default prompt"
        sections .insert (0 ,{"original_time":0.0 ,"prompt":first_prompt })
        sections .sort (key =lambda x :x ["original_time"])

    final_sections =[]
    for i in range (len (sections )):
        start_time =sections [i ]["original_time"]
        prompt_text =sections [i ]["prompt"]
        final_sections .append ((start_time ,prompt_text ))

    print (f"Parsed timestamped prompts (in original order): {final_sections}")
    return final_sections 

def update_iteration_info (vid_len_s ,fps_val ,win_size ):

    try :

        vid_len_s =float (vid_len_s )if vid_len_s is not None else 0.0 
        fps_val =int (fps_val )if fps_val is not None else 0 
        win_size =int (win_size )if win_size is not None else 0 
        print (f"DEBUG update_iteration_info - Converted inputs: vid_len_s={vid_len_s}, fps_val={fps_val}, win_size={win_size}")
    except (ValueError ,TypeError ,AttributeError )as e :
        print (f"Error converting types in update_iteration_info: {e}")
        print (f"Received types: vid_len_s={type(vid_len_s)}, fps_val={type(fps_val)}, win_size={type(win_size)}")

        return "Error: Invalid input types for calculation."

    if fps_val <=0 or win_size <=0 :
        return "Invalid FPS or Latent Window Size."

    try :

        total_frames_needed =vid_len_s *fps_val 
        frames_per_section_calc =win_size *4 

        total_latent_sections =0 
        if frames_per_section_calc >0 :

            frames_to_generate =total_frames_needed -1 
            frames_per_f1_loop =(win_size *4 -3 )
            if frames_per_f1_loop >0 :
                total_f1_loops =math .ceil (frames_to_generate /frames_per_f1_loop )if frames_to_generate >0 else 0 
                total_latent_sections =max (total_f1_loops ,1 )
            else :
                return "Invalid parameters leading to zero frames per section."

        else :
             return "Invalid parameters leading to zero frames per section."

        section_duration_seconds =frames_per_section_calc /fps_val 

        frames_in_one_section =win_size *4 -3 

        timing_description =""
        if abs (section_duration_seconds -1.0 )<0.01 :
            timing_description ="**precisely 1.0 second**"
        else :
            timing_description =f"~**{section_duration_seconds:.2f} seconds**"

        global active_model_name 
        if active_model_name ==MODEL_DISPLAY_NAME_F1 :
             info_text =(
             f"**Generation Info (FramePack F1):** Approx. **{total_latent_sections}** loop(s) will run.\n"
             f"Each loop adds ~**{(frames_in_one_section / fps_val):.2f} seconds** of video time.\n"
             f"(One loop processes {frames_in_one_section} frames internally at {fps_val} FPS).\n"
             f"*Timestamp prompts are based on the *final* video duration.*"
             )
        else :
             info_text =(
             f"**Generation Info (Original):** Approx. **{total_latent_sections}** section(s) will be generated.\n"
             f"Each section represents {timing_description} of the final video time.\n"
             f"(One section processes {frames_in_one_section} frames internally with overlap at {fps_val} FPS).\n"
             f"*Use the **{section_duration_seconds:.2f}s per section** estimate for '[seconds] prompt' timings.*"
             )

        if active_model_name ==MODEL_DISPLAY_NAME_ORIGINAL :
            ideal_lws_float =fps_val /4.0 
            ideal_lws_int =round (ideal_lws_float )
            ideal_lws_clamped =max (1 ,min (ideal_lws_int ,33 ))

            if win_size !=ideal_lws_clamped :
                ideal_duration =(ideal_lws_clamped *4 )/fps_val 
                if abs (ideal_duration -1.0 )<0.01 :
                     info_text +=f"\n\n*Tip: Set Latent Window Size to **{ideal_lws_clamped}** for (near) exact 1-second sections at {fps_val} FPS.*"

        return info_text 
    except Exception as e :
        print (f"Error calculating iteration info: {e}")
        traceback .print_exc ()
        return "Error calculating info."

@torch .no_grad ()
def worker (input_image ,end_image ,prompt ,n_prompt ,seed ,use_random_seed ,total_second_length ,latent_window_size ,steps ,cfg ,gs ,rs ,gpu_memory_preservation ,teacache_threshold ,video_quality ='high',export_gif =False ,export_apng =False ,export_webp =False ,num_generations =1 ,resolution ="640",fps =30 ,

selected_lora_dropdown_values=None,
lora_scales=None,

save_individual_frames_flag =False ,save_intermediate_frames_flag =False ,save_last_frame_flag =False ,use_multiline_prompts_flag =False ,rife_enabled =False ,rife_multiplier ="2x FPS",

active_model :str =MODEL_DISPLAY_NAME_ORIGINAL ,
target_width_from_ui =640 ,target_height_from_ui =640,
save_temp_metadata_ui_value: bool = True # New parameter
):

    global transformer ,text_encoder ,text_encoder_2 ,image_encoder ,vae 
    global individual_frames_folder ,intermediate_individual_frames_folder ,last_frames_folder ,intermediate_last_frames_folder 
    global currently_loaded_lora_info
    global outputs_folder # Ensure outputs_folder is accessible

    if selected_lora_dropdown_values is None:
        selected_lora_dropdown_values = ["None"] * N_LORA
    if lora_scales is None:
        lora_scales = [1.0] * N_LORA

    # Apply LoRA scaling directly to loaded adapters
    active_loras = []
    for i in range(N_LORA):
        adapter_name = currently_loaded_lora_info[i]["adapter_name"]
        scale = lora_scales[i]
        if adapter_name is not None:
            active_loras.append((adapter_name, scale))

    if active_loras:
        print(f"Worker: Applying LoRA scaling for {len(active_loras)} adapters...")
        
        # Apply scaling to each unique adapter
        applied_adapters = set()
        for adapter_name, scale in active_loras:
            if adapter_name not in applied_adapters:
                try:
                    apply_lora_scaling(transformer, adapter_name, scale)
                    applied_adapters.add(adapter_name)
                    print(f"Worker: Applied scaling {scale} to adapter '{adapter_name}'")
                except Exception as e:
                    print(f"Worker ERROR applying scaling to adapter '{adapter_name}': {e}")
                    traceback.print_exc()
        
        # Ensure all adapters are enabled
        if hasattr(transformer, 'enable_adapters'):
            try:
                transformer.enable_adapters()
                print("Worker: Enabled all LoRA adapters")
            except Exception as e:
                print(f"Worker: Error enabling adapters: {e}")
    else:
        # No LoRAs to apply, disable adapters if possible
        if hasattr(transformer, 'disable_adapters'):
            try:
                transformer.disable_adapters()
                print("Worker: No active LoRAs - disabled all adapters")
            except Exception as e:
                print(f"Worker: Error disabling adapters: {e}")

    total_latent_sections =0 
    frames_per_section_calc =latent_window_size *4 
    if frames_per_section_calc <=0 :
        raise ValueError ("Invalid Latent Window Size or FPS leading to zero frames per section")

    total_frames_needed =total_second_length *fps 

    if active_model ==MODEL_DISPLAY_NAME_F1 :

        frames_to_generate =total_frames_needed -1 
        frames_per_f1_loop =latent_window_size *4 -3 
        if frames_per_f1_loop <=0 :raise ValueError ("Invalid LWS for F1 model")
        total_latent_sections =math .ceil (frames_to_generate /frames_per_f1_loop )if frames_to_generate >0 else 0 
        total_latent_sections =max (total_latent_sections ,1 )
        print (f"F1 Model: Calculated {total_latent_sections} generation loops needed.")
    else :
        total_latent_sections =int (max (round (total_frames_needed /frames_per_section_calc ),1 ))
        print (f"Original Model: Calculated {total_latent_sections} sections needed.")

    parsed_prompts =None 
    encoded_prompts ={}
    using_timestamped_prompts =False 
    if not use_multiline_prompts_flag :
        parsed_prompts =parse_simple_timestamped_prompt (prompt ,total_second_length ,latent_window_size ,fps )
        if parsed_prompts :
            using_timestamped_prompts =True 
            print ("Using timestamped prompts.")
        else :
            print ("Timestamped prompt format not detected or invalid, using the entire prompt as one.")
    else :
        print ("Multi-line prompts enabled, skipping timestamp parsing.")

    current_seed =seed 
    all_outputs ={}
    last_used_seed =seed 

    start_time =time .time ()
    generation_times =[]

    estimated_vae_time_per_frame =0.05 
    estimated_save_time =2.0 

    vae_time_history =[]
    save_time_history =[]

    for gen_idx in range (num_generations ):

        gen_start_time =time .time ()

        if stream .input_queue .top ()=='end':
            stream .output_queue .push (('end',None ))
            print ("Worker detected end signal at start of generation")
            return 

        if use_random_seed :
            current_seed =random .randint (1 ,2147483647 )
        elif gen_idx >0 :
            current_seed +=1 

        last_used_seed =current_seed 
        stream .output_queue .push (('seed_update',current_seed ))

        job_id =generate_new_timestamp () # This will be our unique ID for temp metadata
        stream .output_queue .push (('progress',(None ,'',make_progress_bar_html (0 ,f'Starting generation {gen_idx+1}/{num_generations} with seed {current_seed} using {active_model}...'))))
        
        print(f"[Worker Debug] job_id: {job_id}, save_temp_metadata_ui_value: {save_temp_metadata_ui_value}")

        if save_temp_metadata_ui_value:
            print(f"[Worker Debug] Entered block to save temporary metadata for job_id: {job_id}")
            # Prepare metadata dictionary for temporary saving
            # Note: using_timestamped_prompts is determined later. We can save without it for now,
            # or update the temp file later if strictly needed (more complex).
            # For simplicity, save what's known now.
            current_metadata_for_temp = {
                "Model": active_model,
                "Prompt": prompt, # Use the prompt passed to worker
                "Negative Prompt": n_prompt,
                "Seed": current_seed,
                "TeaCache": f"Enabled (Threshold: {teacache_threshold})" if teacache_threshold > 0.0 else "Disabled",
                "Video Length (seconds)": total_second_length,
                "FPS": fps,
                "Latent Window Size": latent_window_size,
                "Steps": steps,
                "CFG Scale": cfg,
                "Distilled CFG Scale": gs,
                "Guidance Rescale": rs,
                "Resolution": resolution,
                "Final Width": target_width_from_ui, # Use actual processing width
                "Final Height": target_height_from_ui, # Use actual processing height
                "Start Frame Provided": input_image is not None, # Check if input_image was provided
                "End Frame Provided": end_image is not None,
                "GPU Inference Preserved Memory (GB)": gpu_memory_preservation,
                "Video Quality": video_quality,
                "RIFE FPS Multiplier": rife_multiplier,
                "Save Temporary Metadata Setting": save_temp_metadata_ui_value,
                # Add other relevant settings that are known at this point
                "Number of Generations Configured": num_generations,
                "Use Random Seed Setting": use_random_seed,
                "Use Multiline Prompts Setting (Worker)": use_multiline_prompts_flag,
            }
            # Add LoRA info if any are selected via dropdowns
            if selected_lora_dropdown_values and lora_scales:
                for i in range(N_LORA):
                    lora_name = selected_lora_dropdown_values[i]
                    lora_scale_val = lora_scales[i]
                    if lora_name and lora_name != "None":
                        if i == 0: # First LoRA
                            current_metadata_for_temp["LoRA"] = lora_name
                            current_metadata_for_temp["LoRA Scale"] = lora_scale_val
                        else: # Subsequent LoRAs
                            current_metadata_for_temp[f"LoRA {i+1} Name"] = lora_name
                            current_metadata_for_temp[f"LoRA {i+1} Scale"] = lora_scale_val
            
            base_folder_for_job_temp_metadata = outputs_folder # Define the base folder for this job's temp metadata
            save_processing_metadata(
                output_path_for_final_file=None, 
                metadata_dict=current_metadata_for_temp.copy(), # Pass a copy
                is_temporary=True,
                job_id_for_temp=job_id,
                base_folder_for_temp_metadata=base_folder_for_job_temp_metadata
            )
            # print(f"Saved temporary metadata for job_id: {job_id}") # Already printed by save_processing_metadata if successful

        try :

            if not high_vram :
                unload_complete_models (
                text_encoder ,text_encoder_2 ,image_encoder ,vae ,transformer 
                )

            stream .output_queue .push (('progress',(None ,'',make_progress_bar_html (0 ,'Text encoding ...'))))

            if not high_vram :

                fake_diffusers_current_device (text_encoder ,gpu )
                load_model_as_complete (text_encoder_2 ,target_device =gpu )

            if using_timestamped_prompts :
                unique_prompts =set (p [1 ]for p in parsed_prompts )
                stream .output_queue .push (('progress',(None ,'',make_progress_bar_html (0 ,f'Encoding {len(unique_prompts)} unique timestamped prompts...'))))
                for p_text in unique_prompts :
                    if p_text not in encoded_prompts :
                         llama_vec_p ,clip_l_pooler_p =encode_prompt_conds (p_text ,text_encoder ,text_encoder_2 ,tokenizer ,tokenizer_2 )
                         llama_vec_p ,llama_attention_mask_p =crop_or_pad_yield_mask (llama_vec_p ,length =512 )
                         encoded_prompts [p_text ]=(llama_vec_p ,llama_attention_mask_p ,clip_l_pooler_p )
                print (f"Pre-encoded {len(encoded_prompts)} unique prompts.")
                if not parsed_prompts :
                    raise ValueError ("Timestamped prompts were detected but parsing resulted in an empty list.")

                initial_prompt_text ="default prompt"
                for t ,p_txt in parsed_prompts :
                     if t ==0.0 :
                         initial_prompt_text =p_txt 
                         break 

                     elif not initial_prompt_text or t <parsed_prompts [0 ][0 ]:
                         initial_prompt_text =p_txt 

                if initial_prompt_text not in encoded_prompts :

                     print (f"Warning: Initial prompt text '{initial_prompt_text}' not found in encoded prompts. Using first available.")
                     initial_prompt_text =list (encoded_prompts .keys ())[0 ]

                llama_vec ,llama_attention_mask ,clip_l_pooler =encoded_prompts [initial_prompt_text ]

            else :
                stream .output_queue .push (('progress',(None ,'',make_progress_bar_html (0 ,'Encoding single prompt...'))))
                llama_vec ,clip_l_pooler =encode_prompt_conds (prompt ,text_encoder ,text_encoder_2 ,tokenizer ,tokenizer_2 )
                llama_vec ,llama_attention_mask =crop_or_pad_yield_mask (llama_vec ,length =512 )
                encoded_prompts [prompt ]=(llama_vec ,llama_attention_mask ,clip_l_pooler )

            if cfg >1.0 :
                llama_vec_n ,clip_l_pooler_n =encode_prompt_conds (n_prompt ,text_encoder ,text_encoder_2 ,tokenizer ,tokenizer_2 )
                llama_vec_n ,llama_attention_mask_n =crop_or_pad_yield_mask (llama_vec_n ,length =512 )
            else :

                first_prompt_key =list (encoded_prompts .keys ())[0 ]
                ref_llama_vec =encoded_prompts [first_prompt_key ][0 ]
                ref_clip_l =encoded_prompts [first_prompt_key ][2 ]
                llama_vec_n ,clip_l_pooler_n =torch .zeros_like (ref_llama_vec ),torch .zeros_like (ref_clip_l )
                ref_llama_mask =encoded_prompts [first_prompt_key ][1 ]
                llama_attention_mask_n =torch .zeros_like (ref_llama_mask )

            target_dtype =transformer .dtype 
            for p_text in encoded_prompts :
                 l_vec ,l_mask ,c_pool =encoded_prompts [p_text ]
                 encoded_prompts [p_text ]=(l_vec .to (target_dtype ),l_mask ,c_pool .to (target_dtype ))

            llama_vec_n =llama_vec_n .to (target_dtype )
            clip_l_pooler_n =clip_l_pooler_n .to (target_dtype )

            height =target_height_from_ui 
            width =target_width_from_ui 
            print (f"UI Target Dimensions: {target_width_from_ui}x{target_height_from_ui}. Using these directly for processing. Resolution Guide: '{resolution}'.")

            stream .output_queue .push (('progress',(None ,'',make_progress_bar_html (0 ,'Processing start frame ...'))))
            input_image_np =resize_and_center_crop (input_image ,target_width =width ,target_height =height )
            try :
                os .makedirs (used_images_folder ,exist_ok =True )
                Image .fromarray (input_image_np ).save (os .path .join (used_images_folder ,f'{job_id}_start.png'))
                print (f"Saved start image to {os.path.join(used_images_folder, f'{job_id}_start.png')}")
            except Exception as e :
                print (f"Error saving start image: {str(e)}")
            input_image_pt =torch .from_numpy (input_image_np ).float ()/127.5 -1 
            input_image_pt =input_image_pt .permute (2 ,0 ,1 )[None ,:,None ]

            has_end_image =end_image is not None 
            end_image_np =None 
            end_image_pt =None 
            end_latent =None 
            if has_end_image :
                stream .output_queue .push (('progress',(None ,'',make_progress_bar_html (0 ,'Processing end frame ...'))))
                end_image_np =resize_and_center_crop (end_image ,target_width =width ,target_height =height )
                try :
                    os .makedirs (used_images_folder ,exist_ok =True )
                    Image .fromarray (end_image_np ).save (os .path .join (used_images_folder ,f'{job_id}_end.png'))
                    print (f"Saved end image to {os.path.join(used_images_folder, f'{job_id}_end.png')}")
                except Exception as e :
                    print (f"Error saving end image: {str(e)}")
                end_image_pt =torch .from_numpy (end_image_np ).float ()/127.5 -1 
                end_image_pt =end_image_pt .permute (2 ,0 ,1 )[None ,:,None ]

            stream .output_queue .push (('progress',(None ,'',make_progress_bar_html (0 ,'VAE encoding ...'))))
            if not high_vram :
                load_model_as_complete (vae ,target_device =gpu )
            start_latent =vae_encode (input_image_pt ,vae )
            if has_end_image :
                end_latent =vae_encode (end_image_pt ,vae )

            if not high_vram :
                 unload_complete_models (vae )

            stream .output_queue .push (('progress',(None ,'',make_progress_bar_html (0 ,'CLIP Vision encoding ...'))))
            if not high_vram :
                load_model_as_complete (image_encoder ,target_device =gpu )

            image_encoder_output =hf_clip_vision_encode (input_image_np ,feature_extractor ,image_encoder )
            image_encoder_last_hidden_state =image_encoder_output .last_hidden_state 
            if has_end_image :
                end_image_encoder_output =hf_clip_vision_encode (end_image_np ,feature_extractor ,image_encoder )
                end_image_encoder_last_hidden_state =end_image_encoder_output .last_hidden_state 

                image_encoder_last_hidden_state =(image_encoder_last_hidden_state +end_image_encoder_last_hidden_state )/2.0 
                print ("Combined start and end frame CLIP vision embeddings.")
            image_encoder_last_hidden_state =image_encoder_last_hidden_state .to (target_dtype )

            if not high_vram :
                unload_complete_models (image_encoder )

            stream .output_queue .push (('progress',(None ,'',make_progress_bar_html (0 ,f'Start sampling ({active_model}) generation {gen_idx+1}/{num_generations}...'))))

            rnd =torch .Generator ("cpu").manual_seed (current_seed )
            num_frames =latent_window_size *4 -3 

            history_latents =None 
            history_pixels =None 
            total_generated_latent_frames =0 

            if active_model ==MODEL_DISPLAY_NAME_F1 :

                history_latents =start_latent .clone ().cpu ()
                total_generated_latent_frames =1 
                print (f"F1 Initial history latent shape: {history_latents.shape}")
            else :

                history_latents =torch .zeros (size =(1 ,16 ,1 +2 +16 ,height //8 ,width //8 ),dtype =torch .float32 ).cpu ()

                total_generated_latent_frames =0 
                print (f"Original Model Initial history latent shape: {history_latents.shape}")

            loop_iterator =None 
            if active_model ==MODEL_DISPLAY_NAME_F1 :
                loop_iterator =range (total_latent_sections )
            else :

                base_latent_paddings =reversed (range (total_latent_sections ))
                if total_latent_sections >4 :
                    latent_paddings =[3 ]+[2 ]*(total_latent_sections -3 )+[1 ,0 ]
                else :
                    latent_paddings =list (base_latent_paddings )
                loop_iterator =enumerate (latent_paddings )

            current_prompt_text_for_callback =prompt 

            for loop_info in loop_iterator :

                latent_padding =0 
                i =0 
                if active_model ==MODEL_DISPLAY_NAME_F1 :
                    i =loop_info 
                    is_last_section =(i ==total_latent_sections -1 )
                    is_first_section =(i ==0 )

                    print (f'F1 Loop {i+1}/{total_latent_sections}, is_last_section = {is_last_section}')
                else :
                    i ,latent_padding =loop_info 
                    is_last_section =latent_padding ==0 
                    is_first_section =(i ==0 )
                    latent_padding_size =latent_padding *latent_window_size 
                    print (f'Original Loop {i+1}/{total_latent_sections}, latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}, is_first_section = {is_first_section}')

                if stream .input_queue .top ()=='end':
                    stream .output_queue .push (('end',None ))
                    print (f"Worker detected end signal during {active_model} loop")
                    try :
                        if not high_vram :unload_complete_models ()
                    except Exception as cleanup_error :print (f"Error during cleanup: {str(cleanup_error)}")
                    return 

                first_prompt_key =list (encoded_prompts .keys ())[0 ]
                active_llama_vec ,active_llama_mask ,active_clip_pooler =encoded_prompts [first_prompt_key ]
                current_prompt_text_for_callback =first_prompt_key 

                if using_timestamped_prompts :

                    current_frames_generated =0 
                    if active_model ==MODEL_DISPLAY_NAME_F1 :

                        current_frames_generated =1 +i *(latent_window_size *4 -3 )
                    else :

                        section_duration_seconds =(latent_window_size *4 )/fps 

                        current_video_time =total_second_length -(i *section_duration_seconds )

                        if current_video_time <0 :current_video_time =0.0 
                        current_frames_generated =int (current_video_time *fps )

                    current_video_time_sec =current_frames_generated /fps 

                    print (f"\n===== PROMPT DEBUG INFO ({active_model}) =====")
                    print (f"Loop/Iter: {i} / {total_latent_sections -1}")
                    print (f"Current video time (estimated): {current_video_time_sec:.2f}s (Frame ~{current_frames_generated})")
                    print (f"Available prompts: {parsed_prompts}")

                    selected_prompt_text =parsed_prompts [0 ][1 ]
                    last_matching_time =parsed_prompts [0 ][0 ]
                    epsilon =1e-4 

                    print (f"Checking against prompts...")
                    for start_time_prompt ,p_text in parsed_prompts :
                        print (f"  - Checking time {start_time_prompt:.2f}s ('{p_text[:20]}...') vs current_video_time {current_video_time_sec:.2f}s")

                        if current_video_time_sec >=(start_time_prompt -epsilon ):
                             selected_prompt_text =p_text 
                             last_matching_time =start_time_prompt 
                             print (f"    - MATCH: Current time {current_video_time_sec:.2f}s >= {start_time_prompt}s. Tentative selection: '{selected_prompt_text[:20]}...'")
                        else :

                            print (f"    - NO MATCH: Current time {current_video_time_sec:.2f}s < {start_time_prompt}s. Stopping search.")
                            break 

                    print (f"Final selected prompt active at/before {current_video_time_sec:.2f}s is the one from {last_matching_time}s: '{selected_prompt_text}'")
                    print (f"===== END DEBUG INFO ({active_model}) =====\n")

                    active_llama_vec ,active_llama_mask ,active_clip_pooler =encoded_prompts [selected_prompt_text ]
                    current_prompt_text_for_callback =selected_prompt_text 
                    print (f'---> Generating section corresponding to video time >= {last_matching_time:.2f}s, Using prompt: "{selected_prompt_text[:50]}..."')

                else :
                     active_llama_vec ,active_llama_mask ,active_clip_pooler =encoded_prompts [prompt ]
                     current_prompt_text_for_callback =prompt 

                latent_indices =None 
                clean_latents =None 
                clean_latent_indices =None 
                clean_latents_2x =None 
                clean_latent_2x_indices =None 
                clean_latents_4x =None 
                clean_latent_4x_indices =None 

                if active_model ==MODEL_DISPLAY_NAME_F1 :

                    total_f1_indices =1 +16 +2 +1 +latent_window_size 
                    indices =torch .arange (0 ,total_f1_indices ).unsqueeze (0 )

                    f1_clean_start_idx ,f1_clean_4x_idx ,f1_clean_2x_idx ,f1_clean_1x_idx ,f1_latent_idx =indices .split ([1 ,16 ,2 ,1 ,latent_window_size ],dim =1 )

                    latent_indices =f1_latent_idx 
                    clean_latent_indices =torch .cat ([f1_clean_start_idx ,f1_clean_1x_idx ],dim =1 )
                    clean_latent_2x_indices =f1_clean_2x_idx 
                    clean_latent_4x_indices =f1_clean_4x_idx 

                    required_history_len =16 +2 +1 
                    if history_latents .shape [2 ]<required_history_len :

                        padding_needed =required_history_len -history_latents .shape [2 ]
                        padding_tensor =torch .zeros (
                        (history_latents .shape [0 ],history_latents .shape [1 ],padding_needed ,history_latents .shape [3 ],history_latents .shape [4 ]),
                        dtype =history_latents .dtype ,device =history_latents .device 
                        )
                        current_history_segment =torch .cat ([padding_tensor ,history_latents ],dim =2 ).to (gpu )
                        print (f"F1 Warning: Padded history by {padding_needed} frames.")
                    else :
                        current_history_segment =history_latents [:,:,-required_history_len :,:,:].to (gpu )

                    clean_latents_4x ,clean_latents_2x ,clean_latents_1x =current_history_segment .split ([16 ,2 ,1 ],dim =2 )

                    clean_latents =torch .cat ([start_latent .to (gpu ),clean_latents_1x ],dim =2 )

                else :

                    latent_padding_size =latent_padding *latent_window_size 
                    total_original_indices =sum ([1 ,latent_padding_size ,latent_window_size ,1 ,2 ,16 ])
                    indices =torch .arange (0 ,total_original_indices ).unsqueeze (0 )
                    clean_latent_indices_pre ,blank_indices ,latent_indices ,clean_latent_indices_post ,clean_latent_2x_indices ,clean_latent_4x_indices =indices .split ([1 ,latent_padding_size ,latent_window_size ,1 ,2 ,16 ],dim =1 )
                    clean_latent_indices =torch .cat ([clean_latent_indices_pre ,clean_latent_indices_post ],dim =1 )

                    clean_latents_pre =start_latent .to (gpu )

                    clean_latents_post_orig ,clean_latents_2x_orig ,clean_latents_4x_orig =history_latents [:,:,:1 +2 +16 ,:,:].to (gpu ).split ([1 ,2 ,16 ],dim =2 )

                    clean_latents_post =clean_latents_post_orig 

                    if has_end_image and is_first_section and end_latent is not None :
                        clean_latents_post =end_latent .to (gpu )
                        print ("Using end_latent for clean_latents_post in the first section.")

                    clean_latents =torch .cat ([clean_latents_pre ,clean_latents_post ],dim =2 )
                    clean_latents_2x =clean_latents_2x_orig 
                    clean_latents_4x =clean_latents_4x_orig 

                if not high_vram :
                    unload_complete_models ()
                    move_model_to_device_with_memory_preservation (transformer ,target_device =gpu ,preserved_memory_gb =gpu_memory_preservation )
                else :

                    if transformer .device !=gpu :
                         print ("Moving transformer to GPU (High VRAM mode)...")
                         transformer .to (gpu )
                
                # Make sure all LoRA modules are also moved to GPU
                if active_loras:
                    print("Verifying all LoRA modules are on GPU...")
                    lora_modules_moved = 0
                    for name, module in transformer.named_modules():
                        if 'lora_' in name.lower():
                            try:
                                if next(module.parameters(), torch.tensor(0)).device != gpu:
                                    module.to(gpu)
                                    lora_modules_moved += 1
                            except Exception as e:
                                print(f"Error moving LoRA module {name} to GPU: {e}")
                    if lora_modules_moved > 0:
                        print(f"Moved {lora_modules_moved} LoRA modules to GPU")

                use_teacache_effective =teacache_threshold >0.0 
                if use_teacache_effective :
                    print (f"TeaCache: Enabled (Threshold: {teacache_threshold})")
                    transformer .initialize_teacache (enable_teacache =True ,num_steps =steps ,rel_l1_thresh =teacache_threshold )
                else :
                    print ("TeaCache: Disabled")
                    transformer .initialize_teacache (enable_teacache =False )

                sampling_start_time =time .time ()

                def callback (d ):
                    preview =d ['denoised']
                    preview =vae_decode_fake (preview )
                    preview =(preview *255.0 ).detach ().cpu ().numpy ().clip (0 ,255 ).astype (np .uint8 )
                    preview =einops .rearrange (preview ,'b c t h w -> (b h) (t w) c')

                    if stream .input_queue .top ()=='end':
                        stream .output_queue .push (('end',None ))
                        print ("\n"+"="*50 )
                        print ("USER REQUESTED TO END GENERATION - STOPPING...")
                        print ("="*50 )
                        raise KeyboardInterrupt ('User ends the task.')

                    current_step =d ['i']+1 
                    percentage =int (100.0 *current_step /steps )
                    elapsed_time =time .time ()-sampling_start_time 
                    time_per_step =elapsed_time /current_step if current_step >0 else 0 
                    remaining_steps =steps -current_step 
                    eta_seconds =time_per_step *remaining_steps 

                    expected_latent_frames_in_section =latent_indices .shape [1 ]
                    expected_output_frames_from_section =num_frames 

                    if current_step ==steps :

                        post_processing_eta =expected_output_frames_from_section *estimated_vae_time_per_frame +estimated_save_time 
                        eta_seconds =post_processing_eta 
                    else :

                        post_processing_eta =expected_output_frames_from_section *estimated_vae_time_per_frame +estimated_save_time 
                        eta_seconds +=post_processing_eta 

                    eta_str =format_time_human_readable (eta_seconds )
                    total_elapsed =time .time ()-gen_start_time 
                    elapsed_str =format_time_human_readable (total_elapsed )
                    hint =f'Sampling {current_step}/{steps} (Gen {gen_idx+1}/{num_generations}, Seed {current_seed}, Loop {i+1}/{total_latent_sections})'

                    total_output_frames_so_far =0 
                    if active_model ==MODEL_DISPLAY_NAME_F1 :

                         total_output_frames_so_far =1 +i *num_frames 

                         total_output_frames_so_far +=int ((current_step /steps )*num_frames )
                    else :

                        if history_pixels is not None :
                             total_output_frames_so_far =history_pixels .shape [2 ]
                        else :
                             total_output_frames_so_far =i *num_frames 

                    desc =f'Total generated frames: ~{int(max(0, total_output_frames_so_far))}, Video length: {max(0, total_output_frames_so_far / fps) :.2f} seconds (FPS-{fps}).'
                    if using_timestamped_prompts :
                        desc +=f' Current Prompt: "{current_prompt_text_for_callback[:50]}..."'
                    time_info =f'Elapsed: {elapsed_str} | ETA: {eta_str}'
                    print (f"\rProgress: {percentage}% | {hint} | {time_info}     ",end ="")
                    stream .output_queue .push (('progress',(preview ,desc ,make_progress_bar_html (percentage ,f"{hint}<br/>{time_info}"))))
                    return 

                try :
                    # Critical check to ensure all LoRA parameters are on GPU before sampling
                    if active_loras:
                        print("Final device check for all LoRA tensors before sampling...")
                        devices_found = set()
                        lora_params_moved = 0
                        for name, param in transformer.named_parameters():
                            if 'lora_' in name.lower():
                                devices_found.add(str(param.device))
                                if param.device.type != 'cuda':
                                    print(f"Moving {name} from {param.device} to {gpu}")
                                    param.data = param.data.to(gpu)
                                    lora_params_moved += 1
                        
                        print(f"LoRA parameter device check complete. Found devices: {devices_found}")
                        if lora_params_moved > 0:
                            print(f"Moved {lora_params_moved} LoRA parameters to GPU before generation")

                    generated_latents =sample_hunyuan (
                    transformer =transformer ,
                    sampler ='unipc',
                    width =width ,
                    height =height ,
                    frames =num_frames ,
                    real_guidance_scale =cfg ,
                    distilled_guidance_scale =gs ,
                    guidance_rescale =rs ,
                    num_inference_steps =steps ,
                    generator =rnd ,
                    prompt_embeds =active_llama_vec .to (gpu ),
                    prompt_embeds_mask =active_llama_mask .to (gpu ),
                    prompt_poolers =active_clip_pooler .to (gpu ),
                    negative_prompt_embeds =llama_vec_n .to (gpu ),
                    negative_prompt_embeds_mask =llama_attention_mask_n .to (gpu ),
                    negative_prompt_poolers =clip_l_pooler_n .to (gpu ),
                    device =gpu ,
                    dtype =target_dtype ,
                    image_embeddings =image_encoder_last_hidden_state .to (gpu ),
                    latent_indices =latent_indices .to (gpu ),
                    clean_latents =clean_latents .to (gpu ),
                    clean_latent_indices =clean_latent_indices .to (gpu ),
                    clean_latents_2x =clean_latents_2x .to (gpu ),
                    clean_latent_2x_indices =clean_latent_2x_indices .to (gpu ),
                    clean_latents_4x =clean_latents_4x .to (gpu ),
                    clean_latent_4x_indices =clean_latent_4x_indices .to (gpu ),
                    callback =callback ,
                    )
                except ConnectionResetError as e :
                    print (f"Connection Reset Error caught during sampling: {str(e)}")
                    print ("Continuing with the process anyway...")
                    if stream .input_queue .top ()=='end':
                        stream .output_queue .push (('end',None ))
                        return 

                    empty_shape =(1 ,16 ,latent_window_size ,height //8 ,width //8 )
                    generated_latents =torch .zeros (empty_shape ,dtype =torch .float32 ).cpu ()
                    print ("Skipping to next generation due to connection error")
                    break 

                section_time =time .time ()-sampling_start_time 
                print (f"\nSection/Loop {i+1} completed sampling in {section_time:.2f} seconds")

                if active_model ==MODEL_DISPLAY_NAME_F1 :
                    history_latents =torch .cat ([history_latents ,generated_latents .cpu ()],dim =2 )
                    total_generated_latent_frames =history_latents .shape [2 ]
                else :

                    current_section_latents =generated_latents .cpu ()
                    if is_last_section :

                        current_section_latents =torch .cat ([start_latent .cpu (),current_section_latents ],dim =2 )

                    history_latents =torch .cat ([current_section_latents ,history_latents ],dim =2 )

                    total_generated_latent_frames +=int (current_section_latents .shape [2 ])

                print (f"Updated history_latents shape (CPU): {history_latents.shape}, Total latent frames: {total_generated_latent_frames}")

                print (f"VAE decoding started... {'Using standard decoding' if high_vram else 'Using memory optimization: VAE offloading'}")
                if not high_vram :
                    offload_model_from_device_for_memory_preservation (transformer ,target_device =gpu ,preserved_memory_gb =gpu_memory_preservation )
                    load_model_as_complete (vae ,target_device =gpu )

                vae_start_time =time .time ()

                latents_to_decode =None 
                if active_model ==MODEL_DISPLAY_NAME_F1 :

                    latents_to_decode =history_latents .to (gpu )
                else :

                    latents_to_decode =history_latents [:,:,:total_generated_latent_frames ,:,:].to (gpu )

                print (f"Latents to decode shape: {latents_to_decode.shape}")

                current_pixels =vae_decode (latents_to_decode ,vae ).cpu ()
                history_pixels =current_pixels 

                vae_time =time .time ()-vae_start_time 
                num_frames_decoded =history_pixels .shape [2 ]
                vae_time_per_frame =vae_time /num_frames_decoded if num_frames_decoded >0 else estimated_vae_time_per_frame 
                vae_time_history .append (vae_time_per_frame )
                if len (vae_time_history )>0 :
                    estimated_vae_time_per_frame =sum (vae_time_history )/len (vae_time_history )
                print (f"VAE decoding completed in {vae_time:.2f} seconds ({vae_time_per_frame:.3f} sec/frame)")
                print (f'Decoded pixel shape {history_pixels.shape}')

                if not high_vram :
                    unload_complete_models (vae )

                is_intermediate =not is_last_section 

                intermediate_suffix =f"_{history_pixels.shape[2]}frames"
                output_filename_base =f"{job_id}"
                if is_intermediate :
                     output_filename_base +=intermediate_suffix 

                output_filename =os .path .join (intermediate_videos_folder if is_intermediate else outputs_folder ,f'{output_filename_base}.mp4')
                webm_output_filename =os .path .join (intermediate_webm_videos_folder if is_intermediate else webm_videos_folder ,f'{output_filename_base}.webm')

                save_start_time =time .time ()
                try :

                    os .makedirs (os .path .dirname (output_filename ),exist_ok =True )
                    save_bcthw_as_mp4 (history_pixels ,output_filename ,fps =fps ,video_quality =video_quality )
                    print (f"Saved MP4 video to {output_filename}")

                    if save_last_frame_flag is True and output_filename and os .path .exists (output_filename ):
                        try :
                            print (f"Attempting to save last frame for {output_filename}")
                            last_frame_base_name_for_save =os .path .splitext (os .path .basename (output_filename ))[0 ]
                            frames_output_dir =os .path .join (
                            intermediate_last_frames_folder if is_intermediate else last_frames_folder ,
                            last_frame_base_name_for_save 
                            )
                            os .makedirs (frames_output_dir ,exist_ok =True )
                            save_last_frame_to_file (history_pixels ,frames_output_dir ,f"{last_frame_base_name_for_save}_lastframe")
                        except Exception as lf_err :
                            print (f"Error saving last frame for {output_filename}: {str(lf_err)}")
                            traceback .print_exc ()

                    if rife_enabled and output_filename and os .path .exists (output_filename ):
                        print (f"RIFE Enabled: Processing {output_filename}")
                        try :
                            cap =cv2 .VideoCapture (output_filename )
                            source_fps =cap .get (cv2 .CAP_PROP_FPS )
                            cap .release ()
                            print (f"Source MP4 FPS: {source_fps:.2f}")

                            if source_fps <=60 :
                                multiplier_val ="4"if rife_multiplier =="4x FPS"else "2"
                                print (f"Using RIFE multiplier: {multiplier_val}x")
                                rife_output_filename =os .path .splitext (output_filename )[0 ]+'_extra_FPS.mp4'
                                print (f"RIFE output filename: {rife_output_filename}")
                                rife_script_path =os .path .abspath (os .path .join (current_dir ,"Practical-RIFE","inference_video.py"))
                                rife_model_path =os .path .abspath (os .path .join (current_dir ,"Practical-RIFE","train_log"))

                                if not os .path .exists (rife_script_path ):print (f"ERROR: RIFE script not found at {rife_script_path}")
                                elif not os .path .exists (rife_model_path ):print (f"ERROR: RIFE model directory not found at {rife_model_path}")
                                else :
                                    cmd =(
                                    f'"{sys.executable}" "{rife_script_path}" '
                                    f'--model="{rife_model_path}" '
                                    f'--multi={multiplier_val} '
                                    f'--video="{os.path.abspath(output_filename)}" '
                                    f'--output="{os.path.abspath(rife_output_filename)}"'
                                    )
                                    print (f"Executing RIFE command: {cmd}")
                                    result =subprocess .run (cmd ,shell =True ,capture_output =True ,text =True ,env =os .environ )
                                    if result .returncode ==0 :
                                        if os .path .exists (rife_output_filename ):
                                            print (f"Successfully applied RIFE. Saved as: {rife_output_filename}")
                                            stream .output_queue .push (('rife_file',rife_output_filename ))
                                        else :
                                             print (f"RIFE command succeeded but output file missing: {rife_output_filename}")
                                             print (f"RIFE stdout:\n{result.stdout}")
                                             print (f"RIFE stderr:\n{result.stderr}")
                                    else :
                                        print (f"Error applying RIFE (return code {result.returncode}).")
                                        print (f"RIFE stdout:\n{result.stdout}")
                                        print (f"RIFE stderr:\n{result.stderr}")
                            else :
                                print (f"Skipping RIFE because source FPS ({source_fps:.2f}) is > 60.")
                        except Exception as rife_err :
                            print (f"Error during RIFE processing for {output_filename}: {str(rife_err)}")
                            traceback .print_exc ()

                    if video_quality =='web_compatible':
                        os .makedirs (os .path .dirname (webm_output_filename ),exist_ok =True )
                        save_bcthw_as_mp4 (history_pixels ,webm_output_filename ,fps =fps ,video_quality =video_quality ,format ='webm')
                        print (f"Saved WebM video to {webm_output_filename}")

                    if ((is_intermediate and save_intermediate_frames_flag )or 
                    (not is_intermediate and save_individual_frames_flag )):
                        frames_output_dir =os .path .join (
                        intermediate_individual_frames_folder if is_intermediate else individual_frames_folder ,
                        os .path .splitext (os .path .basename (output_filename ))[0 ]
                        )
                        from diffusers_helper .utils import save_individual_frames 
                        save_individual_frames (history_pixels ,frames_output_dir ,job_id )
                        print (f"Saved individual frames to {frames_output_dir}")

                except ConnectionResetError as e :
                    print (f"Connection Reset Error during video saving: {str(e)}")
                    print ("Continuing with the process anyway...")
                    output_filename =None 
                    webm_output_filename =None 
                except Exception as e :
                    print (f"Error saving MP4/WebM video or associated last frame: {str(e)}")
                    traceback .print_exc ()
                    output_filename =None 
                    webm_output_filename =None 

                save_metadata_enabled =True 
                if save_metadata_enabled and is_last_section :
                    gen_time_current =time .time ()-gen_start_time 
                    generation_time_seconds =int (gen_time_current )
                    generation_time_formatted =format_time_human_readable (gen_time_current )
                    metadata_prompt =prompt 
                    if using_timestamped_prompts :

                         metadata_prompt =prompt 

                    metadata ={
                    "Model":active_model ,
                    "Prompt":metadata_prompt ,
                    "Negative Prompt":n_prompt ,
                    "Seed":current_seed ,
                    "TeaCache":f"Enabled (Threshold: {teacache_threshold})"if teacache_threshold >0.0 else "Disabled",
                    "Video Length (seconds)":total_second_length ,
                    "FPS":fps ,
                    "Latent Window Size":latent_window_size ,
                    "Steps":steps ,
                    "CFG Scale":cfg ,
                    "Distilled CFG Scale":gs ,
                    "Guidance Rescale":rs ,
                    "Resolution":resolution ,
                    "Final Width":width ,
                    "Final Height":height ,
                    "Generation Time":generation_time_formatted ,
                    "Total Seconds":f"{generation_time_seconds} seconds",
                    "Start Frame Provided":True ,
                    "End Frame Provided":has_end_image ,
                    "Timestamped Prompts Used":using_timestamped_prompts ,
                    "GPU Inference Preserved Memory (GB)":gpu_memory_preservation ,
                    "Video Quality":video_quality ,
                    "RIFE FPS Multiplier":rife_multiplier ,
                    }
                    # Multi-LoRA metadata
                    for i in range(N_LORA):
                        lora_name = selected_lora_dropdown_values[i] if selected_lora_dropdown_values else "None"
                        lora_scale = lora_scales[i] if lora_scales else 1.0
                        if i == 0:
                            if lora_name != "None":
                                metadata["LoRA"] = lora_name
                                metadata["LoRA Scale"] = lora_scale
                        else:
                            if lora_name != "None":
                                metadata[f"LoRA {i+1} Name"] = lora_name
                                metadata[f"LoRA {i+1} Scale"] = lora_scale

                    metadata["Save Temporary Metadata Setting"] = save_temp_metadata_ui_value # Add the setting itself to final metadata

                    final_metadata_saved_successfully = True
                    final_metadata_for_main_output_saved = False
                    if output_filename: 
                        if save_processing_metadata (output_filename ,metadata .copy ()):
                            final_metadata_for_main_output_saved = True
                        else:
                             final_metadata_saved_successfully = False # Though this var isn't used after this block

                    if final_metadata_for_main_output_saved and save_temp_metadata_ui_value:
                        base_folder_for_job_temp_metadata = outputs_folder # Same base folder used for creation
                        delete_temporary_metadata_for_job(job_id, base_folder_for_job_temp_metadata)

                    final_gif_path =os .path .join (gif_videos_folder ,f'{job_id}.gif')
                    if export_gif and os .path .exists (final_gif_path ):
                        save_processing_metadata (final_gif_path ,metadata .copy ())
                    final_apng_path =os .path .join (apng_videos_folder ,f'{job_id}.png')
                    if export_apng and os .path .exists (final_apng_path ):
                         save_processing_metadata (final_apng_path ,metadata .copy ())
                    final_webp_path =os .path .join (webp_videos_folder ,f'{job_id}.webp')
                    if export_webp and os .path .exists (final_webp_path ):
                         save_processing_metadata (final_webp_path ,metadata .copy ())
                    final_webm_path =os .path .join (webm_videos_folder ,f'{job_id}.webm')
                    if video_quality =='web_compatible'and final_webm_path and os .path .exists (final_webm_path ):
                        save_processing_metadata (final_webm_path ,metadata .copy ())
                    rife_final_path =os .path .splitext (output_filename )[0 ]+'_extra_FPS.mp4'if output_filename else None 
                    if rife_enabled and rife_final_path and os .path .exists (rife_final_path ):
                         save_processing_metadata (rife_final_path ,metadata .copy ())

                try :
                    gif_filename =os .path .join (intermediate_gif_videos_folder if is_intermediate else gif_videos_folder ,f'{output_filename_base}.gif')
                    if export_gif :
                        try :
                            os .makedirs (os .path .dirname (gif_filename ),exist_ok =True )
                            save_bcthw_as_gif (history_pixels ,gif_filename ,fps =fps )
                            print (f"Saved GIF animation to {gif_filename}")
                        except Exception as e :print (f"Error saving GIF: {str(e)}")

                    apng_filename =os .path .join (intermediate_apng_videos_folder if is_intermediate else apng_videos_folder ,f'{output_filename_base}.png')
                    if export_apng :
                        try :
                            os .makedirs (os .path .dirname (apng_filename ),exist_ok =True )
                            save_bcthw_as_apng (history_pixels ,apng_filename ,fps =fps )
                            print (f"Saved APNG animation to {apng_filename}")
                        except Exception as e :print (f"Error saving APNG: {str(e)}")

                    webp_filename =os .path .join (intermediate_webp_videos_folder if is_intermediate else webp_videos_folder ,f'{output_filename_base}.webp')
                    if export_webp :
                        try :
                            os .makedirs (os .path .dirname (webp_filename ),exist_ok =True )
                            save_bcthw_as_webp (history_pixels ,webp_filename ,fps =fps )
                            print (f"Saved WebP animation to {webp_filename}")
                        except Exception as e :print (f"Error saving WebP: {str(e)}")
                except ConnectionResetError as e :
                    print (f"Connection Reset Error during additional format saving: {str(e)}")
                    print ("Continuing with the process anyway...")

                save_time =time .time ()-save_start_time 
                save_time_history .append (save_time )
                if len (save_time_history )>0 :
                    estimated_save_time =sum (save_time_history )/len (save_time_history )
                print (f"Saving operations completed in {save_time:.2f} seconds")

                primary_output_file =output_filename 
                if video_quality =='web_compatible'and webm_output_filename and os .path .exists (webm_output_filename ):
                     primary_output_file =webm_output_filename 
                stream .output_queue .push (('file',primary_output_file ))

                if is_last_section :
                    break 

            gen_time_completed =time .time ()-gen_start_time 
            generation_times .append (gen_time_completed )
            avg_gen_time =sum (generation_times )/len (generation_times )
            remaining_gens =num_generations -(gen_idx +1 )
            estimated_remaining_time =avg_gen_time *remaining_gens 

            print (f"\nGeneration {gen_idx+1}/{num_generations} completed in {gen_time_completed:.2f} seconds")
            if remaining_gens >0 :
                print (f"Estimated time for remaining generations: {estimated_remaining_time/60:.1f} minutes")

            stream .output_queue .push (('timing',{'gen_time':gen_time_completed ,'avg_time':avg_gen_time ,'remaining_time':estimated_remaining_time }))

        except KeyboardInterrupt as e :
            if str (e )=='User ends the task.':
                print ("\n"+"="*50 +"\nGENERATION ENDED BY USER\n"+"="*50 )
                if not high_vram :
                    print ("Unloading models from memory...")
                    unload_complete_models (text_encoder ,text_encoder_2 ,image_encoder ,vae ,transformer )
                stream .output_queue .push (('end',None ))
                return 
            else :raise 
        except ConnectionResetError as e :
            print (f"Connection Reset Error outside main processing loop: {str(e)}")
            print ("Trying to continue with next generation if possible...")
            if not high_vram :
                try :unload_complete_models (text_encoder ,text_encoder_2 ,image_encoder ,vae ,transformer )
                except Exception as cleanup_error :print (f"Error during memory cleanup: {str(cleanup_error)}")
            if gen_idx ==num_generations -1 :
                stream .output_queue .push (('end',None ))
                return 
            continue 
        except Exception as e :
            print ("\n"+"="*50 +f"\nERROR DURING GENERATION: {str(e)}\n"+"="*50 )
            traceback .print_exc ()
            print ("="*50 )
            if not high_vram :
                try :unload_complete_models (text_encoder ,text_encoder_2 ,image_encoder ,vae ,transformer )
                except Exception as cleanup_error :print (f"Error during memory cleanup after exception: {str(cleanup_error)}")
            if gen_idx ==num_generations -1 :
                stream .output_queue .push (('end',None ))
                return 
            continue 

    total_time_worker =time .time ()-start_time 
    print (f"\nTotal worker time for {num_generations} generation(s): {total_time_worker:.2f} seconds ({total_time_worker/60:.2f} minutes)")

    if not high_vram :
        unload_complete_models (text_encoder ,text_encoder_2 ,image_encoder ,vae ,transformer )

    stream .output_queue .push (('final_timing',{'total_time':total_time_worker ,'generation_times':generation_times }))
    stream .output_queue .push (('final_seed',last_used_seed ))
    stream .output_queue .push (('end',None ))
    return 

def manage_lora_structure (selected_lora_dropdown_values):
    global transformer, currently_loaded_lora_info, text_encoder, text_encoder_2, image_encoder, vae

    if not isinstance(selected_lora_dropdown_values, list) or len(selected_lora_dropdown_values) != N_LORA:
        raise ValueError(f"Expected a list of {N_LORA} LoRA selections.")

    # Get target adapter names and paths for all slots
    target_adapter_names = []
    target_lora_paths = []
    for display_name in selected_lora_dropdown_values:
        lora_path = get_lora_path_from_name(display_name)
        if lora_path != "none":
            # Use the same adapter name format as load_lora function
            filename = os.path.basename(lora_path)
            adapter_name = str(PurePath(filename).with_suffix('')).replace('.', '_DOT_')
        else:
            adapter_name = None
        target_adapter_names.append(adapter_name)
        target_lora_paths.append(lora_path)

    # Check if any change in adapter names or paths
    changed = False
    for i in range(N_LORA):
        if (currently_loaded_lora_info[i]["adapter_name"] != target_adapter_names[i] or
            currently_loaded_lora_info[i]["lora_path"] != target_lora_paths[i]):
            changed = True
            break

    if changed:
        print(f"LoRA Change Detected: Target={target_adapter_names}, Current={[info['adapter_name'] for info in currently_loaded_lora_info]}")

        transformer_on_cpu = False
        if transformer.device == cpu or (hasattr(transformer, 'model') and transformer.model.device == cpu):
            transformer_on_cpu = True
        elif not high_vram:
            print("Ensuring transformer is on CPU before LoRA structure change (Low VRAM mode)...")
            unload_complete_models(transformer)
            transformer_on_cpu = True
        else:
            try:
                print("Moving transformer to CPU for LoRA structure change (High VRAM mode)...")
                transformer.to(cpu)
                torch.cuda.empty_cache()
                transformer_on_cpu = True
            except Exception as e:
                print(f"Warning: Failed to move transformer to CPU in high VRAM mode: {e}")

        if not transformer_on_cpu:
            print("ERROR: Transformer could not be confirmed on CPU. Aborting LoRA structure change.")
            raise RuntimeError("Failed to ensure transformer is on CPU for LoRA modification.")

        # Unload all previous LoRAs
        any_loaded = any(info["adapter_name"] is not None for info in currently_loaded_lora_info)
        if any_loaded:
            print("Unloading previous LoRA structures...")
            unload_success = safe_unload_lora(transformer, cpu)
            if unload_success:
                print("Successfully unloaded all previous LoRAs.")
            else:
                print("ERROR: Failed to unload previous LoRAs! State may be corrupt.")
        # Reset info
        for i in range(N_LORA):
            currently_loaded_lora_info[i] = {"adapter_name": None, "lora_path": None}

        # Load each unique LoRA file only once
        loaded_in_this_cycle_adapter_names = set()
        for i in range(N_LORA):
            lora_path = target_lora_paths[i]
            adapter_name = target_adapter_names[i]
            if lora_path != "none" and adapter_name not in loaded_in_this_cycle_adapter_names:
                lora_dir, lora_filename = os.path.split(lora_path)
                print(f"Loading new LoRA structure: {adapter_name} from {lora_path}")
                try:
                    load_lora(transformer, lora_dir, lora_filename)
                    print(f"Successfully loaded structure for {adapter_name}.")
                    loaded_in_this_cycle_adapter_names.add(adapter_name)
                except Exception as e:
                    print(f"ERROR loading LoRA structure for {adapter_name}: {e}")
                    traceback.print_exc()
                    continue
            # Update info for this slot
            if lora_path != "none":
                currently_loaded_lora_info[i] = {"adapter_name": adapter_name, "lora_path": lora_path}
            else:
                currently_loaded_lora_info[i] = {"adapter_name": None, "lora_path": None}
        print(f"Final currently_loaded_lora_info: {currently_loaded_lora_info}")
    else:
        print(f"No LoRA structure change needed. Current: {[info['adapter_name'] for info in currently_loaded_lora_info]}")

def switch_active_model (target_model_display_name :str ,progress =gr .Progress () ):

    global transformer ,active_model_name ,currently_loaded_lora_info 

    if target_model_display_name ==active_model_name :
        print (f"Model '{active_model_name}' is already active.")
        return active_model_name ,f"Model '{active_model_name}' is already active."

    progress (0 ,desc =f"Switching model to '{target_model_display_name}'...")
    print (f"Switching model from '{active_model_name}' to '{target_model_display_name}'...")

    # Multi-LoRA: unload all if any are loaded
    any_loaded = any(info["adapter_name"] is not None for info in currently_loaded_lora_info)
    if any_loaded :
        print (f"Unloading LoRA(s) before switching model...")
        progress (0.1 ,desc =f"Unloading LoRA(s)...")
        try :
            if transformer .device !=cpu :
                if not high_vram :
                    unload_complete_models (transformer )
                if transformer .device !=cpu :
                    transformer .to (cpu )
                    torch .cuda .empty_cache ()

            unload_success =safe_unload_lora (transformer ,cpu )
            if unload_success :
                active_loras_before_unload = [info["adapter_name"] for info in currently_loaded_lora_info if info["adapter_name"]]
                if active_loras_before_unload:
                    print(f"Successfully unloaded LoRA(s): {', '.join(active_loras_before_unload)}.")
                else:
                    print("Successfully ensured no LoRAs are active (or none were loaded).")
                currently_loaded_lora_info = [{"adapter_name": None, "lora_path": None} for _ in range(N_LORA)]
            else :
                active_loras_before_attempt = [info["adapter_name"] for info in currently_loaded_lora_info if info["adapter_name"]]
                if active_loras_before_attempt:
                    print(f"Warning: Failed to cleanly unload LoRA(s): {', '.join(active_loras_before_attempt)}. Proceeding with model switch, but LoRA state might be inconsistent.")
                else:
                    print("Warning: Failed to cleanly unload LoRAs (though none might have been loaded). Proceeding with model switch.")
                currently_loaded_lora_info = [{"adapter_name": None, "lora_path": None} for _ in range(N_LORA)]
        except Exception as e :
            print (f"Error unloading LoRA during model switch: {e}")
            traceback .print_exc ()
            currently_loaded_lora_info = [{"adapter_name": None, "lora_path": None} for _ in range(N_LORA)]

    progress (0.3 ,desc =f"Unloading current model '{active_model_name}'...")
    print (f"Unloading current model '{active_model_name}'...")
    try :
        is_dynamic_swap =hasattr (transformer ,'_hf_hook')and isinstance (transformer ._hf_hook ,DynamicSwapInstaller .SwapHook )
        if is_dynamic_swap :
            print ("Uninstalling DynamicSwap from current transformer...")
            DynamicSwapInstaller .uninstall_model (transformer )

        del transformer 
        torch .cuda .empty_cache ()
        print (f"Model '{active_model_name}' unloaded.")
    except Exception as e :
        print (f"Error during model unload: {e}")
        traceback .print_exc ()

    new_model_hf_name =None 
    if target_model_display_name ==MODEL_DISPLAY_NAME_ORIGINAL :
        new_model_hf_name =MODEL_NAME_ORIGINAL 
    elif target_model_display_name ==MODEL_DISPLAY_NAME_F1 :
        new_model_hf_name =MODEL_NAME_F1 
    else :
        error_msg =f"Unknown target model name: {target_model_display_name}"
        print (f"ERROR: {error_msg}")

        return active_model_name ,f"Error: {error_msg}. Model not switched."

    progress (0.5 ,desc =f"Loading new model '{target_model_display_name}' from {new_model_hf_name}...")
    print (f"Loading new model '{target_model_display_name}' from {new_model_hf_name}...")
    try :
        transformer =HunyuanVideoTransformer3DModelPacked .from_pretrained (new_model_hf_name ,torch_dtype =torch .bfloat16 ).cpu ()
        transformer .eval ()
        transformer .high_quality_fp32_output_for_inference =True 
        transformer .to (dtype =torch .bfloat16 )
        transformer .requires_grad_ (False )
        print (f"New model '{target_model_display_name}' loaded to CPU.")

        if not high_vram :
            progress (0.8 ,desc =f"Applying memory optimization...")
            print ("Applying DynamicSwap to new transformer...")
            DynamicSwapInstaller .install_model (transformer ,device =gpu )

        active_model_name =target_model_display_name 
        progress (1.0 ,desc =f"Model switched successfully to '{active_model_name}'.")
        print (f"Model switched successfully to '{active_model_name}'.")

        return active_model_name ,f"Model switched to '{active_model_name}'."

    except Exception as e :
        error_msg =f"Failed to load model '{target_model_display_name}': {e}"
        print (f"ERROR: {error_msg}")
        traceback .print_exc ()

        try :
             print ("Attempting to reload default model as fallback...")
             default_hf_name =MODEL_NAME_ORIGINAL if DEFAULT_MODEL_NAME ==MODEL_DISPLAY_NAME_ORIGINAL else MODEL_NAME_F1 
             transformer =HunyuanVideoTransformer3DModelPacked .from_pretrained (default_hf_name ,torch_dtype =torch .bfloat16 ).cpu ()
             transformer .eval ()
             transformer .high_quality_fp32_output_for_inference =True 
             transformer .to (dtype =torch .bfloat16 )
             transformer .requires_grad_ (False )
             if not high_vram :DynamicSwapInstaller .install_model (transformer ,device =gpu )
             active_model_name =DEFAULT_MODEL_NAME 
             return active_model_name ,f"Error: {error_msg}. Reverted to default model '{active_model_name}'."
        except Exception as fallback_e :
             fatal_error_msg =f"CRITICAL ERROR: Failed to load target model AND fallback model. Error: {fallback_e}"
             print (fatal_error_msg )

             return active_model_name ,fatal_error_msg 

def process (input_image ,end_image ,prompt ,n_prompt ,seed ,use_random_seed ,num_generations ,total_second_length ,latent_window_size ,steps ,cfg ,gs ,rs ,gpu_memory_preservation ,teacache_threshold ,video_quality ='high',export_gif =False ,export_apng =False ,export_webp =False ,save_metadata =True ,resolution ="640",fps =30 ,

lora_scale_1=1.0, lora_scale_2=1.0, lora_scale_3=1.0, lora_scale_4=1.0,
selected_lora_1="None", selected_lora_2="None", selected_lora_3="None", selected_lora_4="None",

use_multiline_prompts =False ,save_individual_frames =False ,save_intermediate_frames =False ,save_last_frame =False ,rife_enabled =False ,rife_multiplier ="2x FPS",

selected_model_display_name =DEFAULT_MODEL_NAME ,
target_width =640 ,target_height =640,
save_temp_metadata_ui_value_from_ui: bool = True
):

    global stream ,currently_loaded_lora_info ,active_model_name 

    if input_image is None :
        print (f"No input image provided. Generating black image of size {target_width}x{target_height}.")
        input_image =np .zeros ((target_height ,target_width ,3 ),dtype =np .uint8 )

    if selected_model_display_name !=active_model_name :
         print (f"Warning: Selected model '{selected_model_display_name}' differs from active model '{active_model_name}'. Using the active model.")

    current_active_model =active_model_name 

    # Multi-LoRA: collect all 4 dropdowns and scales
    selected_lora_dropdown_values = [selected_lora_1, selected_lora_2, selected_lora_3, selected_lora_4]
    lora_scales = [lora_scale_1, lora_scale_2, lora_scale_3, lora_scale_4]

    try :
        manage_lora_structure(selected_lora_dropdown_values)
    except RuntimeError as e :
         print (f"LoRA Management Error: {e}")
         yield None ,None ,f"Error managing LoRA: {e}",'',gr .update (interactive =True ),gr .update (interactive =False ),seed ,''
         return 

    yield None ,None ,'','',gr .update (interactive =False ),gr .update (interactive =True ),seed ,''

    if use_multiline_prompts and prompt .strip ():
        prompt_lines =[line .strip ()for line in prompt .split ('\n')]
        prompt_lines =[line for line in prompt_lines if len (line )>=2 ]
        if not prompt_lines :prompt_lines =[prompt .strip ()]
        print (f"Multi-line enabled: Processing {len(prompt_lines)} prompts individually.")
    else :
        prompt_lines =[prompt .strip ()]
        if not use_multiline_prompts :print ("Multi-line disabled: Passing full prompt to worker for potential timestamp parsing.")
        else :print ("Multi-line enabled, but prompt seems empty or invalid, using as single line.")

    total_prompts_or_loops =len (prompt_lines )
    final_video =None 

    for prompt_idx ,current_prompt_line in enumerate (prompt_lines ):
        stream =AsyncStream ()
        print (f"Starting processing loop {prompt_idx+1}/{total_prompts_or_loops} using model '{current_active_model}'")
        status_msg =f"Processing prompt {prompt_idx+1}/{total_prompts_or_loops} ({current_active_model})"if use_multiline_prompts else f"Starting generation ({current_active_model})"
        yield None ,None ,status_msg ,'',gr .update (interactive =False ),gr .update (interactive =True ),seed ,''

        prompt_to_worker =prompt if not use_multiline_prompts else current_prompt_line 

        # Multi-LoRA: pass all 4 dropdowns and scales
        async_run (worker ,input_image ,end_image ,prompt_to_worker ,n_prompt ,seed ,use_random_seed ,total_second_length ,latent_window_size ,steps ,cfg ,gs ,rs ,gpu_memory_preservation ,teacache_threshold ,video_quality ,export_gif ,export_apng ,export_webp ,num_generations ,resolution ,fps ,
            selected_lora_dropdown_values=selected_lora_dropdown_values,
            lora_scales=lora_scales,
            save_individual_frames_flag =save_individual_frames ,
            save_intermediate_frames_flag =save_intermediate_frames ,
            save_last_frame_flag =save_last_frame ,
            use_multiline_prompts_flag =use_multiline_prompts ,
            rife_enabled =rife_enabled ,rife_multiplier =rife_multiplier ,
            active_model =current_active_model ,
            target_width_from_ui =target_width ,
            target_height_from_ui =target_height ,
            save_temp_metadata_ui_value = save_temp_metadata_ui_value_from_ui # Pass the UI value
        )

        output_filename =None 
        webm_filename =None 
        gif_filename =None 
        apng_filename =None 
        webp_filename =None 
        rife_final_video_path =None 
        current_seed_display =seed 
        timing_info =""
        last_output =None 

        while True :
            flag ,data =stream .output_queue .next ()

            if flag =='seed_update':
                current_seed_display =data 
                yield gr .update (),gr .update (),gr .update (),gr .update (),gr .update (),gr .update (),current_seed_display ,timing_info 

            if flag =='final_seed':
                 pass 

            if flag =='timing':
                gen_time =data ['gen_time']
                avg_time =data ['avg_time']
                remaining_time =data ['remaining_time']
                eta_str =f"{remaining_time/60:.1f} minutes"if remaining_time >60 else f"{remaining_time:.1f} seconds"
                timing_info =f"Last generation: {gen_time:.2f}s | Average: {avg_time:.2f}s | ETA: {eta_str}"
                yield gr .update (),gr .update (),gr .update (),gr .update (),gr .update (),gr .update (),current_seed_display ,timing_info 

            if flag =='final_timing':
                total_time =data ['total_time']
                timing_info =f"Total generation time: {total_time:.2f}s ({total_time/60:.2f} min)"
                yield gr .update (),gr .update (),gr .update (),gr .update (),gr .update (),gr .update (),current_seed_display ,timing_info 

            if flag =='file':
                output_filename =data 
                if output_filename is None :
                    print ("Warning: No primary output file was generated by worker")
                    yield None ,gr .update (),gr .update (),gr .update (),gr .update (interactive =False ),gr .update (interactive =True ),current_seed_display ,timing_info 
                    continue 

                last_output =output_filename 
                final_video =output_filename 

                potential_rife_path =os .path .splitext (output_filename )[0 ]+'_extra_FPS.mp4'
                if rife_enabled and os .path .exists (potential_rife_path ):
                    rife_final_video_path =potential_rife_path 
                    final_video =rife_final_video_path 
                    print (f"RIFE output detected: {rife_final_video_path}")

                display_video =final_video 

                prompt_info =f" (Prompt {prompt_idx+1}/{total_prompts_or_loops}, {current_active_model})"if use_multiline_prompts else f" ({current_active_model})"
                yield display_video ,gr .update (),gr .update (),gr .update (),gr .update (interactive =False ),gr .update (interactive =True ),current_seed_display ,timing_info +prompt_info 

            if flag =='rife_file':

                 rife_video_file =data 
                 print (f"Displaying RIFE-enhanced video: {rife_video_file}")
                 rife_final_video_path =rife_video_file 
                 final_video =rife_video_file 
                 prompt_info =f" (Prompt {prompt_idx+1}/{total_prompts_or_loops}, {current_active_model})"if use_multiline_prompts else f" ({current_active_model})"
                 yield rife_video_file ,gr .update (),gr .update (value =f"RIFE-enhanced video ready ({rife_multiplier})"),gr .update (),gr .update (interactive =False ),gr .update (interactive =True ),current_seed_display ,timing_info +prompt_info 

            if flag =='progress':
                preview ,desc ,html =data 

                model_prompt_info =f" ({current_active_model}"
                if use_multiline_prompts :
                     model_prompt_info +=f", Prompt {prompt_idx+1}/{total_prompts_or_loops}"
                model_prompt_info +=")"

                if html :
                    import re 
                    hint_match =re .search (r'>(.*?)<br',html )
                    if hint_match :
                        hint =hint_match .group (1 )
                        new_hint =f"{hint}{model_prompt_info}"

                        escaped_hint =re .escape (hint )
                        html =re .sub (f">{escaped_hint}<br",f">{new_hint}<br",html ,count =1 )
                    else :
                         html +=f"<span>{model_prompt_info}</span>"

                if desc :
                    desc +=model_prompt_info 

                yield gr .update (),gr .update (visible =True ,value =preview ),desc ,html ,gr .update (interactive =False ),gr .update (interactive =True ),current_seed_display ,timing_info 

            if flag =='end':

                display_video =final_video if final_video else None 

                if prompt_idx ==len (prompt_lines )-1 :
                    yield display_video ,gr .update (visible =False ),'','',gr .update (interactive =True ),gr .update (interactive =False ),current_seed_display ,timing_info 
                else :
                    yield display_video ,gr .update (visible =False ),f"Completed prompt {prompt_idx+1}/{total_prompts_or_loops} ({current_active_model})",'',gr .update (interactive =False ),gr .update (interactive =True ),current_seed_display ,timing_info 
                break 

        if not use_random_seed and prompt_idx <len (prompt_lines )-1 :
             seed +=1 

        if not use_multiline_prompts :
            break 

    display_video =final_video if final_video else None 
    yield display_video ,gr .update (visible =False ),'','',gr .update (interactive =True ),gr .update (interactive =False ),current_seed_display ,timing_info 

def batch_process (input_folder ,output_folder ,batch_end_frame_folder ,prompt ,n_prompt ,seed ,use_random_seed ,total_second_length ,
latent_window_size ,steps ,cfg ,gs ,rs ,gpu_memory_preservation ,teacache_threshold ,
video_quality ='high',export_gif =False ,export_apng =False ,export_webp =False ,
skip_existing =True ,save_metadata =True ,num_generations =1 ,resolution ="640",fps =30 ,
lora_scale_1=1.0, lora_scale_2=1.0, lora_scale_3=1.0, lora_scale_4=1.0,
selected_lora_1="None", selected_lora_2="None", selected_lora_3="None", selected_lora_4="None",
batch_use_multiline_prompts =False ,
batch_save_individual_frames =False ,batch_save_intermediate_frames =False ,batch_save_last_frame =False ,
rife_enabled =False ,rife_multiplier ="2x FPS",
selected_model_display_name =DEFAULT_MODEL_NAME ,
target_width =640 ,target_height =640 ,
batch_auto_resize =True,
save_temp_metadata_ui_value_from_ui: bool = True
):

    global stream ,batch_stop_requested ,currently_loaded_lora_info ,active_model_name 

    print ("Resetting batch stop flag.")
    batch_stop_requested =False 

    if selected_model_display_name !=active_model_name :
         print (f"Warning: Selected batch model '{selected_model_display_name}' differs from active model '{active_model_name}'. Using the active model for the entire batch.")
    current_active_model =active_model_name 

    # Multi-LoRA: collect all 4 dropdowns and scales
    selected_lora_dropdown_values = [selected_lora_1, selected_lora_2, selected_lora_3, selected_lora_4]
    lora_scales = [lora_scale_1, lora_scale_2, lora_scale_3, lora_scale_4]

    try :
        manage_lora_structure(selected_lora_dropdown_values)
    except RuntimeError as e :
         print (f"LoRA Management Error during batch setup: {e}")
         yield None ,None ,f"Error managing LoRA: {e}","",gr .update (interactive =True ),gr .update (interactive =False ),seed ,""
         return 

    if not input_folder or not os .path .exists (input_folder ):
        return None ,f"Input folder does not exist: {input_folder}","","",gr .update (interactive =True ),gr .update (interactive =False ),seed ,""
    if not output_folder :
        output_folder =outputs_batch_folder 
    else :
        try :os .makedirs (output_folder ,exist_ok =True )
        except Exception as e :return None ,f"Error creating output folder: {str(e)}","","",gr .update (interactive =True ),gr .update (interactive =False ),seed ,""

    use_end_frames =batch_end_frame_folder and os .path .isdir (batch_end_frame_folder )
    if batch_end_frame_folder and not use_end_frames :
         print (f"Warning: End frame folder provided but not found or not a directory: {batch_end_frame_folder}. Proceeding without end frames.")

    image_files =get_images_from_folder (input_folder )
    if not image_files :
        return None ,f"No image files found in {input_folder}","","",gr .update (interactive =True ),gr .update (interactive =False ),seed ,""

    yield None ,None ,f"Found {len(image_files)} images to process using model '{current_active_model}'. End frames {'enabled' if use_end_frames else 'disabled'}.","",gr .update (interactive =False ),gr .update (interactive =True ),seed ,""

    final_output =None 
    current_batch_seed =seed 
    # current_adapter_name =currently_loaded_lora_info ["adapter_name"]if currently_loaded_lora_info ["adapter_name"]else "None"

    for idx ,original_image_path_for_loop in enumerate (image_files ):
        if batch_stop_requested :
            print ("Batch stop requested. Exiting batch process.")
            yield final_output ,gr .update (visible =False ),"Batch processing stopped by user.","",gr .update (interactive =True ),gr .update (interactive =False ),current_batch_seed ,""
            return 

        start_image_basename =os .path .basename (original_image_path_for_loop )
        output_filename_base =os .path .splitext (start_image_basename )[0 ]

        input_image_for_worker_np =None 
        effective_width_for_worker =0 
        effective_height_for_worker =0 

        try :
            img_pil_original =Image .open (original_image_path_for_loop )
            if img_pil_original .mode !='RGB':img_pil_original =img_pil_original .convert ('RGB')
            orig_width_pil ,orig_height_pil =img_pil_original .size 

            img_for_processing_pil =img_pil_original 
            effective_width_for_worker =orig_width_pil 
            effective_height_for_worker =orig_height_pil 

            if batch_auto_resize :
                print (f"Processing {start_image_basename} with Auto Resize enabled.")
                # Apply EXIF transpose before getting original dimensions for calculations
                try:
                    print(f"  Applying EXIF transpose to {start_image_basename}...")
                    img_for_processing_pil = ImageOps.exif_transpose(img_for_processing_pil)
                    orig_width_pil, orig_height_pil = img_for_processing_pil.size # Update dimensions after transpose
                    print(f"  Dimensions after EXIF transpose: {orig_width_pil}x{orig_height_pil}")
                except Exception as exif_err:
                    print(f"  Warning: Could not apply EXIF transpose to {start_image_basename}: {exif_err}")
                
                target_res_val =int (resolution )
                target_pixel_count =target_res_val *target_res_val 
                original_pixel_count =orig_width_pil *orig_height_pil 

                if original_pixel_count >0 :
                    scale_factor =math .sqrt (target_pixel_count /original_pixel_count )
                    scaled_width_float =orig_width_pil *scale_factor 
                    scaled_height_float =orig_height_pil *scale_factor 

                    final_crop_width =math .floor (scaled_width_float /32.0 )*32 
                    final_crop_height =math .floor (scaled_height_float /32.0 )*32 

                    final_crop_width =max (32 ,final_crop_width )
                    final_crop_height =max (32 ,final_crop_height )

                    if not (final_crop_width ==orig_width_pil and final_crop_height ==orig_height_pil )and final_crop_width >0 and final_crop_height >0 :
                        print (f"  Auto-resizing {start_image_basename} from {orig_width_pil}x{orig_height_pil}...")
                        print (f"    Target area from resolution '{resolution}': {target_pixel_count}px.")
                        print (f"    Scaled aspect-preserved (float) to: {scaled_width_float:.2f}x{scaled_height_float:.2f}")
                        print (f"    Target final cropped dimensions (divisible by 32): {final_crop_width}x{final_crop_height}")

                        img_aspect =orig_width_pil /orig_height_pil 

                        crop_target_aspect =final_crop_width /final_crop_height if final_crop_height >0 else 1.0 

                        if img_aspect >crop_target_aspect :

                            resize_h_for_crop =final_crop_height 
                            resize_w_for_crop =int (round (final_crop_height *img_aspect ))
                        else :

                            resize_w_for_crop =final_crop_width 
                            resize_h_for_crop =int (round (final_crop_width /img_aspect ))

                        resize_w_for_crop =max (1 ,resize_w_for_crop )
                        resize_h_for_crop =max (1 ,resize_h_for_crop )

                        print (f"    Resizing original to {resize_w_for_crop}x{resize_h_for_crop} for center cropping...")
                        img_resized_for_cropping =img_for_processing_pil .resize ((resize_w_for_crop ,resize_h_for_crop ),Image .Resampling .LANCZOS )

                        current_w_rc ,current_h_rc =img_resized_for_cropping .size 
                        left =(current_w_rc -final_crop_width )/2 
                        top =(current_h_rc -final_crop_height )/2 
                        right =(current_w_rc +final_crop_width )/2 
                        bottom =(current_h_rc +final_crop_height )/2 

                        img_for_processing_pil =img_resized_for_cropping .crop ((int (round (left )),int (round (top )),int (round (right )),int (round (bottom ))))
                        effective_width_for_worker =final_crop_width 
                        effective_height_for_worker =final_crop_height 
                        print (f"    Resized and center-cropped to: {effective_width_for_worker}x{effective_height_for_worker}")
                    else :
                        print (f"  Image {start_image_basename} ({orig_width_pil}x{orig_height_pil}) already meets target criteria or does not require changes. Using original dimensions.")

                else :
                    print (f"  Warning: Original image {start_image_basename} has zero or negative pixels. Using original dimensions {orig_width_pil}x{orig_height_pil}.")

            input_image_for_worker_np =np .array (img_for_processing_pil )
            if len (input_image_for_worker_np .shape )==2 :
                 input_image_for_worker_np =np .stack ([input_image_for_worker_np ]*3 ,axis =2 )

            print (f"  Prepared image {start_image_basename} for worker with final dimensions: {effective_width_for_worker}x{effective_height_for_worker}")

        except Exception as e :
            print (f"Error loading or auto-resizing image {original_image_path_for_loop}: {str(e)}")
            traceback .print_exc ()
            yield None ,None ,f"Error processing {idx+1}/{len(image_files)}: {start_image_basename} - {str(e)}","",gr .update (interactive =False ),gr .update (interactive =True ),current_batch_seed_display ,""
            continue 

        current_prompt_text =prompt 
        custom_prompt =get_prompt_from_txt_file (original_image_path_for_loop )
        if custom_prompt :
            current_prompt_text =custom_prompt 
            print (f"Using custom prompt from txt file for {original_image_path_for_loop}")
        else :
            print (f"Using default batch prompt for {original_image_path_for_loop}")

        if batch_use_multiline_prompts :
            potential_lines =current_prompt_text .split ('\n')
            prompt_lines_or_fulltext =[line .strip ()for line in potential_lines if line .strip ()]
            prompt_lines_or_fulltext =[line for line in prompt_lines_or_fulltext if len (line )>=2 ]
            if not prompt_lines_or_fulltext :prompt_lines_or_fulltext =[current_prompt_text .strip ()]
            print (f"Batch multi-line enabled: Processing {len(prompt_lines_or_fulltext)} prompts for {start_image_basename}")
        else :
            prompt_lines_or_fulltext =[current_prompt_text .strip ()]
            print (f"Batch multi-line disabled: Passing full prompt text to worker for {start_image_basename}")

        total_prompts_or_loops =len (prompt_lines_or_fulltext )

        skip_this_image =False 
        if skip_existing :

            first_prompt_idx =0 
            first_gen_idx =1 
            output_check_suffix =""
            if batch_use_multiline_prompts :
                 output_check_suffix +=f"_p{first_prompt_idx+1}"
            if num_generations >1 :
                 output_check_suffix +=f"_g{first_gen_idx}"
            elif not batch_use_multiline_prompts and num_generations >1 :
                 output_check_suffix +=f"_{first_gen_idx}"

            output_check_mp4 =os .path .join (output_folder ,f"{output_filename_base}{output_check_suffix}.mp4")
            output_check_webm =os .path .join (output_folder ,f"{output_filename_base}{output_check_suffix}.webm")
            output_check_rife =os .path .join (output_folder ,f"{output_filename_base}{output_check_suffix}_extra_FPS.mp4")

            exists_mp4 =os .path .exists (output_check_mp4 )
            exists_webm =video_quality =='web_compatible'and os .path .exists (output_check_webm )
            exists_rife =rife_enabled and os .path .exists (output_check_rife )

            if rife_enabled :
                 skip_this_image =exists_rife or exists_mp4 
            elif video_quality =='web_compatible':
                 skip_this_image =exists_webm or exists_mp4 
            else :
                 skip_this_image =exists_mp4 

        if skip_this_image :
            print (f"Skipping {original_image_path_for_loop} - output already exists")
            yield None ,None ,f"Skipping {idx+1}/{len(image_files)}: {start_image_basename} - already processed","",gr .update (interactive =False ),gr .update (interactive =True ),current_batch_seed ,""
            continue 

        try :
            img =Image .open (original_image_path_for_loop )
            if img .mode !='RGB':img =img .convert ('RGB')
            input_image =np .array (img )
            if len (input_image .shape )==2 :input_image =np .stack ([input_image ]*3 ,axis =2 )
            print (f"Loaded start image {original_image_path_for_loop} with shape {input_image.shape} and dtype {input_image.dtype}")
        except Exception as e :
            print (f"Error loading start image {original_image_path_for_loop}: {str(e)}")
            yield None ,None ,f"Error processing {idx+1}/{len(image_files)}: {start_image_basename} - {str(e)}","",gr .update (interactive =False ),gr .update (interactive =True ),current_batch_seed ,""
            continue 

        current_end_image =None 
        end_image_path_str ="None"
        if use_end_frames :

            potential_end_path =os .path .join (batch_end_frame_folder ,start_image_basename )
            if os .path .exists (potential_end_path ):
                try :
                    end_img_pil = Image.open(potential_end_path)
                    # Apply EXIF transpose to end image as well if it's used
                    try:
                        print(f"  Applying EXIF transpose to end image: {potential_end_path}...")
                        end_img_pil = ImageOps.exif_transpose(end_img_pil)
                        print(f"  Dimensions of end image after EXIF transpose: {end_img_pil.width}x{end_img_pil.height}")
                    except Exception as exif_err_end:
                        print(f"  Warning: Could not apply EXIF transpose to end image {potential_end_path}: {exif_err_end}")

                    if end_img_pil.mode != 'RGB': end_img_pil = end_img_pil.convert('RGB')
                    current_end_image =np .array (end_img_pil)
                    if len (current_end_image .shape )==2 :current_end_image =np .stack ([current_end_image ]*3 ,axis =2 )
                    print (f"Loaded matching end image: {potential_end_path}")
                    end_image_path_str =potential_end_path 
                except Exception as e :
                    print (f"Error loading end image {potential_end_path}: {str(e)}. Processing without end frame.")
                    current_end_image =None 
            else :
                 print (f"No matching end frame found for {start_image_basename} in {batch_end_frame_folder}")

        for prompt_idx ,current_prompt_segment in enumerate (prompt_lines_or_fulltext ):
            if batch_stop_requested :
                print ("Batch stop requested during prompt loop. Exiting batch process.")
                yield final_output ,gr .update (visible =False ),"Batch processing stopped by user.","",gr .update (interactive =True ),gr .update (interactive =False ),current_batch_seed ,""
                return 

            seed_for_this_prompt =current_batch_seed 
            if use_random_seed :

                seed_for_this_prompt =random .randint (1 ,2147483647 )
            elif idx >0 or prompt_idx >0 :

                 seed_for_this_prompt =current_batch_seed +(idx *total_prompts_or_loops )+prompt_idx 

            current_batch_seed_display =seed_for_this_prompt 

            prompt_info =""
            if batch_use_multiline_prompts :
                prompt_info =f" (Prompt {prompt_idx+1}/{total_prompts_or_loops}: {current_prompt_segment[:30]}{'...' if len(current_prompt_segment) > 30 else ''})"
            elif not batch_use_multiline_prompts and custom_prompt :
                 prompt_info =" (Using .txt prompt - potential timestamps)"
            elif not batch_use_multiline_prompts and not custom_prompt :
                 prompt_info =" (Using default prompt - potential timestamps)"

            yield None ,None ,f"Processing {idx+1}/{len(image_files)}: {start_image_basename} ({current_active_model}, End: {os.path.basename(end_image_path_str) if current_end_image is not None else 'No'}) with {num_generations} generation(s){prompt_info}","",gr .update (interactive =False ),gr .update (interactive =True ),current_batch_seed_display ,""

            gen_start_time_batch =time .time ()

            stream =AsyncStream ()

            batch_individual_frames_folder_abs =None 
            batch_intermediate_individual_frames_folder_abs =None 
            batch_last_frames_folder_abs =None 
            batch_intermediate_last_frames_folder_abs =None 

            override_paths_needed =batch_save_individual_frames or batch_save_intermediate_frames or batch_save_last_frame 
            if override_paths_needed :
                batch_base_frame_folder =os .path .join (output_folder ,"frames_output")
                batch_individual_frames_folder_abs =os .path .join (batch_base_frame_folder ,'individual_frames')
                batch_intermediate_individual_frames_folder_abs =os .path .join (batch_individual_frames_folder_abs ,'intermediate_videos')
                batch_last_frames_folder_abs =os .path .join (batch_base_frame_folder ,'last_frames')
                batch_intermediate_last_frames_folder_abs =os .path .join (batch_last_frames_folder_abs ,'intermediate_videos')
                try :
                    os .makedirs (batch_individual_frames_folder_abs ,exist_ok =True )
                    os .makedirs (batch_intermediate_individual_frames_folder_abs ,exist_ok =True )
                    os .makedirs (batch_last_frames_folder_abs ,exist_ok =True )
                    os .makedirs (batch_intermediate_last_frames_folder_abs ,exist_ok =True )
                    print (f"Created/Ensured batch frame output folders in: {batch_base_frame_folder}")
                except Exception as e :
                     print (f"Error creating batch frame folders: {e}")
                     override_paths_needed =False 

            def batch_worker_path_override (*args ,**kwargs ):
                global individual_frames_folder ,intermediate_individual_frames_folder ,last_frames_folder ,intermediate_last_frames_folder 

                orig_individual_frames =individual_frames_folder 
                orig_intermediate_individual_frames =intermediate_individual_frames_folder 
                orig_last_frames =last_frames_folder 
                orig_intermediate_last_frames =intermediate_last_frames_folder 

                individual_frames_folder =batch_individual_frames_folder_abs 
                intermediate_individual_frames_folder =batch_intermediate_individual_frames_folder_abs 
                last_frames_folder =batch_last_frames_folder_abs 
                intermediate_last_frames_folder =batch_intermediate_last_frames_folder_abs 
                print (f"Worker override: Using batch frame paths rooted at {output_folder}")
                try :

                    result =worker (*args ,**kwargs )
                    return result 
                finally :

                    individual_frames_folder =orig_individual_frames 
                    intermediate_individual_frames_folder =orig_intermediate_individual_frames 
                    last_frames_folder =orig_last_frames 
                    intermediate_last_frames_folder =orig_intermediate_last_frames 
                    print ("Worker override: Restored global frame paths.")

            worker_function_to_call =batch_worker_path_override if override_paths_needed else worker 

            async_run (worker_function_to_call ,
            input_image_for_worker_np ,
            current_end_image ,current_prompt_segment ,n_prompt ,
            current_batch_seed_display ,
            False ,
            total_second_length ,latent_window_size ,steps ,cfg ,gs ,rs ,
            gpu_memory_preservation ,teacache_threshold ,video_quality ,export_gif ,
            export_apng ,export_webp ,
            num_generations =num_generations ,
            resolution =resolution ,fps =fps ,
            selected_lora_dropdown_values=selected_lora_dropdown_values,
            lora_scales=lora_scales,
            save_individual_frames_flag =batch_save_individual_frames ,
            save_intermediate_frames_flag =batch_save_intermediate_frames ,
            save_last_frame_flag =batch_save_last_frame ,
            use_multiline_prompts_flag =batch_use_multiline_prompts ,
            rife_enabled =rife_enabled ,rife_multiplier =rife_multiplier ,
            active_model =current_active_model ,
            target_width_from_ui =effective_width_for_worker ,
            target_height_from_ui =effective_height_for_worker ,
            save_temp_metadata_ui_value = save_temp_metadata_ui_value_from_ui # Pass the UI value
            )

            output_filename_from_worker =None 
            last_output_for_ui =None 
            all_outputs_for_image ={}
            generation_count_this_prompt =0 

            while True :
                if batch_stop_requested :
                     print ("Batch stop requested while waiting for worker. Ending loop.")
                     if stream :stream .input_queue .push ('end')
                     break 
                flag ,data =stream .output_queue .next ()

                if flag is None :continue 

                if flag =='seed_update':

                    current_batch_seed_display =data 
                    yield gr .update (),gr .update (),gr .update (),gr .update (),gr .update (),gr .update (),current_batch_seed_display ,gr .update ()

                if flag =='final_seed':

                     pass 

                if flag =='file':
                    output_filename_from_worker =data 
                    if output_filename_from_worker :
                        is_intermediate ='intermediate'in output_filename_from_worker 
                        if not is_intermediate :
                            generation_count_this_prompt +=1 

                            output_file_suffix =""
                            if batch_use_multiline_prompts :
                                output_file_suffix +=f"_p{prompt_idx+1}"
                            if num_generations >1 :

                                output_file_suffix +=f"_g{generation_count_this_prompt}"
                            elif not batch_use_multiline_prompts and num_generations >1 :
                                output_file_suffix +=f"_{generation_count_this_prompt}"

                            final_output_filename_base =f"{output_filename_base}{output_file_suffix}"

                            moved_primary_file =move_and_rename_output_file (
                            output_filename_from_worker ,
                            output_folder ,
                            f"{final_output_filename_base}{os.path.splitext(output_filename_from_worker)[1]}"
                            )

                            if moved_primary_file :
                                output_key =f'primary_{generation_count_this_prompt}'if num_generations >1 else 'primary'
                                all_outputs_for_image [output_key ]=moved_primary_file 
                                last_output_for_ui =moved_primary_file 
                                final_output =moved_primary_file 

                                prompt_status =f" (Prompt {prompt_idx+1}/{total_prompts_or_loops})"if batch_use_multiline_prompts else ""
                                yield last_output_for_ui ,gr .update (visible =False ),f"Processing {idx+1}/{len(image_files)}: {start_image_basename} ({current_active_model}) - Generated video {generation_count_this_prompt}/{num_generations}{prompt_status}","",gr .update (interactive =False ),gr .update (interactive =True ),current_batch_seed_display ,gr .update ()

                                if save_metadata :
                                    gen_time_batch_current =time .time ()-gen_start_time_batch 
                                    generation_time_seconds =int (gen_time_batch_current )
                                    generation_time_formatted =format_time_human_readable (gen_time_batch_current )

                                    final_proc_width_meta =effective_width_for_worker 
                                    final_proc_height_meta =effective_height_for_worker 
                                    print (f"Batch metadata: Using effective dimensions {effective_width_for_worker}x{effective_height_for_worker}. Resolution Guide from UI '{resolution}'.")

                                    has_end_image =current_end_image is not None 

                                    using_timestamped_prompts =not batch_use_multiline_prompts 

                                    metadata ={
                                    "Model":current_active_model ,
                                    "Prompt":current_prompt_segment ,
                                    "Negative Prompt":n_prompt ,

                                    "Seed":current_batch_seed_display +(generation_count_this_prompt -1 )if not use_random_seed else "Random",
                                    "TeaCache":f"Enabled (Threshold: {teacache_threshold})"if teacache_threshold >0.0 else "Disabled",
                                    "Video Length (seconds)":total_second_length ,
                                    "FPS":fps ,
                                    "Latent Window Size":latent_window_size ,
                                    "Steps":steps ,
                                    "CFG Scale":cfg ,
                                    "Distilled CFG Scale":gs ,
                                    "Guidance Rescale":rs ,
                                    "Resolution":resolution ,
                                    "Final Width":final_proc_width_meta ,
                                    "Final Height":final_proc_height_meta ,
                                    "Generation Time":generation_time_formatted ,
                                    "Total Seconds":f"{generation_time_seconds} seconds",
                                    "Start Frame":original_image_path_for_loop ,
                                    "End Frame Provided":has_end_image ,
                                    "Timestamped Prompts Used":using_timestamped_prompts ,
                                    "GPU Inference Preserved Memory (GB)":gpu_memory_preservation ,
                                    "Video Quality":video_quality ,
                                    "RIFE FPS Multiplier":rife_multiplier ,
                                    "Number of Generations Configured":num_generations ,
                                    }
                                    if not batch_use_multiline_prompts :
                                         metadata ["Timestamped Prompts Parsed"]="[Check Worker Logs]"

                                    # if current_adapter_name !="None":
                                    #     metadata ["LoRA"]=current_adapter_name 
                                    #     metadata ["LoRA Scale"]=lora_scale 

                                    # Multi-LoRA metadata for batch
                                    for i in range(N_LORA):
                                        lora_name = selected_lora_dropdown_values[i] # Use the dropdown value which is the display name
                                        lora_scale_val = lora_scales[i]
                                        if lora_name and lora_name != "None":
                                            if i == 0: # First LoRA (already handled by old LoRA/LoRA Scale if present)
                                                if "LoRA" not in metadata: # only add if not covered by legacy single LoRA
                                                    metadata["LoRA"] = lora_name
                                                    metadata["LoRA Scale"] = lora_scale_val
                                            else: # Subsequent LoRAs
                                                metadata[f"LoRA {i+1} Name"] = lora_name
                                                metadata[f"LoRA {i+1} Scale"] = lora_scale_val

                                    save_processing_metadata (moved_primary_file ,metadata .copy ())

                                    worker_output_base =os .path .splitext (output_filename_from_worker )[0 ]
                                    target_output_base =os .path .join (output_folder ,final_output_filename_base )

                                    gif_original_path =f"{worker_output_base}.gif"
                                    if export_gif and os .path .exists (gif_original_path ):
                                        moved_gif =move_and_rename_output_file (gif_original_path ,output_folder ,f"{final_output_filename_base}.gif")
                                        if moved_gif :save_processing_metadata (moved_gif ,metadata .copy ())

                                    apng_original_path =f"{worker_output_base}.png"
                                    if export_apng and os .path .exists (apng_original_path ):
                                         moved_apng =move_and_rename_output_file (apng_original_path ,output_folder ,f"{final_output_filename_base}.png")
                                         if moved_apng :save_processing_metadata (moved_apng ,metadata .copy ())

                                    webp_original_path =f"{worker_output_base}.webp"
                                    if export_webp and os .path .exists (webp_original_path ):
                                         moved_webp =move_and_rename_output_file (webp_original_path ,output_folder ,f"{final_output_filename_base}.webp")
                                         if moved_webp :save_processing_metadata (moved_webp ,metadata .copy ())

                                    webm_original_path =f"{worker_output_base}.webm"
                                    if video_quality =='web_compatible'and moved_primary_file !=webm_original_path and os .path .exists (webm_original_path ):
                                         moved_webm =move_and_rename_output_file (webm_original_path ,output_folder ,f"{final_output_filename_base}.webm")
                                         if moved_webm :save_processing_metadata (moved_webm ,metadata .copy ())

                                    rife_original_path =f"{worker_output_base}_extra_FPS.mp4"
                                    if rife_enabled and os .path .exists (rife_original_path ):
                                        rife_target_path =f"{target_output_base}_extra_FPS.mp4"
                                        try :
                                            shutil .copy2 (rife_original_path ,rife_target_path )
                                            print (f"Copied RIFE enhanced file to batch outputs: {rife_target_path}")
                                            save_processing_metadata (rife_target_path ,metadata .copy ())

                                            last_output_for_ui =rife_target_path 
                                            final_output =rife_target_path 
                                        except Exception as e :
                                            print (f"Error copying RIFE enhanced file to {rife_target_path}: {str(e)}")

                            else :
                                 print (f"ERROR: Failed to move/rename output file {output_filename_from_worker}")

                                 yield last_output_for_ui ,gr .update (visible =False ),f"Error saving output for {idx+1}/{len(image_files)}: {start_image_basename}","",gr .update (interactive =False ),gr .update (interactive =True ),current_batch_seed_display ,gr .update ()

                        else :
                            prompt_status =f" (Prompt {prompt_idx+1}/{total_prompts_or_loops})"if batch_use_multiline_prompts else ""
                            yield output_filename_from_worker ,gr .update (visible =False ),f"Processing {idx+1}/{len(image_files)}: {start_image_basename} ({current_active_model}) - Generating intermediate result{prompt_status}","",gr .update (interactive =False ),gr .update (interactive =True ),current_batch_seed_display ,gr .update ()

                if flag =='progress':
                    preview ,desc ,html =data 

                    batch_prog_info =f"Processing {idx+1}/{len(image_files)}: {start_image_basename} ({current_active_model}"
                    if batch_use_multiline_prompts :
                        batch_prog_info +=f", Prompt {prompt_idx+1}/{total_prompts_or_loops}"
                    batch_prog_info +=")"

                    current_progress_desc =f"{batch_prog_info} - {desc}"if desc else batch_prog_info 
                    progress_html =html if html else make_progress_bar_html (0 ,batch_prog_info )

                    if html :
                         import re 
                         hint_match =re .search (r'>(.*?)<br',html )
                         if hint_match :
                              hint =hint_match .group (1 )
                              new_hint =f"{hint} [{idx+1}/{len(image_files)}]"
                              if batch_use_multiline_prompts :new_hint +=f"[P{prompt_idx+1}]"
                              escaped_hint =re .escape (hint )
                              progress_html =re .sub (f">{escaped_hint}<br",f">{new_hint}<br",html ,count =1 )
                         else :
                             progress_html +=f"<span>{batch_prog_info}</span>"

                    video_update =last_output_for_ui if last_output_for_ui else gr .update ()
                    yield video_update ,gr .update (visible =True ,value =preview ),current_progress_desc ,progress_html ,gr .update (interactive =False ),gr .update (interactive =True ),current_batch_seed_display ,gr .update ()

                if flag =='end':

                    video_update =last_output_for_ui if last_output_for_ui else gr .update ()
                    if prompt_idx ==len (prompt_lines_or_fulltext )-1 :
                        yield video_update ,gr .update (visible =False ),f"Completed {idx+1}/{len(image_files)}: {start_image_basename} ({current_active_model})","",gr .update (interactive =False ),gr .update (interactive =True ),current_batch_seed_display ,gr .update ()
                    else :
                        prompt_status =f" (Completed prompt {prompt_idx+1}/{total_prompts_or_loops}, continuing to next prompt)"
                        yield video_update ,gr .update (visible =False ),f"Processing {idx+1}/{len(image_files)}: {start_image_basename}{prompt_status}","",gr .update (interactive =False ),gr .update (interactive =True ),current_batch_seed_display ,gr .update ()
                    break 

            if batch_stop_requested :
                print ("Batch stop requested after worker finished. Exiting batch process.")
                yield final_output ,gr .update (visible =False ),"Batch processing stopped by user.","",gr .update (interactive =True ),gr .update (interactive =False ),current_batch_seed_display ,""
                return 

            if not batch_use_multiline_prompts :
                 break 

        if not use_random_seed :
             current_batch_seed =seed +(idx +1 )

        if batch_stop_requested :
            print ("Batch stop requested after inner loop. Exiting batch process.")
            yield final_output ,gr .update (visible =False ),"Batch processing stopped by user.","",gr .update (interactive =True ),gr .update (interactive =False ),current_batch_seed_display ,""
            return 

    if not batch_stop_requested :
        yield final_output ,gr .update (visible =False ),f"Batch processing complete. Processed {len(image_files)} images using {current_active_model}.","",gr .update (interactive =True ),gr .update (interactive =False ),current_batch_seed_display ,""
    else :
         yield final_output ,gr .update (visible =False ),"Batch processing stopped by user.","",gr .update (interactive =True ),gr .update (interactive =False ),current_batch_seed_display ,""

    if batch_auto_resize :
        try :
            for file in os .listdir (output_folder ):
                if file .startswith ("temp_"):
                    temp_path =os .path .join (output_folder ,file )
                    try :
                        os .remove (temp_path )
                        print (f"Cleaned up temporary file: {file}")
                    except Exception as e :
                        print (f"Error cleaning up temporary file {file}: {str(e)}")
        except Exception as e :
            print (f"Error during cleanup of temporary files: {str(e)}")

def end_process ():
    global batch_stop_requested 
    print ("\nSending end generation signal...")
    if 'stream'in globals ()and stream :
        stream .input_queue .push ('end')
        print ("End signal sent to current worker.")
    else :
        print ("Stream not initialized, cannot send end signal to worker.")

    print ("Setting batch stop flag...")
    batch_stop_requested =True 

    updates =[gr .update (interactive =True ),gr .update (interactive =False ),gr .update (interactive =True ),gr .update (interactive =False )]
    return updates 

quick_prompts =[
'A character doing some simple body movements.','A talking man.',
'[0] A person stands still\n[2] The person waves hello\n[4] The person claps',
'[0] close up shot, cinematic\n[3] medium shot, cinematic\n[5] wide angle shot, cinematic'
]
quick_prompts =[[x ]for x in quick_prompts ]

def auto_set_window_size (fps_val :int ,current_lws :int ):

    if not isinstance (fps_val ,int )or fps_val <=0 :
        print ("Invalid FPS for auto window size calculation.")
        return gr .update ()

    try :
        ideal_lws_float =fps_val /4.0 
        target_lws =round (ideal_lws_float )
        min_lws =1 
        max_lws =33 
        calculated_lws =max (min_lws ,min (target_lws ,max_lws ))
        resulting_duration =(calculated_lws *4 )/fps_val 

        print (f"Auto-setting LWS: Ideal float LWS for 1s sections={ideal_lws_float:.2f}, Rounded integer LWS={target_lws}, Clamped LWS={calculated_lws}")
        print (f"--> Resulting section duration with LWS={calculated_lws} at {fps_val} FPS will be: {resulting_duration:.3f} seconds")

        if abs (resulting_duration -1.0 )<0.01 :print ("This setting provides (near) exact 1-second sections.")
        else :print (f"Note: This is the closest integer LWS to achieve 1-second sections.")

        if calculated_lws !=current_lws :return gr .update (value =calculated_lws )
        else :
            print (f"Latent Window Size is already optimal ({current_lws}) for ~1s sections.")
            return gr .update ()

    except Exception as e :
        print (f"Error calculating auto window size: {e}")
        traceback .print_exc ()
        return gr .update ()

def update_target_dimensions_from_image (image_array ,resolution_str ):
    if image_array is None :

        print ("Input image cleared, resetting target dimensions to default (640x640).")
        return gr .update (value =640 ),gr .update (value =640 )
    try :
        H ,W ,_ =image_array .shape 
        print (f"Input image uploaded with dimensions: {W}x{H}. Resolution guide: {resolution_str}")

        bucket_h ,bucket_w =find_nearest_bucket (H ,W ,resolution =resolution_str )
        print (f"Suggested bucket dimensions: {bucket_w}x{bucket_h}. Updating target sliders.")
        return gr .update (value =bucket_w ),gr .update (value =bucket_h )
    except Exception as e :
        print (f"Error processing uploaded image for target dimensions: {e}")
        traceback .print_exc ()

        return gr .update (value =640 ),gr .update (value =640 )

def get_nearest_bucket_size (width :int ,height :int ,resolution :str )->tuple [int ,int ]:

    target_res =int (resolution )

    bucket_sizes =[
    (512 ,512 ),(512 ,768 ),(512 ,1024 ),
    (768 ,512 ),(768 ,768 ),(768 ,1024 ),
    (1024 ,512 ),(1024 ,768 ),(1024 ,1024 )
    ]

    aspect_ratio =width /height 

    best_bucket =None 
    min_diff =float ('inf')

    for bucket_w ,bucket_h in bucket_sizes :

        if aspect_ratio >1 :
            new_width =min (bucket_w ,int (bucket_h *aspect_ratio ))
            new_height =int (new_width /aspect_ratio )
        else :
            new_height =min (bucket_h ,int (bucket_w /aspect_ratio ))
            new_width =int (new_height *aspect_ratio )

        target_area =target_res *target_res 
        actual_area =new_width *new_height 
        diff =abs (target_area -actual_area )

        if diff <min_diff :
            min_diff =diff 
            best_bucket =(new_width ,new_height )

    return best_bucket 

css =make_progress_bar_css ()
block =gr .Blocks (css =css ).queue ()
with block :
    gr .Markdown ('# FramePack Improved SECourses App V59 - https://www.patreon.com/posts/126855226')
    with gr .Row ():

        model_selector =gr .Radio (
        label ="Select Model",
        choices =[MODEL_DISPLAY_NAME_ORIGINAL ,MODEL_DISPLAY_NAME_F1 ],
        value =active_model_name ,
        info ="Choose the generation model. Switching will unload the current model and load the new one."
        )
        model_status =gr .Markdown (f"Active model: **{active_model_name}**")

    with gr .Row ():
        with gr .Column ():
            with gr .Tabs ():
                with gr .Tab ("Single Image / Multi-Prompt"):
                    with gr .Row ():
                        with gr .Column ():
                            input_image =gr .Image (sources ='upload',type ="numpy",label ="Start Frame",height =320 )
                        with gr .Column ():

                            end_image =gr .Image (sources ='upload',type ="numpy",label ="End Frame (Optional, Original Model primarily)",height =320 )

                    with gr .Row ():
                        iteration_info_display =gr .Markdown ("Calculating generation info...",elem_id ="iteration-info-display")
                        auto_set_lws_button =gr .Button (value ="Set Window for ~1s Sections (Original Model)",scale =1 )

                    prompt =gr .Textbox (label ="Prompt",value ='',lines =4 ,info ="Use '[seconds] prompt' format on new lines ONLY when 'Use Multi-line Prompts' is OFF. Example [0] starts second 0, [2] starts after 2 seconds passed and so on. Applies to both models.")
                    with gr .Row ():
                        use_multiline_prompts =gr .Checkbox (label ="Use Multi-line Prompts",value =False ,info ="ON: Each line is a separate gen (new seed per line if random). OFF: Try parsing '[secs] prompt' format.")

                        latent_window_size =gr .Slider (label ="Latent Window Size",minimum =1 ,maximum =33 ,value =9 ,step =1 ,visible =True ,info ="Controls generation chunk size. Affects section/loop count and duration (see info above prompt). Default 9 recommended.")
                    example_quick_prompts =gr .Dataset (samples =quick_prompts ,label ='Quick List',samples_per_page =1000 ,components =[prompt ])
                    example_quick_prompts .click (lambda x :x [0 ],inputs =[example_quick_prompts ],outputs =prompt ,show_progress =False ,queue =False )

                    with gr .Row ():
                        save_metadata =gr .Checkbox (label ="Save Processing Metadata",value =True ,info ="Save processing parameters in a text file alongside each video")
                        save_individual_frames =gr .Checkbox (label ="Save Individual Frames",value =False ,info ="Save each frame of the final video as an individual image")
                        save_intermediate_frames =gr .Checkbox (label ="Save Intermediate Frames",value =False ,info ="Save each frame of intermediate videos as individual images")
                        save_last_frame =gr .Checkbox (label ="Save Last Frame Of Generations (MP4 Only)",value =False ,info ="Save only the last frame of each MP4 generation to the last_frames folder")

                    with gr .Row ():
                        start_button =gr .Button (value ="Start Generation",variant ='primary')
                        end_button =gr .Button (value ="End Generation",interactive =False )

                with gr .Tab ("Batch Processing"):
                    batch_input_folder =gr .Textbox (label ="Input Folder Path (Start Frames)",info ="Folder containing starting images to process (sorted naturally)")
                    batch_end_frame_folder =gr .Textbox (label ="End Frame Folder Path (Optional, Original Model primarily)",info ="Folder containing matching end frames (same filename as start frame)")
                    batch_output_folder =gr .Textbox (label ="Output Folder Path (optional)",info ="Leave empty to use the default batch_outputs folder")
                    batch_prompt =gr .Textbox (label ="Default Prompt",value ='',lines =4 ,info ="Used if no matching .txt file exists. Use '[seconds] prompt' format on new lines ONLY when 'Use Multi-line Prompts' is OFF.")

                    with gr .Row ():
                        batch_skip_existing =gr .Checkbox (label ="Skip Existing Files",value =True ,info ="Skip images where the first expected output video already exists")
                        batch_save_metadata =gr .Checkbox (label ="Save Processing Metadata",value =True ,info ="Save processing parameters in a text file alongside each video")
                        batch_use_multiline_prompts =gr .Checkbox (label ="Use Multi-line Prompts",value =False ,info ="ON: Each line in prompt/.txt is a separate gen. OFF: Try parsing '[secs] prompt' format from full prompt/.txt.")
                        batch_auto_resize =gr .Checkbox (label ="Auto Resize to Nearest Bucket",value =True ,info ="Automatically resize input images to nearest bucket size while maintaining aspect ratio")

                    with gr .Row ():
                        batch_save_individual_frames =gr .Checkbox (label ="Save Individual Frames",value =False ,info ="Save each frame of the final video as an individual image (in batch_output/frames_output)")
                        batch_save_intermediate_frames =gr .Checkbox (label ="Save Intermediate Frames",value =False ,info ="Save each frame of intermediate videos as individual images (in batch_output/frames_output)")
                        batch_save_last_frame =gr .Checkbox (label ="Save Last Frame Of Generations (MP4 Only)",value =False ,info ="Save only the last frame of each MP4 generation (in batch_output/frames_output)")

                    with gr .Row ():
                        batch_start_button =gr .Button (value ="Start Batch Processing",variant ='primary')
                        batch_end_button =gr .Button (value ="End Processing",interactive =False )

                    with gr .Row ():
                        open_batch_input_folder =gr .Button (value ="Open Start Folder")
                        open_batch_end_folder =gr .Button (value ="Open End Folder")
                        open_batch_output_folder =gr .Button (value ="Open Output Folder")

                with gr .Tab ("Load from Metadata File"):
                    gr .Markdown ("Upload a `.txt` metadata file (previously saved by this application) to automatically populate the settings. This will overwrite current settings in all tabs.")
                    metadata_file_input =gr .File (label ="Upload Metadata .txt File",file_types =[".txt"],type ="binary")
                    load_from_metadata_button =gr .Button ("Load Settings from File",variant ='primary')
                    metadata_load_status =gr .Markdown ("")

                with gr .Row ():

                    gpu_memory_preservation =gr .Slider (label ="GPU Inference Preserved Memory (GB)",minimum =0 ,maximum =128 ,value =8 ,step =0.1 ,info ="Increase this until you don't use shared VRAM. If you increase unnecessarily it will become slower. However If you make this lower than expected, It will use shared VRAM and will become ultra slow. Follow nvitop and monitor GPU watt usage")

                    def update_memory_for_resolution (res ):

                        res_int =int (res )
                        if res_int >=1440 :return 23 
                        elif res_int >=1320 :return 21 
                        elif res_int >=1200 :return 19 
                        elif res_int >=1080 :return 16 
                        elif res_int >=960 :return 14 
                        elif res_int >=840 :return 12 
                        elif res_int >=720 :return 10 
                        elif res_int >=640 :return 8 
                        else :return 6 
                    

            with gr .Group ("Common Settings"):
                with gr .Row ():
                    num_generations =gr .Slider (label ="Number of Generations",minimum =1 ,maximum =50 ,value =1 ,step =1 ,info ="Generate multiple videos in sequence (per image/prompt)")
                    resolution =gr .Dropdown (label ="Resolution",choices =["1440","1320","1200","1080","960","840","720","640","480","320","240"],value ="640",info ="Output Resolution (bigger than 640 set more Preserved Memory)")
                    resolution.change (fn=update_memory_for_resolution ,inputs=resolution ,outputs=gpu_memory_preservation )
                with gr .Row ():
                    target_width_slider =gr .Slider (label ="Target Width",minimum =256 ,maximum =2048 ,value =640 ,step =32 ,info ="Desired output width. Will be snapped to nearest bucket.")
                    target_height_slider =gr .Slider (label ="Target Height",minimum =256 ,maximum =2048 ,value =640 ,step =32 ,info ="Desired output height. Will be snapped to nearest bucket.")

                with gr .Row ():

                    teacache_threshold =gr .Slider (label ='TeaCache Threshold',minimum =0.0 ,maximum =0.5 ,value =0.15 ,step =0.01 ,info ='0 = Disabled, >0 = Enabled. Higher values = more caching but potentially less detail. Affects both models.')
                    seed =gr .Number (label ="Seed",value =31337 ,precision =0 )
                    use_random_seed =gr .Checkbox (label ="Random Seed",value =True ,info ="Use random seeds instead of fixed/incrementing")

                n_prompt =gr .Textbox (label ="Negative Prompt",value ="",visible =True ,info ="Used when CFG Scale > 1.0 (Primarily for Original Model)")

                with gr .Row ():
                    fps =gr .Slider (label ="FPS",minimum =4 ,maximum =60 ,value =30 ,step =1 ,info ="Output Videos FPS - Directly changes how many frames are generated")
                    total_second_length =gr .Slider (label ="Total Video Length (Seconds)",minimum =1 ,maximum =120 ,value =5 ,step =0.1 )

                with gr .Row ():
                    steps =gr .Slider (label ="Steps",minimum =1 ,maximum =100 ,value =25 ,step =1 ,info ='Default 25 recommended for both models.')

                    gs =gr .Slider (label ="Distilled CFG Scale",minimum =1.0 ,maximum =32.0 ,value =10.0 ,step =0.01 ,info ='Default 10.0 recommended for both models.')

                with gr .Row ():

                    cfg =gr .Slider (label ="CFG Scale",minimum =1.0 ,maximum =32.0 ,value =1.0 ,step =0.01 ,visible =True ,info ='Needs > 1.0 for Negative Prompt. F1 model typically uses 1.0.')

                    rs =gr .Slider (label ="CFG Re-Scale",minimum =0.0 ,maximum =1.0 ,value =0.0 ,step =0.01 ,visible =True ,info ='Default 0.0 recommended for both models.')

                gr .Markdown ("### LoRA Settings (Applies to selected model)")
                with gr .Row ():
                    with gr .Column ():
                        lora_options =scan_lora_files ()
                        selected_lora_1 =gr .Dropdown (label ="Select LoRA 1",choices =[name for name ,_ in lora_options ],value ="None",info ="Select a LoRA to apply")
                        selected_lora_2 =gr .Dropdown (label ="Select LoRA 2",choices =[name for name ,_ in lora_options ],value ="None",info ="Select a second LoRA to apply")
                        selected_lora_3 =gr .Dropdown (label ="Select LoRA 3",choices =[name for name ,_ in lora_options ],value ="None",info ="Select a third LoRA to apply")
                        selected_lora_4 =gr .Dropdown (label ="Select LoRA 4",choices =[name for name ,_ in lora_options ],value ="None",info ="Select a fourth LoRA to apply")
                    with gr .Column ():
                        with gr .Row ():
                             lora_refresh_btn =gr .Button (value ="🔄 Refresh",scale =1 )
                             lora_folder_btn =gr .Button (value ="📁 Open Folder",scale =1 )
                        lora_scale_1 =gr .Slider (label ="LoRA 1 Scale",minimum =0.0 ,maximum =9.0 ,value =1.0 ,step =0.01 ,info ="Adjust the strength of the LoRA 1 effect")
                        lora_scale_2 =gr .Slider (label ="LoRA 2 Scale",minimum =0.0 ,maximum =9.0 ,value =1.0 ,step =0.01 ,info ="Adjust the strength of the LoRA 2 effect")
                        lora_scale_3 =gr .Slider (label ="LoRA 3 Scale",minimum =0.0 ,maximum =9.0 ,value =1.0 ,step =0.01 ,info ="Adjust the strength of the LoRA 3 effect")
                        lora_scale_4 =gr .Slider (label ="LoRA 4 Scale",minimum =0.0 ,maximum =9.0 ,value =1.0 ,step =0.01 ,info ="Adjust the strength of the LoRA 4 effect")



        with gr .Column ():
            preview_image =gr .Image (label ="Next Latents",height =200 ,visible =False )
            result_video =gr .Video (label ="Finished Frames",autoplay =True ,show_share_button =True ,height =512 ,loop =True )
            video_info =gr .HTML ("<div id='video-info'>Generate a video to see information</div>")
            gr .Markdown ('''
            **Notes:**
            - **Original Model:** Generates video back-to-front. Start frame appears late, end frame early. Uses overlapping windows. Timestamp prompts `[secs]` relate to *final* video time, estimated by section duration (see info above prompt).
            - **FramePack F1 Model:** Extends video from the start frame. Start frame is always present. End frame is ignored or has minimal effect. Timestamp prompts `[secs]` relate to *final* video time.
            ''')
            progress_desc =gr .Markdown ('',elem_classes ='no-generating-animation')
            progress_bar =gr .HTML ('',elem_classes ='no-generating-animation')
            timing_display =gr .Markdown ("",label ="Time Information",elem_classes ='no-generating-animation')

            gr .Markdown ("### Presets (Includes selected model)")
            with gr .Row ():
                preset_dropdown =gr .Dropdown (label ="Select Preset",choices =scan_presets (),value =load_last_used_preset_name ()or "Default")
                preset_load_button =gr .Button (value ="Load Preset")
                preset_refresh_button =gr .Button (value ="🔄 Refresh")
            with gr .Row ():
                preset_save_name =gr .Textbox (label ="Save Preset As",placeholder ="Enter preset name...")
                preset_save_button =gr .Button (value ="Save Current Settings")
            preset_status_display =gr .Markdown ("")

            gr .Markdown ("### Folder Options")
            with gr .Row ():
                open_outputs_btn =gr .Button (value ="Open Generations Folder")
                open_batch_outputs_btn =gr .Button (value ="Open Batch Outputs Folder")

            video_quality =gr .Radio (
            label ="Video Quality",
            choices =["high","medium","low","web_compatible"],
            value ="high",
            info ="High: Best quality, Medium: Balanced, Low: Smallest file size, Web Compatible: Best browser compatibility (MP4+WebM)"
            )

            gr .Markdown ("### RIFE Frame Interpolation (MP4 Only)")
            with gr .Row ():
                rife_enabled =gr .Checkbox (label ="Enable RIFE (2x/4x FPS)",value =False ,info ="Increases FPS of generated MP4s using RIFE. Saves as '[filename]_extra_FPS.mp4'")
                rife_multiplier =gr .Radio (choices =["2x FPS","4x FPS"],label ="RIFE FPS Multiplier",value ="2x FPS",info ="Choose the frame rate multiplication factor.")

            gr .Markdown ("### Additional Export Formats")
            gr .Markdown ("Select additional formats to export alongside MP4/WebM:")
            with gr .Row ():
                export_gif =gr .Checkbox (label ="Export as GIF",value =False ,info ="Save animation as GIF")
                export_apng =gr .Checkbox (label ="Export as APNG",value =False ,info ="Save animation as Animated PNG")
                export_webp =gr .Checkbox (label ="Export as WebP",value =False ,info ="Save animation as WebP")
            with gr .Row ():
                save_temp_metadata = gr.Checkbox(label="Save Temporary Metadata", value=True, info="If generation fails or is cancelled, metadata up to that point is kept as a temporary file. This temp file is deleted on successful generation of final metadata.")

    preset_components_list =[
    model_selector ,
    use_multiline_prompts ,save_metadata ,save_individual_frames ,save_intermediate_frames ,save_last_frame ,
    batch_skip_existing ,batch_save_metadata ,batch_use_multiline_prompts ,batch_save_individual_frames ,batch_save_intermediate_frames ,batch_save_last_frame ,
    num_generations ,resolution ,teacache_threshold ,seed ,use_random_seed ,n_prompt ,fps ,total_second_length ,
    latent_window_size ,steps ,gs ,cfg ,rs ,
    target_width_slider ,target_height_slider ,
    selected_lora_1 ,selected_lora_2 ,selected_lora_3 ,selected_lora_4 ,
    lora_scale_1 ,lora_scale_2 ,lora_scale_3 ,lora_scale_4 ,
    gpu_memory_preservation ,video_quality ,rife_enabled ,rife_multiplier ,export_gif ,export_apng ,export_webp,
    save_temp_metadata
    ]
    component_names_for_preset =[
    "model_selector",
    "use_multiline_prompts","save_metadata","save_individual_frames","save_intermediate_frames","save_last_frame",
    "batch_skip_existing","batch_save_metadata","batch_use_multiline_prompts","batch_save_individual_frames","batch_save_intermediate_frames","batch_save_last_frame",
    "num_generations","resolution","teacache_threshold","seed","use_random_seed","n_prompt","fps","total_second_length",
    "latent_window_size","steps","gs","cfg","rs",
    "target_width","target_height",
    "selected_lora_1","selected_lora_2","selected_lora_3","selected_lora_4",
    "lora_scale_1","lora_scale_2","lora_scale_3","lora_scale_4",
    "gpu_memory_preservation","video_quality","rife_enabled","rife_multiplier","export_gif","export_apng","export_webp",
    "save_temp_metadata"
    ]

    def save_preset_action (name :str ,*values ):

        if not name :
            return gr .update (),gr .update (value ="Preset name cannot be empty.")

        preset_data ={}
        if len (values )!=len (component_names_for_preset ):
             msg =f"Error: Mismatched number of values ({len(values)}) and component names ({len(component_names_for_preset)})."
             print (msg )
             return gr .update (),gr .update (value =msg )

        for i ,comp_name in enumerate (component_names_for_preset ):
             preset_data [comp_name ]=values [i ]

        preset_path =get_preset_path (name )
        try :
            with open (preset_path ,'w',encoding ='utf-8')as f :
                json .dump (preset_data ,f ,indent =4 )
            save_last_used_preset_name (name )
            presets =scan_presets ()
            status_msg =f"Preset '{name}' saved successfully."
            print (status_msg )

            return gr .update (choices =presets ,value =name ),gr .update (value =status_msg )
        except Exception as e :
            error_msg =f"Error saving preset '{name}': {e}"
            print (error_msg )

            return gr .update (),gr .update (value =error_msg )

    def load_preset_action (name :str ,progress =gr .Progress () ):

        global active_model_name 

        progress (0 ,desc =f"Loading preset '{name}'...")
        preset_data =load_preset_data (name )
        if preset_data is None :
             error_msg =f"Failed to load preset '{name}'."
             print (error_msg )

             return [gr .update ()for _ in preset_components_list ]+[gr .update (value =error_msg ),gr .update (value =f"Active model: **{active_model_name}**")]+[gr .update ()]

        model_status_update =f"Active model: **{active_model_name}**"
        preset_model =preset_data .get ("model_selector",DEFAULT_MODEL_NAME )
        if preset_model !=active_model_name :
             progress (0.1 ,desc =f"Preset requires model '{preset_model}'. Switching...")
             print (f"Preset '{name}' requires model switch to '{preset_model}'.")
             try :

                 new_active_model ,switch_status_msg =switch_active_model (preset_model ,progress =progress )

                 active_model_name =new_active_model 
                 model_status_update =f"Status: {switch_status_msg}"

                 preset_data ["model_selector"]=active_model_name 
                 print (f"Switch status: {switch_status_msg}")
                 if "Error"in switch_status_msg or "CRITICAL"in switch_status_msg :

                      pass 
             except Exception as switch_err :
                  error_msg =f"Error switching model for preset: {switch_err}"
                  print (error_msg )
                  traceback .print_exc ()
                  model_status_update =f"Error switching model: {error_msg}. Model remains '{active_model_name}'."
                  preset_data ["model_selector"]=active_model_name 
        else :
             progress (0.5 ,desc ="Model already correct. Loading settings...")
             print (f"Preset '{name}' uses the currently active model '{active_model_name}'.")
             model_status_update =f"Active model: **{active_model_name}**"

        updates =[]
        loaded_values :Dict [str ,Any ]={}
        available_loras =[lora_name for lora_name ,_ in scan_lora_files ()]

        # Multi-LoRA: default missing slots to 'None' and 1.0
        for i in range(1, N_LORA+1):
            if f"selected_lora_{i}" not in preset_data:
                preset_data[f"selected_lora_{i}"] = "None"
            if f"lora_scale_{i}" not in preset_data:
                preset_data[f"lora_scale_{i}"] = 1.0

        for i ,comp_name in enumerate (component_names_for_preset ):
            target_component =preset_components_list [i ]

            comp_initial_value =getattr (target_component ,'value',None )

            if comp_name in preset_data :
                value =preset_data [comp_name ]

                if comp_name.startswith("selected_lora_"):
                    if value not in available_loras :
                        print (f"Preset Warning: Saved LoRA '{value}' not found. Setting LoRA to 'None'.")
                        value ="None"
                    updates .append (gr .update (value =value ))
                elif comp_name =="model_selector":

                     updates .append (gr .update (value =active_model_name ))
                     value =active_model_name 
                elif comp_name =="resolution":

                     _VALID_RESOLUTIONS =["1440","1320","1200","1080","960","840","720","640","480","320","240"]
                     if value not in _VALID_RESOLUTIONS :
                          print (f"Preset Warning: Saved resolution '{value}' not valid. Using default ('640').")
                          value ="640"
                     updates .append (gr .update (value =value ))
                else :

                    updates .append (gr .update (value =value ))
                loaded_values [comp_name ]=value 
            else :

                 print (f"Preset Warning: Key '{comp_name}' not found in preset '{name}'. Keeping current value.")
                 updates .append (gr .update ())
                 loaded_values [comp_name ]=comp_initial_value 

        if len (updates )!=len (preset_components_list ):
             error_msg =f"Error applying preset '{name}': Mismatch in component update count."
             print (error_msg )

             return [gr .update ()for _ in preset_components_list ]+[gr .update (value =error_msg ),gr .update (value =f"Active model: **{active_model_name}**")]+[gr .update ()]

        vid_len =loaded_values .get ('total_second_length',5 )
        fps_val =loaded_values .get ('fps',30 )
        win_size =loaded_values .get ('latent_window_size',9 )

        info_text =update_iteration_info (vid_len ,fps_val ,win_size )
        info_update =gr .update (value =info_text )

        save_last_used_preset_name (name )
        status_msg =f"Preset '{name}' loaded."
        print (status_msg )
        preset_status_update =gr .update (value =status_msg )
        model_status_update_gr =gr .update (value =model_status_update )

        return updates +[preset_status_update ,model_status_update_gr ,info_update ]

    def refresh_presets_action ():

        presets =scan_presets ()
        last_used =load_last_used_preset_name ()
        selected =last_used if last_used in presets else "Default"
        return gr .update (choices =presets ,value =selected )

    lora_refresh_btn .click (fn =lambda : [refresh_loras() for _ in range(N_LORA)],outputs =[selected_lora_1, selected_lora_2, selected_lora_3, selected_lora_4])
    lora_folder_btn .click (fn =lambda :open_folder (loras_folder ),inputs =None ,outputs =None )

    ips =[input_image ,end_image ,prompt ,n_prompt ,seed ,use_random_seed ,num_generations ,total_second_length ,latent_window_size ,steps ,cfg ,gs ,rs ,gpu_memory_preservation ,teacache_threshold ,video_quality ,export_gif ,export_apng ,export_webp ,save_metadata ,resolution ,fps ,
    lora_scale_1, lora_scale_2, lora_scale_3, lora_scale_4,
    selected_lora_1, selected_lora_2, selected_lora_3, selected_lora_4,
    use_multiline_prompts ,save_individual_frames ,save_intermediate_frames ,save_last_frame ,rife_enabled ,rife_multiplier ,
    model_selector ,
    target_width_slider ,target_height_slider ,
    save_temp_metadata # Add the checkbox component here
    ]
    start_button .click (fn =process ,inputs =ips ,outputs =[result_video ,preview_image ,progress_desc ,progress_bar ,start_button ,end_button ,seed ,timing_display ])

    end_button .click (fn =end_process ,inputs =None ,outputs =[start_button ,end_button ,batch_start_button ,batch_end_button ],cancels =[])

    open_outputs_btn .click (fn =lambda :open_folder (outputs_folder ),inputs =None ,outputs =None )
    open_batch_outputs_btn .click (fn =lambda :open_folder (outputs_batch_folder ),inputs =None ,outputs =None )

    batch_folder_status_text =gr .Textbox (visible =False )
    open_batch_input_folder .click (fn =lambda x :open_folder (x )if x and os .path .isdir (x )else f"Folder not found or invalid: {x}",inputs =[batch_input_folder ],outputs =[batch_folder_status_text ])
    open_batch_end_folder .click (fn =lambda x :open_folder (x )if x and os .path .isdir (x )else f"Folder not found or invalid: {x}",inputs =[batch_end_frame_folder ],outputs =[batch_folder_status_text ])
    open_batch_output_folder .click (fn =lambda x :open_folder (x if x and os .path .isdir (x )else outputs_batch_folder ),inputs =[batch_output_folder ],outputs =[batch_folder_status_text ])

    batch_ips =[batch_input_folder ,batch_output_folder ,batch_end_frame_folder ,batch_prompt ,n_prompt ,seed ,use_random_seed ,
    total_second_length ,latent_window_size ,steps ,cfg ,gs ,rs ,gpu_memory_preservation ,
    teacache_threshold ,video_quality ,export_gif ,export_apng ,export_webp ,batch_skip_existing ,
    batch_save_metadata ,num_generations ,resolution ,fps ,
    lora_scale_1, lora_scale_2, lora_scale_3, lora_scale_4,
    selected_lora_1, selected_lora_2, selected_lora_3, selected_lora_4,
    batch_use_multiline_prompts ,
    batch_save_individual_frames ,batch_save_intermediate_frames ,batch_save_last_frame ,
    rife_enabled ,rife_multiplier ,
    model_selector ,
    target_width_slider ,target_height_slider ,
    batch_auto_resize,
    save_temp_metadata # Add the checkbox component here
    ]
    batch_start_button .click (fn =batch_process ,inputs =batch_ips ,outputs =[result_video ,preview_image ,progress_desc ,progress_bar ,batch_start_button ,batch_end_button ,seed ,timing_display ])

    batch_end_button .click (fn =end_process ,inputs =None ,outputs =[start_button ,end_button ,batch_start_button ,batch_end_button ],cancels =[])

    preset_save_button .click (
    fn =save_preset_action ,
    inputs =[preset_save_name ]+preset_components_list ,
    outputs =[preset_dropdown ,preset_status_display ]
    )

    preset_load_button .click (
    fn =load_preset_action ,
    inputs =[preset_dropdown ],
    outputs =preset_components_list +[preset_status_display ,model_status ,iteration_info_display ]
    )
    preset_refresh_button .click (
    fn =refresh_presets_action ,
    inputs =[],
    outputs =[preset_dropdown ]
    )

    auto_set_lws_button .click (
    fn =auto_set_window_size ,
    inputs =[fps ,latent_window_size ],
    outputs =[latent_window_size ]
    )

    iteration_info_inputs =[total_second_length ,fps ,latent_window_size ]

    def update_iter_info_ui (vid_len ,fps_val ,win_size ,current_model ):

        return update_iteration_info (vid_len ,fps_val ,win_size )

    model_selector .change (
    fn =update_iter_info_ui ,
    inputs =iteration_info_inputs +[model_selector ],
    outputs =iteration_info_display ,
    queue =False 
    )

    for comp in iteration_info_inputs :
        comp .change (
        fn =update_iter_info_ui ,
        inputs =iteration_info_inputs +[model_selector ],
        outputs =iteration_info_display ,
        queue =False 
        )

    model_selector .change (
    fn =switch_active_model ,
    inputs =[model_selector ],
    outputs =[model_selector ,model_status ],
    show_progress ="full"
    )

    load_from_metadata_button .click (
    fn =load_settings_from_metadata_file ,
    inputs =[metadata_file_input ],
    outputs =preset_components_list +[metadata_load_status ,model_status ,iteration_info_display ],
    show_progress ="full"
    )

    input_image .upload (
    fn =update_target_dimensions_from_image ,
    inputs =[input_image ,resolution ],
    outputs =[target_width_slider ,target_height_slider ],
    queue =False 
    )
    input_image .clear (
    fn =update_target_dimensions_from_image ,
    inputs =[input_image ,resolution ],
    outputs =[target_width_slider ,target_height_slider ],
    queue =False 
    )

    video_info_js ="""
    function updateVideoInfo() {
        const videoResultDiv = document.querySelector('#result_video');
        if (!videoResultDiv) return;
        const videoElement = videoResultDiv.querySelector('video');

        if (videoElement && videoElement.currentSrc && videoElement.currentSrc.startsWith('http')) { // Check if src is loaded
            const infoDiv = document.getElementById('video-info');
            if (!infoDiv) return;
            const displayInfo = () => {
                if (videoElement.videoWidth && videoElement.videoHeight && videoElement.duration && isFinite(videoElement.duration)) {
                     const format = videoElement.currentSrc ? videoElement.currentSrc.split('.').pop().toUpperCase().split('?')[0] : 'N/A'; // Handle potential query strings
                     infoDiv.innerHTML = `<p>Resolution: ${videoElement.videoWidth}x${videoElement.videoHeight} | Duration: ${videoElement.duration.toFixed(2)}s | Format: ${format}</p>`;
                } else if (videoElement.readyState < 1) {
                     infoDiv.innerHTML = '<p>Loading video info...</p>';
                } else {
                     // Sometimes duration might be Infinity initially
                     infoDiv.innerHTML = `<p>Resolution: ${videoElement.videoWidth}x${videoElement.videoHeight} | Duration: Loading... | Format: ${videoElement.currentSrc ? videoElement.currentSrc.split('.').pop().toUpperCase().split('?')[0] : 'N/A'}</p>`;
                }
            };
            // Use 'loadeddata' or 'durationchange' as they often fire when duration is known
            videoElement.removeEventListener('loadeddata', displayInfo);
            videoElement.addEventListener('loadeddata', displayInfo);
            videoElement.removeEventListener('durationchange', displayInfo);
            videoElement.addEventListener('durationchange', displayInfo);

            // Initial check if data is already available
            if (videoElement.readyState >= 2) { // HAVE_CURRENT_DATA or more
                displayInfo();
            } else {
                 infoDiv.innerHTML = '<p>Loading video info...</p>';
            }
        } else {
             const infoDiv = document.getElementById('video-info');
             if (infoDiv) infoDiv.innerHTML = "<div>Generate a video to see information</div>";
        }
    }
    // Debounce function
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    };

    const debouncedUpdateVideoInfo = debounce(updateVideoInfo, 250); // Update max 4 times per second

    // Use MutationObserver to detect when the video src changes
    const observerCallback = function(mutationsList, observer) {
        for(const mutation of mutationsList) {
            // Check if the result_video div or its children changed, or if attributes changed (like src)
             if (mutation.type === 'childList' || mutation.type === 'attributes') {
                 const videoResultDiv = document.querySelector('#result_video');
                 // Check if the mutation target is the video element itself or its container
                 if (videoResultDiv && (mutation.target === videoResultDiv || videoResultDiv.contains(mutation.target) || mutation.target.tagName === 'VIDEO')) {
                    debouncedUpdateVideoInfo(); // Use debounced update
                    break; // No need to check other mutations for this change
                 }
            }
        }
    };
    const observer = new MutationObserver(observerCallback);
    // Observe changes in the body, looking for subtree modifications and attribute changes
    observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['src'] });

    // Initial call on load
    if (document.readyState === 'complete') {
      // Already loaded
      // debouncedUpdateVideoInfo(); // Call directly if loaded
    } else {
      // Wait for DOM content
      document.addEventListener('DOMContentLoaded', debouncedUpdateVideoInfo);
    }
    // Add listener for Gradio updates as well
    if (typeof window.gradio_config !== 'undefined') {
        window.addEventListener('gradio:update', debouncedUpdateVideoInfo);
    }
    """
    result_video .elem_id ="result_video"

    def apply_preset_and_init_info_on_startup ():
        global active_model_name ,transformer 

        print ("Applying preset and initializing info on startup...")
        initial_values ={}
        for i ,comp in enumerate (preset_components_list ):

             default_value =getattr (comp ,'value',None )

             if component_names_for_preset [i ]=="target_width"and default_value is None :
                 default_value =640 
             if component_names_for_preset [i ]=="target_height"and default_value is None :
                 default_value =640 
             initial_values [component_names_for_preset [i ]]=default_value 

        initial_values ["model_selector"]=active_model_name 

        if "target_width"not in initial_values :initial_values ["target_width"]=640 
        if "target_height"not in initial_values :initial_values ["target_height"]=640 
        # Multi-LoRA: default missing slots to 'None' and 1.0
        for i in range(1, N_LORA+1):
            if f"selected_lora_{i}" not in initial_values:
                initial_values[f"selected_lora_{i}"] = "None"
            if f"lora_scale_{i}" not in initial_values:
                initial_values[f"lora_scale_{i}"] = 1.0
        create_default_preset_if_needed (initial_values )

        preset_to_load =load_last_used_preset_name ()
        available_presets =scan_presets ()
        if preset_to_load not in available_presets :
            print (f"Last used preset '{preset_to_load}' not found or invalid, loading 'Default'.")
            preset_to_load ="Default"
        else :
            print (f"Loading last used preset: '{preset_to_load}'")

        preset_data =load_preset_data (preset_to_load )
        if preset_data is None and preset_to_load !="Default":
             print (f"Failed to load '{preset_to_load}', attempting to load 'Default'.")
             preset_to_load ="Default"
             preset_data =load_preset_data (preset_to_load )
        elif preset_data is None and preset_to_load =="Default":
             print (f"Critical Error: Failed to load 'Default' preset data. Using hardcoded component defaults.")

             preset_data =initial_values 

        startup_model_status =f"Active model: **{active_model_name}**"
        preset_model =preset_data .get ("model_selector",DEFAULT_MODEL_NAME )
        if preset_model !=active_model_name :
             print (f"Startup: Preset '{preset_to_load}' requires model '{preset_model}'. Switching...")

             startup_progress =gr .Progress ()
             startup_progress (0 ,desc =f"Loading model '{preset_model}' for startup preset...")
             try :

                 new_active_model ,switch_status_msg =switch_active_model (preset_model ,progress =startup_progress )
                 active_model_name =new_active_model 
                 startup_model_status =f"Status: {switch_status_msg}"
                 preset_data ["model_selector"]=active_model_name 
                 print (f"Startup model switch status: {switch_status_msg}")
             except Exception as startup_switch_err :
                  error_msg =f"Startup Error switching model: {startup_switch_err}"
                  print (error_msg )
                  traceback .print_exc ()
                  startup_model_status =f"Error: {error_msg}. Model remains '{active_model_name}'."
                  preset_data ["model_selector"]=active_model_name 
        else :
             print (f"Startup: Preset '{preset_to_load}' uses initially loaded model '{active_model_name}'.")
             startup_model_status =f"Active model: **{active_model_name}**"

        preset_updates =[]
        loaded_values_startup :Dict [str ,Any ]={}
        available_loras_startup =[lora_name for lora_name ,_ in scan_lora_files ()]

        for i ,comp_name in enumerate (component_names_for_preset ):
            comp_initial_value =initial_values .get (comp_name )
            value_to_set =comp_initial_value 
            if comp_name in preset_data :
                 value_from_preset =preset_data [comp_name ]

                 if comp_name =="model_selector":
                      value_to_set =active_model_name 
                 elif comp_name =="selected_lora":
                      if value_from_preset not in available_loras_startup :
                           print (f"Startup Warning: Preset LoRA '{value_from_preset}' not found. Setting LoRA to 'None'.")
                           value_to_set ="None"
                      else :
                           value_to_set =value_from_preset 
                 elif comp_name =="resolution":

                      _VALID_RESOLUTIONS =["1440","1320","1200","1080","960","840","720","640","480","320","240"]
                      if value_from_preset not in _VALID_RESOLUTIONS :
                           print (f"Startup Warning: Preset resolution '{value_from_preset}' not valid. Using default ('640').")
                           value_to_set ="640"
                      else :
                           value_to_set =value_from_preset 
                 else :

                      value_to_set =value_from_preset 
            else :
                 print (f"Startup Warning: Key '{comp_name}' missing in '{preset_to_load}'. Using component's default.")
                 value_to_set =comp_initial_value 

            preset_updates .append (gr .update (value =value_to_set ))
            loaded_values_startup [comp_name ]=value_to_set 

        initial_vid_len =loaded_values_startup .get ('total_second_length',5 )
        initial_fps =loaded_values_startup .get ('fps',30 )
        initial_win_size =loaded_values_startup .get ('latent_window_size',9 )

        initial_info_text =update_iteration_info (initial_vid_len ,initial_fps ,initial_win_size )

        return [gr .update (choices =available_presets ,value =preset_to_load )]+preset_updates +[startup_model_status ,initial_info_text ]

    block .load (
    fn =apply_preset_and_init_info_on_startup ,
    inputs =[],
    outputs =[preset_dropdown ]+preset_components_list +[model_status ,iteration_info_display ]
    )

    block .load (None ,None ,None ,js =video_info_js )

def get_available_drives ():

    available_paths =[]
    if platform .system ()=="Windows":
        import string 
        from ctypes import windll 
        drives =[]
        bitmask =windll .kernel32 .GetLogicalDrives ()
        for letter in string .ascii_uppercase :
            if bitmask &1 :drives .append (f"{letter}:\\")
            bitmask >>=1 
        available_paths =drives 
    elif platform .system ()=="Darwin":
         available_paths =["/","/Volumes"]
    else :
        available_paths =["/","/mnt","/media"]

        home_dir =os .path .expanduser ("~")
        if home_dir not in available_paths :
            available_paths .append (home_dir )

    existing_paths =[p for p in available_paths if os .path .exists (p )and os .path .isdir (p )]

    cwd =os .getcwd ()
    script_dir =os .path .dirname (os .path .abspath (__file__ ))
    if cwd not in existing_paths :existing_paths .append (cwd )
    if script_dir not in existing_paths :existing_paths .append (script_dir )

    for folder in [outputs_folder ,loras_folder ,presets_folder ,outputs_batch_folder ]:
         abs_folder =os .path .abspath (folder )
         if abs_folder not in existing_paths and os .path .exists (abs_folder ):
             existing_paths .append (abs_folder )

    print (f"Allowed Gradio paths: {list(set(existing_paths))}")
    return list (set (existing_paths ))

block .launch (
share =args .share ,
inbrowser =True ,
allowed_paths =get_available_drives ()
)
