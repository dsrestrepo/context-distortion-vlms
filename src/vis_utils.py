####### VISUALIZATION UTILITIES #######
# Most of the code is inspired and adapted from: https://github.com/zjysteven/VLM-Visualizer
# many are copied from https://github.com/mattneary/attention/blob/master/attention/attention.py

# Go and check the original repository for more details! THANKS!

from io import BytesIO
import requests
import os
import numpy as np
import cv2
from IPython.display import display, HTML
import html
import weasyprint
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

#from utils import (
#    load_image, 
#    aggregate_llm_attention, aggregate_vit_attention,
#    heterogenous_stack,
#    show_mask_on_image
#)



def aggregate_llm_attention(attn):
    '''Extract average attention vector'''
    avged = []
    for layer in attn:
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = torch.concat((
            # We zero the first entry because it's what's called
            # null attention (https://aclanthology.org/W19-4808.pdf)
            torch.tensor([0.]),
            # usually there's only one item in attns_per_head but
            # on the first generation, there's a row for each token
            # in the prompt as well, so take [-1]
            attns_per_head[-1][1:].cpu(),
            # attns_per_head[-1].cpu(),
            # add zero for the final generated token, which never
            # gets any attention
            torch.tensor([0.]),
        ))
        avged.append(vec / vec.sum())
    return torch.stack(avged).mean(dim=0)


def aggregate_vit_attention(attn, select_layer=-2, all_prev_layers=True):
    '''Assuming LLaVA-style `select_layer` which is -2 by default'''
    if all_prev_layers:
        avged = []
        for i, layer in enumerate(attn):
            if i > len(attn) + select_layer:
                break
            layer_attns = layer.squeeze(0)
            attns_per_head = layer_attns.mean(dim=0) # average over heads
            vec = attns_per_head[1:, 1:].cpu() # the first token is <CLS>
            avged.append(vec / vec.sum(-1, keepdim=True)) # normalize
        return torch.stack(avged).mean(dim=0)
    else:
        layer = attn[select_layer]
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = attns_per_head[1:, 1:].cpu()
        return vec / vec.sum(-1, keepdim=True)


def heterogenous_stack(vecs):
    '''Pad vectors with zeros then stack'''
    max_length = max(v.shape[0] for v in vecs)
    return torch.stack([
        torch.concat((v, torch.zeros(max_length - v.shape[0])))
        for v in vecs
    ])


def load_image(image_path_or_url):
    if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_path_or_url).convert('RGB')
    return image


def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_HSV)
    hm = np.float32(heatmap) / 255
    cam = hm + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam), heatmap



def construct_llm_attention_matrix(output_attentions):
    """
    Constructs the LLM attention matrix from the given output attentions.

    Parameters:
        output_attentions (list or tensor): The attention outputs from the model. 
            It is assumed that output_attentions[0] contains the prompt attentions
            and that the rest correspond to additional generations.

    Returns:
        torch.Tensor: The LLM attention matrix of shape [N, N], where N is the total 
        number of input tokens (both image and text) plus output tokens.
    """
    # Aggregate prompt attentions for generation 0
    aggregated_prompt_attention = []
    for layer in output_attentions[0]:
        # Remove batch dimension
        layer_attns = layer.squeeze(0)
        
        # Average attention across all heads
        attns_per_head = layer_attns.mean(dim=0)
        
        # Process the attention matrix:
        # Exclude the last token and clone to CPU for modification.
        cur = attns_per_head[:-1].cpu().clone()
        # Following the practice in `aggregate_llm_attention`:
        # Set the attention for the bos token (first column) to zero for all tokens (except bos itself)
        cur[1:, 0] = 0.
        # Normalize attentions (without the bos token)
        cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
        
        aggregated_prompt_attention.append(cur)
    
    # Average the aggregated attentions from all layers for generation 0
    aggregated_prompt_attention = torch.stack(aggregated_prompt_attention).mean(dim=0)
    
    # Combine prompt attentions with additional generation attentions.
    # Note: `heterogenous_stack` and `aggregate_llm_attention` are assumed to be defined externally.
    llm_attn_matrix = heterogenous_stack(
        [torch.tensor([1])] +
        list(aggregated_prompt_attention) +
        list(map(aggregate_llm_attention, output_attentions))
    )
    
    return llm_attn_matrix



############################################
##  Calculate & Visualize Image Attention ##
############################################

def calculate_image_attention(
    llm_attn_matrix,         # 2D array (full seq_len x seq_len) that merges the average attentions from LLM
    vit_attn_matrix,         # 2D array [n_image_tokens, n_image_tokens]
    tokenizer,
    image,
    output_ids,
    vision_token_start,
    vision_token_end,
    output_token_start,
    output_token_end,
    patch_grid_size=24,
    num_images_per_row=8,
    overlay=True,
    return_attentions=True,
    display=True,
    normalize=True
):
    """
    - llm_attn_matrix: (total_seq_len, total_seq_len) final LLM attn, typically after averaging layers & heads.
    - vit_attn_matrix: Each entry is the aggregated patch-level attention from the ViT model. Shape = (num_image_tokens, num_image_tokens).
    - tokenizer: Your LLaVA or LLaVA-Med tokenizer to decode tokens or do any conversions.
    - image: The original PIL (or np.array) image to overlay.
    - output_ids: List of generated token IDs (for labeling).
    - vision_token_start, vision_token_end: The slice of positions in the LLM's sequence that correspond to image tokens.
    - output_token_start, output_token_end: The slice that correspond to *generated* tokens.
    - patch_grid_size: Usually 24 (336 / 14).
    - overlay: If True, overlay the heatmap on top of the original image. If False, just show raw heatmap.
    - return_attentions: if True, returns a list of per‐generated‐token patch‐attention arrays.
    - display: If True, display the image attention in the notebook.
    - normalize: If True, normalize the patch attention to [0, 1].
    """

    # Convert the PIL image to numpy BGR array.
    # If image is PIL, do: np_img = np.array(image)[:, :, ::-1]
    # If image is a cv2 image, adapt accordingly.
    np_img = np.array(image)[:, :, ::-1]

    generated_token_indices = list(range(output_token_start, output_token_end))
    num_generated_tokens = len(generated_token_indices)

    
    # If we're going to display, set up the figure once
    if display:
        num_rows = num_generated_tokens // num_images_per_row
        if num_generated_tokens % num_images_per_row != 0:
            num_rows += 1
        fig, axes = plt.subplots(
            num_rows, 
            num_images_per_row, 
            figsize=(10, 10 * num_rows / num_images_per_row), 
            dpi=150
        )
        plt.subplots_adjust(wspace=0.05, hspace=0.2)

        # If there's only one row, make axes into a list
        if num_rows == 1 and num_images_per_row == 1:
            axes = np.array([axes])
        elif num_rows == 1:
            axes = axes[np.newaxis, :]


    

    # To store patch-level attention arrays
    patch_attn_list = []

    # Iterate over each generated token
    #for idx, ax in enumerate(axes.flatten()):
    for idx in range(num_generated_tokens):
        
        # The position of this token in the LLM’s sequence
        cur_token_pos = generated_token_indices[idx]
        # Pull out how that single token attends to all vision tokens
        # shape: (vision_token_count,)
        attn_weights_over_vis = llm_attn_matrix[cur_token_pos, vision_token_start:vision_token_end]
        
        # Multiply each vision-token weight by the corresponding patch attention in the ViT
        # vit_attn_matrix should have shape [n_vision_tokens, n_vision_tokens].
        # attn_weights_over_vis has shape [n_vision_tokens].
        # For each patch i, we sum over all vision tokens
        patch_attention = []
        for patch_i, patch_vit_map in enumerate(vit_attn_matrix):
            # patch_vit_map might be shape [patch_grid_size^2].
            patch_attention.append(patch_vit_map * attn_weights_over_vis[patch_i])
        patch_attention = torch.stack(patch_attention, dim=0).sum(dim=0)

        # Reshape to (patch_grid_size, patch_grid_size)
        patch_attention = patch_attention.reshape(patch_grid_size, patch_grid_size)
        if normalize:
            patch_attention = patch_attention / (patch_attention.max() + 1e-8)

        # Optionally store this patch attention map
        if return_attentions:
            patch_attn_list.append(patch_attention.detach().cpu().numpy())

        if display:
            # Identify the correct axis
            row_idx = idx // num_images_per_row
            col_idx = idx % num_images_per_row
            ax = axes[row_idx, col_idx] if len(axes.shape) == 2 else axes[row_idx][col_idx]
            
            # Now scale up to full image resolution
            patch_attention_t = patch_attention.unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
            # Resample to match the actual image shape
            big_attn = F.interpolate(
                patch_attention_t, 
                size=(image.size[1], image.size[0]), 
                mode='bicubic', 
                align_corners=False
            ).squeeze().cpu().numpy()
    
            # Overlay image and attention
            img_with_attn, heatmap = show_mask_on_image(np_img, big_attn)
            ax.imshow(heatmap if not overlay else img_with_attn)
    
            # Label with the token text
            token_str = tokenizer.decode([output_ids[idx]], add_special_tokens=False).strip()
            ax.set_title(token_str, fontsize=7, pad=1)
            ax.axis("off")

    if display:
        total_plots = num_rows * num_images_per_row
        if total_plots > num_generated_tokens:
            # For each unused subplot, remove it
            for leftover_idx in range(num_generated_tokens, total_plots):
                row_idx = leftover_idx // num_images_per_row
                col_idx = leftover_idx % num_images_per_row
                fig.delaxes(axes[row_idx, col_idx])

        plt.suptitle("Image Attention by Generated Token", fontsize=10)
        plt.show()


    if return_attentions:
        return patch_attn_list




############################################
##  Calculate & Visualize Text Attention  ##
############################################


def clean_token(token):
    """
    Remove the subword marker (e.g., "▁") from the token and adjust spacing.
    """
    if token.startswith("▁"):
        return token[1:]
    if token == '<0x0A>':
        return " "
    return token

def html_to_png(html_str, output_png="attention.png"):
    "Convert attention to png"
    html_obj = weasyprint.HTML(string=html_str)
    html_obj.write_png(output_png)

def build_text_html_line(attn_row, input_tokens, gen_token):
    # Normalize attention for color mapping
    norm_attn = (attn_row - attn_row.min()) / (attn_row.max() - attn_row.min() + 1e-8)
    token_html = ""
    for token, weight in zip(input_tokens, norm_attn):
        safe_token = html.escape(token)
        token_html += f'<span style="background-color: rgba(0,0,255,{weight:.2f}); padding:1px; margin:1px;">{safe_token}</span>'
    safe_gen_token = html.escape(gen_token)
    return f"<div style='font-family: monospace; margin-bottom: 0px;'>{token_html}</div>"
    #return f"<div style='font-family: monospace; margin-bottom: 0px;'><strong>{safe_gen_token}:</strong> {token_html}</div>"


def calculate_text_attention(
    llm_attn_matrix,
    text_token_indices,
    output_token_indices,
    tokenizer,
    input_ids,
    output_ids,
    save=False,
    display_attention=True,
    normalize=True
):
    """
    Creates and visualizes a heatmap:
      rows = each generated token
      cols = each input text token
    using a single 2D attention matrix (llm_attn_matrix).

    Args:
      llm_attn_matrix: shape [total_seq_len, total_seq_len], the LLM attention 
                       already averaged across heads/layers as you did for image attn.
      text_token_indices: list of positions for the "input text" in the prompt 
                          (excluding image tokens).
      output_token_indices: list of positions for the "generated tokens" in the final seq.
      tokenizer: to decode the input_ids / output_ids for labeling.
      input_ids: the original prompt tokens (including text + image placeholder).
      output_ids: the newly generated tokens, same as the ones you used for image attn.
    """

    attention_array = []

    for out_i, out_token_pos in enumerate(output_token_indices):
        # Row in llm_attn_matrix that corresponds to this output token
        row = llm_attn_matrix[out_token_pos]

        # Slice out only the text token columns, ignoring vision tokens
        # shape = (num_text_tokens,)
        text_token_indices = torch.tensor(text_token_indices, dtype=torch.long)
        text_attn = row[text_token_indices]


        # (Optional) if there's a chance that some part of the row is all zeros 
        # (e.g. during early generation), you can renormalize:
        sum_val = text_attn.sum()
        if sum_val > 0:
            if normalize:
                text_attn = text_attn / sum_val

        attention_array.append(text_attn.cpu().numpy())

    # Attention to every out token # shape [num_output_tokens, num_text_tokens]
    attention_array = np.stack(attention_array, axis=0) 

    if display_attention:
        # Build token labels
        # Remove image placeholder from input_ids
        #vision_token_start = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0].item()
        #input_ids_not_img = torch.cat([input_ids[0:vision_token_start], input_ids[vision_token_start+1:]])
        
        vision_token_positions = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
        
        if len(vision_token_positions) > 0:
            vision_token_start = vision_token_positions[0].item()
            input_ids_not_img = torch.cat([input_ids[:vision_token_start], input_ids[vision_token_start+1:]])
        else:
            input_ids_not_img = input_ids
    
        # Convert tokens to strings and clean them.
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids_not_img)
        input_tokens = [clean_token(tok) for tok in input_tokens]
        gen_tokens   = tokenizer.convert_ids_to_tokens(output_ids)
        gen_tokens   = [clean_token(tok) for tok in gen_tokens]

        
        for i in range(attention_array.shape[0]):
            html_lines = build_text_html_line(attention_array[i], input_tokens, gen_tokens[i])
            display(HTML(html_lines))
            
            if save:
                if '</s>' in gen_tokens[i]:
                    pass
                else:
                    # Optionally, render it to an image
                    out_filename = f"gen_token_{i}_({gen_tokens[i]}).png"
                    html_to_png(html_lines, out_filename)
                
    return attention_array




def visualize_image_and_text_attentions(
    # LLM and ViT attentions
    llm_attn_matrix,         # shape [total_seq_len, total_seq_len]
    vit_attn_matrix,         # shape [n_vision_tokens, patch_grid_size^2] (aggregated ViT patch attention)
    # indexing info
    text_token_indices,      # which positions in the prompt are text tokens (excluding image tokens)
    output_token_indices,    # which positions are generated tokens
    vision_token_start,      # start index of vision tokens in the LLM input
    vision_token_end,        # end index of vision tokens
    output_token_start,      # start index of generated tokens
    output_token_end,        # end index of generated tokens
    # token data & prompt
    tokenizer,
    input_ids,
    output_ids,
    # image data
    image,
    patch_grid_size=24,
    overlay=True,
    show_figure=True,
    save_dir='attentions',
    image_attention_dir='images',
    text_attention_dir="text",
    save=False,
    normalize=True
):
    """
    Gathers both image and text attention for each generated token, then displays
    them one by one:
      1) The patch-level image overlay.
      2) The text-token highlight for that same token.

    Args:
      llm_attn_matrix: 2D array [total_seq_len, total_seq_len], LLM cross-token attentions
      vit_attn_matrix: 2D array [n_vision_tokens, patch_grid_size^2], aggregated from ViT
      text_token_indices: list of positions in the LLM prompt that correspond to text (not image).
      output_token_indices: list of positions for generated tokens.
      vision_token_start, vision_token_end: index range for vision tokens.
      output_token_start, output_token_end: index range for output tokens.
      tokenizer, input_ids, output_ids: for decoding tokens & building text highlights.
      image: PIL image used for overlay.
      patch_grid_size: typically 24 for 336×336 images with 14×14 patches.
      overlay: whether to overlay heatmap on image or show it separately.
    """

    if save:
        image_attention_dir = os.path.join(save_dir, image_attention_dir)
        text_attention_dir = os.path.join(save_dir, text_attention_dir)
        os.makedirs(image_attention_dir, exist_ok=True) 
        os.makedirs(text_attention_dir, exist_ok=True) 
    
    # GET IMAGE ATTENTIONS (no plotting)
    
    img_attn_list = calculate_image_attention(
        llm_attn_matrix=llm_attn_matrix,
        vit_attn_matrix=vit_attn_matrix,
        tokenizer=tokenizer,
        image=image,
        output_ids=output_ids,
        vision_token_start=vision_token_start,
        vision_token_end=vision_token_end,
        output_token_start=output_token_start,
        output_token_end=output_token_end,
        patch_grid_size=patch_grid_size,
        overlay=overlay,
        return_attentions=True,   
        display=False,
        normalize=normalize
    )
    
    # GET TEXT ATTENTIONS
    text_attn_array = calculate_text_attention(
        llm_attn_matrix=llm_attn_matrix,
        text_token_indices=text_token_indices,
        output_token_indices=output_token_indices,
        tokenizer=tokenizer,
        input_ids=input_ids,
        output_ids=output_ids,
        display_attention=False ,
        normalize=normalize
    )


    # Optionally show in the notebook
    if show_figure:
        fig, ax = plt.subplots(figsize=(4,4), dpi=200)
        ax.imshow(image)
        ax.set_title(f"Original Image", fontsize=20)
        ax.axis("off")
        if save:
            plt.savefig(f"{image_attention_dir}/Original_image.jpg")
        plt.show()


    #print(len(img_attn_list))
    #print(img_attn_list[0].shape)

    #print(len(text_attn_array))
    #print(text_attn_array[0].shape)
    
    # LOOP PER GENERATED TOKEN & DISPLAY BOTH    
    num_generated_tokens = len(output_token_indices)
    np_img = np.array(image)[:, :, ::-1]   # BGR if needed
    
    # Convert token IDs to strings
    vision_token_positions = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    
    if len(vision_token_positions) > 0:
        vision_token_start = vision_token_positions[0].item()
        input_ids_not_img = torch.cat([input_ids[:vision_token_start], input_ids[vision_token_start+1:]])
    else:
        input_ids_not_img = input_ids
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids_not_img)
    gen_tokens   = tokenizer.convert_ids_to_tokens(output_ids)
        
    input_tokens = [clean_token(tok) for tok in input_tokens]
    gen_tokens   = [clean_token(tok) for tok in gen_tokens]
    
    # Iterate over each generated token
    for i in range(num_generated_tokens):
        
        # IMAGE ATTENTION
        patch_attention = img_attn_list[i]  # shape (patch_grid_size, patch_grid_size)
        
        # Upscale to image resolution
        patch_t = torch.tensor(patch_attention).unsqueeze(0).unsqueeze(0).float()
        big_attn = F.interpolate(
            patch_t,
            size=(image.size[1], image.size[0]),
            mode='bicubic',
            align_corners=False
        ).squeeze().cpu().numpy()
        
        # Show the image overlay in a figure
        from IPython.display import display

        
        # We assume you have a show_mask_on_image() function
        img_with_attn, heatmap = show_mask_on_image(np_img, big_attn)
        final_img = heatmap if not overlay else img_with_attn


        # Optionally show in the notebook
        if show_figure:
            fig, ax = plt.subplots(figsize=(4,4), dpi=200)
            ax.imshow(final_img)
            ax.set_title(f"Image attn → '{gen_tokens[i].strip()}'", fontsize=20)
            ax.axis("off")
            if save:
                if '</s>' in gen_tokens[i]:
                    pass
                else:
                    plt.savefig(f"{image_attention_dir}/Image attn → '{gen_tokens[i].strip()}'.jpg")
            plt.show()

        
        # Convert final_img (NumPy, BGR) into a PIL image (RGB)
        final_img_rgb = final_img#[:, :, ::-1]  
        overlay_pil = Image.fromarray(final_img_rgb)


        # TEXT ATTENTION
        text_row = text_attn_array[i]
        html_str = build_text_html_line(text_row, input_tokens, gen_tokens[i])
        if show_figure:
            display(HTML(html_str))
            
            if save:
                if '</s>' in gen_tokens[i]:
                    pass
                else:
                    # Optionally, render it to an image
                    out_filename = f"{text_attention_dir}/gen_token_{i}_({gen_tokens[i]}).png"
                    html_to_png(html_str, out_filename)


import torch
import numpy as np
from PIL import Image

def input_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Converts a PyTorch tensor normalized with ImageNet statistics
    to a PIL Image.
    
    Args:
        tensor (torch.Tensor): Tensor with shape (C, H, W) or (B, C, H, W)
                               normalized using ImageNet stats.
    
    Returns:
        Image.Image: The corresponding PIL Image.
    """
    # If a batch is provided, take the first image.
    if tensor.ndim == 4:
        tensor = tensor[0]
    
    # Define ImageNet normalization parameters:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Unnormalize the tensor:
    unnormalized_img_tensor = tensor.cpu() * std + mean
    
    # Convert to a NumPy array and change shape from (C, H, W) to (H, W, C):
    unnormalized_img = unnormalized_img_tensor.numpy().transpose(1, 2, 0)
    
    # Scale from [0, 1] to [0, 255] and convert to uint8:
    unnormalized_img = (unnormalized_img * 255).astype(np.uint8)
    
    # Convert the NumPy array to a PIL Image:
    return Image.fromarray(unnormalized_img)
