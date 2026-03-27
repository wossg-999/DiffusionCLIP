from typing import Union, List, Optional
import math
import os
import numpy as np
import torch
from pkg_resources import packaging
from torch import nn
from torch.nn import functional as F
from .clip_model import CLIP, instantiate_from_config
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from .UNetWrapper import UNetWrapper
from sklearn.cluster import KMeans
from omegaconf import OmegaConf

class ProjectLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_replicas, stack=False, is_array=True):
        super(ProjectLayer, self).__init__()

        self.head = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_replicas)])
        self.num_replicas = num_replicas
        self.stack = stack
        self.is_array = is_array

    def forward(self, tokens):
        out_tokens = []
        for i in range(self.num_replicas):
            if self.is_array:
                temp = self.head[i](tokens[i][:, 1:, :]) # for ViT, we exclude the class token and only extract patch tokens here.
            else:
                temp = self.head[i](tokens)

            out_tokens.append(temp)

        if self.stack:
            out_tokens = torch.stack(out_tokens, dim=1)

        return out_tokens

class PromptLayer(nn.Module):
    def __init__(self, channel, length, depth, is_text, prompting_type, enabled=True):
        super(PromptLayer, self).__init__()

        self.channel = channel
        self.length = length
        self.depth = depth
        self.is_text = is_text
        self.enabled = enabled

        self.prompting_type = prompting_type

        if self.enabled: # only when enabled, the parameters should be constructed
            if 'S' in prompting_type: # static prompts
                # learnable
                self.static_prompts = nn.ParameterList(
                    [nn.Parameter(torch.empty(self.length, self.channel))
                     for _ in range(self.depth)])

                for single_para in self.static_prompts:
                    nn.init.normal_(single_para, std=0.02)

            if 'D' in prompting_type: # dynamic prompts
                self.dynamic_prompts = [0.] # place holder

    def set_dynamic_prompts(self, dynamic_prompts):
        self.dynamic_prompts = dynamic_prompts

    def forward_text(self, resblock, indx, x, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None):
        if self.enabled:
            length = self.length

            # only prompt the first J layers
            if indx < self.depth:
                if 'S' in self.prompting_type and 'D' in self.prompting_type: # both
                    static_prompts = self.static_prompts[indx].unsqueeze(0).expand(x.shape[1], -1, -1)
                    textual_context = self.dynamic_prompts + static_prompts
                elif 'S' in self.prompting_type:  # static
                    static_prompts = self.static_prompts[indx].unsqueeze(0).expand(x.shape[1], -1, -1)
                    textual_context = static_prompts
                elif 'D' in self.prompting_type:  # dynamic
                    textual_context = self.dynamic_prompts
                else:
                    print('You should at least choose one type of prompts when the prompting branches are not none.')
                    raise NotImplementedError

            if indx == 0:  # for the first layer
                x = x
            else:
                if indx < self.depth:  # replace with learnalbe tokens
                    prefix = x[:1, :, :]
                    suffix = x[1 + length:, :, :]
                    textual_context = textual_context.permute(1, 0, 2).half()
                    x = torch.cat([prefix, textual_context, suffix], dim=0)
                else:  # keep the same
                    x = x
        else:
            x = x

        x, attn_tmp = resblock(q_x=x, k_x=k_x, v_x= v_x, attn_mask=attn_mask)

        return x, attn_tmp

    def forward_visual(self, resblock, indx, x, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None):
        if self.enabled:
            length = self.length

            # only prompt the first J layers
            if indx < self.depth:
                if 'S' in self.prompting_type and 'D' in self.prompting_type: # both
                    static_prompts = self.static_prompts[indx].unsqueeze(0).expand(x.shape[1], -1, -1)
                    visual_context = self.dynamic_prompts + static_prompts
                elif 'S' in self.prompting_type:  # static
                    static_prompts = self.static_prompts[indx].unsqueeze(0).expand(x.shape[1], -1, -1)
                    visual_context = static_prompts
                elif 'D' in self.prompting_type:  # dynamic
                    visual_context = self.dynamic_prompts
                else:
                    print('You should at least choose one type of prompts when the prompting branches are not none.')
                    raise NotImplementedError


            if indx == 0:  # for the first layer
                visual_context = visual_context.permute(1, 0, 2).half()
                x = torch.cat([x, visual_context], dim=0)
            else:
                if indx < self.depth:  # replace with learnalbe tokens
                    prefix = x[0:x.shape[0] - length, :, :]
                    visual_context = visual_context.permute(1, 0, 2).half()
                    x = torch.cat([prefix, visual_context], dim=0)
                else:  # keep the same
                    x = x
        else:
            x = x

        x, attn_tmp = resblock(q_x=x, k_x=k_x, v_x= v_x, attn_mask=attn_mask)

        if self.enabled:
            tokens = x[:x.shape[0] - length, :, :]
        else:
            tokens = x

        return x, tokens, attn_tmp

    def forward(self, resblock, indx, x, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None):
        if self.is_text:
            return self.forward_text(resblock, indx, x, k_x, v_x, attn_mask)
        else:
            return self.forward_visual(resblock, indx, x, k_x, v_x, attn_mask)
# by thk
# class PromptLayer(nn.Module):
#     def __init__(self, channel, length, depth, is_text, prompting_type, sparsity_lambda=0.01, enabled=True):
#         super(PromptLayer, self).__init__()

#         self.channel = channel
#         self.length = length
#         self.depth = depth
#         self.is_text = is_text
#         self.enabled = enabled
#         self.sparsity_lambda = sparsity_lambda
#         self.prompting_type = prompting_type

#         if self.enabled:  # Only initialize parameters when enabled
#             # Static prompts: learnable prompts initialized for each depth
#             if 'S' in prompting_type:
#                 self.static_prompts = nn.ParameterList([
#                     nn.Parameter(torch.empty(self.length, self.channel)) for _ in range(self.depth)
#                 ])
#                 for param in self.static_prompts:
#                     nn.init.normal_(param, std=0.02)
            
#             # Placeholder for dynamic prompts, allowing later assignment
#             if 'D' in prompting_type:
#                 self.dynamic_prompts = [None] * self.depth

#     def set_dynamic_prompts(self, dynamic_prompts):
#         """Sets dynamic prompts externally to allow flexibility."""
#         self.dynamic_prompts = dynamic_prompts

#     def compute_sparse_loss(self, prompts):
#         """Calculates L1 sparse regularization loss."""
#         sparse_loss = self.sparsity_lambda * torch.norm(prompts, p=1)
#         return sparse_loss

#     def get_prompt_context(self, indx):
#         """Generates prompt context based on layer index and prompt type."""
#         if 'S' in self.prompting_type and 'D' in self.prompting_type:  # Both static and dynamic prompts
#             static_prompt = self.static_prompts[indx].unsqueeze(0)
#             prompt_context = self.dynamic_prompts[indx] + static_prompt
#         elif 'S' in self.prompting_type:  # Static prompts only
#             prompt_context = self.static_prompts[indx].unsqueeze(0)
#         elif 'D' in self.prompting_type:  # Dynamic prompts only
#             prompt_context = self.dynamic_prompts[indx]
#         else:
#             raise NotImplementedError("At least one prompt type ('S' or 'D') must be enabled.")
        
#         return prompt_context.expand(-1, -1, self.channel)  # Ensure prompt expands to correct shape

#     def forward_prompt(self, resblock, indx, x, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None):
#         """Handles the forward pass for prompts with text or visual context."""
#         if self.enabled and indx < self.depth:
#             prompt_context = self.get_prompt_context(indx)
#             sparse_loss = self.compute_sparse_loss(prompt_context)

#             # Handle text-specific behavior
#             if self.is_text:
#                 prefix = x[:1, :, :]
#                 suffix = x[1 + self.length:, :, :] if indx < self.depth else x[1:, :, :]
#                 prompt_context = prompt_context.permute(1, 0, 2).half()
#                 x = torch.cat([prefix, prompt_context, suffix], dim=0)
#             else:  # Visual-specific behavior
#                 prefix = x[:x.shape[0] - self.length, :, :] if indx < self.depth else x
#                 prompt_context = prompt_context.permute(1, 0, 2).half()
#                 x = torch.cat([prefix, prompt_context], dim=0)
#         else:
#             sparse_loss = 0.0  # No sparse loss when not enabled
#             x = x

#         x, attn_tmp = resblock(q_x=x, k_x=k_x, v_x=v_x, attn_mask=attn_mask)
#         return x, attn_tmp, sparse_loss

#     def forward(self, resblock, indx, x, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None):
#         """Unified forward pass for both text and visual contexts."""
#         x, attn_tmp, sparse_loss = self.forward_prompt(resblock, indx, x, k_x, v_x, attn_mask)
#         if self.is_text:
#             tokens = x
#         else:  # For visual, truncate the prompt length
#             tokens = x[:x.shape[0] - self.length, :, :] if self.enabled else x
#         return x, tokens, attn_tmp, sparse_loss



class TextEmbebddingLayer(nn.Module):
    def __init__(self, fixed):
        super(TextEmbebddingLayer, self).__init__()
        self.tokenizer = _Tokenizer()
        self.ensemble_text_features = {}
        self.prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw',
                              '{} without defect',
                              '{} without damage']
        self.prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
        self.prompt_state = [self.prompt_normal, self.prompt_abnormal]
        self.prompt_templates = ['a bad photo of a {}.',
                                 'a low resolution photo of the {}.',
                                 'a bad photo of the {}.',
                                 'a cropped photo of the {}.',
                                 ]
        self.fixed = fixed

    def tokenize(self, texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[
        torch.IntTensor, torch.LongTensor]:
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        else:
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    ## TODO: text layeer is not compitable with multiple batches...
    def forward(self, model, texts, device):
        text_feature_list = []

        for indx, text in enumerate(texts):

            if self.fixed:
                if self.ensemble_text_features.get(text) is None:
                    text_features = self.encode_text(model, text, device)
                    self.ensemble_text_features[text] = text_features
                else:
                    text_features = self.ensemble_text_features[text]
            else:
                text_features = self.encode_text(model, text, device)
                self.ensemble_text_features[text] = text_features

            text_feature_list.append(text_features)

        text_features = torch.stack(text_feature_list, dim=0)
        text_features = F.normalize(text_features, dim=1)

        return text_features

    def encode_text(self, model, text, device):
        text_features = []
        for i in range(len(self.prompt_state)):
            text = text.replace('-', ' ')
            prompted_state = [state.format(text) for state in self.prompt_state[i]]
            prompted_sentence = []
            for s in prompted_state:
                for template in self.prompt_templates:
                    prompted_sentence.append(template.format(s))
            prompted_sentence = self.tokenize(prompted_sentence, context_length=77).to(device)

            class_embeddings = model.encode_text(prompted_sentence)

            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            text_features.append(class_embedding)

        text_features = torch.stack(text_features, dim=1)

        return text_features


# Note: the implementation of HSF is slightly different to the reported one, since we found that the upgraded one is more stable.
class HybridSemanticFusion(nn.Module):
    def __init__(self, k_clusters):
        super(HybridSemanticFusion, self).__init__()
        self.k_clusters = k_clusters
        self.n_aggregate_patch_tokens = k_clusters * 5
        self.cluster_performer = KMeans(n_clusters=self.k_clusters, n_init="auto")

    # @torch.no_grad()
    def forward(self, patch_tokens: list, anomaly_maps: list):
        anomaly_map = torch.mean(torch.stack(anomaly_maps, dim=1), dim=1)
        anomaly_map = torch.softmax(anomaly_map, dim=2)[:, :, 1] # B, L

        # extract most abnormal feats
        selected_abnormal_tokens = []
        k = min(anomaly_map.shape[1], self.n_aggregate_patch_tokens)
        top_k_indices = torch.topk(anomaly_map, k=k, dim=1).indices
        for layer in range(len(patch_tokens)):
            selected_tokens = patch_tokens[layer]. \
                gather(dim=1, index=top_k_indices.unsqueeze(-1).
                       expand(-1, -1, patch_tokens[layer].shape[-1]))
            selected_abnormal_tokens.append(selected_tokens)

        # use kmeans to extract these centriods
        # Stack the data_preprocess
        stacked_data = torch.cat(selected_abnormal_tokens, dim=2)

        batch_cluster_centers = []
        # Perform K-Means clustering
        for b in range(stacked_data.shape[0]):
            cluster_labels = self.cluster_performer.fit_predict(stacked_data[b, :, :].detach().cpu().numpy())

            # Initialize a list to store the cluster centers
            cluster_centers = []

            # Extract cluster centers for each cluster
            for cluster_id in range(self.k_clusters):
                collected_cluster_data = []
                for abnormal_tokens in selected_abnormal_tokens:
                    cluster_data = abnormal_tokens[b, :, :][cluster_labels == cluster_id]
                    collected_cluster_data.append(cluster_data)
                collected_cluster_data = torch.cat(collected_cluster_data, dim=0)
                cluster_center = torch.mean(collected_cluster_data, dim=0, keepdim=True)
                cluster_centers.append(cluster_center)

            # Normalize the cluster centers
            cluster_centers = torch.cat(cluster_centers, dim=0)
            cluster_centers = torch.mean(cluster_centers, dim=0)
            batch_cluster_centers.append(cluster_centers)

        batch_cluster_centers = torch.stack(batch_cluster_centers, dim=0)
        batch_cluster_centers = F.normalize(batch_cluster_centers, dim=1)

        return batch_cluster_centers

class StableDiffusionFeatureExtractor(nn.Module):
    def __init__(self, sd_config_path: str, sd_ckpt_path: str = '', sd_vae_ckpt_path: str = '', base_size: int = 512,
                 timestep: int = 50, use_attn: bool = True,
                 image_mean: Optional[List[float]] = None, image_std: Optional[List[float]] = None):
        super().__init__()
        config = OmegaConf.load(sd_config_path)
        config.model.params.concat_mode = False
        config.model.params.conditioning_key = 'crossattn'

        full_ckpt_has_vae = False
        if sd_ckpt_path:
            checkpoint = torch.load(sd_ckpt_path, map_location='cpu')
            state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
            if isinstance(state_dict, dict):
                full_ckpt_has_vae = any(key.startswith('first_stage_model.') for key in state_dict.keys())
            config.model.params.ckpt_path = sd_ckpt_path

        self.use_vae_latent = False
        if sd_vae_ckpt_path and 'first_stage_config' in config.model.params:
            config.model.params.first_stage_config.params.ckpt_path = sd_vae_ckpt_path
            if sd_ckpt_path:
                config.model.params.ignore_keys = ['first_stage_model']
            self.use_vae_latent = True
        elif full_ckpt_has_vae and 'first_stage_config' in config.model.params:
            self.use_vae_latent = True
        else:
            if 'first_stage_config' in config.model.params:
                del config.model.params['first_stage_config']
        self.context_dim = config.model.params.unet_config.params.context_dim
        if image_mean is None:
            image_mean = [0.48145466, 0.4578275, 0.40821073]
        if image_std is None:
            image_std = [0.26862954, 0.26130258, 0.27577711]
        self.register_buffer('image_mean', torch.tensor(image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('image_std', torch.tensor(image_std).view(1, 3, 1, 1), persistent=False)
        sd_model = instantiate_from_config(config.model)
        self.sd_model = sd_model
        self.unet_wrapper = UNetWrapper(sd_model.model, use_attn=use_attn, base_size=base_size)
        self.base_size = base_size
        self.timestep = timestep
        self.has_first_stage = self.use_vae_latent and hasattr(sd_model, 'first_stage_model') and sd_model.first_stage_model is not None

        if self.has_first_stage:
            self.sd_model.eval()
            for param in self.sd_model.parameters():
                param.requires_grad = False
        else:
            self.sd_model = None
            for param in self.unet_wrapper.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        restored_image = image * self.image_std.to(image.dtype) + self.image_mean.to(image.dtype)
        restored_image = restored_image.clamp(0.0, 1.0)
        if restored_image.shape[-1] != self.base_size or restored_image.shape[-2] != self.base_size:
            restored_image = F.interpolate(restored_image, size=(self.base_size, self.base_size), mode='bilinear', align_corners=False)
        vae_input = restored_image * 2.0 - 1.0

        if self.has_first_stage:
            latents = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(vae_input)).detach()
        else:
            latents = F.interpolate(vae_input, size=(64, 64), mode='bilinear', align_corners=False)
            latents = latents.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)

        t = torch.full((latents.shape[0],), self.timestep, device=latents.device, dtype=torch.long)
        context = torch.zeros(latents.shape[0], 77, self.context_dim, device=latents.device, dtype=latents.dtype)
        out_list = self.unet_wrapper(latents, t, c_crossattn=[context])
        return out_list


class SDTokenAdapter(nn.Module):
    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_dim, kernel_size=1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, feat: torch.Tensor, target_tokens: int):
        feat = self.proj(feat)
        b, c, h, w = feat.shape
        feat = feat.view(b, c, h * w).transpose(1, 2)
        if feat.shape[1] != target_tokens:
            feat = F.interpolate(feat.transpose(1, 2), size=target_tokens, mode='linear', align_corners=False).transpose(1, 2)
        feat = self.norm(feat)
        return feat


class FeatureFusionBlock(nn.Module):
    def __init__(self, token_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
            nn.Sigmoid()
        )
        self.out_norm = nn.LayerNorm(token_dim)

    def forward(self, vit_tokens: torch.Tensor, sd_tokens: torch.Tensor):
        gate = self.gate(torch.cat([vit_tokens, sd_tokens], dim=-1))
        fused = vit_tokens + gate * sd_tokens
        return self.out_norm(fused)


class AdaCLIP(nn.Module):
    def __init__(self, freeze_clip: CLIP, text_channel: int, visual_channel: int,
                 prompting_length: int, prompting_depth: int, prompting_branch: str, prompting_type: str,
                 use_hsf: bool, k_clusters: int,
                 output_layers: list, device: str, image_size: int,
                 use_sd_feature: bool = False, sd_config_path: str = 'method/v1-inference.yaml',
                 sd_ckpt_path: str = '', sd_vae_ckpt_path: str = '', sd_timestep: int = 50, sd_base_size: int = 512,
                 sd_use_attention: bool = True):
        super(AdaCLIP, self).__init__()
        self.freeze_clip = freeze_clip

        self.visual = self.freeze_clip.visual
        self.transformer = self.freeze_clip.transformer
        self.token_embedding = self.freeze_clip.token_embedding
        self.positional_embedding = self.freeze_clip.positional_embedding
        self.ln_final = self.freeze_clip.ln_final
        self.text_projection = self.freeze_clip.text_projection
        self.attn_mask = self.freeze_clip.attn_mask

        self.output_layers = output_layers

        self.prompting_branch = prompting_branch
        self.prompting_type = prompting_type
        self.prompting_depth = prompting_depth
        self.prompting_length = prompting_length
        self.use_hsf = use_hsf
        self.k_clusters = k_clusters
        self.use_sd_feature = use_sd_feature

        if 'L' in self.prompting_branch:
            self.enable_text_prompt = True
        else:
            self.enable_text_prompt = False

        if 'V' in self.prompting_branch:
            self.enable_visual_prompt = True
        else:
            self.enable_visual_prompt = False

        self.text_embedding_layer = TextEmbebddingLayer(fixed=(not self.enable_text_prompt))
        self.text_prompter = PromptLayer(text_channel, prompting_length, prompting_depth, is_text=True,
                                         prompting_type=prompting_type,
                                         enabled=self.enable_text_prompt)
        self.visual_prompter = PromptLayer(visual_channel, prompting_length, prompting_depth, is_text=False,
                                           prompting_type=prompting_type,
                                           enabled=self.enable_visual_prompt)

        self.patch_token_layer = ProjectLayer(
            visual_channel,
            text_channel,
            len(output_layers), stack=False, is_array=True
        )

        self.cls_token_layer = ProjectLayer(
            text_channel,
            text_channel,
            1, stack=False, is_array=False
        )

        if 'D' in self.prompting_type: # dynamic prompts
            self.dynamic_visual_prompt_generator = ProjectLayer(text_channel,
                                                                visual_channel,
                                                                prompting_length,
                                                                stack=True,
                                                                is_array=False)
            self.dynamic_text_prompt_generator = ProjectLayer(text_channel,
                                                              text_channel,
                                                              prompting_length,
                                                              stack=True,
                                                              is_array=False)

        if self.use_hsf:
            self.HSF = HybridSemanticFusion(k_clusters)

        if self.use_sd_feature:
            sd_config_abspath = sd_config_path
            if not os.path.isabs(sd_config_abspath):
                candidate_in_method = os.path.normpath(os.path.join(os.path.dirname(__file__), sd_config_path))
                candidate_in_repo = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', sd_config_path))
                sd_config_abspath = candidate_in_method if os.path.exists(candidate_in_method) else candidate_in_repo
            image_mean = getattr(self.visual, 'image_mean', [0.48145466, 0.4578275, 0.40821073])
            image_std = getattr(self.visual, 'image_std', [0.26862954, 0.26130258, 0.27577711])
            self.sd_extractor = StableDiffusionFeatureExtractor(
                sd_config_path=sd_config_abspath,
                sd_ckpt_path=sd_ckpt_path,
                sd_vae_ckpt_path=sd_vae_ckpt_path,
                base_size=sd_base_size,
                timestep=sd_timestep,
                use_attn=sd_use_attention,
                image_mean=image_mean,
                image_std=image_std,
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 3, image_size, image_size)
                sd_feat_channels = [feat.shape[1] for feat in self.sd_extractor(dummy)]
            self.sd_token_adapters = nn.ModuleList([
                SDTokenAdapter(in_channels=sd_feat_channels[i], out_dim=visual_channel)
                for i in range(len(output_layers))
            ])
            self.sd_fusion_blocks = nn.ModuleList([
                FeatureFusionBlock(token_dim=visual_channel)
                for _ in range(len(output_layers))
            ])

        self.image_size = image_size
        self.device = device

    def generate_and_set_dynamic_promtps(self, image):
        with torch.no_grad():
            # extract image features
            image_features, _ = self.visual.forward(image, self.output_layers)

        dynamic_visual_prompts = self.dynamic_visual_prompt_generator(image_features)
        dynamic_text_prompts = self.dynamic_text_prompt_generator(image_features)

        self.visual_prompter.set_dynamic_prompts(dynamic_visual_prompts)
        self.text_prompter.set_dynamic_prompts(dynamic_text_prompts)


    def encode_image(self, image):

        x = image
        if self.visual.input_patchnorm:
            x = x.reshape(x.shape[0], x.shape[1],
                          self.visual.grid_size[0],
                          self.visual.patch_size[0],
                          self.visual.grid_size[1],
                          self.visual.patch_size[1])
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.visual.grid_size[0] * self.visual.grid_size[1], -1)
            x = self.visual.patchnorm_pre_ln(x)
            x = self.visual.conv1(x)
        else:
            x = self.visual.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)

        x = torch.cat(
            [self.visual.class_embedding.to(x.dtype) +
             torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)

        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)

        patch_embedding = x
        x = x.permute(1, 0, 2)
        patch_tokens = []

        sd_tokens_per_layer = []
        if self.use_sd_feature:
            sd_features = self.sd_extractor(image)
            sd_features = sd_features[:len(self.output_layers)]
            token_count = x.shape[0] - 1
            for layer_idx, sd_feat in enumerate(sd_features):
                sd_tokens = self.sd_token_adapters[layer_idx](sd_feat, token_count)
                sd_tokens_per_layer.append(sd_tokens)

        selected_layer_idx = 0
        for indx, r in enumerate(self.visual.transformer.resblocks):
            x, tokens, attn_tmp = self.visual_prompter(r, indx, x, k_x=None, v_x=None, attn_mask=None)

            if (indx + 1) in self.output_layers:
                if self.use_sd_feature and selected_layer_idx < len(sd_tokens_per_layer):
                    cls_token = tokens[:1, :, :]
                    vit_patch_tokens = tokens[1:, :, :].permute(1, 0, 2)
                    fused_patch_tokens = self.sd_fusion_blocks[selected_layer_idx](
                        vit_patch_tokens,
                        sd_tokens_per_layer[selected_layer_idx]
                    )
                    fused_patch_tokens = fused_patch_tokens.permute(1, 0, 2)
                    tokens = torch.cat([cls_token, fused_patch_tokens], dim=0)
                    x = torch.cat([tokens, x[tokens.shape[0]:, :, :]], dim=0)
                patch_tokens.append(tokens)
                selected_layer_idx += 1

        x = x.permute(1, 0, 2)
        patch_tokens = [patch_tokens[t].permute(1, 0, 2) for t in range(len(patch_tokens))]

        if self.visual.attn_pool is not None:
            x = self.visual.attn_pool(x)
            x = self.visual.ln_post(x)
            pooled, tokens = self.visual._global_pool(x)
        else:
            pooled, tokens = self.visual._global_pool(x)
            pooled = self.visual.ln_post(pooled)

        if self.visual.proj is not None:
            pooled = pooled @ self.visual.proj

        return pooled, patch_tokens, patch_embedding

    def proj_visual_tokens(self, image_features, patch_tokens):

        # for patch tokens
        proj_patch_tokens = self.patch_token_layer(patch_tokens)
        for layer in range(len(proj_patch_tokens)):
            proj_patch_tokens[layer] /= proj_patch_tokens[layer].norm(dim=-1, keepdim=True)

        # for cls tokens
        proj_cls_tokens = self.cls_token_layer(image_features)[0]
        proj_cls_tokens /= proj_cls_tokens.norm(dim=-1, keepdim=True)

        return proj_cls_tokens, proj_patch_tokens

    def encode_text(self, text):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        for indx, r in enumerate(self.transformer.resblocks):
            # add prompt here
            x, attn_tmp = self.text_prompter(r, indx, x, k_x=None, v_x=None, attn_mask=self.attn_mask)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def visual_text_similarity(self, image_feature, patch_token, text_feature, aggregation):
        anomaly_maps = []

        for layer in range(len(patch_token)):
            anomaly_map = (100.0 * patch_token[layer] @ text_feature)
            anomaly_maps.append(anomaly_map)

        if self.use_hsf:
            alpha = 0.2
            clustered_feature = self.HSF.forward(patch_token, anomaly_maps)
            # aggregate the class token and the clustered features for more comprehensive information
            cur_image_feature = alpha * clustered_feature + (1 - alpha) * image_feature
            cur_image_feature = F.normalize(cur_image_feature, dim=1)
        else:
            cur_image_feature = image_feature

        anomaly_score = (100.0 * cur_image_feature.unsqueeze(1) @ text_feature)
        anomaly_score = anomaly_score.squeeze(1)
        anomaly_score = torch.softmax(anomaly_score, dim=1)

        # NOTE: this bilinear interpolation is not unreproducible and may occasionally lead to unstable ZSAD performance.
        for i in range(len(anomaly_maps)):
            B, L, C = anomaly_maps[i].shape
            H = int(np.sqrt(L))
            anomaly_maps[i] = anomaly_maps[i].permute(0, 2, 1).view(B, 2, H, H)
            anomaly_maps[i] = F.interpolate(anomaly_maps[i], size=self.image_size, mode='bilinear', align_corners=True)

        if aggregation: # in the test stage, we firstly aggregate logits from all hierarchies and then do the softmax normalization
            anomaly_map = torch.mean(torch.stack(anomaly_maps, dim=1), dim=1)
            anomaly_map = torch.softmax(anomaly_map, dim=1)
            anomaly_map = (anomaly_map[:, 1:, :, :] + 1 - anomaly_map[:, 0:1, :, :]) / 2.0
            anomaly_score = anomaly_score[:, 1]
            return anomaly_map, anomaly_score
        else: # otherwise, we do the softmax normalization for individual hierarchies
            for i in range(len(anomaly_maps)):
                anomaly_maps[i] = torch.softmax(anomaly_maps[i], dim=1)
            return anomaly_maps, anomaly_score

    def extract_feat(self, image, cls_name):
        if 'D' in self.prompting_type:
            self.generate_and_set_dynamic_promtps(image) # generate and set dynamic prompts for corresponding prompters

        if self.enable_visual_prompt:
            image_features, patch_tokens, _ = self.encode_image(image)
        else:
            with torch.no_grad():
                image_features, patch_tokens, _ = self.encode_image(image)

        if self.enable_text_prompt:
            text_features = self.text_embedding_layer(self, cls_name, self.device)
        else:
            with torch.no_grad():
                text_features = self.text_embedding_layer(self, cls_name, self.device)

        proj_cls_tokens, proj_patch_tokens = self.proj_visual_tokens(image_features, patch_tokens)

        return proj_cls_tokens, proj_patch_tokens, text_features

    @torch.cuda.amp.autocast()
    def forward(self, image, cls_name, aggregation=True):
        # extract features for images and texts
        image_features, patch_tokens, text_features = self.extract_feat(image, cls_name)
        anomaly_map, anomaly_score = self.visual_text_similarity(image_features, patch_tokens, text_features, aggregation)

        if aggregation:
            anomaly_map = anomaly_map # tensor
            anomaly_score = anomaly_score
            anomaly_map = anomaly_map.squeeze(1)

            return anomaly_map, anomaly_score
        else:
            anomaly_maps = anomaly_map # list
            anomaly_score = anomaly_score

            return anomaly_maps, anomaly_score

