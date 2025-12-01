import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union
from diffusers.pipelines import FluxKontextPipeline
from diffusers.pipelines.flux.pipeline_flux_kontext import retrieve_timesteps, calculate_shift
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

class ReInvFluxKontextPipeline(FluxKontextPipeline):
    def prepare_latents(
        self,
        image_1: Optional[torch.Tensor],
        image_2: Optional[torch.Tensor],
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, height, width)

        image_latents_1 = image_latents_2 = image_ids = None
        if image_1 is not None:
            image_1 = image_1.to(device=device, dtype=dtype)
            if image_1.shape[1] != self.latent_channels:
                image_latents_1 = self._encode_vae_image(image=image_1, generator=generator)
            else:
                image_latents_1 = image_1
            if batch_size > image_latents_1.shape[0] and batch_size % image_latents_1.shape[0] == 0:
                # expand init_latents for batch_size
                additional_image_per_prompt = batch_size // image_latents_1.shape[0]
                image_latents_1 = torch.cat([image_latents_1] * additional_image_per_prompt, dim=0)
            elif batch_size > image_latents_1.shape[0] and batch_size % image_latents_1.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image_latents_1.shape[0]} to {batch_size} text prompts."
                )
            else:
                image_latents_1 = torch.cat([image_latents_1], dim=0)

            image_latent_height, image_latent_width = image_latents_1.shape[2:]
            image_latents_1 = self._pack_latents(
                image_latents_1, batch_size, num_channels_latents, image_latent_height, image_latent_width
            )
            image_ids = self._prepare_latent_image_ids(
                batch_size, image_latent_height // 2, image_latent_width // 2, device, dtype
            )
            # image ids are the same as latent ids with the first dimension set to 1 instead of 0
            image_ids[..., 0] = 1

        if image_2 is not None:
            image_2 = image_2.to(device=device, dtype=dtype)
            if image_2.shape[1] != self.latent_channels:
                image_latents_2 = self._encode_vae_image(image=image_2, generator=generator)
            else:
                image_latents_2 = image_2
            if batch_size > image_latents_2.shape[0] and batch_size % image_latents_2.shape[0] == 0:
                # expand init_latents for batch_size
                additional_image_per_prompt = batch_size // image_latents_2.shape[0]
                image_latents_2 = torch.cat([image_latents_2] * additional_image_per_prompt, dim=0)
            elif batch_size > image_latents_2.shape[0] and batch_size % image_latents_2.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image_latents_2.shape[0]} to {batch_size} text prompts."
                )
            else:
                image_latents_2 = torch.cat([image_latents_2], dim=0)

            image_latent_height, image_latent_width = image_latents_2.shape[2:]
            image_latents_2 = self._pack_latents(
                image_latents_2, batch_size, num_channels_latents, image_latent_height, image_latent_width
            )
            image_ids = self._prepare_latent_image_ids(
                batch_size, image_latent_height // 2, image_latent_width // 2, device, dtype
            )
            # image ids are the same as latent ids with the first dimension set to 1 instead of 0
            image_ids[..., 0] = 1

        latent_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        if latents is None:
            from diffusers.utils.torch_utils import randn_tensor
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents_1, image_latents_2, latent_ids, image_ids

    @torch.no_grad()
    def __call__(
        self, 
        num_inference_steps, 
        image, 
        image_2, 
        latents=None, 
        prompt="", 
        prompt_2="", 
        guidance_scale=3.5, 
        height=1024, 
        width=1024, 
        start_step=0, 
        mask=None,
        eta=1.0,
        timesteps=None,
        sigmas=None,
        v_star=False
    ):
        # 1. Prepare prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            device='cuda',
            num_images_per_prompt=1,
            max_sequence_length=512,
            lora_scale=None,
        )
        (
            prompt_embeds_2,
            pooled_prompt_embeds_2,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt_2,
            prompt_2=None,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            device='cuda',
            num_images_per_prompt=1,
            max_sequence_length=512,
            lora_scale=None,
        )

        # 2. Preprocess image
        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            img = image[0] if isinstance(image, list) else image
            image_height, image_width = self.image_processor.get_default_height_width(img)
            image = self.image_processor.resize(image, image_height, image_width)
            image = self.image_processor.preprocess(image, image_height, image_width)

        if image_2 is not None and not (isinstance(image_2, torch.Tensor) and image_2.size(1) == self.latent_channels):
            img = image_2[0] if isinstance(image_2, list) else image_2
            image_height, image_width = self.image_processor.get_default_height_width(img)
            image_2 = self.image_processor.resize(image_2, image_height, image_width)
            image_2 = self.image_processor.preprocess(image_2, image_height, image_width)

        num_channels_latents = self.transformer.config.in_channels // 4

        if mask is not None and not (isinstance(mask, torch.Tensor) and mask.size(1) == self.latent_channels):
            img = mask[0] if isinstance(mask, list) else mask
            image_height, image_width = self.image_processor.get_default_height_width(img)
            mask = self.image_processor.resize(mask, image_height, image_width)
            mask = self.image_processor.preprocess(mask, image_height, image_width)
            mask = F.interpolate(mask, size=(height//num_channels_latents, width//num_channels_latents), mode='nearest').view(1, -1)

        # 3. Prepare latent variables
        latents, image_latents_1, image_latents_2, latent_ids, image_ids = self.prepare_latents(
            image,
            image_2,
            1,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            'cuda',
            None,
            latents,
        )
        if image_ids is not None:
            latent_ids = torch.cat([latent_ids, image_ids], dim=0)  # dim 0 is sequence dimension

        # 4. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if timesteps is None:
            image_seq_len = latents.shape[1]
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                'cuda',
                sigmas=sigmas,
                mu=mu,
            )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device='cuda', dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # generate
        self.scheduler.set_begin_index(0)
        z_t, z_1, z_1_ref = latents, image_latents_1, image_latents_2
        z_t_list = [z_t]
        v_dict = {}
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self._current_timestep = t

                latent_model_input = z_t
                if i < start_step:
                    latent_model_input = torch.cat([z_t, z_1], dim=1)
                    pooled_prompt_embeds = pooled_prompt_embeds
                    prompt_embeds = prompt_embeds

                    if v_star:
                        v_t_star = (z_t - z_1) / (t / 1000)
                        z_t = z_t + v_t_star * (self.scheduler.sigmas[i+1] - self.scheduler.sigmas[i])
                        progress_bar.update()
                        continue
                elif i >= start_step:
                    latent_model_input = torch.cat([z_t, z_1_ref], dim=1)
                    pooled_prompt_embeds = pooled_prompt_embeds_2
                    prompt_embeds = prompt_embeds_2
                timestep = t.expand(z_t.shape[0]).to(z_t.dtype)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
                v = noise_pred[:, : latents.size(1)]

                delta_t = (self.scheduler.sigmas[i+1] - self.scheduler.sigmas[i])
                if i < start_step or mask is None:
                    z_t = z_t + v * delta_t
                else:
                    v_t_star = (z_t - z_1) / (t / 1000)
                    z_t[mask == -1] = z_t[mask == -1] + ( v_t_star[mask == -1] * eta + v[mask == -1] * (1 - eta) ) * delta_t
                    z_t[mask == 1] = z_t[mask == 1] + v[mask == 1] * delta_t

                z_t_list.append(z_t)

                progress_bar.update()

        self._current_timestep = None

        latents = self._unpack_latents(z_t, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type='pil')

        # Offload all models
        self.maybe_free_model_hooks()

        return FluxPipelineOutput(images=image)