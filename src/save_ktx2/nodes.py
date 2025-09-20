import os
import folder_paths
import torch
import numpy as np
from pyktx.ktx_texture2 import KtxTexture2
from pyktx.ktx_texture_create_info import KtxTextureCreateInfo
from pyktx.ktx_texture_create_storage import KtxTextureCreateStorage
from pyktx.ktx_astc_params import KtxAstcParams
from pyktx.ktx_pack_astc_block_dimension import KtxPackAstcBlockDimension
from pyktx.ktx_pack_astc_encoder_mode import KtxPackAstcEncoderMode
from pyktx.ktx_pack_astc_quality_levels import KtxPackAstcQualityLevels
from pyktx.ktx_basis_params import KtxBasisParams
from pyktx.ktx_pack_uastc_flag_bits import KtxPackUastcFlagBits
from pyktx.ktx_transcode_fmt import KtxTranscodeFmt
from pyktx.ktx_transcode_flag_bits import KtxTranscodeFlagBits
from pyktx.vk_format import VkFormat
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from comfy.cli_args import args

class SaveKtx2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."})
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image"
    DESCRIPTION = "Saves the input images to .ktx2 format."

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory(), images[0].shape[1], images[0].shape[0])
        results = list()

        for (batch_number, image) in enumerate(images):
            H, W, _ = image.shape
            img_data = image.mul(255).clamp(0, 255).to(torch.uint8).cpu().contiguous().numpy()
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))

            img_png = Image.fromarray(img_data)
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file = f"{filename_with_batch_num}.png"
            img_png.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": "output"
            })

            fmt = VkFormat.VK_FORMAT_R8G8B8_SRGB
            img_ktx2 = img_data[::-1].tobytes()

            tex_astc = KtxTexture2.create(KtxTextureCreateInfo(
                gl_internal_format=None,
                base_width=W,
                base_height=H,
                base_depth=1,
                vk_format=fmt,
            ), KtxTextureCreateStorage.ALLOC)
            tex_astc.set_image_from_memory(0, 0, 0, img_ktx2)
            tex_astc.compress_astc(KtxAstcParams(
                block_dimension=KtxPackAstcBlockDimension.D8x8,
                mode=KtxPackAstcEncoderMode.LDR,
                perceptual=True,
                quality_level=KtxPackAstcQualityLevels.EXHAUSTIVE,
                thread_count=os.cpu_count(),
                verbose=True
            ))
            file = f"{filename_with_batch_num}.png.astc.ktx2"
            tex_astc.write_to_named_file(os.path.join(full_output_folder, file))
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": "output"
            })

            tex_dxt1 = KtxTexture2.create(KtxTextureCreateInfo(
                gl_internal_format=None,
                base_width=W,
                base_height=H,
                base_depth=1,
                vk_format=fmt,
            ), KtxTextureCreateStorage.ALLOC)
            tex_dxt1.set_image_from_memory(0, 0, 0, img_ktx2)
            tex_dxt1.compress_basis(KtxBasisParams(
                compression_level=5,
                quality_level=255,
                thread_count=os.cpu_count(),
                uastc=True,
                # uastc_flags=KtxPackUastcFlagBits.SLOWER
            ))
            tex_dxt1.transcode_basis(KtxTranscodeFmt.BC1_RGB, KtxTranscodeFlagBits.HIGH_QUALITY)
            file = f"{filename_with_batch_num}.png.dxt1.ktx2"
            tex_dxt1.write_to_named_file(os.path.join(full_output_folder, file))
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": "output"
            })

        return { "ui": { "images": results } }

NODE_CLASS_MAPPINGS = {
    "Save KTX2": SaveKtx2
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Save KTX2": "Save KTX2"
}
