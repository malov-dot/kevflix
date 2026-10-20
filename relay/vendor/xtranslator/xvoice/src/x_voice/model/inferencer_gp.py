from __future__ import annotations

import os
import torch
import threading
from datetime import timedelta
import json
from random import choice
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from x_voice.infer.utils_infer import load_checkpoint, load_vocoder
from x_voice.model import CFM
from x_voice.model.dataset import DynamicBatchSampler, collate_fn_gp_inference
from x_voice.model.utils import default, exists,  convert_char_to_pinyin,  str_to_list_ipa_all # trim_text, available_texts, create_derangement,
from x_voice.model.modules import MelSpec
import torchaudio
from concurrent.futures import ThreadPoolExecutor


class Inferencer_gp:
    def __init__(
        self,
        model: CFM,
        checkpoint_path: str,
        root_path: str,
        batch_size_per_gpu=32,
        batch_size_type: str = "sample",
        max_samples=32,
        accelerate_kwargs: dict = dict(),
        mel_spec_type: str = "vocos", 
        tokenizer: str = "char",
        nfe_step: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        cfg_schedule=None,
        cfg_decay_time=0.0,
        reverse=True,
        layered=False,
        cfg_strength2=0.0,
        save_wav=False,
    ):
        process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=43200))
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs, process_group_kwargs],
            **accelerate_kwargs,
        )

        self.checkpoint_path = checkpoint_path
        self.root_path = root_path
        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.mel_spec_type = mel_spec_type
        self.tokenizer = tokenizer
        self.nfe_step = nfe_step
        self.cfg_strength = cfg_strength
        self.sway_sampling_coef = sway_sampling_coef
        self.cfg_schedule = cfg_schedule
        self.cfg_decay_time = cfg_decay_time
        self.layered = layered
        self.cfg_strength2 = cfg_strength2
        self.save_wav = save_wav
        self.executor = ThreadPoolExecutor(max_workers=16)

        dtype = torch.float32 if self.mel_spec_type == "bigvgan" else None
        from x_voice.infer.utils_infer import load_checkpoint
        model = load_checkpoint(model, checkpoint_path, str(self.accelerator.device), dtype=dtype, use_ema=True)
        if self.mel_spec_type == "vocos":
            vocoder_local_path = "my_vocoder/vocos-mel-24khz"
        elif self.mel_spec_type == "bigvgan":
            vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"
        self.vocoder = load_vocoder(vocoder_name=self.mel_spec_type, is_local=True, local_path=vocoder_local_path)
        self.model = self.accelerator.prepare(model)
        self.model.eval()

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def inference(self, dataset: Dataset, num_workers=16, resumable_with_seed: int = 666):
        generator = torch.Generator()
        if exists(resumable_with_seed):
            generator.manual_seed(resumable_with_seed)

        if self.batch_size_type == "sample":
            dataloader = DataLoader(
                dataset,
                collate_fn=collate_fn_gp_inference,
                num_workers=num_workers,
                pin_memory=True,
                batch_size=self.batch_size_per_gpu,
                shuffle=True,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(dataset)
            batch_sampler = DynamicBatchSampler(
                sampler,
                self.batch_size_per_gpu,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,
                drop_residual=False,
                # drop_last=False,
            )
            dataloader = DataLoader(
                dataset,
                collate_fn=collate_fn_gp_inference,
                num_workers=num_workers,
                pin_memory=True,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(f"batch_size_type must be 'sample' or 'frame'")

        dataloader = self.accelerator.prepare(dataloader)

        progress_bar = tqdm(
            range(len(dataloader)),
            desc="Inferencing",
            disable=not self.accelerator.is_local_main_process,
        )

        for batch in dataloader:
            mel_spec = batch["mel"].permute(0, 2, 1)
            
            self.process_batch(
                mel_spec, 
                text=batch["gen_text"], 
                lens=batch["mel_lengths"], 
                total_lens=batch["total_mel_len"],
                rel_paths=batch["rel_paths"],
                ref_text=batch["ref_text"],
                ref_text_ipa=batch["ref_text_ipa"],
                gen_text_ipa=batch["gen_text_ipa"],
                language_ids=batch["language_ids"]
            )
            
            progress_bar.update(1)

        self.accelerator.wait_for_everyone()
        print("Inference finished.")

    def process_batch(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722
        text: int["b nt"] | list[str],  # target text # noqa: F722 
        lens: int["b"] | None = None,  # reference mel lens # noqa: F821
        total_lens: int["b"] | None = None,
        rel_paths: str["b nt"] |list[str] = None, 
        ref_text: int["b nt"] | list[str] = None,
        ref_text_ipa: int["b nt"] | list[str] = None,
        gen_text_ipa: int["b nt"] | list[str] = None,
        language_ids: list[str] | None = None,
    ):
        all_files_exist = True
        for i in range(len(rel_paths)):
            pt_file_path = os.path.join(self.root_path, f"{rel_paths[i]}.pt")
            json_file_path = os.path.join(self.root_path, f"{rel_paths[i]}.json")
            
            if not (os.path.exists(pt_file_path) and os.path.exists(json_file_path)):
                all_files_exist = False
                break
        
        if all_files_exist:
            # Skip this batch.
            return

        batch, seq_len, device = *inp.shape[:2], self.accelerator.device

        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)
        
        text_source_input = []
        duration = torch.tensor(total_lens, dtype=torch.long, device=device)
        for i in range(batch):
            original = ref_text_ipa[i]
            changed = gen_text_ipa[i]
            if self.tokenizer == "pinyin":
                text_source_input.append(convert_char_to_pinyin([changed + " " + original])[0])
            elif self.tokenizer.startswith("ipa"):
                text_source_input.append(str_to_list_ipa_all(changed + " " + original, self.tokenizer))
            else:
                text_source_input.append(changed + " " + original)
        
        with torch.inference_mode():
            generated, _ = self.accelerator.unwrap_model(self.model).sample(
                cond=inp,
                text=text_source_input, # list[list[str]]
                duration=duration,
                lens=lens,
                steps=self.nfe_step,
                cfg_strength=self.cfg_strength,
                sway_sampling_coef=self.sway_sampling_coef,
                language_ids=language_ids,
                cfg_schedule=self.cfg_schedule,
                cfg_decay_time=self.cfg_decay_time,
                layered=self.layered,
                cfg_strength2=self.cfg_strength2,
                reverse=True,
            )
            generated = generated.to(torch.float32)
            generated_cpu = generated.cpu()

            for i in range(batch):
                curr_total_len = duration[i].item()
                curr_ref_len = lens[i].item()
                curr_gen_len = curr_total_len - curr_ref_len
                rel_path_no_suffix = os.path.splitext(rel_paths[i])[0]
                output_dir = os.path.join(self.root_path, os.path.dirname(rel_path_no_suffix))

                curr_gen_part = generated_cpu[i, :curr_gen_len, :]
                curr_gen_text_ipa = gen_text_ipa[i]
                curr_gen_text = text[i]
                
                pt_save_path = os.path.join(self.root_path, f"{rel_path_no_suffix}.pt")
                json_save_path = os.path.join(self.root_path, f"{rel_path_no_suffix}.json")
                wav_save_path = os.path.join(self.root_path, f"{rel_path_no_suffix}.wav")

                curr_gen_wav = None
                if self.save_wav:
                    gen_need_spec = curr_gen_part.unsqueeze(0).permute(0, 2, 1).to(generated.device)
                    if self.mel_spec_type == "vocos":
                        curr_gen_wav = self.vocoder.decode(gen_need_spec).cpu()
                    elif self.mel_spec_type == "bigvgan":
                        curr_gen_wav = self.vocoder(gen_need_spec).squeeze(0).cpu()

                self.executor.submit(
                    self.save_worker, 
                    curr_gen_part.clone(), 
                    pt_save_path, 
                    json_save_path, 
                    curr_gen_len, 
                    curr_gen_text, 
                    curr_gen_text_ipa,
                    wav_save_path,
                    curr_gen_wav,
                )
            
    def save_worker(self, tensor_data, pt_path, json_path, gen_len_val, text_content, text_ipa, wav_path=None, wav_tensor=None):
        os.makedirs(os.path.dirname(pt_path), exist_ok=True)
        torch.save(tensor_data, pt_path)
        metadata = {"gen_len": gen_len_val, "text": text_content, "text_ipa": text_ipa}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        if wav_path is not None and wav_tensor is not None:
            torchaudio.save(wav_path, wav_tensor, 24000)
