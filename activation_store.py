# MIT License

# Copyright (c) 2025 bartbussmann

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



import torch
from transformer_lens.hook_points import HookedRootModule
from datasets import load_dataset
from sae import TrainingConfig
import transformer_lens.utils as utils
import numpy as np
from loading import load_with_retries

import os
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class ReplayBufferNew:
    def __init__(self, buffer_size: int, batch_size: int, device: str):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = None
        self.position = 0                # write head (0..buffer_size-1)
        self.size = 0                    # number of valid entries (0..buffer_size)
        self.used_positions = set()      # indices sampled since last “repay”
        self.device = device

    def _write_circular(self, start: int, x: torch.Tensor):
        """Write tensor x starting at index `start` with circular wrap; return new head index."""
        n = x.shape[0]
        end = start + n
        if end <= self.buffer_size:
            self.buffer[start:end] = x
        else:
            first = self.buffer_size - start
            self.buffer[start:] = x[:first]
            self.buffer[:end % self.buffer_size] = x[first:]
        return end % self.buffer_size

    def add(self, activations: torch.Tensor) -> int:
        """
        Add activations to the buffer.
        Returns the number of entries that overwrote previously sampled indices
        (i.e., how many we ‘repaid’ toward clearing used_positions).
        """
        #print(f"inside add")
        activations = activations.cpu()
        if self.buffer is None:
            self.buffer = torch.zeros(
                (self.buffer_size,) + activations.shape[1:], dtype=activations.dtype
            )
            print(f"buffer is none, initing it as tensor with shape {self.buffer.shape}")

        n = activations.shape[0]
        repaid = 0

        # 1) Overwrite sampled slots first (reuse them)
        if self.used_positions and n > 0:
            take = min(len(self.used_positions), n)
            #print(f"buffer partially filled, cur_size is {len(self.used_positions)}")
            # Grab any `take` positions from the set
            positions = list(self.used_positions)[:take]
            self.buffer[positions] = activations[:take]
            self.used_positions.difference_update(positions)
            repaid = take
            remaining = activations[take:]
            #print(f"after writing to buffer {remaining.shape} acts left")
        else:
            remaining = activations

        # 2) Write the remaining activations at the circular head
        if remaining.numel() > 0:
            self.position = self._write_circular(self.position, remaining)

        # 3) Update the number of valid entries
        self.size = min(self.buffer_size, self.size + n)
        #print(f"cur buffer size after add is {self.size}")

        return repaid

    def sample(self) -> torch.Tensor:
        if self.buffer is None or self.size == 0:
            raise RuntimeError("Buffer is empty")

        # Sample from all valid slots (or full buffer if full), excluding those already used
        all_valid = set(range(self.buffer_size if self.size == self.buffer_size else self.size))
        available_positions = all_valid - self.used_positions

        if len(available_positions) < self.batch_size:
            # Not enough fresh entries → allow reuse this step, then reset freshness window
            self.used_positions.clear()
            available_positions = all_valid
            print("Warning! Training on some tokens a second time")

        sample_positions = torch.tensor(list(available_positions))
        perm = torch.randperm(len(sample_positions))[:self.batch_size]
        sample_positions = sample_positions[perm]
        self.used_positions.update(sample_positions.tolist())

        return self.buffer[sample_positions].to(self.device)

    def is_full(self) -> bool:
        return self.size == self.buffer_size

    @torch.inference_mode()
    def get_mean_std(self):
        if self.buffer is None or self.size == 0:
            raise RuntimeError("Buffer is empty")
        view = self.buffer if self.size == self.buffer_size else self.buffer[:self.size]
        return view.mean().item(), view.std(-1).mean().item()

    def get_occupancy(self):
        # Real capacity utilization
        return 0.0 if self.buffer is None else (self.size / self.buffer_size)

    def get_fresh_fraction(self) -> float:
        """
        Fraction of the *next* batch that can be drawn from entries not sampled
        since the last reset. With the repay logic in get_batch(), this should
        stay ~1.0 in steady state.
        """
        if self.buffer is None or self.batch_size == 0:
            return 0.0
        valid = self.buffer_size if self.size == self.buffer_size else self.size
        fresh = max(0, valid - len(self.used_positions))
        return min(1.0, fresh / self.batch_size)


class ActivationsStore:
    def __init__(
        self,
        model,
        cfg: TrainingConfig,
    ):
        self.model = model
        self.cfg = cfg
        self.context_size = cfg.seq_len
        self.model_batch_size = cfg.model_batch_size
        self.device = cfg.device
        self.num_batches_in_buffer = cfg.num_batches_in_buffer
        if "qwen" in model.cfg.tokenizer_name.lower():
            self.prepend_bos = False
            print("Qwen tokenizer detected, setting prepend_bos_token to False.")
        else:
            self.prepend_bos = True
        self.dataset = iter(load_with_retries(
            lambda: load_dataset(cfg.dataset_path, split="train", streaming=True),
            "loading activation dataset",
        ))
        

        self.hook_point = utils.get_act_name(cfg.hook_point, cfg.layer)
        print(f"self.hook_point = {self.hook_point}")

        self.tokens_column = self._get_tokens_column()
        self.cfg = cfg
        self.tokenizer = model.tokenizer

        self.compiled_forward = torch.compile(self.model.run_with_cache)

        buffer_size = self.num_batches_in_buffer * cfg.batch_size

        self.replay_buffer = ReplayBufferNew(
            buffer_size=buffer_size,
            batch_size=cfg.batch_size,
            device=cfg.device,
        )
        # Initial fill of buffer
        print("Filling buffer...")
        self._fill_buffer()
        print("Buffer filled.")

        self.mean, self.std = self.replay_buffer.get_mean_std()

    def _get_tokens_column(self):
        sample = load_with_retries(
            lambda: next(self.dataset), "loading first dataset sample"
        )
        if "tokens" in sample:
            return "tokens"
        elif "input_ids" in sample:
            return "input_ids"
        elif "text" in sample:
            return "text"
        else:
            raise ValueError("Dataset must have a 'tokens', 'input_ids', or 'text' column.")

    def get_batch_tokens(self):    
        all_tokens = []
        while len(all_tokens) < self.model_batch_size * (self.context_size + 1):
            try:
                seq = next(self.dataset)
            except StopIteration:
                self.dataset = self.dataset.shuffle()
                self.iter_dataset = iter(self.dataset)
                seq = next(self.dataset)
            if self.tokens_column == "text":
                tokens = self.model.to_tokens(seq["text"], truncate=True, move_to_device=True, prepend_bos=self.prepend_bos).squeeze(0)
            else:
                tokens = seq[self.tokens_column]
            all_tokens.extend(tokens)
        token_tensor = torch.tensor(all_tokens, dtype=torch.long, device=self.device)[:self.model_batch_size * (self.context_size + 1)]

        return token_tensor.view(self.model_batch_size, self.context_size + 1)


    def get_activations(self, batch_tokens: torch.Tensor):      
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, cache = self.model.run_with_cache(
                    batch_tokens,
                    names_filter=[self.hook_point],
                    stop_at_layer=self.cfg.layer + 1,
                )
        return cache[self.hook_point][:, 1:, :]  # drop BOS   

    def _fill_buffer(self):
        # Fill until the buffer is full using the correct validity metric
        while self.replay_buffer.size < self.replay_buffer.buffer_size:
            batch_tokens = self.get_batch_tokens()
            activations = self.get_activations(batch_tokens)
            if isinstance(self.hook_point, tuple):
                activations = (activations[0].reshape(-1, self.cfg.act_size), activations[1].reshape(-1, self.cfg.act_size))  # drop BOS
            else:
                activations = activations.reshape(-1, self.cfg.act_size)
            self.replay_buffer.add(activations)    

    def get_batch(self, add_new: bool = True):
        outp = self.replay_buffer.sample()

        # REPAY: ensure we actually overwrite at least projector_batch_size sampled slots
        if add_new:
            repaid = 0
            while repaid < self.cfg.batch_size:
                new_activations = self.get_activations(self.get_batch_tokens())
                if isinstance(self.hook_point, tuple):
                    new_activations = (new_activations[0].reshape(-1, self.cfg.act_size), new_activations[1].reshape(-1, self.cfg.act_size)) 
                else:
                    new_activations = new_activations.reshape(-1, self.cfg.act_size)
               
                repaid += self.replay_buffer.add(new_activations)

        # At this point, used_positions should be ~0 and fresh_fraction near 1
        return outp
