import os
import torch
import random
import numpy as np
import pdb

from torch.utils.data import Sampler, BatchSampler

from transformers import Trainer
from transformers.trainer import (
    has_length,
)
from typing import List, Optional, Dict, Sequence


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    assert len(mm_indices) > 0, "Should have at least one multimodal sample."
    assert len(lang_indices) > 0, "Should have at least one language sample."

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) >= megabatch_size:
        megabatches = [additional_batch[:megabatch_size]] + megabatches
        additional_batch = additional_batch[megabatch_size:]

    if len(additional_batch) > 0:
        megabatches.append(additional_batch)

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class StrictlyBalancedTaskBatchSampler(BatchSampler):
    """
    Strictly balanced task batch sampler. Ensures each batch contains tasks at the requested
    proportion. When a task runs out of samples, subsequent batches are filled from the
    remaining tasks.
    """
    
    def __init__(self, dataset, batch_size, drop_last=False, debug=False, track_batches=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.debug = debug
        self.track_batches = track_batches
        
        # Storage for per-batch indices
        self.batch_indices = []
        self.batch_task_counts = []
        
        # Collect sample indices grouped by task type and the reverse mapping
        self.indices_by_task = {}
        self.task_type_by_index = {}
        
        for i, sample in enumerate(dataset.list_data_dict):
            task_type = sample["task_type"]
            if task_type not in self.indices_by_task:
                self.indices_by_task[task_type] = []
            self.indices_by_task[task_type].append(i)
            self.task_type_by_index[i] = task_type
        
        self.task_types = list(self.indices_by_task.keys())
        self.num_tasks = len(self.task_types)
        
        # Compute the task distribution
        self.task_sizes = {t: len(indices) for t, indices in self.indices_by_task.items()}
        total_samples = sum(self.task_sizes.values())
        
        if self.debug:
            print(f"Task distribution in dataset:")
            for task, size in self.task_sizes.items():
                print(f"  - {task}: {size} samples ({size/total_samples*100:.2f}%)")
        
        # Decide the target proportion of each task per batch
        if self.num_tasks == 2:
            self.target_task_ratios = {t: 0.5 for t in self.task_types}
        else:
            self.target_task_ratios = {t: 1/self.num_tasks for t in self.task_types}
        
        # Compute the number of samples per task in a batch
        self.samples_per_task_in_batch = {}
        remaining = batch_size
        
        for i, task in enumerate(self.task_types):
            if i == self.num_tasks - 1:
                self.samples_per_task_in_batch[task] = remaining
            else:
                count = int(batch_size * self.target_task_ratios[task])
                self.samples_per_task_in_batch[task] = count
                remaining -= count
        
        if self.debug:
            print(f"Target samples per task in each batch (size {batch_size}):")
            for task, count in self.samples_per_task_in_batch.items():
                print(f"  - {task}: {count} samples ({count/batch_size*100:.2f}%)")
        
        # Estimate the total number of batches (accounting for fill-in from other tasks)
        total_batches = total_samples // batch_size
        if not drop_last and total_samples % batch_size > 0:
            total_batches += 1
        self.total_batches = total_batches
    
    def __iter__(self):
        # Reset the per-batch records
        self.batch_indices = []
        self.batch_task_counts = []
        
        # Copy and shuffle the indices for each task type
        indices_by_task = {}
        for task_type in self.task_types:
            indices = self.indices_by_task[task_type].copy()
            random.shuffle(indices)
            indices_by_task[task_type] = indices
        
        # Track the consumed position for each task type
        position_by_task = {t: 0 for t in self.task_types}
        
        # Batch counter
        batch_count = 0
        
        # Keep yielding batches until every sample is consumed
        total_samples_left = sum(len(indices_by_task[t]) - position_by_task[t] for t in self.task_types)
        
        while total_samples_left > 0:
            # If the remaining samples cannot fill a batch and drop_last is set, stop
            if total_samples_left < self.batch_size and self.drop_last:
                break
            
            # Target batch size
            target_batch_size = min(self.batch_size, total_samples_left)
            
            # Determine which tasks still have samples available
            available_tasks = [t for t in self.task_types if position_by_task[t] < len(indices_by_task[t])]
            
            # If no tasks are available, stop generating batches
            if not available_tasks:
                break
            
            batch = []
            
            if len(available_tasks) == self.num_tasks:
                # All tasks still have samples, use the standard balanced layout
                for task in self.task_types:
                    # Do not exceed the number of available samples
                    samples_for_task = min(
                        self.samples_per_task_in_batch[task],
                        len(indices_by_task[task]) - position_by_task[task]
                    )
                    
                    start = position_by_task[task]
                    end = start + samples_for_task
                    task_samples = indices_by_task[task][start:end]
                    batch.extend(task_samples)
                    position_by_task[task] = end
            else:
                # Some tasks are exhausted; redistribute the batch slots
                # Compute how many samples each remaining task should contribute
                remaining_slots = target_batch_size
                samples_per_available_task = {}
                
                if len(available_tasks) == 1:
                    # Only one task has samples left, fill the batch entirely from it
                    task = available_tasks[0]
                    samples_per_available_task[task] = min(
                        remaining_slots,
                        len(indices_by_task[task]) - position_by_task[task]
                    )
                else:
                    # Multiple tasks have samples left, allocate proportionally
                    # Compute the total weight of the available tasks
                    available_weights = sum(self.target_task_ratios[t] for t in available_tasks)
                    
                    # Reallocate the slots according to the ratios
                    for i, task in enumerate(available_tasks):
                        if i == len(available_tasks) - 1:
                            # Last task absorbs all the remaining slots
                            alloc = remaining_slots
                        else:
                            # Allocate by weight
                            alloc = int(target_batch_size * (self.target_task_ratios[task] / available_weights))
                            remaining_slots -= alloc
                        
                        # Do not exceed the number of available samples
                        samples_per_available_task[task] = min(
                            alloc,
                            len(indices_by_task[task]) - position_by_task[task]
                        )
                
                # Collect the allocated samples
                for task, samples_count in samples_per_available_task.items():
                    start = position_by_task[task]
                    end = start + samples_count
                    task_samples = indices_by_task[task][start:end]
                    batch.extend(task_samples)
                    position_by_task[task] = end
                
                # If there are still empty slots, try another allocation pass
                remaining_batch_slots = target_batch_size - len(batch)
                
                if remaining_batch_slots > 0:
                    # Re-check which tasks still have samples available
                    still_available = [t for t in self.task_types if position_by_task[t] < len(indices_by_task[t])]
                    
                    while remaining_batch_slots > 0 and still_available:
                        # Distribute the remaining slots in round-robin order
                        for task in list(still_available):  # iterate over a copy to allow modification
                            if position_by_task[task] >= len(indices_by_task[task]):
                                still_available.remove(task)
                                continue
                            
                            # Add one sample
                            batch.append(indices_by_task[task][position_by_task[task]])
                            position_by_task[task] += 1
                            remaining_batch_slots -= 1
                            
                            if remaining_batch_slots == 0:
                                break
            
            # Shuffle the batch
            random.shuffle(batch)
            
            # Validate the task distribution inside the batch
            if self.debug or self.track_batches:
                task_types_in_batch = [self.task_type_by_index[idx] for idx in batch]
                task_count = {}
                for t in task_types_in_batch:
                    task_count[t] = task_count.get(t, 0) + 1
                
                if self.debug:
                    available_msg = "All tasks available" if len(available_tasks) == self.num_tasks else f"Available tasks: {available_tasks}"
                    print(f"Batch {batch_count} - {available_msg}")
                    print(f"Batch {batch_count} - Task distribution: {task_count}")
                    print(f"Batch {batch_count} - Batch size: {len(batch)}")
                
                # Track batch indices and task counts
                if self.track_batches:
                    self.batch_indices.append(batch)
                    self.batch_task_counts.append(task_count)
            
            batch_count += 1
            yield batch
            
            # Update the total number of remaining samples
            total_samples_left = sum(len(indices_by_task[t]) - position_by_task[t] for t in self.task_types)
    
    def __len__(self):
        return self.total_batches
    
    def get_batch_statistics(self):
        """
        Return batch statistics, including per-batch indices and task counts.
        """
        if not self.track_batches:
            raise ValueError("Batch tracking is not enabled. Initialize with track_batches=True to use this method.")
        
        # Compute task ratios for each batch
        batch_ratios = []
        for counts in self.batch_task_counts:
            batch_size = sum(counts.values())
            ratios = {task: count/batch_size for task, count in counts.items()}
            batch_ratios.append(ratios)
        
        # Compute the average task ratios
        avg_ratios = {}
        for task in self.task_types:
            task_ratios = [ratios.get(task, 0) for ratios in batch_ratios]
            avg_ratios[task] = sum(task_ratios) / len(batch_ratios) if batch_ratios else 0
        
        return {
            "batch_indices": self.batch_indices,
            "batch_task_counts": self.batch_task_counts,
            "batch_ratios": batch_ratios,
            "average_ratios": avg_ratios,
            "total_batches": len(self.batch_indices)
        }
    
    def print_batch_summary(self):
        """
        Print a summary of the batch statistics.
        """
        if not self.track_batches:
            raise ValueError("Batch tracking is not enabled. Initialize with track_batches=True to use this method.")
        
        stats = self.get_batch_statistics()
        
        print(f"Batch Summary Statistics:")
        print(f"Total batches: {stats['total_batches']}")
        print(f"Average task ratios: {stats['average_ratios']}")
        
        print(f"\nBatch-by-batch statistics:")
        for i, (counts, ratios) in enumerate(zip(stats['batch_task_counts'], stats['batch_ratios'])):
            batch_size = sum(counts.values())
            print(f"Batch {i} (size {batch_size}):")
            for task in self.task_types:
                count = counts.get(task, 0)
                ratio = ratios.get(task, 0)
                print(f"  - {task}: {count} samples ({ratio*100:.2f}%)")


class BalancedTaskCollator:
    """
    Balanced-task data collator that wraps a base collator.
    """
    def __init__(self, base_collator):
        self.base_collator = base_collator
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return self.base_collator(instances)


class GraphLLaVATrainer(Trainer):
    """
    GraphLLaVA trainer that supports strictly balanced tasks.
    """
    def __init__(self, balance_tasks=False, debug_sampler=False, track_batches=False, **kwargs):
        self.balance_tasks = balance_tasks
        self.debug_sampler = debug_sampler
        self.track_batches = track_batches
        super().__init__(**kwargs)
        
        if self.balance_tasks and hasattr(self, 'data_collator'):
            self.data_collator = BalancedTaskCollator(self.data_collator)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        
        if self.balance_tasks:
            return None
        elif self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()
    
    def get_train_dataloader(self):
        if self.balance_tasks:
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")
            
            # Use the balanced-task batch sampler
            batch_sampler = StrictlyBalancedTaskBatchSampler(
                dataset=self.train_dataset,
                batch_size=self.args.train_batch_size * max(1, self.args.n_gpu),
                drop_last=self.args.dataloader_drop_last,
                debug=self.debug_sampler,
                track_batches=self.track_batches
            )
            
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        else:
            return super().get_train_dataloader()

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            projector_list = ['mm_projector_graph', 'mm_projector_image', 'mm_projector_text', 'mm_projector_node', 
                             "mm_projector_nc", "mm_projector_lp", "mm_projector_shared_first_layer", 
                             "mm_projector_shared", "common_projector", "expert_projectors", "router", "cross_task_attention"]
            
            for proj_key in projector_list:
                if not hasattr(self.model.get_model(), proj_key):
                    continue
                
                keys_to_match = [proj_key]
                
                if getattr(self.args, "use_graph_start_end", False):
                    keys_to_match.extend(['embed_tokens', 'embed_in'])
                if getattr(self.args, "mm_use_graph_special_token", False):
                    keys_to_match.extend(['special_token_emb'])

                weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

                if self.args.local_rank == 0 or self.args.local_rank == -1:
                    self.model.config.save_pretrained(output_dir)
                    torch.save(weight_to_save, os.path.join(output_dir, f'{proj_key}.bin'))
            if hasattr(self.model.get_model(), "task_token_embeddings"):
                task_tokens = {}
                for name, param in self.model.get_model().task_token_embeddings.named_parameters():
                    task_tokens[name] = maybe_zero_3(param, ignore_status=True, name=name).cpu()
                
                if self.args.local_rank == 0 or self.args.local_rank == -1:
                    torch.save(task_tokens, os.path.join(output_dir, 'task_token_embeddings.bin'))
            if getattr(self.args, 'lora_enable', False):               
                state_dict = get_peft_state_maybe_zero_3(
                        self.model.named_parameters(), self.args.lora_bias
                    )
                non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                        self.model.named_parameters()
                    )
                if self.args.local_rank == 0 or self.args.local_rank == -1:
                    self.model.config.save_pretrained(output_dir)
                    self.model.save_pretrained(output_dir, state_dict=state_dict)
                    torch.save(non_lora_state_dict, os.path.join(output_dir, 'non_lora_trainables.bin'))
        else:
            super(GraphLLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(GraphLLaVATrainer, self)._save(output_dir, state_dict)
            
    def print_batch_statistics(self):
        if not self.balance_tasks or not self.track_batches:
            raise ValueError("This method requires balance_tasks=True and track_batches=True.")
        
        # Extract the sampler from the dataloader
        dataloader = self.get_train_dataloader()
        batch_sampler = dataloader.batch_sampler
        
        # Print statistics
        batch_sampler.print_batch_summary()
        
        return batch_sampler.get_batch_statistics()