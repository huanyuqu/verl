# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This trainer supports model-agonistic model initialization with huggingface
"""

from collections import deque
import uuid
from pprint import pprint

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from recipe.one_step_off_policy.utils import need_critic
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_reference_policy, need_reward_model
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.tracking import ValidationGenerationsLogger
from recipe.one_step_off_policy.ray_trainer import OneStepOffRayTrainer


class GenerationBatchFuture:
    """
    Wrapper class for encapsulating batch generation results
    """

    def __init__(self, epoch, batch, gen_batch_output):
        """
        :param epoch: current epoch
        :param batch: Input batch data
        :param gen_batch_output: Generated sequences from the main model (DataProtoFuture)
        """
        self.epoch = epoch
        self.batch = batch
        self.gen_batch_output = gen_batch_output

    def get(self):
        """
        Get the actual results by calling get() method on gen_batch_output

        Returns:
            tuple: (batch, gen_batch_result)
                - batch: Original input batch data
                - gen_batch_result: Result from gen_batch_output.get() or gen_batch_output itself
        """
        # Call get() method on gen_batch_output if available
        if hasattr(self.gen_batch_output, "get"):
            gen_batch_result = self.gen_batch_output.get()
        else:
            gen_batch_result = self.gen_batch_output

        return self.epoch, self.batch, gen_batch_result


class HalfStepOffRayTrainer(OneStepOffRayTrainer):
    def __init__(self, config, tokenizer, role_worker_mapping,  
                 resource_pool_manager, ray_worker_group_cls = ..., processor=None, reward_fn=None, val_reward_fn=None, train_dataset = None, val_dataset = None, collate_fn=None, train_sampler = None, device_name="cuda"):
        super().__init__(config, tokenizer, role_worker_mapping, 
                         resource_pool_manager, ray_worker_group_cls, processor, reward_fn, val_reward_fn, train_dataset, val_dataset, collate_fn, train_sampler, device_name)

    def _launch_rollout_tasks_for_batch(self, batch_dict):
        """
        Launch asynchronous generation tasks for each prompt in the batch.
        Each task corresponds to `n` rollouts for a single prompt.
        """
        batch = DataProto.from_single_dict(batch_dict)
        num_rollouts_per_prompt = self.config.actor_rollout_ref.rollout.n

        # Prepare the generation inputs by popping generation-specific keys
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "interaction_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("interaction_kwargs")

        gen_batch_base = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        pending_futures = []
        
        # Iterate over each original prompt in the batch
        for i in range(len(gen_batch_base)):
            original_data_for_prompt = batch[i : i + 1]
            gen_input_for_prompt = gen_batch_base[i : i + 1]
            gen_input_repeated = gen_input_for_prompt.repeat(
                repeat_times=num_rollouts_per_prompt, interleave=False
            )

            # Launch an independent async generation task for this prompt's rollouts
            gen_future = self.rollout_wg.async_generate_sequences(gen_input_repeated)
            
            # Store the future and the corresponding original data
            pending_futures.append((gen_future, original_data_for_prompt))
            
        return pending_futures

    def _collate_experiences(self, completed_rollouts):
        """
        Collate completed rollout experiences into a single DataProto batch for training.
        """
        if not completed_rollouts:
            return None

        all_original_data = []
        all_gen_outputs = []

        for original_data, gen_output in completed_rollouts:
            all_original_data.append(original_data)
            all_gen_outputs.append(gen_output)

        # Concatenate all original data parts and all generation outputs
        collated_original_data = DataProto.cat(all_original_data)
        collated_gen_outputs = DataProto.cat(all_gen_outputs)

        # Repeat the original data to match the number of rollouts per prompt
        num_rollouts_per_prompt = self.config.actor_rollout_ref.rollout.n
        collated_original_data = collated_original_data.repeat(
            repeat_times=num_rollouts_per_prompt, interleave=True
        )

        # Combine with generation results to form the final training batch
        final_batch = collated_original_data.union(collated_gen_outputs)
        return final_batch

    def _perform_ppo_step(self, batch, epoch, logger):
        """
        Execute a single PPO training step on the given batch of experiences.
        """
        metrics = {}
        timing_raw = {}
        is_last_step = self.global_steps >= self.total_training_steps

        with marked_timer("step", timing_raw):
            batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
            )
            
            batch.batch["response_mask"] = compute_response_mask(batch)
            if self.config.trainer.balance_batch:
                self._balance_batch(batch, metrics=metrics)

            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

            with marked_timer("reward", timing_raw, color="yellow"):
                if self.use_rm:
                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                    batch = batch.union(reward_tensor)
                if self.config.reward_model.launch_reward_fn_async:
                    future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                else:
                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

            with marked_timer("old_log_prob", timing_raw, color="blue"):
                old_log_prob = self.actor_wg.compute_log_prob(batch)
                batch = batch.union(old_log_prob)

            if self.use_reference_policy:
                with marked_timer("ref", timing_raw, color="olive"):
                    if not self.ref_in_actor:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    else:
                        ref_log_prob = self.actor_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

            if self.use_critic:
                with marked_timer("values", timing_raw, color="cyan"):
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)

            with marked_timer("adv", timing_raw, color="brown"):
                if self.config.reward_model.launch_reward_fn_async:
                    reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                batch.batch["token_level_scores"] = reward_tensor
                if reward_extra_infos_dict:
                    batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                if self.config.algorithm.use_kl_in_reward:
                    batch, kl_metrics = apply_kl_penalty(
                        batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                    )
                    metrics.update(kl_metrics)
                else:
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                batch = compute_advantage(
                    batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                    norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                    config=self.config.algorithm,
                )

            if self.use_critic:
                with marked_timer("update_critic", timing_raw, color="pink"):
                    critic_output = self.critic_wg.update_critic(batch)
                metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

            if self.config.trainer.critic_warmup <= self.global_steps:
                with marked_timer("update_actor", timing_raw, color="red"):
                    batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    actor_output = self.actor_wg.update_actor(batch)
                metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

        # --- Logging and Validation ---
        if (self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and
            (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)):
            with marked_timer("testing", timing_raw, color="green"):
                val_metrics = self._validate()
            metrics.update(val_metrics)

        if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
            with marked_timer("save_checkpoint", timing_raw, color="green"):
                self._save_checkpoint()

        metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

        logger.log(data=metrics, step=self.global_steps)

    def fit(self):
        """
        The training loop of PPO.
        This version implements a pipelined approach where the next rollout (N+1)
        starts immediately after the current training step (N) finishes, running
        concurrently with any stragglers from rollout (N).
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        continuous_iterator = self._create_continuous_iterator()
        training_trigger_ratio = self.config.trainer.get("training_trigger_ratio", 0.9)

        # State for managing asynchronous pipeline
        straggler_futures = []  # Futures of stragglers from the previous rollout
        completed_stragglers = []  # Completed results from previous stragglers
        current_rollout_futures = [] # Futures for the current rollout batch

        # --- Kick-off the first rollout ---
        try:
            epoch, first_batch_dict = next(continuous_iterator)
            self.sync_rollout_weights()
            current_rollout_futures = self._launch_rollout_tasks_for_batch(first_batch_dict)
        except StopIteration:
            print("Dataset is empty. Training finished.")
            progress_bar.close()
            return
        except Exception as e:
            print(f"Error in launch_rollout_tasks_for_batch: {e}")
            return None

        while self.global_steps < self.total_training_steps:
            # ------------------------------------------------------------------
            # STEP 1: PREPARE TRAINING DATA (for Train(N))
            # ------------------------------------------------------------------
            # Wait for all stragglers from the previous step (Rollout(N-1)) to complete.
            # For the first iteration, this list is empty.
            if straggler_futures:
                print(f"Waiting for {len(straggler_futures)} stragglers from the previous step to complete...")
                remaining_results = ray.get([f[0] for f in straggler_futures])
                for i, (future, data) in enumerate(straggler_futures):
                    completed_stragglers.append((data, remaining_results[i]))
                straggler_futures.clear()

            # Wait for the trigger ratio of the current rollout (Rollout(N)) to complete.
            trigger_threshold = int(len(current_rollout_futures) * training_trigger_ratio)
            completed_early_birds = []
            
            if current_rollout_futures:
                print(f"Waiting for {trigger_threshold}/{len(current_rollout_futures)} early-bird rollouts to complete...")
                while len(completed_early_birds) < trigger_threshold and current_rollout_futures:
                    ready_refs, _ = ray.wait([f[0] for f in current_rollout_futures], num_returns=1)
                    ready_futures_set = set(ready_refs)
                    
                    remaining_futures_for_current_rollout = []
                    for future, data in current_rollout_futures:
                        if future in ready_futures_set:
                            gen_output = ray.get(future)
                            completed_early_birds.append((data, gen_output))
                        else:
                            remaining_futures_for_current_rollout.append((future, data))
                    current_rollout_futures = remaining_futures_for_current_rollout
            
            # The remaining tasks in `current_rollout_futures` are the stragglers for this step (Train(N)).
            # Move them to the straggler buffer to be handled in the next iteration.
            straggler_futures.extend(current_rollout_futures)
            current_rollout_futures = []

            # Combine "early-bird" samples with completed stragglers from the PREVIOUS batch.
            experiences_to_train = completed_early_birds + completed_stragglers
            completed_stragglers.clear()

            # ------------------------------------------------------------------
            # STEP 2: TRAIN (N) and LAUNCH ROLLOUT (N+1)
            # ------------------------------------------------------------------
            training_batch = self._collate_experiences(experiences_to_train)
            
            if training_batch:
                self.global_steps += 1
                progress_bar.update(1)
                
                # Perform the PPO step (blocking call)
                self._perform_ppo_step(training_batch, epoch, logger)

                # Immediately launch the next rollout (Rollout(N+1)) after training is done.
                if self.global_steps < self.total_training_steps:
                    try:
                        epoch, next_batch_dict = next(continuous_iterator)
                        print(f"Train step {self.global_steps} finished. Launching next rollout batch...")
                        self.sync_rollout_weights()
                        current_rollout_futures = self._launch_rollout_tasks_for_batch(next_batch_dict)
                    except StopIteration:
                        print("Reached end of dataset.")
                        # No more data to launch, loop will continue to process remaining stragglers.
            else:
                print(f"Skipping training for step {self.global_steps} due to empty batch.")
                if not straggler_futures:
                    # If there's no training data and no pending stragglers, we might be done.
                    break

        progress_bar.close()
        print("Training finished.")