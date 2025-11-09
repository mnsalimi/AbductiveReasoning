#!/usr/bin/env python3
"""
Multi-Evaluation Orchestrator
Runs multiple evaluation scripts with configurable parameters and organized logging.
Finds the best checkpoint from the training directory and passes it to all
sub-scripts for consistent evaluation.
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import concurrent.futures
import time
import threading

# ============================================================================
# CONFIGURATION SECTION - EASILY MODIFIABLE
# ============================================================================

# Shared model/training paths (will be injected into evaluation scripts)
RAW_MODEL_PATH = "/home/moein_salimi/PLLMS/unsloth-Qwen2.5-3B-Instruct-unsloth-bnb-4bit"
TRAINING_DIR = "/home/moein_salimi/users/Nima/AbductiveReasoning/GRPO/results/dt11.04.22:13_e20_unsloth_Qwen2.5_3B_Instruct_unsloth_bnb_4bit_bnb_4bit_lr1e-05_t0.7_ε0.2_r64_b16"
BASE_OUTPUT_DIR = "/home/moein_salimi/users/Nima/AbductiveReasoning/GRPO"

# List of evaluation scripts to run
EVALUATION_SCRIPTS = [
    {
        'script': 'evaluate_aimo_raw_vs_finetuned.py',
        'name': 'AIMO Dataset Evaluation',
        'output_subdir': 'aimo_evaluation_results',
        'params': {
            'split': 'test',
        },
        'override_terminal': False
    },
    {
        'script': 'evaluate_aime_raw_vs_finetuned.py',
        'name': 'AIME 2025 Dataset Evaluation',
        'output_subdir': 'aime_evaluation_results',
        'params': {
            'split': 'train'
        },
        'override_terminal': False
    },
    {
        'script': 'evaluate_copa_raw_vs_finetuned_guess_cause.py',
        'name': 'COPA Dataset Evaluation (Guess Cause)',
        'output_subdir': 'copa_evaluation_guess_cause_results',
        'params': {
            'split': 'train',
        },
        'override_terminal': False
    },
    {
        'script': 'evaluate_copa_raw_vs_finetuned_guess_effect.py',
        'name': 'COPA Dataset Evaluation (Guess effect)',
        'output_subdir': 'copa_evaluation_guess_effect_results',
        'params': {
            'split': 'train',
        },
        'override_terminal': False
    },
    {
        'script': 'evaluate_art_raw_vs_finetuned.py',
        'name': 'ART Dataset Evaluation',
        'output_subdir': 'art_evaluation_results',
        'params': {
        },
        'override_terminal': False
    },
    {
        'script': 'evaluate_goEmotion_raw_vs_finetuned.py',
        'name': 'GoEmotion Dataset Evaluation',
        'output_subdir': 'goEmotion_evaluation_results',
        'params': {
            'split': 'test',
        },
        'override_terminal': False
    },
    {
        'script': 'evaluate_gsm8k_raw_vs_finetuned.py',
        'name': 'GSM8K Dataset Evaluation',
        'output_subdir': 'gsm8k_evaluation_results',
        'params': {
            'split': 'test',
        },
        'override_terminal': False
    },
]

# Default parameters shared across all scripts
DEFAULT_PARAMS = {
    'cuda_device': '3',
    'batch_size': 1,
    'max_samples': None,
    'skip_raw': False,
    'skip_finetuned': False,
    'checkpoint_path': None,
    'checkpoint_dir': None,
}

# Parallel execution settings
DEFAULT_PARALLEL_COUNT = 1

# CUDA devices for parallel execution (will cycle through these IN ORDER)
CUDA_DEVICES = ['3']

# Output directory for consolidated orchestrator results
ORCHESTRATOR_OUTPUT_DIR = './multi_evaluation_results'

# ============================================================================
# END OF CONFIGURATION SECTION
# ============================================================================


class EvaluationOrchestrator:
    """Manages execution of multiple evaluation scripts with organized logging."""
    
    def __init__(self, output_dir: str, parallel_count: int = 1,
        raw_model_path: str = None, training_dir: str = None,
        base_output_dir: str = None, realtime_logs: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parallel_count = parallel_count
        self.realtime_logs = realtime_logs
        
        # Thread-safe printing lock
        self.print_lock = threading.Lock()
        
        # Store paths for injection
        self.raw_model_path = raw_model_path or RAW_MODEL_PATH
        self.training_dir = training_dir or TRAINING_DIR
        self.base_output_dir = base_output_dir or BASE_OUTPUT_DIR
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.output_dir / f'run_{timestamp}'
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Master log file
        self.master_log = self.run_dir / 'master_log.txt'
        
    def find_best_checkpoint(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Finds the best checkpoint based on 'val_metrics.json' or falls back to the latest.
        Returns (path_to_checkpoint, reason_string).
        """
        print(f"\n{'='*70}")
        print(f"🔍 CHECKPOINT SELECTION PROCESS")
        print(f"{'='*70}")
        
        val_metrics_path = os.path.join(self.training_dir, "val_metrics.json")
        checkpoint_dir = os.path.join(self.training_dir, "checkpoint")
        
        print(f"📁 Training directory: {self.training_dir}")
        print(f"📁 Checkpoint directory: {checkpoint_dir}")
        print(f"📄 Val metrics file: {val_metrics_path}")
        print()

        # Check if checkpoint directory exists
        if not os.path.exists(checkpoint_dir):
            reason = "Checkpoint directory not found."
            print(f"❌ {reason}")
            print(f"{'='*70}\n")
            return None, reason

        # Find all checkpoints
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
        if not checkpoints:
            reason = "No checkpoints found in the directory."
            print(f"❌ {reason}")
            print(f"{'='*70}\n")
            return None, reason
        
        print(f"✅ Found {len(checkpoints)} checkpoint(s):")
        
        # Parse checkpoint steps and display them
        try:
            checkpoint_steps = [(int(c.split('-')[1]), c) for c in checkpoints]
            checkpoint_steps.sort()
            
            # Display all checkpoints
            for step, name in checkpoint_steps:
                print(f"   • {name} (step {step})")
            
            latest_checkpoint_name = checkpoint_steps[-1][1]
            latest_checkpoint_step = checkpoint_steps[-1][0]
            latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint_name)
            
            print(f"\n📌 Latest checkpoint: {latest_checkpoint_name} (step {latest_checkpoint_step})")
            
        except (ValueError, IndexError) as e:
            reason = f"Could not parse checkpoint numbers: {e}"
            print(f"❌ {reason}")
            print(f"{'='*70}\n")
            return None, reason

        # Check if validation metrics exist
        if not os.path.exists(val_metrics_path):
            reason = "No val_metrics.json found, using latest checkpoint."
            print(f"\n⚠️  {reason}")
            print(f"🎯 Selected: {latest_checkpoint_path}")
            print(f"{'='*70}\n")
            return latest_checkpoint_path, reason

        # Load and analyze validation metrics
        try:
            print(f"\n📊 Loading validation metrics...")
            with open(val_metrics_path, 'r') as f:
                val_metrics = json.load(f)
            
            print(f"✅ Found metrics for {len(val_metrics)} epoch(s)")
            print(f"\n{'Epoch':<8} {'Avg Reward':<12} {'Status'}")
            print(f"{'-'*40}")
            
            # Find best epoch and display all metrics
            best_score = -float('inf')
            best_epoch = -1.0
            
            epochs_sorted = sorted([(float(k), v) for k, v in val_metrics.items()], 
                                key=lambda x: x[0])
            
            for epoch, metrics in epochs_sorted:
                avg_reward = metrics.get('avg_reward', -float('inf'))
                is_best = ""
                if avg_reward > best_score:
                    best_score = avg_reward
                    best_epoch = epoch
                    is_best = "⭐ BEST"
                
                print(f"{epoch:<8.1f} {avg_reward:<12.4f} {is_best}")
            
            if best_epoch == -1.0:
                reason = "Could not find 'avg_reward' in val_metrics.json, using latest."
                print(f"\n⚠️  {reason}")
                print(f"🎯 Selected: {latest_checkpoint_path}")
                print(f"{'='*70}\n")
                return latest_checkpoint_path, reason
            
            print(f"\n⭐ Best validation performance:")
            print(f"   Epoch: {best_epoch}")
            print(f"   Avg Reward: {best_score:.4f}")
            
            # Calculate steps per epoch
            max_checkpoint_step = checkpoint_steps[-1][0]
            
            # Try to get actual step from metrics if available
            best_epoch_key = str(int(best_epoch))
            actual_step = None
            if best_epoch_key in val_metrics and 'step' in val_metrics[best_epoch_key]:
                actual_step = val_metrics[best_epoch_key]['step']
                print(f"   Actual Step: {actual_step} (from metrics)")
            
            # Calculate target step
            if actual_step is not None:
                target_step = actual_step
                print(f"\n📍 Using actual step from metrics: {target_step}")
            else:
                # Estimate based on available data
                max_epoch = max(float(k) for k in val_metrics.keys())
                estimated_steps_per_epoch = max_checkpoint_step / 20.0  # Assuming 20 epochs
                target_step = int(best_epoch * estimated_steps_per_epoch)
                
                print(f"\n🔢 Estimation details:")
                print(f"   Max checkpoint step: {max_checkpoint_step}")
                print(f"   Assumed total epochs: 20.0")
                print(f"   Max epoch in metrics: {max_epoch}")
                print(f"   Estimated steps/epoch: {estimated_steps_per_epoch:.2f}")
                print(f"   Target step (epoch {best_epoch} × {estimated_steps_per_epoch:.2f}): {target_step}")
                
                if max_epoch < 20.0:
                    print(f"\n⚠️  WARNING: Max epoch ({max_epoch}) is less than assumed 20 epochs!")
                    print(f"   This may indicate incomplete training or incorrect assumption.")
                    # Provide alternative calculation
                    alt_steps_per_epoch = max_checkpoint_step / max_epoch
                    alt_target_step = int(best_epoch * alt_steps_per_epoch)
                    print(f"   Alternative calculation: {max_checkpoint_step}/{max_epoch:.1f} = {alt_steps_per_epoch:.2f} steps/epoch")
                    print(f"   Alternative target step: {alt_target_step}")
            
            # Find nearest checkpoint
            print(f"\n🔍 Finding nearest checkpoint to step {target_step}...")
            print(f"\n{'Checkpoint':<20} {'Step':<8} {'Distance':<10} {'Status'}")
            print(f"{'-'*50}")
            
            best_checkpoint_tuple = None
            min_distance = float('inf')
            
            for step, name in checkpoint_steps:
                distance = abs(step - target_step)
                is_selected = ""
                if distance < min_distance:
                    min_distance = distance
                    best_checkpoint_tuple = (step, name)
                    is_selected = "✅ SELECTED"
                
                print(f"{name:<20} {step:<8} {distance:<10} {is_selected}")
            
            best_checkpoint_name = best_checkpoint_tuple[1]
            best_checkpoint_step = best_checkpoint_tuple[0]
            best_checkpoint_path = os.path.join(checkpoint_dir, best_checkpoint_name)
            
            step_difference = best_checkpoint_step - target_step
            step_diff_sign = "+" if step_difference > 0 else ""
            
            print(f"\n🎯 SELECTED CHECKPOINT:")
            print(f"   Path: {best_checkpoint_path}")
            print(f"   Step: {best_checkpoint_step}")
            print(f"   Target step: {target_step}")
            print(f"   Difference: {step_diff_sign}{step_difference} steps")
            print(f"   Distance: {min_distance} steps ({min_distance/max_checkpoint_step*100:.1f}% of max)")
            
            # Sanity check warnings
            if min_distance > estimated_steps_per_epoch:
                print(f"\n⚠️  WARNING: Selected checkpoint is {min_distance} steps away from target!")
                print(f"   This is more than one epoch's worth of steps ({estimated_steps_per_epoch:.1f}).")
                print(f"   Consider checking your checkpoint saving frequency or validation schedule.")
            
            if best_checkpoint_step == latest_checkpoint_step:
                print(f"\n⚠️  NOTE: Selected checkpoint is the latest one.")
                print(f"   This might indicate overfitting or that training should continue.")
            
            reason = (f"Highest validation avg_reward ({best_score:.4f}) at epoch {best_epoch}, "
                    f"target step {target_step}, selected checkpoint at step {best_checkpoint_step} "
                    f"({step_diff_sign}{step_difference} steps from target).")
            
            print(f"{'='*70}\n")
            return best_checkpoint_path, reason

        except (json.JSONDecodeError, KeyError, Exception) as e:
            reason = f"Error processing val_metrics.json ({e}), using latest checkpoint."
            print(f"\n❌ {reason}")
            print(f"🎯 Selected: {latest_checkpoint_path}")
            print(f"{'='*70}\n")
            return latest_checkpoint_path, reason


    def inject_paths_into_script(self, script_config: Dict) -> Dict[str, str]:
        """Create environment variables to inject paths into evaluation scripts."""
        output_dir = os.path.join(self.base_output_dir, 
        script_config.get('output_subdir', 'evaluation_results'))
        
        return {
            'EVAL_RAW_MODEL_PATH': self.raw_model_path,
            'EVAL_TRAINING_DIR': self.training_dir,
            'EVAL_OUTPUT_DIR': output_dir,
        }
    
    def build_command_args(self, script_config: Dict, terminal_args: Dict, 
                          cuda_device: str) -> List[str]:
        """Build command line arguments for a script."""
        override = script_config.get('override_terminal', False)
        script_params = script_config.get('params', {})
        
        # Determine parameter priority
        if override:
            final_params = {**DEFAULT_PARAMS, **terminal_args, **script_params}
        else:
            final_params = {**DEFAULT_PARAMS, **script_params, **terminal_args}
        
        # Override cuda_device for parallel execution
        final_params['cuda_device'] = cuda_device
        
        # Build argument list
        args = []
        for key, value in final_params.items():
            if value is None:
                continue
            
            arg_name = f'--{key}'
            
            # Handle boolean flags
            if isinstance(value, bool):
                if value:
                    args.append(arg_name)
            else:
                args.extend([arg_name, str(value)])
        
        return args
    
    def stream_output(self, pipe, log_file, script_name: str, stream_name: str):
        """Stream output from pipe to both console and log file in real-time."""
        try:
            for line in iter(pipe.readline, ''):
                if line:
                    # Write to log file immediately
                    log_file.write(line)
                    log_file.flush()
                    
                    # Print to console with thread-safe lock
                    if self.realtime_logs:
                        with self.print_lock:
                            # Add prefix to identify which script is outputting
                            prefix = f"[{script_name}] "
                            print(f"{prefix}{line}", end='')
                            sys.stdout.flush()
        except Exception as e:
            with self.print_lock:
                print(f"Error streaming {stream_name} for {script_name}: {e}")
    
    def run_single_evaluation(self, script_config: Dict, terminal_args: Dict,
                            cuda_device: str, index: int) -> Dict[str, Any]:
        """Run a single evaluation script and capture its output with real-time streaming."""
        script_path = script_config['script']
        script_name = script_config['name']
        
        with self.print_lock:
            print(f"\n{'='*70}")
            print(f"[{index + 1}/{len(EVALUATION_SCRIPTS)}] Starting: {script_name}")
            print(f"Script: {script_path}")
            print(f"CUDA Device: {cuda_device}")
            print(f"{'='*70}\n")
        
        # Build command
        cmd_args = self.build_command_args(script_config, terminal_args, cuda_device)
        command = [sys.executable, script_path] + cmd_args
        
        # Prepare environment with path injection
        env = os.environ.copy()
        env.update(self.inject_paths_into_script(script_config))
        
        # Create individual log file
        log_filename = f"{index + 1:02d}_{Path(script_path).stem}.txt"
        log_path = self.run_dir / log_filename
        
        # Record start time
        start_time = time.time()
        start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        result = {
            'index': index,
            'name': script_name,
            'script': script_path,
            'cuda_device': cuda_device,
            'start_time': start_datetime,
            'command': ' '.join(command),
            'success': False,
            'error': None,
            'log_file': str(log_path),
            'duration_seconds': 0,
            'raw_model_path': self.raw_model_path,
            'training_dir': self.training_dir,
            'output_dir': env['EVAL_OUTPUT_DIR']
        }
        
        try:
            # Open log file for writing
            with open(log_path, 'w', encoding='utf-8', buffering=1) as log_file:
                # Write header to log file
                log_file.write(f"{'='*70}\n")
                log_file.write(f"EVALUATION: {script_name}\n")
                log_file.write(f"{'='*70}\n")
                log_file.write(f"Script: {script_path}\n")
                log_file.write(f"CUDA Device: {cuda_device}\n")
                log_file.write(f"Start Time: {start_datetime}\n")
                log_file.write(f"Command: {' '.join(command)}\n")
                log_file.write(f"\nPATH CONFIGURATION:\n")
                log_file.write(f"  Raw Model: {self.raw_model_path}\n")
                log_file.write(f"  Training Dir: {self.training_dir}\n")
                log_file.write(f"  Output Dir: {env['EVAL_OUTPUT_DIR']}\n")
                log_file.write(f"{'='*70}\n\n")
                log_file.write("OUTPUT:\n")
                log_file.write("-" * 70 + "\n")
                log_file.flush()
                
                # Start subprocess with pipes
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Merge stderr into stdout
                    text=True,
                    bufsize=1,  # Line buffered
                    env=env,
                    universal_newlines=True
                )
                
                # Stream output in real-time
                self.stream_output(process.stdout, log_file, script_name, "stdout")
                
                # Wait for process to complete
                return_code = process.wait()
                
                # Calculate duration
                duration = time.time() - start_time
                result['duration_seconds'] = duration
                result['return_code'] = return_code
                result['success'] = (return_code == 0)
                
                # Write footer to log file
                log_file.write("\n" + "-" * 70 + "\n")
                log_file.write(f"\nDuration: {duration:.2f} seconds ({duration/60:.1f} minutes)\n")
                log_file.write(f"Return Code: {return_code}\n")
                log_file.write(f"Status: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}\n")
                log_file.write(f"{'='*70}\n")
            
            # Print summary
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            with self.print_lock:
                print(f"\n{status} - {script_name} (Duration: {duration:.2f}s / {duration/60:.1f}m)")
                
                if not result['success']:
                    result['error'] = f"Script exited with code {return_code}"
                    print(f"   Error: {result['error']}")
                    print(f"   Check log file: {log_path}\n")
                else:
                    print(f"   Log file: {log_path}\n")
            
        except Exception as e:
            duration = time.time() - start_time
            result['duration_seconds'] = duration
            result['error'] = str(e)
            
            # Write error to log file
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"EXCEPTION OCCURRED\n")
                f.write(f"Duration: {duration:.2f} seconds\n")
                f.write(f"Exception: {str(e)}\n")
                f.write(f"{'='*70}\n")
            
            with self.print_lock:
                print(f"\n❌ EXCEPTION - {script_name}: {str(e)}")
                print(f"   Check log file: {log_path}\n")
        
        return result
    
    def run_all_evaluations(self, terminal_args: Dict, find_best: bool):
        """Run all evaluation scripts with parallel execution support."""
        print(f"\n{'='*70}")
        print(f"🚀 MULTI-EVALUATION ORCHESTRATOR")
        print(f"{'='*70}")
        print(f"Total Scripts: {len(EVALUATION_SCRIPTS)}")
        print(f"Parallel Count: {self.parallel_count}")
        print(f"CUDA Devices Pool: {CUDA_DEVICES}")
        print(f"Real-time Logs: {'Enabled' if self.realtime_logs else 'Disabled'}")
        print(f"Output Directory: {self.run_dir}")
        print(f"\nPATH CONFIGURATION:")
        print(f"  Raw Model: {self.raw_model_path}")
        print(f"  Training Dir: {self.training_dir}")
        print(f"  Base Output: {self.base_output_dir}")
        
        # --- NEW: Best Checkpoint Finder Logic ---
        print(f"\nCHECKPOINT SELECTION:")
        if 'checkpoint_path' in terminal_args:
            print(f"  Mode: Manual (provided via --checkpoint_path)")
            print(f"  Using: {terminal_args['checkpoint_path']}")
        elif find_best:
            print(f"  Mode: Automatic (searching for best checkpoint...)")
            best_path, reason = self.find_best_checkpoint()
            if best_path:
                terminal_args['checkpoint_path'] = best_path
                print(f"  ✅ Found: {best_path}")
                print(f"     Reason: {reason}")
            else:
                print(f"  ⚠️ WARNING: Could not find best checkpoint. Reason: {reason}")
                print(f"     Sub-scripts will use their own default behavior.")
        else:
            print("  Mode: Disabled (via --no-find-best-checkpoint)")
            print("  Sub-scripts will use their own default behavior.")
        print(f"{'='*70}\n")
        
        results = []
        overall_start = time.time()
        
        if self.parallel_count == 1:
            # Sequential execution
            for idx, script_config in enumerate(EVALUATION_SCRIPTS):
                cuda_device = CUDA_DEVICES[idx % len(CUDA_DEVICES)]
                result = self.run_single_evaluation(script_config, terminal_args, 
                                                   cuda_device, idx)
                results.append(result)
        else:
            # Parallel execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_count) as executor:
                futures = []
                for idx, script_config in enumerate(EVALUATION_SCRIPTS):
                    cuda_device = CUDA_DEVICES[idx % len(CUDA_DEVICES)]
                    future = executor.submit(
                        self.run_single_evaluation,
                        script_config, terminal_args, cuda_device, idx
                    )
                    futures.append(future)
                
                # Wait for all to complete
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
        
        # Sort results by index to maintain order
        results.sort(key=lambda x: x['index'])
        
        overall_duration = time.time() - overall_start
        
        # Write master log
        self.write_master_log(results, overall_duration, terminal_args)
        
        # Print final summary
        self.print_summary(results, overall_duration)
        
        return results
    
    def write_master_log(self, results: List[Dict], overall_duration: float,
                        terminal_args: Dict):
        """Write consolidated master log file."""
        with open(self.master_log, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("MULTI-EVALUATION ORCHESTRATOR - MASTER LOG\n")
            f.write("="*70 + "\n")
            f.write(f"Run Directory: {self.run_dir}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Duration: {overall_duration:.2f} seconds ({overall_duration/60:.1f} minutes)\n")
            f.write(f"Parallel Count: {self.parallel_count}\n")
            f.write(f"Real-time Logs: {'Enabled' if self.realtime_logs else 'Disabled'}\n")
            f.write(f"Total Scripts: {len(EVALUATION_SCRIPTS)}\n")
            f.write(f"\nPATH CONFIGURATION:\n")
            f.write(f"  Raw Model: {self.raw_model_path}\n")
            f.write(f"  Training Dir: {self.training_dir}\n")
            f.write(f"  Base Output: {self.base_output_dir}\n")
            f.write(f"\nCUDA DEVICES POOL: {CUDA_DEVICES}\n")
            f.write("="*70 + "\n\n")
            
            # Terminal arguments
            f.write("TERMINAL ARGUMENTS & CHECKPOINT:\n")
            f.write("-"*70 + "\n")
            if terminal_args:
                for key, value in terminal_args.items():
                    if value is not None:
                        f.write(f"  --{key}: {value}\n")
            else:
                f.write("  (none provided)\n")
            f.write("\n")
            
            # Summary table
            f.write("EXECUTION SUMMARY:\n")
            f.write("-"*70 + "\n")
            success_count = sum(1 for r in results if r['success'])
            failed_count = len(results) - success_count
            f.write(f"✅ Successful: {success_count}/{len(results)}\n")
            f.write(f"❌ Failed: {failed_count}/{len(results)}\n")
            f.write("\n")
            
            # Individual results
            f.write("INDIVIDUAL RESULTS:\n")
            f.write("="*70 + "\n\n")
            
            for result in results:
                f.write(f"[{result['index'] + 1}] {result['name']}\n")
                f.write("-"*70 + "\n")
                f.write(f"Script: {result['script']}\n")
                f.write(f"CUDA Device: {result['cuda_device']}\n")
                f.write(f"Start Time: {result['start_time']}\n")
                f.write(f"Duration: {result['duration_seconds']:.2f} seconds ({result['duration_seconds']/60:.1f} minutes)\n")
                f.write(f"Status: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}\n")
                
                if result.get('return_code') is not None:
                    f.write(f"Return Code: {result['return_code']}\n")
                
                if result.get('error'):
                    f.write(f"Error: {result['error']}\n")
                
                f.write(f"Output Dir: {result['output_dir']}\n")
                f.write(f"Log File: {result['log_file']}\n")
                f.write(f"Command: {result['command']}\n")
                f.write("\n")
            
            f.write("="*70 + "\n")
            f.write("END OF MASTER LOG\n")
            f.write("="*70 + "\n")
    
    def print_summary(self, results: List[Dict], overall_duration: float):
        """Print final summary to console."""
        print(f"\n{'='*70}")
        print(f"📊 FINAL SUMMARY")
        print(f"{'='*70}")
        
        success_count = sum(1 for r in results if r['success'])
        failed_count = len(results) - success_count
        
        print(f"✅ Successful: {success_count}/{len(results)}")
        print(f"❌ Failed: {failed_count}/{len(results)}")
        print(f"⏱️  Total Duration: {overall_duration:.2f} seconds ({overall_duration/60:.1f} minutes)")
        print(f"📁 Results Directory: {self.run_dir}")
        print(f"📄 Master Log: {self.master_log}")
        print(f"{'='*70}\n")
        
        if failed_count > 0:
            print("Failed evaluations:")
            for result in results:
                if not result['success']:
                    print(f"  ❌ {result['name']}")
                    print(f"     Log: {result['log_file']}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description='Run multiple evaluation scripts with organized logging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all evaluations sequentially, automatically finding the best checkpoint
  python run_evaluations.py
  
  # Run with a specific checkpoint for all evaluations
  python run_evaluations.py --checkpoint_path /path/to/checkpoint-640
  
  # Run evaluations but disable the automatic checkpoint finder
  python run_evaluations.py --no-find-best-checkpoint
  
  # Run 2 evaluations in parallel on GPUs 2 and 3
  python run_evaluations.py --parallel 2
        """
    )
    
    # Orchestrator-specific arguments
    parser.add_argument('--parallel', type=int, default=DEFAULT_PARALLEL_COUNT,
                       help=f'Number of scripts to run in parallel (default: {DEFAULT_PARALLEL_COUNT})')
    parser.add_argument('--output_dir', type=str, default=ORCHESTRATOR_OUTPUT_DIR,
                       help=f'Output directory for orchestrator results (default: {ORCHESTRATOR_OUTPUT_DIR})')
    parser.add_argument('--no_realtime', action='store_true',
                       help='Disable real-time log streaming to console (logs still written to files)')
    
    # Path override arguments
    parser.add_argument('--raw_model_path', type=str, default=None,
                       help=f'Override RAW_MODEL_PATH (default: {RAW_MODEL_PATH})')
    parser.add_argument('--training_dir', type=str, default=None,
                       help=f'Override TRAINING_DIR (default: {TRAINING_DIR})')
    parser.add_argument('--base_output_dir', type=str, default=None,
                       help=f'Override BASE_OUTPUT_DIR (default: {BASE_OUTPUT_DIR})')
    
    # Common evaluation arguments
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate')
    parser.add_argument('--cuda_device', type=str, default=None,
                       help='CUDA device (only used if parallel=1, otherwise cycles through CUDA_DEVICES)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for evaluation')
    parser.add_argument('--split', type=str, default=None,
                       choices=['train', 'test', 'validation'],
                       help='Dataset split to use')
    parser.add_argument('--skip_raw', action='store_true',
                       help='Skip raw model evaluation')
    parser.add_argument('--skip_finetuned', action='store_true',
                       help='Skip fine-tuned model evaluation')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to a specific checkpoint. Overrides automatic finding.')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Path to directory containing checkpoints')
    parser.add_argument('--no-find-best-checkpoint', action='store_false', dest='find_best_checkpoint',
                       help='Disable the automatic best checkpoint finding logic.')

    
    args = parser.parse_args()
    
    # Extract terminal arguments
    terminal_args = {
        'max_samples': args.max_samples,
        'cuda_device': args.cuda_device,
        'batch_size': args.batch_size,
        'split': args.split,
        'skip_raw': args.skip_raw,
        'skip_finetuned': args.skip_finetuned,
        'checkpoint_path': args.checkpoint_path,
        'checkpoint_dir': args.checkpoint_dir,
    }
    # Clean out any arguments that were not provided
    terminal_args = {k: v for k, v in terminal_args.items() if v is not None}
    
    # Create orchestrator and run
    orchestrator = EvaluationOrchestrator(
        args.output_dir, 
        args.parallel,
        args.raw_model_path,
        args.training_dir,
        args.base_output_dir,
        realtime_logs=not args.no_realtime
    )
    results = orchestrator.run_all_evaluations(terminal_args, args.find_best_checkpoint)
    
    # Exit with error code if any evaluation failed
    failed_count = sum(1 for r in results if not r['success'])
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == '__main__':
    main()
