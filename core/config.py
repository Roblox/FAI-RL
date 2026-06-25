# rl_finetuning/core/config.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Literal
import yaml
import sys
import os

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config_validation import validate_api_config


@dataclass
class ModelConfig:
    """Configuration for model settings."""
    base_model_name: str
    torch_dtype: str = "bfloat16"
    low_cpu_mem_usage: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention: bool = False
    # Quantization configuration for QLoRA
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # S3 connection overrides (only used when base_model_name starts with s3://).
    # When unset, boto3 uses its default credential/region resolution chain.
    s3_region: Optional[str] = None
    s3_endpoint_url: Optional[str] = None

    ignore_mismatched_sizes: bool = False

    # LoRA configuration
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    lora_bias: str = "none"

    # Multimodal (vision-language) configuration. Only used by the sft_vlm
    # algorithm. When True, the vision encoder is frozen and only the language
    # model / projector (and LoRA adapters, if enabled) are trained -- the
    # standard recipe for VLM SFT.
    freeze_vision_tower: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_model_name": self.base_model_name,
            "torch_dtype": self.torch_dtype,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "load_in_8bit": self.load_in_8bit,
            "load_in_4bit": self.load_in_4bit,
            "use_flash_attention": self.use_flash_attention,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "s3_region": self.s3_region,
            "s3_endpoint_url": self.s3_endpoint_url,
            "ignore_mismatched_sizes": self.ignore_mismatched_sizes,
            "use_lora": self.use_lora,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "lora_bias": self.lora_bias,
            "freeze_vision_tower": self.freeze_vision_tower,
        }


@dataclass
class DatasetInfo:
    """Configuration for a single dataset."""
    name: str
    split: str = "train"
    subset: Optional[str] = None
    text_column: str = "text"
    prompt_column: str = "prompt"
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"
    answer_column: str = "answer"
    dataset_columns: Optional[List[str]] = None

    # Multimodal (sft_vlm) columns. image_column holds an HTTP(S) URL, a local
    # path, raw bytes, a PIL image, or a list thereof (multiple images per row).
    # question_column / response_column hold the user prompt and target answer.
    # Alternatively, messages_column points at a column already in conversational
    # format (a list of {role, content} dicts) -- when set it takes precedence.
    image_column: Optional[str] = None
    question_column: str = "question"
    response_column: str = "response"
    messages_column: Optional[str] = None

    # S3 connection overrides (only used when name starts with s3://).
    # When unset, boto3 uses its default credential/region resolution chain.
    s3_region: Optional[str] = None
    s3_endpoint_url: Optional[str] = None


@dataclass
class DataConfig:
    """Configuration for dataset settings."""
    datasets: List[DatasetInfo] = field(default_factory=list)
    max_length: int = 512
    max_prompt_length: int = 256
    remove_unused_columns: bool = False
    system_prompt: Optional[str] = None
    prompt_column: str = "prompt"
    dataset_num_proc: int = 1

    # Multimodal (sft_vlm) image-fetch settings. image_cache_dir, when set,
    # caches downloaded images on disk so they are not re-fetched every epoch.
    image_cache_dir: Optional[str] = None
    image_fetch_timeout: int = 10
    image_fetch_retries: int = 3
    # If set, downscale any fetched image whose width*height exceeds this many
    # pixels (preserving aspect ratio). Bounds the number of vision tokens /
    # sequence length so high-resolution images don't blow up memory.
    max_image_pixels: Optional[int] = None
    # S3 connection overrides used when an image value is an s3:// URI (e.g.
    # PNGs stored in a bucket). Independent of each dataset's s3_region /
    # s3_endpoint_url (which govern the dataset file). When unset, boto3 uses
    # its default credential/region resolution chain.
    image_s3_region: Optional[str] = None
    image_s3_endpoint_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "datasets": [
                {
                    "name": ds.name,
                    "split": ds.split,
                    "subset": ds.subset,
                    "text_column": ds.text_column,
                    "prompt_column": ds.prompt_column,
                    "chosen_column": ds.chosen_column,
                    "rejected_column": ds.rejected_column,
                    "answer_column": ds.answer_column,
                    "dataset_columns": ds.dataset_columns,
                    "image_column": ds.image_column,
                    "question_column": ds.question_column,
                    "response_column": ds.response_column,
                    "messages_column": ds.messages_column,
                    "s3_region": ds.s3_region,
                    "s3_endpoint_url": ds.s3_endpoint_url,
                }
                for ds in self.datasets
            ],
            "max_length": self.max_length,
            "max_prompt_length": self.max_prompt_length,
            "remove_unused_columns": self.remove_unused_columns,
            "system_prompt": self.system_prompt,
            "prompt_column": self.prompt_column,
            "dataset_num_proc": self.dataset_num_proc,
            "image_cache_dir": self.image_cache_dir,
            "image_fetch_timeout": self.image_fetch_timeout,
            "image_fetch_retries": self.image_fetch_retries,
            "max_image_pixels": self.max_image_pixels,
            "image_s3_region": self.image_s3_region,
            "image_s3_endpoint_url": self.image_s3_endpoint_url,
        }


@dataclass
class TrainingConfig:
    """Configuration for training settings."""
    output_dir: str
    algorithm: Optional[str] = None
    
    # Training hyperparameters
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-6
    num_train_epochs: int = 3
    max_steps: int = -1
    warmup_steps: int = 50
    
    # GRPO/GSPO specific parameters (optional for other algorithms)
    num_generations: int = 8                    # Number of generations for GRPO/GSPO
    
    # GSPO specific parameters (optional for other algorithms)
    # Reference: https://swift.readthedocs.io/en/v3.7/Instruction/GRPO/AdvancedResearch/GSPO.html
    beta: float = 0.0                           # zero kl regularization  
    epsilon: float = 3e-4                       # from paper section 5.1
    epsilon_high: float = 4e-4                  # from paper section 5.1
    steps_per_generation: int = 4               # each batch of rollout data is partitioned into four minibatches for gradient updates
    importance_sampling_level: str = "sequence" # GSPO uses sequence-level importance sampling
    
    # Logging and inference
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    
    # Optimization
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    
    # DeepSpeed config path. If set, the launcher uses `deepspeed --num_gpus=N`
    # with this config; if unset, the launcher uses plain `torchrun` (DDP).
    # LoRA recipes leave this unset (DDP replicates the frozen base on every
    # GPU and only allreduces the small adapter grads). Full-FT recipes
    # typically point this at configs/deepspeed/zero1_config.json (sharded
    # optimizer states, no parameter sharding). configs/deepspeed/zero3_config.json
    # is available as an escape hatch but is NOT recommended for LoRA on MoE
    # models -- per-rank expert routing diverges and the _ALLGATHER_BASE
    # collective deadlocks.
    deepspeed_config: Optional[str] = None
    
    # DataLoader
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False
    dataloader_drop_last: bool = True
    
    # DDP
    ddp_find_unused_parameters: bool = False

    # Miscellaneous
    save_only_model: bool = True
    prediction_loss_only: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging."""
    enabled: bool = True
    project: str = "rl"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: list = field(default_factory=list)
    # Optional W&B API key. If set in the recipe, it is exported as the
    # WANDB_API_KEY env var before wandb.init so non-interactive runs can
    # authenticate without a prior `wandb login`. Falls back to the existing
    # env var if left as None.
    api_key: Optional[str] = None
    # W&B server base URL. Defaults to the public SaaS endpoint; override for
    # self-hosted / dedicated cloud deployments. Exported as WANDB_BASE_URL
    # before wandb.init so HF Trainer / TRL subprocesses also pick it up.
    base_url: str = "https://api.wandb.ai"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "project": self.project,
            "entity": self.entity,
            "name": self.name,
            "tags": self.tags,
            # Redact api_key to avoid leaking secrets into logs / saved configs.
            "api_key": "***" if self.api_key else None,
            "base_url": self.base_url,
        }


@dataclass
class S3Config:
    """Configuration for uploading checkpoints to S3."""
    enabled: bool = False
    bucket: str = ""
    prefix: str = ""
    region: Optional[str] = None
    endpoint_url: Optional[str] = None
    upload_final_model: bool = True
    upload_checkpoints: bool = True
    delete_local_after_upload: bool = False

    # Upload backend.
    #   "auto"   -> use s5cmd if it's on $PATH, else boto3
    #   "s5cmd"  -> always use s5cmd (errors if not installed)
    #   "boto3"  -> always use boto3 (slow but no extra binary)
    # On EKS pods we measured s5cmd ~1.25 GiB/s vs boto3 ~150 MiB/s for the
    # same workload (Qwen3-30B final/ dir, 90 GiB), so "auto" is a safe default.
    uploader: Literal["auto", "boto3", "s5cmd"] = "auto"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


@dataclass
class InferenceConfig:
    """Configuration for inference settings."""
    # Model configuration - either model_paths (for local models) or model (for API models)
    # model_paths: List of checkpoint paths for batch inference
    model_paths: Optional[List[str]] = None
    model: Optional[str] = None
    
    # API configuration
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    
    # Dataset configuration
    dataset_name: str = "Roblox/FAI-RL-inference-dataset"
    dataset_split: str = "test"
    output_file: str = "outputs/inference_results.json"
    system_prompt: str = ""
    
    # Dataset column configuration
    dataset_columns: List[str] = field(default_factory=lambda: ["persona", "prompt"])
    response_column: str = "response"
    checkpoint_column: str = "checkpoint"  # Column name for checkpoint identifier in multi-checkpoint inference

    # Multimodal (VLM) inference. Setting image_column enables VLM mode: it names
    # the dataset column holding an image URL / local path (or a list of them) for
    # each row. The image(s) are fetched into PIL and fed to the processor
    # alongside the templated text prompt. The fetch knobs mirror DataConfig's.
    # When image_column is unset, inference runs text-only exactly as before.
    image_column: Optional[str] = None
    image_cache_dir: Optional[str] = None
    image_fetch_timeout: int = 10
    image_fetch_retries: int = 3
    max_image_pixels: Optional[int] = None
    # S3 overrides for image values that are s3:// URIs (e.g. PNGs in a bucket).
    # When unset, boto3 uses its default credential/region resolution chain.
    image_s3_region: Optional[str] = None
    image_s3_endpoint_url: Optional[str] = None

    # S3 connection settings (used when model_paths entries start with s3://)
    s3_region: Optional[str] = None
    s3_endpoint_url: Optional[str] = None

    # S3 result upload settings
    s3_upload_results: bool = False
    s3_bucket: str = ""
    s3_prefix: str = ""
    s3_uploader: str = "auto"

    # Generation parameters
    temperature: float = 1.0
    top_p: float = 0.9
    max_new_tokens: int = 200
    do_sample: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


@dataclass
class EvaluationConfig:
    """Configuration for evaluation settings."""
    # Model configuration - either model_paths (for local models) or model (for API models)
    # model_paths: List of checkpoint paths for batch evaluation
    model_paths: Optional[List[str]] = None
    model: Optional[str] = None
    
    # API configuration
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    
    # Dataset configuration
    dataset_name: str = "cais/mmlu"
    dataset_subset: Optional[str] = None
    dataset_split: str = "test" 
    output_file: str = "outputs/evaluation_results.csv"
    system_prompt: str = ""
    
    # Dataset column configuration
    dataset_columns: List[str] = field(default_factory=lambda: ["question", "choices", "answer"])
    ground_truth_column: str = "answer"
    response_column: str = "response"
    checkpoint_column: str = "checkpoint"  # Column name for checkpoint identifier in multi-checkpoint evaluation
    
    # Prompt configuration
    prompt_template: Optional[str] = None
    
    # Multiple choice configuration
    choice_labels: List[str] = field(default_factory=lambda: ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])
    
    # S3 connection settings (used when model_paths entries start with s3://)
    s3_region: Optional[str] = None
    s3_endpoint_url: Optional[str] = None

    # S3 result upload settings
    s3_upload_results: bool = False
    s3_bucket: str = ""
    s3_prefix: str = ""
    s3_uploader: str = "auto"

    # Generation parameters
    temperature: float = 1.0
    top_p: float = 0.9
    max_new_tokens: int = 10
    do_sample: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


@dataclass
class ExperimentConfig:
    """Main configuration class that combines all settings."""
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    wandb: WandbConfig
    s3: S3Config = field(default_factory=S3Config)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Handle datasets configuration
        data_config = config_dict['data'].copy()
        if 'datasets' in data_config:
            data_config['datasets'] = [
                DatasetInfo(**ds) for ds in data_config['datasets']
            ]
        
        return cls(
            model=ModelConfig(**config_dict['model']),
            data=DataConfig(**data_config),
            training=TrainingConfig(**config_dict['training']),
            wandb=WandbConfig(**config_dict.get('wandb', {})),
            s3=S3Config(**config_dict.get('s3', {})),
        )
    
    @classmethod
    def load_inference_config(cls, config_path: str) -> 'InferenceConfig':
        """Load inference configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = InferenceConfig(**config_dict['inference'])
        
        # Validate API configuration
        validate_api_config(config)
        
        return config
    
    @classmethod
    def load_eval_config(cls, config_path: str) -> 'EvaluationConfig':
        """Load evaluation configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = EvaluationConfig(**config_dict['evaluation'])
        
        # Validate API configuration
        validate_api_config(config)
        
        return config
    
    def to_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'model': self.model.to_dict(),
            'data': self.data.to_dict(),
            'training': self.training.to_dict(),
            'wandb': self.wandb.to_dict(),
            's3': self.s3.to_dict(),
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
