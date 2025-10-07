# rl_finetuning/core/config.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import yaml


@dataclass
class ModelConfig:
    """Configuration for model settings."""
    base_model_name: str
    torch_dtype: str = "bfloat16"
    low_cpu_mem_usage: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention: bool = False
    value_model_name: Optional[str] = None  # For PPO: value and reward models
    
    # Quantization configuration for QLoRA
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # LoRA configuration
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    lora_bias: str = "none"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_model_name": self.base_model_name,
            "torch_dtype": self.torch_dtype,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "load_in_8bit": self.load_in_8bit,
            "load_in_4bit": self.load_in_4bit,
            "use_flash_attention": self.use_flash_attention,
            "value_model_name": self.value_model_name,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "use_lora": self.use_lora,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "lora_bias": self.lora_bias,
        }


@dataclass
class DatasetInfo:
    """Configuration for a single dataset."""
    name: str
    split: str = "train"
    subset: Optional[str] = None
    prompt_column: str = "prompt"
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"
    answer_column: str = "answer"
    dataset_columns: Optional[List[str]] = None


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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "datasets": [
                {
                    "name": ds.name,
                    "split": ds.split,
                    "subset": ds.subset,
                    "prompt_column": ds.prompt_column,
                    "chosen_column": ds.chosen_column,
                    "rejected_column": ds.rejected_column,
                    "answer_column": ds.answer_column,
                    "dataset_columns": ds.dataset_columns,
                }
                for ds in self.datasets
            ],
            "max_length": self.max_length,
            "max_prompt_length": self.max_prompt_length,
            "remove_unused_columns": self.remove_unused_columns,
            "system_prompt": self.system_prompt,
            "prompt_column": self.prompt_column,
            "dataset_num_proc": self.dataset_num_proc,
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
    
    # GSPO specific parameters (optional for other algorithms)
    # Reference: https://swift.readthedocs.io/en/v3.7/Instruction/GRPO/AdvancedResearch/GSPO.html
    beta: float = 0.0                           # zero kl regularization
    group_size: int = 4                         # Group size for sequence grouping  
    epsilon: float = 3e-4                       # from paper section 5.1
    epsilon_high: float = 4e-4                  # from paper section 5.1
    steps_per_generation: int = 4               # each batch of rollout data is partitioned into four minibatches for gradient updates
    importance_sampling_level: str = "sequence" # GSPO uses sequence-level importance sampling
    
    # PPO specific parameters (used by PPO trainer)
    gamma: float = 1.0                          # Discount factor
    lam: float = 0.95                           # GAE lambda
    cliprange: float = 0.2                      # PPO clipping range
    cliprange_value: float = 0.2                # Value function clipping range
    vf_coef: float = 0.1                        # Value function coefficient
    
    # Logging and inference
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    
    # Optimization
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    
    # DeepSpeed
    deepspeed_config: Optional[str] = None
    
    # DataLoader
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False
    dataloader_drop_last: bool = True
    
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "project": self.project,
            "entity": self.entity,
            "name": self.name,
            "tags": self.tags,
        }


@dataclass
class InferenceConfig:
    """Configuration for inference settings."""
    # Model configuration - either model_path (for local models) or model (for API models)
    model_path: Optional[str] = None
    model: Optional[str] = None
    
    # API configuration
    mlp_api_key: Optional[str] = None
    
    # Dataset configuration
    dataset_name: str = "Roblox/FAI-RL-inference-dataset"
    dataset_split: str = "test"
    output_file: str = "outputs/inference_results.json"
    system_prompt: str = ""
    
    # Dataset column configuration
    dataset_columns: List[str] = field(default_factory=lambda: ["persona", "prompt"])
    
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
    # Model configuration
    model_path: Optional[str] = None
    model: Optional[str] = None
    
    # API configuration  
    mlp_api_key: Optional[str] = None
    
    # Dataset configuration
    dataset_name: str = "cais/mmlu"
    dataset_subset: Optional[str] = None
    dataset_split: str = "test" 
    output_file: str = "outputs/evaluation_results.csv"
    system_prompt: str = ""
    
    # Dataset column configuration
    dataset_columns: List[str] = field(default_factory=lambda: ["question", "choices", "answer"])
    ground_truth_column: str = "answer"
    
    # Prompt configuration
    prompt_template: Optional[str] = None
    
    # Multiple choice configuration
    output_type: str = "numerical"  # "numerical" or "multiple_choice"
    choice_labels: List[str] = field(default_factory=lambda: ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])
    
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
        )
    
    @classmethod
    def load_inference_config(cls, config_path: str) -> 'InferenceConfig':
        """Load inference configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = InferenceConfig(**config_dict['inference'])
        
        # Validate API key if present
        if hasattr(config, 'mlp_api_key') and config.mlp_api_key:
            if config.mlp_api_key == "YOUR_MLP_API_KEY":
                raise ValueError(
                    "Error: mlp_api_key is still set to the placeholder 'YOUR_MLP_API_KEY'. "
                    "Please replace it with your actual API key in the configuration file."
                )
        
        return config
    
    @classmethod
    def load_eval_config(cls, config_path: str) -> 'EvaluationConfig':
        """Load evaluation configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = EvaluationConfig(**config_dict['evaluation'])
        
        # Validate API key if present
        if hasattr(config, 'mlp_api_key') and config.mlp_api_key:
            if config.mlp_api_key == "YOUR_MLP_API_KEY":
                raise ValueError(
                    "Error: mlp_api_key is still set to the placeholder 'YOUR_MLP_API_KEY'. "
                    "Please replace it with your actual API key in the configuration file."
                )
        
        return config
    
    def to_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'model': self.model.to_dict(),
            'data': self.data.to_dict(),
            'training': self.training.to_dict(),
            'wandb': self.wandb.to_dict(),
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
