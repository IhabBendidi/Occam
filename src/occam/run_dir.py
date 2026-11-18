"""
Output Management System for Occam
Provides centralized output organization with versioning and proper structure.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import shutil


class OutputManager:
    """
    Centralized output management system with versioning and organization.
    
    Creates a structured output directory with the following format:
    outputs/
    ├── experiments/
    │   ├── exp_001_training_YYYY-MM-DD_HH-MM-SS/
    │   │   ├── checkpoints/
    │   │   ├── logs/
    │   │   ├── config/
    │   │   └── metadata.json
    │   ├── exp_002_inference_YYYY-MM-DD_HH-MM-SS/
    │   │   ├── embeddings/
    │   │   ├── reports/
    │   │   ├── processed_data/
    │   │   └── metadata.json
    │   └── ...
    ├── latest_training/  -> symlink to latest training experiment
    ├── latest_inference/ -> symlink to latest inference experiment
    └── experiment_registry.json
    """
    
    def __init__(self, base_output_dir: str = "outputs"):
        """
        Initialize the output manager.
        
        Args:
            base_output_dir: Base directory for all outputs
        """
        self.base_output_dir = Path(base_output_dir)
        self.experiments_dir = self.base_output_dir / "experiments"
        self.registry_file = self.base_output_dir / "experiment_registry.json"
        
        # Create base directory structure
        self._initialize_directories()
        
        # Load or create experiment registry
        self.registry = self._load_registry()
    
    def _initialize_directories(self):
        """Create the base directory structure."""
        self.base_output_dir.mkdir(exist_ok=True)
        self.experiments_dir.mkdir(exist_ok=True)
    
    def _load_registry(self):
        """Load experiment registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                # Handle corrupted registry file
                print(f"⚠️ Warning: Registry file corrupted ({e})")
                print(f"🔧 Creating backup and starting with fresh registry...")
                
                # Backup corrupted file
                backup_file = self.registry_file.with_suffix('.json.corrupted')
                import shutil
                shutil.move(str(self.registry_file), str(backup_file))
                print(f"💾 Corrupted registry backed up to: {backup_file}")
                
                # Return fresh registry structure
                return {
                    "experiments": [],
                    "next_id": 1,
                    "created": datetime.now().isoformat()
                }
        else:
            return {
                "experiments": [],
                "next_id": 1,
                "created": datetime.now().isoformat()
            }
    
    def _save_registry(self):
        """Save the experiment registry."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def _get_next_experiment_id(self) -> str:
        """Get the next experiment ID."""
        exp_id = f"exp_{self.registry['next_id']:03d}"
        self.registry['next_id'] += 1
        return exp_id
    
    def create_experiment(self, 
                         experiment_type: str,
                         description: str = "",
                         config: Optional[Dict[str, Any]] = None,
                         tags: Optional[List[str]] = None) -> 'ExperimentManager':
        """
        Create a new experiment with proper versioning and structure.
        
        Args:
            experiment_type: Type of experiment ('training', 'inference', 'testing')
            description: Human-readable description of the experiment
            config: Configuration dictionary to save
            tags: Optional tags for the experiment
            
        Returns:
            ExperimentManager instance for this experiment
        """
        # Generate experiment directory name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_id = self._get_next_experiment_id()
        exp_name = f"{exp_id}_{experiment_type}_{timestamp}"
        
        if description:
            # Sanitize description for filename
            desc_clean = "".join(c for c in description if c.isalnum() or c in "._-")[:30]
            exp_name = f"{exp_id}_{experiment_type}_{desc_clean}_{timestamp}"
        
        exp_dir = self.experiments_dir / exp_name
        exp_dir.mkdir(exist_ok=True)
        
        # Create experiment metadata
        metadata = {
            "experiment_id": exp_id,
            "experiment_type": experiment_type,
            "description": description,
            "tags": tags or [],
            "created": datetime.now().isoformat(),
            "directory": str(exp_dir),
            "config": config or {}
        }
        
        # Save metadata
        metadata_file = exp_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update registry
        self.registry["experiments"].append(metadata)
        self._save_registry()
        
        # Create symlink to latest experiment of this type
        latest_link = self.base_output_dir / f"latest_{experiment_type}"
        if latest_link.exists() or latest_link.is_symlink():
            try:
                latest_link.unlink()
            except FileNotFoundError:
                # Symlink doesn't exist, nothing to unlink
                pass
        
        try:
            latest_link.symlink_to(exp_dir.relative_to(self.base_output_dir))
        except FileExistsError:
            # Symlink already exists (possible race condition in distributed training)
            # Try to remove and recreate once more
            try:
                latest_link.unlink()
                latest_link.symlink_to(exp_dir.relative_to(self.base_output_dir))
            except (FileNotFoundError, FileExistsError):
                # If it still fails, just continue - symlink creation is not critical
                print(f"⚠️ Warning: Could not create symlink {latest_link}")
                pass
        
        print(f"📁 Created experiment: {exp_name}")
        print(f"📍 Location: {exp_dir}")
        
        return ExperimentManager(exp_dir, metadata, self)
    
    def get_experiment(self, experiment_id: str) -> Optional['ExperimentManager']:
        """Get an existing experiment by ID."""
        for exp_metadata in self.registry["experiments"]:
            if exp_metadata["experiment_id"] == experiment_id:
                exp_dir = Path(exp_metadata["directory"])
                if exp_dir.exists():
                    return ExperimentManager(exp_dir, exp_metadata, self)
        return None
    
    def list_experiments(self, experiment_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all experiments, optionally filtered by type."""
        experiments = self.registry["experiments"]
        if experiment_type:
            experiments = [exp for exp in experiments if exp["experiment_type"] == experiment_type]
        return experiments
    
    def get_latest_experiment(self, experiment_type: str) -> Optional['ExperimentManager']:
        """Get the latest experiment of a specific type."""
        latest_link = self.base_output_dir / f"latest_{experiment_type}"
        if latest_link.exists():
            exp_dir = latest_link.resolve()
            metadata_file = exp_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                return ExperimentManager(exp_dir, metadata, self)
        return None
    
    def cleanup_old_experiments(self, keep_last_n: int = 10, experiment_type: Optional[str] = None):
        """Clean up old experiments, keeping only the last N."""
        experiments = self.list_experiments(experiment_type)
        experiments.sort(key=lambda x: x["created"], reverse=True)
        
        to_remove = experiments[keep_last_n:]
        for exp in to_remove:
            exp_dir = Path(exp["directory"])
            if exp_dir.exists():
                print(f"🗑️  Removing old experiment: {exp['experiment_id']}")
                shutil.rmtree(exp_dir)
                self.registry["experiments"].remove(exp)
        
        if to_remove:
            self._save_registry()


class ExperimentManager:
    """
    Manages a single experiment's outputs and structure.
    """
    
    def __init__(self, experiment_dir: Path, metadata: Dict[str, Any], output_manager: OutputManager):
        self.experiment_dir = experiment_dir
        self.metadata = metadata
        self.output_manager = output_manager
        self.experiment_type = metadata["experiment_type"]
        
        # Create appropriate subdirectories based on experiment type
        self._setup_subdirectories()
    
    def _setup_subdirectories(self):
        """Create subdirectories based on experiment type."""
        if self.experiment_type == "training":
            self.checkpoints_dir = self.experiment_dir / "checkpoints"
            self.logs_dir = self.experiment_dir / "logs"
            self.config_dir = self.experiment_dir / "config"
            self.plots_dir = self.experiment_dir / "plots"
            
            self.checkpoints_dir.mkdir(exist_ok=True)
            self.logs_dir.mkdir(exist_ok=True)
            self.config_dir.mkdir(exist_ok=True)
            self.plots_dir.mkdir(exist_ok=True)
            
        elif self.experiment_type == "inference":
            self.embeddings_dir = self.experiment_dir / "embeddings"
            self.reports_dir = self.experiment_dir / "reports"
            self.processed_data_dir = self.experiment_dir / "processed_data"
            self.plots_dir = self.experiment_dir / "plots"
            
            self.embeddings_dir.mkdir(exist_ok=True)
            self.reports_dir.mkdir(exist_ok=True)
            self.processed_data_dir.mkdir(exist_ok=True)
            self.plots_dir.mkdir(exist_ok=True)
            
        elif self.experiment_type == "testing":
            self.results_dir = self.experiment_dir / "results"
            self.logs_dir = self.experiment_dir / "logs"
            self.plots_dir = self.experiment_dir / "plots"
            
            self.results_dir.mkdir(exist_ok=True)
            self.logs_dir.mkdir(exist_ok=True)
            self.plots_dir.mkdir(exist_ok=True)
    
    def get_path(self, subdir: str, filename: str = "") -> Path:
        """
        Get a path within the experiment directory.
        
        Args:
            subdir: Subdirectory name ('checkpoints', 'logs', 'embeddings', etc.)
            filename: Optional filename to append
            
        Returns:
            Full path to the requested location
        """
        subdir_path = self.experiment_dir / subdir
        subdir_path.mkdir(exist_ok=True)
        
        if filename:
            return subdir_path / filename
        return subdir_path
    
    def save_config(self, config: Dict[str, Any], filename: str = "config.json"):
        """Save configuration to the experiment."""
        if hasattr(self, 'config_dir'):
            config_path = self.config_dir / filename
        else:
            config_path = self.experiment_dir / filename
            
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"💾 Config saved: {config_path}")
    
    def log_progress(self, message: str, level: str = "INFO"):
        """Log a progress message to the experiment log."""
        if hasattr(self, 'logs_dir'):
            log_file = self.logs_dir / "experiment.log"
        else:
            log_file = self.experiment_dir / "experiment.log"
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"
        
        with open(log_file, 'a') as f:
            f.write(log_entry)
        
        print(f"📝 {log_entry.strip()}")
    
    def update_metadata(self, updates: Dict[str, Any]):
        """Update experiment metadata."""
        self.metadata.update(updates)
        self.metadata["last_updated"] = datetime.now().isoformat()
        
        # Save updated metadata
        metadata_file = self.experiment_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Update registry
        for i, exp in enumerate(self.output_manager.registry["experiments"]):
            if exp["experiment_id"] == self.metadata["experiment_id"]:
                self.output_manager.registry["experiments"][i] = self.metadata
                break
        self.output_manager._save_registry()
    
    def get_summary(self) -> str:
        """Get a summary of the experiment."""
        summary = f"""
Experiment Summary
==================
ID: {self.metadata['experiment_id']}
Type: {self.metadata['experiment_type']}
Description: {self.metadata.get('description', 'No description')}
Created: {self.metadata['created']}
Directory: {self.experiment_dir}
Tags: {', '.join(self.metadata.get('tags', []))}
"""
        return summary


def get_output_manager(base_dir: str = "outputs") -> OutputManager:
    """Get a singleton instance of the output manager."""
    if not hasattr(get_output_manager, '_instance'):
        get_output_manager._instance = OutputManager(base_dir)
    return get_output_manager._instance 