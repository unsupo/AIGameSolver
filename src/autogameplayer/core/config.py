from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    # Server Configuration
    server_host: str = "localhost"
    server_port: int = 8000
    transport: str = "sse"
    
    # Vision Configuration
    vision_model: str = "facebook/dinov2-small"
    
    # Checkpoint Configuration (Centralized slot semantics)
    bootstrap_slot: int = 0  # Global start point / Pallet Town
    tas_trigger_slot: int = 1 # Slot that triggers TAS auto-recording
    rolling_save_start: int = 2
    rolling_save_end: int = 7
    milestone_tmp_slot: int = 99 # Temporary slot for successful individuals in trainer
    
    # Memory Detective Defaults
    # Game Boy (GB/GBC) WRAM: 0xC000-0xE000
    # GBA WRAM: 0x02000000-0x02040000
    gb_memory_range: tuple[int, int] = (0xC000, 0xE000)
    gba_memory_range: tuple[int, int] = (0x02000000, 0x02040000)
    
    # LLM Configuration
    llm_provider: str = "openai" # "openai", "ollama", "local"
    llm_api_key: str = "sk-no-key-required"
    llm_base_url: str = "http://localhost:11434/v1" # Default to Ollama
    llm_model: str = "llama3.2-vision" # Default to a common local vision model
    default_controller: str = "gb"
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent.parent
    data_dir: Path = base_dir / "data"
    
    @property
    def roms_dir(self) -> Path: return self.data_dir / "roms"
    @property
    def models_dir(self) -> Path: return self.data_dir / "models"
    @property
    def saves_dir(self) -> Path: return self.data_dir / "saves"
    @property
    def knowledge_dir(self) -> Path: return self.data_dir / "knowledge"
    @property
    def bios_dir(self) -> Path: return self.data_dir / "bios"
    @property
    def datasets_dir(self) -> Path: return self.data_dir / "datasets"
    @property
    def skills_dir(self) -> Path: return self.data_dir / "skills"
    
    @property
    def is_ollama(self) -> bool:
        return self.llm_provider == "ollama" or "11434" in str(self.llm_base_url)

    @property
    def default_embedding_model(self) -> str:
        return "nomic-embed-text" if self.is_ollama else "text-embedding-3-small"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AGP_",
        extra="ignore"
    )

settings = Settings()
