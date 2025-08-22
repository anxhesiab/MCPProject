# mcp_system/__init__.py
from pathlib import Path
from dotenv import load_dotenv

# Automatically load "../.env" when any module in this package is imported.
# Adjust the relative path if your .env lives elsewhere.
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=False)
