"""Allow running as: python -m autoforecast"""
from dotenv import load_dotenv
load_dotenv()

from .orchestrator import main

main()
