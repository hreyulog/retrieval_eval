import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict
from tabulate import tabulate 


class Logger:
    _instance = None

    def __init__(self, source: str, results_dir: Path):
        self.source = source
        self.results_dir = results_dir
        self._setup_console_logger()

    @classmethod
    def create(cls, source: str):
        if cls._instance is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{source}_{timestamp}"
            results_dir = Path("results") / experiment_name
            results_dir.mkdir(parents=True, exist_ok=True)
            cls._instance = cls(experiment_name, results_dir)
        return cls._instance

    @classmethod
    def get(cls):
        if cls._instance is None:
            raise RuntimeError("Logger not initialized. Call Logger.create(source) first.")
        return cls._instance

    def _setup_console_logger(self):
        log_file = self.results_dir / "logs.log"

        self.logger = logging.getLogger("RAG_EVAL")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
                datefmt="%Y-%m-%d %H:%M:%S"
            )

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setFormatter(formatter)

            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)


    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def save_json(self, data: Dict, filename: str) -> Path:
        if not filename.endswith('.json'):
            filename += '.json'

        filepath = self.results_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

        self.logger.info(f"Saved JSON data to {filepath}")
        return filepath

    def info_summary(self, data: Dict, title: str = "Evaluation Summary"):
        table = tabulate(data.items(), headers=["Metric", "Value"], tablefmt="grid", floatfmt=".4f")
        summary_msg = f"\n{title}\n{table}"
        self.logger.info(summary_msg)
        print(summary_msg)
