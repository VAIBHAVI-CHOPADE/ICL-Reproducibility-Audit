# model_clients.py
import os, json, subprocess
from typing import Any, Dict

class LLMClient:
    """
    Compatible with all Ollama versions (including 0.12.x that lack -o/--option).
    Uses environment variables for decoding options.
    """
    def __init__(self, model_name="mistral:latest", decoding=None, timeout_sec=180):
        self.model_name = model_name.replace("ollama:", "")
        self.decoding = decoding or {}
        self.timeout_sec = timeout_sec

    def _set_env(self):
        """
        Map decoding options to environment variables that Ollama honors.
        Older builds (<0.1.26) ignore CLI flags, so this method sets them globally.
        """
        env = os.environ.copy()
        d = self.decoding
        # Temperature → OLLAMA_TEMPERATURE
        if "temperature" in d:
            env["OLLAMA_TEMPERATURE"] = str(d["temperature"])
        if "top_p" in d:
            env["OLLAMA_TOP_P"] = str(d["top_p"])
        if "top_k" in d:
            env["OLLAMA_TOP_K"] = str(d["top_k"])
        if "max_tokens" in d:
            env["OLLAMA_NUM_PREDICT"] = str(d["max_tokens"])
        if "seed" in d:
            env["OLLAMA_SEED"] = str(d["seed"])
        return env

    def call(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a single [INST] prompt to the Ollama model.
        Returns plain text output.
        """
        prompt = f"[INST]{system_prompt}\n{user_prompt}[/INST]"
        env = self._set_env()

        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name],
                input=prompt.encode("utf-8"),
                capture_output=True,
                timeout=self.timeout_sec,
                env=env
            )
        except subprocess.TimeoutExpired:
            print("❌ Timeout during Ollama run.")
            return "ERROR"
        except Exception as e:
            print("⚠️ Exception calling model:", e)
            return "ERROR"

        if result.returncode != 0:
            print("❌ Ollama error:", result.stderr.decode("utf-8")[:400])
            return "ERROR"

        return result.stdout.decode("utf-8", errors="ignore").strip()

