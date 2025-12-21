from xtalk.model_types import Agent
from typing import Union


class EchoAgent(Agent):
    """A simple agent that echoes user input."""

    def generate(self, input) -> str:
        if isinstance(input, dict):
            return input["content"]
        return input

    def clone(self) -> "EchoAgent":
        return EchoAgent()
