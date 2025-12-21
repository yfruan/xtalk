from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    AIMessageChunk,
)
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGenerationChunk
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Any, Dict, Iterator, List, Optional
from ..log_utils import logger


class LocalQwenChatModel(BaseChatModel):
    """LangChain adapter for a local Qwen model"""

    model_name: str = "Qwen/Qwen3-0.6B"
    max_new_tokens: int = 32768
    temperature: float = 0.7

    tokenizer: Any = None
    model: Any = None

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 32768,
        temperature: float = 0.7,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device_map
        )

    def _convert_messages_to_qwen_format(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, str]]:
        """Convert LangChain message objects into the Qwen chat schema"""
        qwen_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                qwen_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                qwen_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                qwen_messages.append({"role": "assistant", "content": message.content})
            else:
                # Treat everything else as a user message
                qwen_messages.append({"role": "user", "content": str(message.content)})
        return qwen_messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response"""
        # Convert messages to Qwen format
        qwen_messages = self._convert_messages_to_qwen_format(messages)

        # Prepare model input
        text = self.tokenizer.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Run generation
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        # Extract newly generated tokens
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        # Decode
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip(
            "\n"
        )

        # Wrap output in ChatResult
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the response tokens"""
        # Convert messages to Qwen format
        qwen_messages = self._convert_messages_to_qwen_format(messages)

        # Prepare prompt text while disabling thinking mode
        text = self.tokenizer.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # 禁用推理模式
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Use TextIteratorStreamer for real streaming generation
        from transformers import TextIteratorStreamer
        from threading import Thread

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,  # Skip prompt tokens in the stream
            skip_special_tokens=True,  # Skip special tokens
            timeout=30.0,  # Timeout
        )

        # Generation parameters
        generation_kwargs = dict(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
            **kwargs,
        )

        # Run generation on a background thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Stream output chunks
        generated_text = ""
        try:
            for new_text in streamer:
                if new_text:
                    generated_text += new_text
                    yield ChatGenerationChunk(message=AIMessageChunk(content=new_text))
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
        finally:
            thread.join()  # Ensure the generation thread finishes

    @property
    def _llm_type(self) -> str:
        return "local_qwen"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
