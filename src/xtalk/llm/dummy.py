# filepath: c:\xcc\data\code\xtalk\src\xtalk\llm_model\dummy.py
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class DummyChatModel(BaseChatModel):
    """
    LangChain ChatModel for tests that ignores inputs and returns a fixed reply.
    """

    # Default response (can be overridden via constructor)
    default_response: str = (
        'The term "psychology" can refer to the entirety of humans\' internal mental activities. It can also denote an organism\'s subjective reflection of the objective world, as well as the processes and phenomena related to mental activity, such as emotion, thinking, and behavior. In addition, "psychology" is often used to refer to the academic discipline that studies human psychological phenomena, mental functions, and behavior.'
    )

    def __init__(self, default_response: Optional[str] = None, **kwargs: Any):
        super().__init__(**kwargs)
        if default_response is not None:
            self.default_response = default_response

    def _apply_stop(self, content: str, stop: Optional[List[str]]) -> str:
        if not stop:
            return content
        # Find earliest stop token occurrence and trim the response
        cut = len(content)
        for s in stop:
            if not s:
                continue
            idx = content.find(s)
            if idx != -1:
                cut = min(cut, idx)
        return content[:cut]

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        content = self._apply_stop(self.default_response, stop)
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
        content = self._apply_stop(self.default_response, stop)
        # Emit full content in one chunk; split further if needed
        yield ChatGenerationChunk(message=AIMessageChunk(content=content))

    @property
    def _llm_type(self) -> str:
        return "dummy"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"default_response": self.default_response}
