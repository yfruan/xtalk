from typing import Optional, TypedDict, Any, Literal


class PipelineContext(TypedDict, total=False):
    """
    Extensible pipeline context fields.

    Notes:
    - thought: internal reasoning/tool intermediate state (optional)
    - caption: scene description from multimodal/audio understanding (optional)
    - speaker_id: current speaker identifier (optional)
    - text_to_embed: text waiting to be written to vector DB (optional)
    - vector_store_instance: per-session vector store instance (e.g., Chroma)

    Defined here separately to avoid circular imports with `llm_agent`.
    """

    thought: Optional[str]
    caption: Optional[str]
    speaker_id: Optional[str]
    embedding_status: Optional[Literal["idle", "processing", "finished"]]
    text_to_embed: Optional[str]
    vector_store_instance: Optional[Any]
