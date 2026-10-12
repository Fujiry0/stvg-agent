"""Temporal grounding prompts."""

def build_temporal_prompt(
    model_family: str,
    query_text: str,
    text_only: bool = False,
    selected_frame_ids: list[int] | None = None,
) -> str:
    """Build a temporal grounding prompt specification."""
    query = query_text.strip()

    if model_family == "gemini":
        return (
            "Answer with time ranges and do not output explanation. "
            f"What is the single time range corresponding to the text query: \"{query}\"? "
            "Output format:"
            "[start, end]"
        )

    if model_family == "gpt":
        return (
            "The input images are frames from a video. Output the frame indexes that "
            f"correspond to the text query: \"{query}\". Only output the index range, "
            "for example, 2-4, 6-8."
        )

    if model_family == "internvl":
        return (
            f"Give you a textual query: {query}\n"
            "When does the described content occur in the video?\n"
            "Please return the timestamp in seconds. "
            "Output format:"
            "[start, end]"
        )

    if model_family == "qwen":
        return (
            f"Give you a textual query: {query}\n"
            "When does the described content occur in the video?\n"
            "Please return the timestamp in seconds. "
            "Output format:"
            "[start, end]"
        )

    if model_family == "eagle":
        return (
            f"Give you a textual query: {query}\n"
            "When does the described content occur in the video?\n"
            "Please return the timestamp in seconds. "
            "Output format:"
            "[start, end]"
        )

    if model_family == "llava_st":
        # official example: "Give you a textual query: 'person takes a laptop from the shelf'. When does the described content occur in the video? Please return the start and end timestamps."
        return (
            f"Give you a textual query: '{query}'. "
            "When does the described content occur in the video? "
            "Please return the start and end timestamps."
        )


    raise ValueError("Invalid model inputs")

