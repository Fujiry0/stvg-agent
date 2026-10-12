"""Spatial grounding prompts."""

def build_spatial_prompt(
    model_family: str,
    query_text: str,
    text_only: bool = False
) -> str:
    """Build a spatial grounding prompt specification."""
    query = query_text.strip()

    if model_family == "gemini":
        return (
            "Answer with bounding boxes and do not output explanation. "
            "Bounding box format: [y_min, x_min, y_max, x_max] with values in [0, 1000] normalized to the video frame.\n"
            f"What are all the positions corresponding to the text query: \"{query}\"? "
            "Output format: "
            "[{\"timestamp\":\"00:30\", \"box_2d\":[100, 200, 300, 400]},\n"
            " {\"timestamp\":\"05:00\", \"box_2d\":[150, 250, 350, 450]}]\n"
        )
    if model_family == "gpt":
        return (
            "Given the frames of video, please find all the objects corresponding "
            f"to the text query:\n\"{query}\".\n"
            "Output only a JSON array.\n"
            "Frame index format: 0-based integer.\n"
            "Bounding box format: [x0, y0, x1, y1] with normalized value in 0.000 ~ 1.000\n"
            "Example:\n"
            "[{\"frame_id\": 1, \"bbox_2d\": [x_min, y_min, x_max, y_max]}, \n"
            "{\"frame_id\": 2, \"bbox_2d\": [x_min, y_min, x_max, y_max]}, ...]."
        )
    if model_family == "internvl":
        return (
            f"Given the query \"{query}\", for each frame, detect and localize "
            "the visual content described by the given textual query in JSON "
            "format. If the visual content does not exist in a frame, skip "
            "that frame.\n"
            "Output Format: "
            "[{\"timestamp\": 1.0, \"bbox_2d\": [x_min, y_min, x_max, y_max]},"
            "{\"timestamp\": 2.0, \"bbox_2d\": [x_min, y_min, x_max, y_max]}, ...]."
        )
    if model_family == "qwen":
        return (
            f"Given the query \"{query}\", for each frame, detect and localize "
            "the visual content described by the given textual query in JSON "
            "format. If the visual content does not exist in a frame, skip "
            "that frame.\n"
            "Output Format: "
            "[{\"timestamp\": 1.0, \"bbox_2d\": [x_min, y_min, x_max, y_max]},"
            "{\"timestamp\": 2.0, \"bbox_2d\": [x_min, y_min, x_max, y_max]}, ...]."
        )
    if model_family == "eagle":
        return (
            f"Given the query \"{query}\", for each frame, detect and localize "
            "the visual content described by the given textual query in JSON "
            "format. If the visual content does not exist in a frame, skip "
            "that frame.\n"
            "Output Format: "
            "[{\"timestamp\": 1.0, \"bbox_2d\": [x_min, y_min, x_max, y_max]},"
            "{\"timestamp\": 2.0, \"bbox_2d\": [x_min, y_min, x_max, y_max]}, ...]."
        )

    if model_family == "llava_st":
        start_token = "<TEMP-000>"
        end_token = "<TEMP-099>"

        return (
            f"Between {{{start_token}{end_token}}}, \"{query}\". "
            "Please describe the location of the corresponding subject/object in this video. "
            "Please give the spatial bounding box corresponding to each timestamp in the time period."
        )

    raise ValueError(f"Unsupported model family: {model_family}")
