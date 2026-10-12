"""FrameImage / InputImages backed lazily by a video file.

Same interface as SpatialClaw's ``kernel_types/frame_image.py`` and
``kernel_types/input_images.py``: ``InputImages`` is a ``list`` of
``FrameImage`` (a PIL-compatible wrapper carrying ``.frame_index``), with
``.frame_indices`` and slicing that preserves the type. Frames are decoded on
first access from the video (SpatialClaw loads them from an extracted frame
cache instead); like SpatialClaw's default (``video_max_fps=None``) every native
frame is included, and images are downscaled to ``image_max_long_edge``.
"""

from typing import List, Optional

from PIL import Image


class VideoSource:
    """Shared decord reader for all frames of one video (opened lazily)."""

    def __init__(self, path: str, max_edge: Optional[int] = None):
        self.path = path
        self.max_edge = max_edge
        self._vr = None

    @property
    def reader(self):
        if self._vr is None:
            from decord import VideoReader, cpu
            self._vr = VideoReader(self.path, ctx=cpu(0))
        return self._vr

    def load(self, frame_index: int) -> Image.Image:
        img = Image.fromarray(self.reader[frame_index].asnumpy())
        if self.max_edge and max(img.size) > self.max_edge:
            scale = self.max_edge / max(img.size)
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        return img


class FrameImage:
    """PIL Image wrapper that carries a frame_index. Loads lazily from its VideoSource."""

    __slots__ = ("_image", "_source", "frame_index")

    def __init__(self, image_or_source, frame_index: int):
        if isinstance(image_or_source, VideoSource):
            self._image, self._source = None, image_or_source
        else:
            self._image, self._source = image_or_source, None
        self.frame_index = frame_index

    @property
    def image(self) -> Image.Image:
        if self._image is None:
            self._image = self._source.load(self.frame_index)
        return self._image

    @property
    def is_loaded(self) -> bool:
        return self._image is not None

    def __getattr__(self, name):
        if name in ("_image", "_source", "frame_index"):
            raise AttributeError(name)
        return getattr(self.image, name)

    def __repr__(self) -> str:
        if self._image is not None:
            return f"FrameImage(frame_index={self.frame_index}, size={self._image.size})"
        return f"FrameImage(frame_index={self.frame_index}, not loaded)"


# --time_label_stills: the video fps, so still frames given to the VLM (vlm.ask_with_thinking) or shown with show()
# get their time drawn in the top-left, like every video clip. None = no label (set by the kernel injection).
LABEL_FPS: Optional[float] = None


def labeled_image(frame: "FrameImage") -> Image.Image:
    """The frame's image, with its time (frame_index / LABEL_FPS) drawn in the top-left when LABEL_FPS is set."""
    if not LABEL_FPS:
        return frame.image
    from stvg_agent.kernel_types.video_clip import annotate_frame
    return annotate_frame(frame.image, frame.frame_index / LABEL_FPS)


class InputImages(list):
    """List of FrameImage (video frames); ``frame_indices`` gives absolute video frame indices."""

    def __init__(self, images: List, frame_indices: Optional[List[int]] = None):
        if not frame_indices:
            frame_indices = list(range(len(images)))
        super().__init__([img if isinstance(img, FrameImage) else FrameImage(img, idx)
                          for img, idx in zip(images, frame_indices)])

    @classmethod
    def from_video(cls, source: VideoSource, frame_indices: List[int]) -> "InputImages":
        new = cls.__new__(cls)
        list.__init__(new, [FrameImage(source, i) for i in frame_indices])
        return new

    @property
    def frame_indices(self) -> List[int]:
        return [fi.frame_index for fi in self]

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(key, slice):
            new = InputImages.__new__(InputImages)
            list.__init__(new, result)
            return new
        return result

    def __repr__(self) -> str:
        fi = self.frame_indices
        fi_str = f"[{fi[0]}, {fi[1]}, ..., {fi[-2]}, {fi[-1]}]" if len(fi) > 6 else str(fi)
        return f"InputImages({len(self)} images, frame_indices={fi_str})"
