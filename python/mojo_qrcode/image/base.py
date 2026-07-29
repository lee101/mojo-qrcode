from __future__ import annotations

import abc


class BaseImage:
    kind = None
    allowed_kinds = None
    needs_context = False
    needs_processing = False
    needs_drawrect = True

    def __init__(self, border, width, box_size, *args, **kwargs):
        self.border = border
        self.width = width
        self.box_size = box_size
        self.pixel_size = (width + border * 2) * box_size
        self.modules = kwargs.pop("qrcode_modules")
        self._img = self.new_image(**kwargs)
        self.init_new_image()

    @abc.abstractmethod
    def drawrect(self, row, col):
        pass

    @abc.abstractmethod
    def save(self, stream, kind=None):
        pass

    @abc.abstractmethod
    def new_image(self, **kwargs):
        pass

    def init_new_image(self):
        pass

    def get_image(self, **kwargs):
        return self._img

    def pixel_box(self, row, col):
        x = (col + self.border) * self.box_size
        y = (row + self.border) * self.box_size
        return (
            (x, y),
            (x + self.box_size - 1, y + self.box_size - 1),
        )

    def check_kind(self, kind, transform=None):
        if kind is None:
            kind = self.kind
        allowed = not self.allowed_kinds or kind in self.allowed_kinds
        if transform:
            kind = transform(kind)
            if not allowed:
                allowed = kind in self.allowed_kinds
        if not allowed:
            raise ValueError(f"Cannot set {type(self).__name__} type to {kind}")
        return kind

    def is_eye(self, row, col):
        return (
            (row < 7 and col < 7)
            or (row < 7 and self.width - col < 8)
            or (self.width - row < 8 and col < 7)
        )
