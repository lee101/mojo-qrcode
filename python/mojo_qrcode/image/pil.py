from PIL import Image, ImageDraw

from .base import BaseImage


class PilImage(BaseImage):
    kind = "PNG"

    def new_image(self, **kwargs):
        back_color = kwargs.get("back_color", "white")
        fill_color = kwargs.get("fill_color", "black")
        if isinstance(fill_color, str):
            fill_color = fill_color.lower()
        if isinstance(back_color, str):
            back_color = back_color.lower()
        if fill_color == "black" and back_color == "white":
            mode, fill_color, back_color = "1", 0, 255
        elif back_color == "transparent":
            mode, back_color = "RGBA", None
        else:
            mode = "RGB"
        image = Image.new(mode, (self.pixel_size, self.pixel_size), back_color)
        self.fill_color = fill_color
        self._drawer = ImageDraw.Draw(image)
        return image

    def drawrect(self, row, col):
        self._drawer.rectangle(self.pixel_box(row, col), fill=self.fill_color)

    def save(self, stream, format=None, **kwargs):
        kind = kwargs.pop("kind", self.kind)
        self._img.save(stream, format=format or kind, **kwargs)

    def __getattr__(self, name):
        return getattr(self._img, name)
