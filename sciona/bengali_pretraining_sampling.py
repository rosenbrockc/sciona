"""Source classifier augmentation draws, including Cutout decision and center."""
from sciona.bengali_font_sampling import FontParameterSampler
from sciona.bengali_pretraining_augmentation import PretrainingTransform


class PretrainingParameterSampler(FontParameterSampler):
    def draw(self):
        geometry=self._draw_geometry(20)
        self.python.random()  # BasicTransform still makes the Cutout decision.
        y=self.python.randint(0,224)
        x=self.python.randint(0,224)
        return PretrainingTransform(*geometry,y,x)
