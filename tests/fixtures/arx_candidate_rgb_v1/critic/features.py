import numpy as np


class FeatureExtractor:
    def reset(self, observation, images, config):
        if config:
            raise ValueError("this fixture accepts no extractor config")

    def extract(self, observation, images):
        image = images["front_rgb"]
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("expected uint8 HWC RGB")
        value = float(image[:, :, 0].mean(dtype=np.float64) / 255.0)
        return {"visual.mean_red": {"valid": True, "value": value}}

    def close(self):
        pass
