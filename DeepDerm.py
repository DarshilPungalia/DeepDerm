import tensorflow as tf
import numpy as np
import pandas as pd
import os
from typing import Literal, List
import joblib
from pathlib import Path


class LesionDetector():
    def __init__(self, data_type : Literal['image', 'image & tabular'], model_dir = r'Models'):
        self.input_type = data_type
        self.model_dir = model_dir
        self.pipeline_path = os.path.join(model_dir, "MultiModal-Pipeline.joblib")
        self.image_model_path = os.path.join(model_dir, "Model1.keras")
        self.multimodal_model_path = os.path.join(model_dir, "MultiModal.keras")
        self.device = tf.config.list_logical_devices('GPU')[0].name if tf.config.list_physical_devices('GPU') else '/device:CPU:0'
        self.tab_loader = {
            '.csv' : pd.read_csv,
            '.xlsx': pd.read_excel
        }
        self.img_loader = {
            '.jpeg' : tf.image.decode_jpeg,
            '.jpg'  : tf.image.decode_jpeg,
            '.png'  : tf.image.decode_png
        }
    
    @staticmethod
    def path_validator(path : str):
        '''Checks if the input path exists or not.'''
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File does not exist: {path}")
            return True
        
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Unicode error in path '{path}': {e}")
        
    def _load_single_image(self, path: str) -> tf.Tensor:
        '''Used to load a single image for the given path.'''
        ext = Path(path).suffix
        if self.path_validator(path):
            image_data = tf.io.read_file(path)
            if ext in self.img_loader:
                image = self.img_loader[ext](image_data, channels=3)
            else:
                raise ValueError(f"Unsupported image type {ext}. Supported: {list(self.img_loader.keys())}")
            image = tf.image.resize(image, [64, 64])
            return tf.cast(image, tf.float32)  # safer than float16
        
    def _load_model(self, path: str):
        '''Caches models'''
        if not hasattr(self, '_model_cache'):
            self._model_cache = {}
        if path not in self._model_cache:
            self._model_cache[path] = tf.keras.models.load_model(path)
        return self._model_cache[path]

    def _load_pipeline(self):
        '''Caches pipeline'''
        if not hasattr(self, '_pipeline'):
            self._pipeline = joblib.load(self.pipeline_path)
        return self._pipeline
    
    @staticmethod
    def _validate_device(device: str):
        '''Checks if a given Physical Device can be accessed in the enviroment.'''
        available = [d.name for d in tf.config.list_logical_devices()]
        if device not in available:
            raise ValueError(f"Requested device {device} not found. Available: {available}")
  
    @staticmethod
    def _probabilities(predictions):
        proba = tf.math.sigmoid(predictions)
        return proba * 100
    
    @staticmethod
    def _extract_prediction_values(tensor: tf.Tensor):
        array = tensor.numpy()
        if array.size == 1:
            return array.item()  
        return array.squeeze().tolist()


    def load_and_preprocess_images(self, path: str | List[str]) -> List[tf.Tensor]:
        '''Loads a single or multiple images from given path(s).'''
        contents = []
        try:
            if isinstance(path, str):
                image = self._load_single_image(path)
                contents.append(image)
            
            elif isinstance(path, list):
                for loc in path:
                    image = self._load_single_image(path=loc)  
                    contents.append(image)
        
        except Exception as e:
            raise RuntimeError(f"Error loading Image(s): {e}")

        return contents
    
    def load_and_preprocess_tabular(self, path: str) -> np.ndarray:
        '''Used to load and transform tabular data.'''
        if self.path_validator(path):
            ext = Path(path).suffix
            if ext in self.tab_loader:
                tabular = self.tab_loader[ext](path)
            else:
                raise ValueError(f"File type {ext} is not supported. Supported file types are '.csv', '.xlsx'")
            
            try:
                pipeline = self._load_pipeline()

            except Exception as e:
                raise RuntimeError(f"Failed to load pipeline from {self.pipeline_path}: {e}")
            
            try:
                tab = pipeline.transform(tabular)
            
            except Exception as e:
                raise RuntimeError(f"Ensure that data is formatted correctly': {e}")
            
        return tab
    
    def predict(self, img_path: str | List[str] = None, tab_path: str = None, device = None, batch_size = 16)-> np.ndarray:
        '''Predict lesion type using just images or a combination of images and patient metadata.'''
        comp_device = device or self.device
        self._validate_device(comp_device)
        with tf.device(comp_device):
            if self.input_type == 'image':
                if img_path is None:
                    raise ValueError("Image path must be provided for image-only prediction.")
                images = self.load_and_preprocess_images(img_path)
                model = self._load_model(self.image_model_path)
                images = np.array(images).reshape((-1, 64, 64, 3))
                preds = model.predict(images, batch_size=batch_size, verbose=0)
                proba = self._probabilities(preds)
                return self._extract_prediction_values(proba)

            elif self.input_type == 'image & tabular':
                if img_path is None or tab_path is None:
                    raise ValueError("Both image and tabular paths are required for multimodal prediction.")
                
                images = self.load_and_preprocess_images(img_path)
                tabular = self.load_and_preprocess_tabular(tab_path)
                model = self._load_model(self.multimodal_model_path)
                images = np.array(images).reshape((-1, 64, 64, 3))
                if not images.shape[0] == tabular.shape[0]:
                    raise ValueError(f"Batch size mismatch: {images.shape[0]} image samples vs {tabular.shape[0]} tabular samples. "
                                        "Ensure both inputs have the same number of samples (N).")
                preds = model.predict((images, tabular), batch_size=batch_size, verbose=0)
                proba = self._probabilities(preds)
                return self._extract_prediction_values(proba)

            else:
                raise ValueError(f"Unsupported input type: {self.input_type}")


