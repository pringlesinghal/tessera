# tessera_ml/models/__init__.py
"""Model modules for tessera_ml"""

from .modules import TransformerEncoder, ProjectionHead
from .ssl_model import MultimodalBTModel, BarlowTwinsLoss, compute_cross_correlation
from .quantization import FakeQuantizeRepresentation, quantize_tensor_symmetric, dequantize_tensor_symmetric

__all__ = [
    "TransformerEncoder",
    "ProjectionHead",
    "MultimodalBTModel",
    "BarlowTwinsLoss",
    "compute_cross_correlation",
    "FakeQuantizeRepresentation",
    "quantize_tensor_symmetric",
    "dequantize_tensor_symmetric",
]

