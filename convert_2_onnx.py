import argparse
import os
import sys
import tensorflow as tf
import tf2onnx


def convert(model_path: str, output_path: str, opset: int = 13) -> None:
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)

    print(f"📥 Loading Keras model from: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    model.summary()

    # Infer input shape from the model itself (e.g. (None, 224, 224, 3))
    input_shape = model.input_shape
    print(f"🔎 Detected input shape: {input_shape}")

    spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)

    print(f"🔄 Converting to ONNX (opset {opset})...")
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=opset,
        output_path=output_path,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Saved ONNX model to: {output_path} ({size_mb:.2f} MB)")


def verify(output_path: str) -> None:
    """Quick sanity check that the ONNX model loads and runs."""
    try:
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        input_meta = session.get_inputs()[0]
        shape = [d if isinstance(d, int) else 1 for d in input_meta.shape]

        dummy_input = np.random.rand(*shape).astype(np.float32)
        outputs = session.run(None, {input_meta.name: dummy_input})

        print(f"🧪 Verification passed. Output shape: {outputs[0].shape}")
    except ImportError:
        print("Skipping verification (install onnxruntime to enable: pip install onnxruntime)")
    except Exception as e:
        print(f"Verification failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FeastAI Keras model to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        default="food_classifier.keras",
        help="Path to the trained .keras model file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="food_classifier.onnx",
        help="Path to save the converted .onnx model",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version (13 is broadly compatible)",
    )
    args = parser.parse_args()

    convert(args.model, args.output, args.opset)
    verify(args.output)