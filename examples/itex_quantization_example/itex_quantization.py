import argparse
import os

from intel_extension_for_tensorflow.python.optimize.quantization import from_model, default_static_qconfig, default_dataset


def model_quantize(model_path, output):
  converter=from_model(model_path)
  converter.optimizations = default_static_qconfig()
  converter.representative_dataset = default_dataset('dummy_v2')(input_shape=(224, 224, 3), label_shape=(1, ))
  # TODO: Support user dataset.
  # converter.representative_dataset = itex.quantization.dataset(user_dataset)
  q_model = converter.convert()
  q_model.save(output)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--model_path", help="path of savemodel", required=True)
  parser.add_argument("-o", "--output", help="path of quantized model")
  args = parser.parse_args()
  if (not args.output):
    output = os.path.join(os.path.dirname(args.model_path), "int8_" + os.path.basename(args.model_path))
  else:
    output = args.output

  model_quantize(args.model_path, output)
  