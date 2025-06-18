# Sine Function Prediction Model

This project aims to predict the sine function using a TensorFlow model with the C API. The model is pre-trained and ready for inference.

## General Overview

- **Model**: A two-layer neural network that predicts the sine function.
- **Training**: The model was trained using Python and exported as a SavedModel.
- **Inference**: The model is executed using the TensorFlow C API in C.

## Prerequisites

- **TensorFlow C API**: You need to install the TensorFlow C API. Follow these steps:
  1. Download the TensorFlow C API from the [official TensorFlow website](https://www.tensorflow.org/install/lang_c).
  2. Extract the downloaded files to a directory, e.g., `/Users/ahmet/Downloads/libtensorflow-cpu-darwin-arm64/`.
  3. Copy the necessary libraries to `/usr/local/lib/`:
     ```sh
     cp /Users/ahmet/Downloads/libtensorflow-cpu-darwin-arm64/lib/libtensorflow.2.dylib /usr/local/lib/
     ```

- **GCC**: Ensure you have GCC installed on your system. You can install it using Homebrew:
  ```sh
  brew install gcc
  ```

## Installation and Execution

1. **Clone the Repository**:
   ```sh
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Compile the C Code**:
   ```sh
   gcc -o sin_runner_clean sin_runner_clean.c -I/Users/ahmet/Downloads/libtensorflow-cpu-darwin-arm64/include -L/usr/local/lib -ltensorflow
   ```

3. **Add Library Path**:
   ```sh
   install_name_tool -add_rpath /usr/local/lib sin_runner_clean
   ```

4. **Run the Executable**:
   ```sh
   ./sin_runner_clean
   ```

## Results

The model successfully predicts the sine function. Example output:

```
sin(0.0) ≈ 0.001404 (actual: 0.000000)
sin(0.1) ≈ 0.156484 (actual: 0.099833)
sin(0.2) ≈ 0.290399 (actual: 0.198669)
sin(0.3) ≈ 0.392158 (actual: 0.295520)
sin(0.4) ≈ 0.462153 (actual: 0.389418)
sin(0.5) ≈ 0.507548 (actual: 0.479426)
```

## Efficiency Improvements

- **Batch Processing**: Consider processing inputs in batches to improve throughput.
- **Model Quantization**: Use TensorFlow's quantization tools to reduce model size and improve inference speed.
- **Hardware Acceleration**: Utilize GPU or TPU for faster inference if available.
- **Optimize C Code**: Profile the C code to identify bottlenecks and optimize accordingly.

## Troubleshooting

- If you encounter issues with the TensorFlow library, ensure that the library path is correctly set and that the library is installed in the specified directory.
- For any other issues, please refer to the [TensorFlow C API documentation](https://www.tensorflow.org/install/lang_c). # -sin-inference-c-tensorflow
