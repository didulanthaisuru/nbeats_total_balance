# GPU Setup Guide for T4x2 GPU Server

## Overview
This guide helps you set up the GPU environment for running the NBEATSx time series forecasting model on your T4x2 GPU server.

## GPU Optimizations Included

The `research_code_gpu.py` file now includes the following GPU optimizations:

### 1. **GPU Detection and Configuration**
- Automatic CUDA availability detection
- GPU device selection (cuda:0 for first GPU)
- CUDA version and GPU information display
- Memory monitoring capabilities

### 2. **PyTorch GPU Optimizations**
- CUDA backend optimizations (`cudnn.benchmark = True`)
- GPU random seed setting for reproducibility
- Memory usage monitoring function

### 3. **NeuralForecast GPU Optimizations**
- `num_workers=0` for GPU compatibility
- GPU memory monitoring before and after training
- Optimized batch processing for GPU

### 4. **Memory Management**
- Real-time GPU memory usage tracking
- Memory allocation and reservation monitoring
- Automatic cleanup after training

## Installation Steps

### Step 1: Install PyTorch with CUDA Support

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 2: Install Other Dependencies
```bash
pip install -r requirements_gpu.txt
```

### Step 3: Verify GPU Setup
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

## Running the GPU-Optimized Code

### 1. **Execute the Script**
```bash
python research_code_gpu.py
```

### 2. **Expected Output**
The script will display:
- GPU configuration information
- CUDA availability status
- GPU memory usage during training
- Training progress with GPU acceleration

### 3. **Performance Benefits**
- **Faster Training**: GPU acceleration for neural network operations
- **Memory Efficiency**: Optimized memory usage for T4x2 setup
- **Scalability**: Better handling of large datasets and complex models

## Troubleshooting

### Common Issues

1. **CUDA Not Available**
   - Ensure PyTorch is installed with CUDA support
   - Check NVIDIA drivers are installed
   - Verify CUDA toolkit installation

2. **Out of Memory Errors**
   - Reduce batch size in the model parameters
   - Monitor GPU memory usage with the built-in function
   - Consider using gradient checkpointing for large models

3. **Slow Performance**
   - Ensure `cudnn.benchmark = True` is set
   - Check if GPU is being utilized (should show high GPU usage)
   - Verify batch size is appropriate for your GPU memory

### Memory Optimization Tips

1. **Batch Size Tuning**
   - Start with batch_size=32 for T4x2
   - Adjust based on available GPU memory
   - Monitor memory usage during training

2. **Model Architecture**
   - The current setup uses standard NBEATSx architecture
   - Consider reducing `n_blocks` if memory is limited
   - Monitor `input_size` parameter impact on memory

## Kaggle Environment

Since you're running on Kaggle:
- Kaggle provides GPU-enabled environments
- No local setup required
- Simply upload `research_code_gpu.py` and run
- GPU optimizations will automatically activate

## Monitoring and Logging

The script includes comprehensive logging:
- GPU memory usage before/after training
- Training duration tracking
- Model performance metrics
- Error handling and recovery

## Next Steps

1. Run the GPU-optimized script on your T4x2 server
2. Monitor GPU memory usage during training
3. Adjust hyperparameters based on performance
4. Scale up batch size if memory allows for faster training

## Support

If you encounter issues:
1. Check GPU memory usage with `print_gpu_memory_usage()`
2. Verify CUDA installation with PyTorch
3. Monitor training logs for error messages
4. Adjust batch size or model parameters as needed 