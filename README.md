<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

**Table of Contents** _generated with
[DocToc](https://github.com/thlorenz/doctoc)_

- [Godot ExecuTorch Module](#godot-executorch-module)
  - [⚠️ **WARNING: EXPERIMENTAL PROJECT - NOT READY FOR PRODUCTION** ⚠️](#-warning-experimental-project---not-ready-for-production-)
  - [Overview](#overview)
  - [Installation](#installation)
    - [Building the Module into Godot](#building-the-module-into-godot)
    - [Alternative: Development Setup](#alternative-development-setup)
  - [Usage in Godot](#usage-in-godot)
    - [Basic Setup](#basic-setup)
    - [Running Inference](#running-inference)
    - [Using MCP Tools](#using-mcp-tools)
    - [Performance Monitoring](#performance-monitoring)
  - [Module Architecture](#module-architecture)
    - [Core Components](#core-components)
    - [Class Hierarchy](#class-hierarchy)
  - [Module Integration Benefits](#module-integration-benefits)
  - [Development](#development)
    - [Building for Different Platforms](#building-for-different-platforms)
    - [Testing the Module](#testing-the-module)
  - [Deployment](#deployment)
  - [Troubleshooting](#troubleshooting)
    - [Common Issues](#common-issues)
  - [License](#license)
- [ExecuTorchResource for Godot Engine](#executorchresource-for-godot-engine)
  - [Overview](#overview-1)
  - [Architecture](#architecture)
    - [High-Level API (ExecuTorch Module Class)](#high-level-api-executorch-module-class)
    - [Low-Level API (Custom Memory Management)](#low-level-api-custom-memory-management)
  - [Key Features](#key-features)
    - [1. **Flexible Memory Management**](#1-flexible-memory-management)
    - [2. **Performance Optimization**](#2-performance-optimization)
    - [3. **Resource Integration**](#3-resource-integration)
    - [4. **Developer-Friendly APIs**](#4-developer-friendly-apis)
  - [Usage Examples](#usage-examples)
    - [Basic Linear Regression](#basic-linear-regression)
    - [Advanced Configuration](#advanced-configuration)
    - [Godot Integration (GDScript)](#godot-integration-gdscript)
  - [Implementation Details](#implementation-details)
    - [ExecuTorch C++ API Integration](#executorch-c-api-integration)
    - [Memory Management](#memory-management)
    - [Performance Profiling](#performance-profiling)
  - [Testing](#testing)
    - [Test Coverage](#test-coverage)
    - [Linear Regression Test Cases](#linear-regression-test-cases)
  - [Building](#building)
    - [As Godot Module](#as-godot-module)
    - [Standalone Testing](#standalone-testing)
  - [Integration with ExecuTorch](#integration-with-executorch)
    - [Supported ExecuTorch Features](#supported-executorch-features)
    - [ExecuTorch Version Compatibility](#executorch-version-compatibility)
  - [Performance Characteristics](#performance-characteristics)
    - [Memory Usage](#memory-usage)
    - [Inference Speed](#inference-speed)
    - [Platform Support](#platform-support)
  - [Future Enhancements](#future-enhancements)
  - [License](#license-1)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Godot ExecuTorch Module

A native C++ module for Godot Engine that provides ExecuTorch machine learning
inference capabilities with built-in MCP (Model Context Protocol) support.

## ⚠️ **WARNING: EXPERIMENTAL PROJECT - NOT READY FOR PRODUCTION** ⚠️

> **🚧 WORK IN PROGRESS 🚧**
>
> This project is currently in **early development** and is **NOT ready for
> production use**.
>
> **Current Status:**
>
> - 🔴 **API is unstable** and subject to breaking changes
> - 🔴 **Limited testing** - may contain bugs and memory leaks
> - 🔴 **Incomplete features** - many planned features are not implemented
> - 🔴 **No official releases** - use at your own risk
> - 🔴 **Documentation may be outdated** or incorrect
>
> **Do not use this in:**
>
> - Production games or applications
> - Commercial projects
> - Critical systems
>
> **This is for experimentation and development only!**

## Overview

This module integrates directly into Godot Engine as a C++ module (not a
GDExtension), providing seamless access to ExecuTorch models from within Godot
projects.

## Installation

### Building the Module into Godot

1. **Clone/Copy the module into Godot's modules directory:**

   ```bash
   cd /path/to/godot/source
   cp -r /path/to/godot_executorch_module modules/executorch
   ```

2. **Build Godot with the module:**

   ```bash
   # Build for your target platform
   scons platform=linuxbsd target=template_debug
   # or
   scons platform=windows target=template_debug
   # or
   scons platform=osx target=template_debug
   ```

3. **The module will be built into the Godot binary automatically**

### Alternative: Development Setup

For development, you can also build just the module:

```bash
cd godot_executorch_module
g++ -std=c++17 -fPIC -shared *.cpp -o libexecutorch_module.so
```

## Usage in Godot

### Basic Setup

```gdscript
extends Node

var runtime: ExecuTorchRuntime

func _ready():
    # The class is automatically available after building the module
    runtime = ExecuTorchRuntime.new()

    # Load a model
    if runtime.load_model_from_file("res://models/linear_regression.pte"):
        print("Model loaded successfully!")
    else:
        print("Failed to load model")
```

### Running Inference

```gdscript
func run_prediction(input_value: float):
    if not runtime.is_model_loaded():
        print("No model loaded")
        return

    # Prepare input data
    var inputs = {
        "input_0": [input_value]
    }

    # Run inference
    var result = runtime.run_inference(inputs)

    if result.has("output_0"):
        var prediction = result["output_0"][0]
        print("Input: ", input_value, " -> Prediction: ", prediction)
        return prediction

    return null
```

### Using MCP Tools

```gdscript
func test_mcp_features():
    # List available tools
    var tools = runtime.list_mcp_tools()
    print("Available MCP tools: ", tools)

    # Get model information
    var model_info = runtime.get_model_info()
    print("Model info: ", model_info)

    # Health check
    var health = runtime.health_check()
    print("Health status: ", health)

    # Direct tool call
    var tool_result = runtime.call_mcp_tool("run_inference", {
        "input_0": [2.5]
    })
    print("Tool result: ", tool_result)
```

### Performance Monitoring

```gdscript
func benchmark_model():
    runtime.reset_performance_stats()

    # Run multiple inferences
    for i in range(100):
        runtime.run_inference({"input_0": [float(i)]})

    print("Total inferences: ", runtime.get_total_inferences())
    print("Last inference time: ", runtime.get_last_inference_time(), "ms")
```

## Module Architecture

### Core Components

- **ExecuTorchRuntime**: Main interface class exposed to GDScript
- **ExecuTorchModel**: Internal model wrapper and inference engine
- **MCPServerInternal**: Built-in Model Context Protocol server
- **register_types**: Godot module registration and lifecycle

### Class Hierarchy

```
ExecuTorchRuntime (exposed to GDScript)
├── ExecuTorchModel (internal model handling)
└── MCPServerInternal (MCP protocol implementation)
```

## Module Integration Benefits

Compared to GDExtension, the C++ module approach provides:

- **Native Integration**: Direct access to Godot's core APIs
- **Better Performance**: No marshalling overhead between GDScript and C++
- **Simplified Distribution**: Built directly into the engine binary
- **Easier Debugging**: Full access to Godot's debugging infrastructure
- **Reduced Dependencies**: No external .so/.dll files needed

## Development

### Building for Different Platforms

**Linux:**

```bash
scons platform=linuxbsd target=template_debug
```

**Windows:**

```bash
scons platform=windows target=template_debug
```

**macOS:**

```bash
scons platform=osx target=template_debug
```

**Android:**

```bash
scons platform=android target=template_debug android_arch=arm64v8
```

### Testing the Module

The module includes the same linear regression test case:

```gdscript
extends Node

func _ready():
    test_linear_regression()

func test_linear_regression():
    var runtime = ExecuTorchRuntime.new()

    # Test with known linear function: y = 2x + 3
    var test_cases = [
        {"input": 0.0, "expected": 3.0},   # 2*0 + 3 = 3
        {"input": 1.0, "expected": 5.0},   # 2*1 + 3 = 5
        {"input": 2.0, "expected": 7.0},   # 2*2 + 3 = 7
        {"input": -1.0, "expected": 1.0}   # 2*(-1) + 3 = 1
    ]

    for test_case in test_cases:
        var result = runtime.run_inference({
            "input_0": [test_case.input]
        })

        if result.has("output_0"):
            var actual = result["output_0"][0]
            var expected = test_case.expected
            var error = abs(actual - expected)

            print("Input: ", test_case.input)
            print("Expected: ", expected, ", Actual: ", actual)
            print("Error: ", error)
            print("✓ Test passed" if error < 0.1 else "✗ Test failed")
            print("---")
```

## Deployment

When distributing your Godot project:

1. **Export Templates**: Use export templates built with the module
2. **No Additional Files**: The module is built into the engine binary
3. **Cross-Platform**: Works on all platforms where Godot runs
4. **Model Files**: Include your `.pte` model files in the project

## Troubleshooting

### Common Issues

**Module not found:**

- Ensure the module is in `modules/executorch/` in Godot source
- Rebuild Godot completely after adding the module

**Compilation errors:**

- Check that all source files are present
- Verify C++17 support is available
- Update SCons if needed

**Runtime errors:**

- Check model file paths and formats
- Verify ExecuTorch model compatibility
- Enable verbose logging for debugging

## License

This module follows the same license as your main project. Ensure ExecuTorch
licensing compatibility for commercial use.

# ExecuTorchResource for Godot Engine

A comprehensive implementation of ExecuTorch (.pte) files as Godot Engine
resources, providing both high-level and low-level APIs for machine learning
inference.

## Overview

The ExecuTorchResource class makes PyTorch ExecuTorch models first-class
citizens in Godot Engine by implementing them as native Godot Resources. This
allows for seamless integration with Godot's resource system, including:

- Native .pte file loading and saving
- Resource Inspector integration
- Memory-efficient model management
- Both simplified and advanced APIs

## Architecture

### High-Level API (ExecuTorch Module Class)

The high-level interface uses ExecuTorch's Module class for simplified model
loading and execution:

```cpp
// High-level usage
auto resource = std::make_unique<ExecuTorchResource>();
resource->load_from_file("model.pte");
Dictionary result = resource->forward(inputs);
```

### Low-Level API (Custom Memory Management)

The low-level interface provides direct control over memory allocation, operator
placement, and execution:

```cpp
// Low-level usage
resource->configure_memory(ExecuTorchResource::MEMORY_POLICY_STATIC, 2 * 1024 * 1024);
resource->set_optimization_level(ExecuTorchResource::OPTIMIZATION_AGGRESSIVE);
resource->enable_profiling(true);
```

## Key Features

### 1. **Flexible Memory Management**

- **Auto Policy**: Automatic memory management (default)
- **Static Policy**: Pre-allocated memory pool for embedded systems
- **Custom Policy**: User-defined memory allocators

### 2. **Performance Optimization**

- **Optimization Levels**: None, Basic, Aggressive
- **Profiling Support**: Detailed performance metrics
- **Memory Monitoring**: Real-time allocation tracking

### 3. **Resource Integration**

- **Native Godot Resource**: Full Inspector support
- **PCK Integration**: Models can be embedded in game packages
- **Streaming**: Large models can be loaded from disk

### 4. **Developer-Friendly APIs**

- **Simple Forward Pass**: Dictionary-based input/output
- **Array Interface**: Direct array processing
- **Metadata Access**: Model introspection capabilities

## Usage Examples

### Basic Linear Regression

```cpp
#include "executorch_resource.h"

// Create and configure resource
auto model = std::make_unique<ExecuTorchResource>();
model->load_from_file("linear_regression.pte");

// Run inference: y = 2x + 3
Dictionary inputs;
inputs["input_0"] = std::vector<float>{1.0f};

Dictionary outputs = model->forward(inputs);
float result = outputs["output_0"][0]; // Expected: 5.0
```

### Advanced Configuration

```cpp
// Configure memory for embedded system
model->configure_memory(
    ExecuTorchResource::MEMORY_POLICY_STATIC,
    1024 * 1024  // 1MB static pool
);

// Enable aggressive optimizations
model->set_optimization_level(ExecuTorchResource::OPTIMIZATION_AGGRESSIVE);

// Enable detailed profiling
model->enable_profiling(true);

// Monitor performance
Dictionary memory_info = model->get_memory_info();
double inference_time = model->get_last_inference_time();
```

### Godot Integration (GDScript)

```gdscript
extends Node

var model: ExecuTorchResource

func _ready():
    model = load("res://models/my_model.pte")

    if model.is_loaded():
        var result = model.forward({"input_0": [2.5]})
        print("Prediction: ", result["output_0"][0])
```

## Implementation Details

### ExecuTorch C++ API Integration

The implementation leverages ExecuTorch's C++ APIs:

```cpp
// High-level Module class usage
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/executor/method.h>

// Create module from .pte file
auto module = std::make_unique<torch::executor::Module>(file_path);

// Execute inference
auto outputs = module->forward({inputs});
```

### Memory Management

```cpp
// Low-level memory control
#include <executorch/runtime/platform/runtime.h>
#include <executorch/runtime/core/memory_allocator.h>

// Configure custom allocator
auto allocator = std::make_unique<torch::executor::MemoryAllocator>(
    memory_pool, pool_size
);
```

### Performance Profiling

```cpp
// Enable detailed profiling
#include <executorch/runtime/core/event_tracer.h>

auto tracer = std::make_unique<torch::executor::EventTracer>();
tracer->start_profiling();
```

## Testing

The implementation includes comprehensive unit tests using doctest:

```bash
# Build and run tests
chmod +x build_and_test.sh
./build_and_test.sh
```

### Test Coverage

- **Resource Lifecycle**: Creation, loading, unloading
- **Memory Management**: All memory policies and configurations
- **Performance Tracking**: Inference timing and statistics
- **Linear Regression**: End-to-end mathematical validation
- **Error Handling**: Invalid inputs and edge cases

### Linear Regression Test Cases

The tests validate a linear regression model `y = 2x + 3`:

| Input | Expected Output | Test Case      |
| ----- | --------------- | -------------- |
| 0.0   | 3.0             | Zero input     |
| 1.0   | 5.0             | Unit input     |
| 2.0   | 7.0             | Double input   |
| -1.0  | 1.0             | Negative input |

## Building

### As Godot Module

1. Copy to Godot's modules directory:

   ```bash
   cp -r godot-executorch /path/to/godot/modules/executorch
   ```

2. Build Godot with the module:
   ```bash
   scons platform=linuxbsd target=template_debug
   ```

### Standalone Testing

```bash
# Compile and test
cd godot-executorch
chmod +x build_and_test.sh
./build_and_test.sh
```

## Integration with ExecuTorch

### Supported ExecuTorch Features

- **Model Loading**: .pte file format support
- **Inference Execution**: Forward pass execution
- **Memory Management**: Custom allocators and memory pools
- **Operator Support**: All ExecuTorch operators
- **Platform Support**: Mobile, embedded, desktop

### ExecuTorch Version Compatibility

This implementation is designed for ExecuTorch v0.1+ and supports:

- PyTorch 2.0+ model export
- Quantized models (INT8, FP16)
- Mobile-optimized operators
- Custom operators (if registered)

## Performance Characteristics

### Memory Usage

- **High-Level API**: Automatic memory management
- **Low-Level API**: User-controlled allocation
- **Static Pools**: Predictable memory usage for embedded systems

### Inference Speed

- **Optimization Levels**: Configurable performance vs. accuracy trade-offs
- **Profiling**: Detailed per-operator timing
- **Memory Locality**: Optimized data layout

### Platform Support

- **Desktop**: Linux, Windows, macOS
- **Mobile**: Android, iOS
- **Embedded**: ARM, RISC-V (with static memory)

## Future Enhancements

- **GPU Support**: Integration with ExecuTorch GPU backends
- **Quantization**: Runtime quantization support
- **Streaming**: Large model streaming and paging
- **Batch Processing**: Multi-sample inference batching
- **Custom Operators**: Easy registration of custom ops

## License

This implementation follows the ExecuTorch and Godot Engine licensing terms.
Ensure compatibility for commercial use.
