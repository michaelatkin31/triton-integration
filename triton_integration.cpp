/**
 * @file triton_integration.cpp
 * 
 * Description:
 *   This file contains a C++ API for compiling and executing Triton kernels.
 *
 * Instructions:
 * 1. Initialize a conda env. Install all dependencies such that test_aot.py runs to completion.
 * 2. export LD_LIBRARY_PATH=/opt/conda/envs/NAME_OF_ENVIRONMENT/lib:$LD_LIBRARY_PATH
 * 3. export PYTHONPATH="/path/to/the/directory/containing/triton_module.py:$PYTHONPATH"
 * 
 * Compile:
 * g++ triton_integration.cpp -o triton_integration -I/opt/conda/envs/triton/include/python3.12 -L/opt/conda/envs/triton/lib -lpython3.12 -lpthread
 *
 * Run:
 * ./triton_integration
 *
 */
#include <Python.h>
#include <iostream>
#include <string>
#include <boost/process.hpp>
#include <filesystem>

struct TritonConfig {
    std::string dtype;
    int BM; // Block size in the M dimension
    int BN; // Block size in the N dimension
    int BK; // Block size in the K dimension
    int M;
    int N;
    int K;
};

/**
 * @struct TritonHandler
 * @brief Handler for managing paths related to Triton kernel execution.
 * 
 * @var TritonHandler::path Path to the compiled Triton kernel or related resources.
 */
struct TritonHandler {
    std::string path;
};

/**
 * @brief Constructs a TritonConfig with the specified parameters.
 * 
 * @param dtype Data type of the elements in the matrices.
 * @param BM Block size in the M dimension.
 * @param BN Block size in the N dimension.
 * @param BK Block size in the K dimension.
 * @param M Total size in the M dimension.
 * @param N Total size in the N dimension.
 * @param K Total size in the K dimension.
 * @return TritonConfig The constructed configuration object.
 */
TritonConfig get_triton_config(const std::string& dtype, int BM, int BN, int BK, int M, int N, int K) {
    TritonConfig config;
    config.dtype = dtype;
    config.BM = BM;
    config.BN = BN;
    config.BK = BK;
    config.M = M;
    config.N = N;
    config.K = K;
    return config;
}

/**
 * @brief Compiles the Triton kernel using the provided configuration.
 * 
 * Initializes the Python interpreter, imports a Python module for compiling Triton kernels (triton_module.py),
 * and calls the compile function within the module. The path to the
 * compiled kernel and related resources is returned in a TritonHandler.
 * 
 * @param config Configuration for the Triton kernel compilation.
 * @return TritonHandler Handler containing the path to the compiled kernel and resources.
 */
TritonHandler compile_triton_kernel(TritonConfig config) {
    // Initialize the Python Interpreter
    Py_Initialize();

    // Define the Python script filename and function name
    const char *scriptFilename = "triton_module";
    const char *functionName = "compile";

    PyObject *pName, *pModule, *pFunc;
    PyObject *pArgs, *pValue;

    // Convert the filename and function name to Python objects
    pName = PyUnicode_DecodeFSDefault(scriptFilename);

    // Import the module
    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    TritonHandler handler;
    handler.path = "";

    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, functionName);

        if (pFunc && PyCallable_Check(pFunc)) {
            pArgs = PyTuple_New(7); 
            PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(config.dtype.c_str()));
            PyTuple_SetItem(pArgs, 1, PyLong_FromLong(config.BM));
            PyTuple_SetItem(pArgs, 2, PyLong_FromLong(config.BN));
            PyTuple_SetItem(pArgs, 3, PyLong_FromLong(config.BK));
            PyTuple_SetItem(pArgs, 4, PyLong_FromLong(config.M));
            PyTuple_SetItem(pArgs, 5, PyLong_FromLong(config.N));
            PyTuple_SetItem(pArgs, 6, PyLong_FromLong(config.K));

            pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);

            if (pValue != NULL) {
                if (PyUnicode_Check(pValue)) {
                    // Assuming the Python function returns a string (path)
                    PyObject* tempBytes = PyUnicode_AsEncodedString(pValue, "UTF-8", "strict"); // Convert Unicode to bytes
                    if (tempBytes != NULL) {
                        handler.path = PyBytes_AS_STRING(tempBytes); // Convert bytes to C string
                        Py_DECREF(tempBytes);
                    }
                }
                Py_DECREF(pValue);
            } else {
                PyErr_Print();
            }
            Py_XDECREF(pFunc);
            Py_DECREF(pModule);
        } else {
            if (PyErr_Occurred())
                PyErr_Print();
            std::cerr << "Cannot find function \"" << functionName << "\"" << std::endl;
        }
    } else {
        PyErr_Print();
        std::cerr << "Failed to load \"" << scriptFilename << "\"" << std::endl;
    }

    // Clean up and close the Python Interpreter
    Py_Finalize();

    return handler;
}

/**
 * @brief Executes the compiled Triton kernel with the specified input matrices.
 * 
 * Copies the input matrices to the working directory, constructs the command to execute
 * the kernel, and manages the environment for the execution. Outputs are written to the
 * specified output matrix file path.
 * 
 * @param handler Handler containing the path to the compiled kernel or resources.
 * @param matrixAFilePath Path to the input matrix A. (1-line CSV)
 * @param matrixBFilePath Path to the input matrix B. (1-line CSV)
 * @param outputMatrixFilepath Path where the output matrix will be written. (1-line CSV)
 */
void run_triton_kernel(const TritonHandler& handler, const std::string& matrixAFilePath, const std::string& matrixBFilePath, const std::string& outputMatrixFilepath) {
    namespace bp = boost::process; // Namespace alias for convenience
    namespace fs = std::filesystem; // Namespace alias for std::filesystem

    // Working directory
    std::string workDir = handler.path;

    // Copy a.csv and b.csv to workDir
    fs::copy(matrixAFilePath, fs::path(workDir) / fs::path(matrixAFilePath).filename(), fs::copy_options::overwrite_existing);
    fs::copy(matrixBFilePath, fs::path(workDir) / fs::path(matrixBFilePath).filename(), fs::copy_options::overwrite_existing);

    // Construct the command to run './test' with file paths as arguments
    // Adjust the file paths to point to the new location in workDir
    std::string command = "./test " + (fs::path(workDir) / fs::path(matrixAFilePath).filename()).string() + " " + (fs::path(workDir) / fs::path(matrixBFilePath).filename()).string() + " " + outputMatrixFilepath;

    // Environment: Copy current environment and modify LD_LIBRARY_PATH
    bp::environment env = boost::this_process::environment(); // Copy current environment
    env["LD_LIBRARY_PATH"] = workDir; // Set LD_LIBRARY_PATH to handler.path

    // Execute the command
    bp::child c(command, bp::start_dir=workDir, env, (bp::std_out & bp::std_err) > stdout); // Redirect stdout and stderr to the parent's stdout
    // Wait for the process to finish
    c.wait();

    // Check the result of the execution
    int result = c.exit_code();
    if (result != 0) {
        std::cerr << "The command failed to execute properly." << std::endl;
    }
}


/**
 * @brief Main function demonstrating the usage of Triton integration.
 * 
 * Constructs a TritonConfig, compiles the Triton kernel, and executes it with input matrices
 * living in the data/ directory. Demonstrates the workflow from configuration to execution.
 * 
 * @return int Exit code of the application.
 */
int main() {
    TritonConfig config = get_triton_config("fp16", 16, 16, 16, 16, 16, 16);
    TritonHandler handler = compile_triton_kernel(config);
    std::cout << "Kernel in path: " << handler.path << std::endl;

    std::string basePath = "./data/"; // Assuming current directory with "./", "data/" is the subdirectory
    std::string matrixAPath = basePath + "a.csv";
    std::string matrixBPath = basePath + "b.csv";
    std::string outputMatrixPath = "/home/ubuntu/triton-integration/data/c.csv"; // TODO change this to a valid, absolute filepath on your machine. Using a relative filepath throws an error, and I wasn't able to figure out the issue

    std::cout << "Running kernel with arguments: " << handler.path << " " << matrixAPath << " " << matrixBPath << " " << outputMatrixPath << std::endl;
    run_triton_kernel(handler, matrixAPath, matrixBPath, outputMatrixPath);
    std::cout << "Kernel result written to " << outputMatrixPath << std::endl;
    std::cout << "Test completed." << std::endl;
    return 0;
}

