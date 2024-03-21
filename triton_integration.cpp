#include <Python.h>
#include <iostream>
#include <string>

struct TritonConfig {
    std::string dtype;
    int BM;
    int BN;
    int BK;
    int M;
    int N;
    int K;
};

struct TritonHandler {
    std::string path;
};

// Function to get the Triton configuration.
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

    std::string resultPath = "";

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
                        resultPath = PyBytes_AS_STRING(tempBytes); // Convert bytes to C string
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

    TritonHandler handler;
    handler.path = resultPath;
    return handler;
}

void run_triton_kernel(const TritonHandler& handler, const std::string& filepath1, const std::string& filepath2, const std::string& filepath3) {
    // Construct the command to set environment variables, change directory, and run './test' within the handler's path
    std::string command = "export LD_LIBRARY_PATH=/opt/conda/envs/triton/lib:$LD_LIBRARY_PATH && "
                          "export PYTHONPATH=\"/home/ubuntu/triton-integration:$PYTHONPATH\" && "
                          "cd " + handler.path + " && ./test " + filepath1 + " " + filepath2 + " " + filepath3;

    // Use system() to execute the command
    int result = std::system(command.c_str());

    // Check the result of the execution
    if (result != 0) {
        std::cerr << "The command failed to execute properly." << std::endl;
    }
}

// void run_triton_kernel(TritonHandler handler, const std::string& filepath1, const std::string& filepath2, const std::string& filepath3) {
//     std::cout << "hola muchacho" << std::endl;
//     // Initialize the Python Interpreter
//     Py_Initialize();
//     std::cout << "hola muchacho 2" << std::endl;
//     // Define the Python script filename and function name
//     const char *scriptFilename = "triton_module";
//     std::cout << "hola muchacho 3" << std::endl;
//     const char *functionName = "run";

//     std::cout << "hola muchacho 4" << std::endl;
//     PyObject *pName, *pModule, *pFunc;
//     std::cout << "hola muchacho 5" << std::endl;
//     PyObject *pArgs, *pValue;
//     std::cout << "hola muchacho 6" << std::endl;

//     // Convert the filename and function name to Python objects
//     std::cout << "scriptFilename: " << scriptFilename << std::endl;
//     pName = PyUnicode_DecodeFSDefault(scriptFilename);
//     std::cout << "hola muchacho 7" << std::endl;


//     // Import the module
//     pModule = PyImport_Import(pName);
//     std::cout << "hola muchacho 8" << std::endl;
//     Py_DECREF(pName);

//     std::cout << "hello" << std::endl;

//     if (pModule != NULL) {
//         pFunc = PyObject_GetAttrString(pModule, functionName);

//         if (pFunc && PyCallable_Check(pFunc)) {
//             pArgs = PyTuple_New(4); 
//             PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(handler.path.c_str())); // Handler's path as the first argument
//             PyTuple_SetItem(pArgs, 1, PyUnicode_FromString(filepath1.c_str()));
//             PyTuple_SetItem(pArgs, 2, PyUnicode_FromString(filepath2.c_str()));
//             PyTuple_SetItem(pArgs, 3, PyUnicode_FromString(filepath3.c_str()));

//             // Call the Python function
//             pValue = PyObject_CallObject(pFunc, pArgs);
//             Py_DECREF(pArgs);

//             if (pValue != NULL) {
//                 // Handle the function return value if necessary
//                 std::cout << "Python run function executed successfully." << std::endl;
//                 Py_DECREF(pValue);
//             } else {
//                 PyErr_Print();
//                 std::cerr << "Python run function call failed." << std::endl;
//             }
//             Py_XDECREF(pFunc);
//             Py_DECREF(pModule);
//         } else {
//             if (PyErr_Occurred())
//                 PyErr_Print();
//             std::cerr << "Cannot find function 'run' in module." << std::endl;
//         }
//     } else {
//         PyErr_Print();
//         std::cerr << "Failed to load module 'triton_module'." << std::endl;
//     }

//     // Clean up and close the Python Interpreter
//     Py_Finalize();
// }

// Example usage
int main() {
    TritonConfig config = get_triton_config("fp16", 16, 16, 16, 16, 16, 16);
    TritonHandler handler = compile_triton_kernel(config);
    std::cout << "Kernel in path: " << handler.path << std::endl;

    std::string basePath = "./data/"; // Assuming current directory with "./", "data/" is the subdirectory
    std::string fileAPath = basePath + "a.csv";
    std::string fileBPath = basePath + "b.csv";
    std::string fileCPath = basePath + "c.csv";
    std::cout << "Running kernel with arguments: " << handler.path << " " << fileAPath << " " << fileBPath << " " << fileCPath << std::endl;
    run_triton_kernel(handler, fileAPath, fileBPath, fileCPath);
    std::cout << "Test completed." << std::endl;
    return 0;
}
