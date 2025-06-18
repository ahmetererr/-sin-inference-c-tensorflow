#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <tensorflow/c/c_api.h>

/**
 * Deallocator used to free tensor data.
 * This function is a no-op and is used to prevent TensorFlow from freeing the tensor data.
 *
 * Args:
 *     data (void*): Pointer to the tensor data.
 *     a (size_t): Size of the data.
 *     b (void*): Additional context (unused).
 */
void NoOpDeallocator(void* data, size_t a, void* b) {}

/**
 * Lists all operations in the graph.
 * This function iterates through the graph and prints the names of all operations.
 *
 * Args:
 *     graph (TF_Graph*): TensorFlow graph to list operations from.
 */
void PrintOperations(TF_Graph* graph) {
    printf("\n=== All Operations in the Graph ===\n");
    size_t pos = 0;
    TF_Operation* oper;
    while ((oper = TF_GraphNextOperation(graph, &pos)) != NULL) {
        printf("Operation found: %s\n", TF_OperationName(oper));
    }
    printf("=== End ===\n\n");
}

/**
 * Prints tensor information.
 * This function prints the shape, type, and size of the given tensor.
 *
 * Args:
 *     tensor (TF_Tensor*): TensorFlow tensor to print information for.
 */
void print_tensor_info(TF_Tensor* tensor) {
    int num_dims = TF_NumDims(tensor);
    printf("Tensor shape: [");
    for (int i = 0; i < num_dims; i++) {
        printf("%lld", TF_Dim(tensor, i));
        if (i < num_dims - 1) printf(", ");
    }
    printf("]\n");
    printf("Tensor type: %d\n", TF_TensorType(tensor));
    printf("Tensor size: %zu bytes\n", TF_TensorByteSize(tensor));
}

int main() {
    TF_Status* status = TF_NewStatus();
    TF_Graph* graph = TF_NewGraph();
    TF_SessionOptions* sess_opts = TF_NewSessionOptions();
    const char* tags = "serve";
    const char* export_dir = "./sin_model_capi";

    // Load the SavedModel
    TF_Session* session = TF_LoadSessionFromSavedModel(
        sess_opts, NULL, export_dir, &tags, 1, graph, NULL, status);

    if (TF_GetCode(status) != TF_OK) {
        printf("Failed to load model: %s\n", TF_Message(status));
        TF_DeleteStatus(status);
        TF_DeleteGraph(graph);
        TF_DeleteSessionOptions(sess_opts);
        return 1;
    }
    printf("Model loaded successfully.\n");

    PrintOperations(graph);

    // Find input and output operations
    TF_Operation* input_op = TF_GraphOperationByName(graph, "serving_default_x");
    TF_Operation* output_op = TF_GraphOperationByName(graph, "StatefulPartitionedCall");

    // Error checking
    if (!input_op || !output_op) {
        printf("❌ Input or output operation not found!\n");
        return 1;
    }

    TF_Output input = {input_op, 0};
    TF_Output output = {output_op, 0};

    // Prepare input data
    float input_data[] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    int64_t input_dims[] = {6, 1};
    TF_Tensor* input_tensor = TF_AllocateTensor(TF_FLOAT, input_dims, 2, sizeof(float) * 6);
    memcpy(TF_TensorData(input_tensor), input_data, sizeof(float) * 6);

    printf("\nInput tensor information:\n");
    print_tensor_info(input_tensor);

    TF_Tensor* output_tensor = NULL;
    TF_Status* run_status = TF_NewStatus();
    int success = 0;

    // Perform inference with TF_SessionRun
    TF_SessionRun(session, NULL,
                 &input, &input_tensor, 1,
                 &output, &output_tensor, 1,
                 NULL, 0, NULL, run_status);

    if (TF_GetCode(run_status) == TF_OK) {
        success = 1;
        printf("Success!\n");
    } else {
        printf("Error: %s\n", TF_Message(run_status));
    }

    if (success) {
        printf("\nOutput tensor information:\n");
        print_tensor_info(output_tensor);
        float* output_data = (float*)TF_TensorData(output_tensor);
        printf("\nResults:\n");
        for (int i = 0; i < 6; i++) {
            printf("sin(%.1f) ≈ %.6f (actual: %.6f)\n", 
                   input_data[i], output_data[i], sinf(input_data[i]));
        }
    }

    TF_DeleteTensor(input_tensor);
    if (output_tensor != NULL) {
        TF_DeleteTensor(output_tensor);
    }
    TF_DeleteStatus(run_status);
    TF_DeleteSession(session, status);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
    TF_DeleteSessionOptions(sess_opts);
    return 0;
} 