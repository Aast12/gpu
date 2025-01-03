#!/bin/bash

# Define the executable and arguments
EXECUTABLE="./build-release/hw2_ex2"
MATRIX_DIMS=(
    "64 128 64"
    "1024 1023 8193"
    "1200 1023 8193"
    "1200 1200 8193"
    "1200 1200 8500"
)
OUTPUT_FILE="ex2.csv"

# Write the header to the CSV file
echo "MatrixA_Rows,MatrixA_Cols,MatrixB_Cols,HostToDeviceTime,GPUExecutionTime,DeviceToHostTime" > $OUTPUT_FILE

# Loop over matrix dimensions
for DIMS in "${MATRIX_DIMS[@]}"; do
    read -r A K B <<< "$DIMS"
    echo "Running with matrix dimensions $A x $K x $B"
    OUTPUT=$($EXECUTABLE $A $K $B)

    echo "$OUTPUT"

    # Extract times from the output
    HOST_TO_DEVICE_TIME=$(echo "$OUTPUT" | grep "Data Copy Host to Device Time" | awk '{print $7}')
    GPU_EXECUTION_TIME=$(echo "$OUTPUT" | grep "GPU Execution Time" | awk '{print $4}')
    DEVICE_TO_HOST_TIME=$(echo "$OUTPUT" | grep "Data Copy Device to Host Time" | awk '{print $7}')

    # Write the extracted times to the CSV file
    echo "$A,$K,$B,$HOST_TO_DEVICE_TIME,$GPU_EXECUTION_TIME,$DEVICE_TO_HOST_TIME" >> $OUTPUT_FILE
done
