#!/bin/bash

# Define the executable and arguments
EXECUTABLE="./hw2_ex1"
INPUT_LENGTHS=(1024 65536 16777216 4294967296)
BLOCK_SIZES=(256 512 1024)
OUTPUT_FILE="times.csv"

# Write the header to the CSV file
echo "InputLength,BlockSize,HostToDeviceTime,GPUExecutionTime,DeviceToHostTime" > $OUTPUT_FILE
# Loop over input lengths and block sizes
for LENGTH in "${INPUT_LENGTHS[@]}"; do
  for BLOCK_SIZE in "${BLOCK_SIZES[@]}"; do
    echo "Running with input length $LENGTH and block size $BLOCK_SIZE"
    OUTPUT=$($EXECUTABLE $LENGTH $BLOCK_SIZE)

    # Extract times from the output
    HOST_TO_DEVICE_TIME=$(echo "$OUTPUT" | grep "Data Copy Host to Device Time" | awk '{print $7}')
    GPU_EXECUTION_TIME=$(echo "$OUTPUT" | grep "GPU Execution Time" | awk '{print $4}')
    DEVICE_TO_HOST_TIME=$(echo "$OUTPUT" | grep "Data Copy Device to Host Time" | awk '{print $7}')

    # Write the extracted times to the CSV file
    echo "$LENGTH,$BLOCK_SIZE,$HOST_TO_DEVICE_TIME,$GPU_EXECUTION_TIME,$DEVICE_TO_HOST_TIME" >> $OUTPUT_FILE
  done
done