#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/Im2Col.cu"
#else

static inline void THNN_(Im2Col_shapeCheck)(
                         THCState *state,
                         THCTensor *input,
                         THCTensor *gradOutput,
                         int kH, int kW, int dH, int dW,
                         int padH, int padW, int sH, int sW) {

  THArgCheck(kW > 0 && kH > 0, 9,
             "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(sW > 0 && sH > 0, 11,
             "stride should be greater than zero, but got sH: %d sW: %d", sH, sW);
  THArgCheck(dW > 0 && dH > 0, 11,
             "dilation should be greater than zero, but got dH: %d dW: %d", dH, dW);

  int ndim = input->nDimension;
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  THCUNN_argCheck(state, ndim == 3, 2, input,
                  "3D input tensor expected but got: %s");

  int nInputPlane  = input->size[dimf];
  int inputHeight  = input->size[dimh];
  int inputWidth   = input->size[dimw];
  int outputHeight = (inputHeight + 2*padH - kH - ((kH - 1)*(dH - 1))) / sH + 1;
  int outputWidth  = (inputWidth + 2*padW - kW - ((kW - 1)*(dW - 1))) / sW + 1;
  int nOutputPlane = nInputPlane * kW * kH;
  int outputLength = outputHeight * outputWidth;

  if (outputWidth < 1 || outputHeight < 1)
      THError("Given input size: (%d x %d x %d). "
              "Calculated output size: (%d x %d). Output size is too small",
              nInputPlane,inputHeight,inputWidth,nOutputPlane,outputLength);

  if (gradOutput != NULL) {
    THCUNN_check_dim_size(state, gradOutput, ndim, 0, nOutputPlane);
    THCUNN_check_dim_size(state, gradOutput, ndim, 1, outputLength);
  }
}

void THNN_(Im2Col_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int kH, int kW,
           int dH, int dW,
           int padH, int padW,
           int sH, int sW) {

  THCUNN_assertSameGPU(state, 2, input, output);

  THNN_(Im2Col_shapeCheck)
       (state, input, NULL, kH, kW, dH, dW, padH, padW, sH, sW);

  input = THCTensor_(newContiguous)(state, input);
  int batch = 1;
  if (input->nDimension == 3) {
      // Force batch
      batch = 0;
      THCTensor_(resize4d)(state, input, 1, input->size[0], input->size[1], input->size[2]);
  }

  // Batch_size + input planes
  int batchSize = input->size[0];

  // Params:
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;
  int inputHeight  = input->size[dimh];
  int inputWidth   = input->size[dimw];
  int nInputPlane = input->size[dimf];
  int outputHeight = (inputHeight + 2*padH - (dH * (kH - 1)) + 1) / sH + 1;
  int outputWidth  = (inputWidth + 2*padW - (dW * (kW - 1)) + 1) / sW + 1;
  int nOutputPlane = nInputPlane * kW * kH;
  int outputLength = outputHeight*outputWidth;

  // Resize output
  THCTensor_(resize3d)(state, output, batchSize, nOutputPlane, outputLength);

  // Helpers
  THCTensor *input_n = THCTensor_(new)(state);
  THCTensor *output_n = THCTensor_(new)(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt++) {
      // Matrix multiply per output:
      THCTensor_(select)(state, input_n, input, 0, elt);
      THCTensor_(select)(state, output_n, output, 0, elt);

      THCTensor_(zero)(state, output_n);
      // Extract columns:
      im2col(
        THCState_getCurrentStream(state),
        THCTensor_(data)(state, input_n),
        nInputPlane, inputHeight, inputWidth, kH, kW, padH, padW, sH, sW,
        dH, dW, THCTensor_(data)(state, output_n)
      );
  }

  // Free
  THCTensor_(free)(state, input_n);
  THCTensor_(free)(state, output_n);

  // Resize output
  if (batch == 0) {
      THCTensor_(resize2d)(state, output, nOutputPlane, outputLength);
      THCTensor_(resize3d)(state, input, nInputPlane, inputHeight, inputWidth);
  }

  THCTensor_(free)(state, input);
}

void THNN_(Im2Col_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int kH, int kW,
           int dH, int dW,
           int padH, int padW,
           int sH, int sW) {

  THCUNN_assertSameGPU(state, 3, input, gradOutput, gradInput);

  input = THCTensor_(newContiguous)(state, input);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  int batch = 1;
  if (input->nDimension == 3) {
      // Force batch
      batch = 0;
      THCTensor_(resize4d)(state, input, 1, input->size[0], input->size[1], input->size[2]);
      THCTensor_(resize3d)(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1]);
  }

  THNN_(Im2Col_shapeCheck)
       (state, input, gradOutput, kH, kW, dH, dW, padH, padW, sH, sW);

  // Batch size + input planes
  int batchSize = input->size[0];

  // Params
  int nInputPlane = input->size[1];
  int inputHeight  = input->size[2];
  int inputWidth   = input->size[3];

  int outputHeight = (inputHeight + 2*padH - (dH * (kH - 1)) + 1) / sH + 1;
  int outputWidth  = (inputWidth + 2*padW - (dW * (kW - 1)) + 1) / sW + 1;
  int nOutputPlane = nInputPlane * kW * kH;
  int outputLength = outputHeight*outputWidth;

  // Resize output
  THCTensor_(resize4d)(state, gradInput, batchSize, nInputPlane, inputHeight, inputWidth);

  // Helpers
  THCTensor *gradInput_n = THCTensor_(new)(state);
  THCTensor *gradOutput_n = THCTensor_(new)(state);

  for (int elt = 0; elt < batchSize; elt++) {
      THCTensor_(select)(state, gradInput_n, gradInput, 0, elt);
      THCTensor_(select)(state, gradOutput_n, gradOutput, 0, elt);

      THCTensor_(zero)(state, gradInput_n);
      // Unpack columns back into input:
      col2im<real, accreal>(
        THCState_getCurrentStream(state),
        THCTensor_(data)(state, gradOutput_n),
        nInputPlane, inputHeight, inputWidth,
        outputHeight, outputWidth, kH, kW, padH, padW, sH, sW,
        dH, dW, THCTensor_(data)(state, gradInput_n)
      );
  }

  // Free
  THCTensor_(free)(state, gradInput_n);
  THCTensor_(free)(state, gradOutput_n);

  // Resize output
  if (batch == 0) {
      THCTensor_(resize2d)(state, gradOutput, nOutputPlane, outputLength);
      THCTensor_(resize3d)(state, input, nInputPlane, inputHeight, inputWidth);
      THCTensor_(resize3d)(state, gradInput, nInputPlane, inputHeight, inputWidth);
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
}

#endif
