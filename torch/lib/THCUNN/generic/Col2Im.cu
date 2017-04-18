#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/Col2Im.cu"
#else

static inline void THNN_(Col2Im_shapeCheck)(
                         THCState *state,
                         THCTensor *input,
                         THCTensor *gradOutput,
                         int outputHeight, int outputWidth,
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

  if (ndim == 4) {
      dimf++;
      dimh++;
      dimw++;
  }

  THCUNN_argCheck(state, ndim == 2 || ndim == 3, 2, input,
                  "2D or 3D input tensor expected but got %s");

  long nInputPlane  = input->size[dimf];
  long inputLength  = input->size[dimh];

  long nOutputPlane = nInputPlane / (kW * kH);

  if (outputWidth < 1 || outputHeight < 1)
      THError("Given input size: (%d x %d). "
              "Calculated output size: (%d x %d x %d). Output size is too small",
              nInputPlane,inputLength,nOutputPlane,outputHeight,outputWidth);

  /*if (gradOutput != NULL) {
    THCUNN_check_dim_size(state, gradOutput, 4, dimf, nOutputPlane);
    THCUNN_check_dim_size(state, gradOutput, 4, dimh, outputHeight);
    THCUNN_check_dim_size(state, gradOutput, 4, dimw, outputWidth);
  }*/
}

void THNN_(Col2Im_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int outputHeight, int outputWidth,
           int kH, int kW,
           int dH, int dW,
           int padH, int padW,
           int sH, int sW) {

  THCUNN_assertSameGPU(state, 2, input, output);

  THNN_(Col2Im_shapeCheck)
       (state, input, NULL, outputHeight, outputWidth, kH, kW, dH, dW, padH, padW, sH, sW);

  int batch = 1;
  if (input->nDimension == 2) {
      // Force batch
      batch = 0;
      THCTensor_(resize3d)(state, input, 1, input->size[0], input->size[1]);
  }

  // Batch_size + input planes
  long batchSize = input->size[0];

  // Params:
  long nInputPlane = input->size[1];
  long inputLength = input->size[2];

  long nOutputPlane = nInputPlane / (kW * kH);

  input = THCTensor_(newContiguous)(state, input);

  // Resize output
  THCTensor_(resize4d)(state, output, batchSize, nOutputPlane, outputHeight, outputWidth);

  // Helpers
  THCTensor *input_n = THCTensor_(new)(state);
  THCTensor *output_n = THCTensor_(new)(state);

  int height_col = (outputHeight + 2 * padH - (dH * (kH - 1) + 1))
                   / sH + 1;
  int width_col = (outputWidth + 2 * padW - (dW * (kW - 1) + 1))
                   / sW + 1;

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt++) {
    THCTensor_(select)(state, input_n, input, 0, elt);
    THCTensor_(select)(state, output_n, output, 0, elt);

    THCTensor_(zero)(state, output_n);
    // Unpack columns:
    col2im<real, accreal>(
        THCState_getCurrentStream(state),
        THCTensor_(data)(state, input_n),
        nOutputPlane,
        outputHeight, outputWidth,
        height_col, width_col,
        kH, kW,
        padH, padW,
        sH, sW,
        dH, dW, THCTensor_(data)(state, output_n)
     );
  }

  // Free
  THCTensor_(free)(state, input_n);
  THCTensor_(free)(state, output_n);

  // Resize output
  if (batch == 0) {
      THCTensor_(resize3d)(state, output, nOutputPlane, outputHeight, outputWidth);
      THCTensor_(resize2d)(state, input, nInputPlane, inputLength);
  }

  THCTensor_(free)(state, input);
}

void THNN_(Col2Im_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int outputHeight, int outputWidth,
           int kH, int kW,
           int dH, int dW,
           int padH, int padW,
           int sH, int sW) {

  THCUNN_assertSameGPU(state, 3, input, gradOutput, gradInput);

  input = THCTensor_(newContiguous)(state, input);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  int batch = 1;
  if (input->nDimension == 2) {
      // Force batch
      batch = 0;
      THCTensor_(resize3d)(state, input, 1, input->size[0], input->size[1]);
  }

  THNN_(Col2Im_shapeCheck)
       (state, input, gradOutput, outputHeight, outputWidth, kH, kW, dH, dW, padH, padW, sH, sW);

  // Batch_size + input planes
  long batchSize = input->size[0];

  // Params:
  long nInputPlane = input->size[1];
  long inputLength = input->size[2];

  long nOutputPlane = nInputPlane / (kW * kH);

  // Resize output
  THCTensor_(resize3d)(state, gradInput, batchSize, nInputPlane, inputLength);

  // Helpers
  THCTensor *gradInput_n = THCTensor_(new)(state);
  THCTensor *gradOutput_n = THCTensor_(new)(state);

  for (int elt = 0; elt < batchSize; elt++) {

    THCTensor_(select)(state, gradInput_n, gradInput, 0, elt);
    THCTensor_(select)(state, gradOutput_n, gradOutput, 0, elt);

    THCTensor_(zero)(state, gradInput_n);
    // Extract columns:
    im2col(
        THCState_getCurrentStream(state),
        THCTensor_(data)(state, gradOutput_n),
        nOutputPlane, outputHeight, outputWidth, kH, kW, padH, padW, sH, sW,
        dH, dW, THCTensor_(data)(state, gradInput_n)
    );

  }

  // Free
  THCTensor_(free)(state, gradInput_n);
  THCTensor_(free)(state, gradOutput_n);

  // Resize output
  if (batch == 0) {
      THCTensor_(resize3d)(state, gradOutput, nOutputPlane, outputHeight, outputWidth);
      THCTensor_(resize2d)(state, input, nInputPlane, inputLength);
      THCTensor_(resize2d)(state, gradInput, nInputPlane, inputLength);
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
}

#endif
