#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Col2Im.c"
#else

void THNN_(Col2Im_updateOutput)(
        THNNState *state,
        THTensor *input,
        THTensor *output,
        int outputHeight, int outputWidth,
        int kH, int kW,
        int dH, int dW,
        int padH, int padW,
        int sH, int sW)
{
    THAssertMsg(false, "Not implemented for CPU");
}

void THNN_(Col2Im_updateGradInput)(
        THNNState *state,
        THTensor *input,
        THTensor *gradOutput,
        THTensor *gradInput,
        int outputHeight, int outputWidth,
        int kH, int kW,
        int dH, int dW,
        int padH, int padW,
        int sH, int sW)
{
    THAssertMsg(false, "Not implemented for CPU");
}

#endif
