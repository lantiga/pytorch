#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Im2Col.c"
#else

void THNN_(Im2Col_updateOutput)(
        THNNState *state,
        THTensor *input,
        THTensor *output,
        int kH, int kW,
        int dH, int dW,
        int padH, int padW,
        int sH, int sW)
{
    THAssertMsg(false, "Not implemented for CPU");
}

void THNN_(Im2Col_updateGradInput)(
        THNNState *state,
        THTensor *input,
        THTensor *gradOutput,
        THTensor *gradInput,
        int kH, int kW,
        int dH, int dW,
        int padH, int padW,
        int sH, int sW)
{
    THAssertMsg(false, "Not implemented for CPU");
}

#endif
