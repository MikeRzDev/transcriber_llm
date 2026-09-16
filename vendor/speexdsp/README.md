# SpeexDSP 1.2.1

Source: https://downloads.xiph.org/releases/speex/speexdsp-1.2.1.tar.gz
SHA-256: `8c777343e4a6399569c72abc38a95b24db56882c83dbdb6c6424a5f4aeb54d3d`

Unmodified subset needed for the preprocessor, with its BSD license in COPYING.
`include/speex/speexdsp_config_types.h` is generated locally using stdint types.
The root build.rs compiles floating-point C with the bundled smallft FFT.
mdf.c supplies the residual-echo symbol referenced by preprocess.c; the app
does not enable echo cancellation. No system Speex installation is required.
