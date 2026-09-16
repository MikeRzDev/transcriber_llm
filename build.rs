fn main() {
    let root = "vendor/speexdsp";
    println!("cargo:rerun-if-changed={root}");
    let mut build = cc::Build::new();
    build
        .include(format!("{root}/include"))
        .include(format!("{root}/libspeexdsp"))
        .define("FLOATING_POINT", None)
        .define("USE_SMALLFT", None)
        .define("EXPORT", "")
        .warnings(false);
    for source in ["preprocess", "mdf", "fftwrap", "filterbank", "smallft"] {
        build.file(format!("{root}/libspeexdsp/{source}.c"));
    }
    build.compile("transcriber_speexdsp");
}
