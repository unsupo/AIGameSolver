import os
from cffi import FFI

ffi = FFI()

# Minimal C definitions to get us running without requiring full headers
ffi.cdef("""
    typedef uint32_t color_t;
    struct mCore;
    struct mCoreConfig { ...; };
    
    struct mCore* mCoreFind(const char* path);
    bool mCoreLoadFile(struct mCore* core, const char* path);
    void mCoreInitConfig(struct mCore* core, const char* port);
    bool mCoreAutoloadSave(struct mCore* core);
    bool mCoreAutoloadPatch(struct mCore* core);
    bool mCoreAutoloadCheats(struct mCore* core);
    
    void mCoreReset(struct mCore*);
    void mCoreRunFrame(struct mCore*);
    void mCoreSetKeys(struct mCore*, uint32_t keys);
    void mCoreDeinit(struct mCore*);

    extern const char* const projectVersion;
    
    enum mPlatform {
        mPLATFORM_NONE = -1,
        mPLATFORM_GBA = 0,
        mPLATFORM_GB = 1,
    };
""")

import ctypes.util

lib_path = os.environ.get("LIBMGBA_PATH")
if not lib_path:
    lib_path = ctypes.util.find_library("mgba")
if not lib_path:
    # Common locations as fallback
    fallbacks = [
        "/opt/homebrew/opt/mgba/lib/libmgba.dylib",
        "/usr/local/lib/libmgba.so",
        "libmgba.so"
    ]
    for p in fallbacks:
        if os.path.exists(p):
            lib_path = p
            break

if not lib_path:
    raise ImportError("Could not find libmgba. Please set LIBMGBA_PATH or install mgba.")

lib = ffi.dlopen(lib_path)
