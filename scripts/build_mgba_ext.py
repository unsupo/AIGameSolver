from cffi import FFI

ffi = FFI()

ffi.set_source(
    "_mgba_ext",
    """
    #include <mgba/core/core.h>
    #include <mgba/core/interface.h>
    #include <mgba-util/vfs.h>
    """,
    include_dirs=["/opt/homebrew/opt/mgba/include"],
    library_dirs=["/opt/homebrew/opt/mgba/lib"],
    libraries=["mgba"],
    extra_compile_args=["-std=c99", "-Wno-unused-variable"],
)

# We just declare the specific functions and struct pointers we need.
# In API mode, CFFI parses this and generates C code that knows the exact layout.
# We don't need to define the whole struct! We can just treat it as an opaque pointer if we only pass it around, 
# or we can define it partially and CFFI will use the C compiler's knowledge of the struct.
# Wait, if we want to access fields, we define them. If we just call functions, opaque is fine!
# BUT mGBA relies on function pointers INSIDE the struct: `core->runFrame(core)`.
# CFFI API mode: If we define `struct mCore { void (*runFrame)(struct mCore*); ... };`, CFFI will generate code to access `runFrame`. It uses `offsetof` in the generated C code!
# Let's add the "..." to let CFFI know there are other fields.

ffi.cdef("""
    typedef uint32_t color_t;
    struct mCore;
    struct VFile;
    
    struct mCore {
        bool (*init)(struct mCore*);
        void (*deinit)(struct mCore*);
        void (*reset)(struct mCore*);
        void (*runFrame)(struct mCore*);
        void (*setKeys)(struct mCore*, uint32_t keys);
        bool (*loadROM)(struct mCore*, struct VFile* vf);
        bool (*loadSave)(struct mCore*, struct VFile* vf);
        ...;
    };

    struct mCore* mCoreFind(const char* path);
    bool mCoreLoadFile(struct mCore* core, const char* path);
    void mCoreInitConfig(struct mCore* core, const char* port);
    
    struct VFile* VFileOpen(const char* path, int flags);
    bool mCoreTakeScreenshotVF(struct mCore* core, struct VFile* vf);
    
    // We also need VFile struct definition for size and close if we want them, but we might just need it opaque.
    struct VFile {
        bool (*close)(struct VFile* vf);
        ssize_t (*size)(struct VFile* vf);
        ...;
    };
""")

if __name__ == "__main__":
    ffi.compile(verbose=True)
