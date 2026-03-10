#include <mgba/core/core.h>
#include <mgba/core/interface.h>
#include <mgba-util/vfs.h>
#include <stdlib.h>
#include <string.h>

static struct mCore* core = NULL;

// Initialize the GBA core with a ROM
int gba_init(const char* rom_path) {
    core = mCoreFind(rom_path);
    if (!core) return -1;

    core->init(core);
    mCoreInitConfig(core, NULL);
    
    if (!mCoreLoadFile(core, rom_path)) {
        return -2;
    }

    core->reset(core);
    return 0;
}

// Run exactly one frame
void gba_step() {
    if (core) {
        core->runFrame(core);
    }
}

// Set keys using a bitmask
void gba_set_keys(uint32_t keys) {
    if (core) {
        core->setKeys(core, keys);
    }
}

// Read a byte from memory
uint8_t gba_read_memory(uint32_t address) {
    if (core) {
        return core->rawRead8(core, address, 0);
    }
    return 0;
}

// Write a byte to memory
void gba_write_memory(uint32_t address, uint8_t value) {
    if (core) {
        core->rawWrite8(core, address, 0, value);
    }
}

// Copy pixels into a pre-allocated buffer (RGBA, 240x160)
void gba_get_pixels(uint32_t* buffer) {
    if (core) {
        core->setVideoBuffer(core, (color_t*)buffer, 240);
        // We run a dummy frame or rely on the last rendered state
        // Actually, setVideoBuffer sets where the NEXT frame will draw.
    }
}

void gba_cleanup() {
    if (core) {
        core->deinit(core);
        core = NULL;
    }
}
