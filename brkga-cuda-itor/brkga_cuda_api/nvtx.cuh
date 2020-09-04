/**
 * \file nvtx.cuh
 * \brief NVIDIA Tools Extension (NVTX) utilitary macros.
 */
#ifndef __NVTX_H__
#define __NVTX_H__

#ifdef USE_NVTX
#pragma message "Compiling with the use of NVTX!"

#include <nvToolsExt.h>
#include <nvToolsExtCuda.h>
#include <sys/syscall.h>
#include <unistd.h>

// Basic colors
#define WHITE 0xffffffff
#define GRAY 0xff707070
#define BLACK 0xff000000
#define RED 0xffff0000
#define GREEN 0xff00ff00
#define BLUE 0xff0000ff
#define CYAN 0xff00ffff
#define MAGENTA 0xffff00ff
#define YELLOW 0xffffff00

// Additional colors
#define MY_GREEN 0xFF0D804A
#define MY_PURPLE 0xFF410D80
#define MY_PINK 0xFF800D43
#define MY_ORANGE 0xFFD18330
#define MY_RED 0xFF1CB8A5
#define MY_BLUE 0xFF0F58B8
#define MY_CYAN 0xFF1CB8A5
#define MY_YELLOW 0xFFD1A930

/**
 * \brief Global NVTX Event Attributes structure.
 *
 * Instead of allocating a new struct every time we want to push a tag, we reuse
 * the same global structure and only modify the relevant fields when the
 * NVTX_PUSH_RANGE macro is invoked.
 */
static nvtxEventAttributes_t nvtx_attribs = {0};

/**
 * \brief Push a range to the NVTX stack.
 *
 * \param label the name of the tag
 * \param hex   the ARGB color hexadecimal (e.g. (uint32_t) 0xFF001122)
 *
 * \todo        Add support for label formatting (like printf).
 * \todo        Add support for optional color.
 */
#define NVTX_PUSH_RANGE(label, hex)                                            \
  do {                                                                         \
    nvtx_attribs.version = NVTX_VERSION;                                       \
    nvtx_attribs.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                         \
    nvtx_attribs.colorType = NVTX_COLOR_ARGB;                                  \
    nvtx_attribs.color = (hex);                                                \
    nvtx_attribs.messageType = NVTX_MESSAGE_TYPE_ASCII;                        \
    nvtx_attribs.message.ascii = label;                                        \
    nvtxRangePushEx(&nvtx_attribs);                                            \
  } while (0)

/**
 * \brief Push a range to the NVTX stack with the name of current function.
 *
 * \param hex   the ARGB color hexadecimal (e.g. (uint32_t) 0xFF001122)
 */
#define NVTX_PUSH_FUNCTION(hex) NVTX_PUSH_RANGE(__FUNCTION__, hex)

/**
 * \brief Pop a range from the NVTX stack.
 */
#define NVTX_POP_RANGE() nvtxRangePop()

/**
 * \brief Place a NVTX Mark.
 *
 * \param hex   the ARGB color hexadecimal (e.g. (uint32_t) 0xFF001122)
 */
#define NVTX_MARK(label, hex)                                                  \
  do {                                                                         \
    nvtx_attribs.color = (hex);                                                \
    nvtx_attribs.messageType = NVTX_MESSAGE_TYPE_ASCII;                        \
    nvtx_attribs.message.ascii = label;                                        \
    nvtxMarkEx(&nvtx_attribs);                                                 \
  } while (0)

/**
 * \brief Name a CUDA Device.
 *
 * \param device  the device number
 * \param label   the device name/label
 */
#define NVTX_NAME_DEVICE(device, name)                                         \
  nvtxNameCuDeviceA((CUdevice)(device), name)

/**
 * \brief Name a host thread.
 *
 * \param label     the thread name
 */
#define NVTX_NAME_THREAD(name) nvtxNameOsThreadA(syscall(SYS_gettid), name)

/**
 * \brief Give a name to a stream.
 *
 * \param stream    CUDA Stream object
 * \param label     Stream name string
 */
#define NVTX_NAME_STREAM(stream, label) nvtxNameCuStreamA(stream, label)

#else // USE_NVTX

// Do nothing when not using NVTX.
#define NVTX_PUSH_RANGE(label, hex)
#define NVTX_PUSH_FUNCTION(hex)
#define NVTX_POP_RANGE()
#define NVTX_MARK(label, hex)
#define NVTX_NAME_DEVICE(device, label)
#define NVTX_NAME_THREAD(name)
#define NVTX_NAME_STREAM(stream, label)

#endif // USE_NVTX

#endif /* __NVTX_H__ */
