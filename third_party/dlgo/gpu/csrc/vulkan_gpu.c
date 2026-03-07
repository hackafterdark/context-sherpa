#include "vulkan_gpu.h"

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#define LOAD_VULKAN() LoadLibraryA("vulkan-1.dll")
#define GET_PROC(lib, name) (void*)GetProcAddress((HMODULE)(lib), name)
#define CLOSE_LIB(lib) FreeLibrary((HMODULE)(lib))
typedef HMODULE LibHandle;
#else
#include <dlfcn.h>
#define LOAD_VULKAN() dlopen("libvulkan.so.1", RTLD_NOW)
#define GET_PROC(lib, name) dlsym(lib, name)
#define CLOSE_LIB(lib) dlclose(lib)
typedef void* LibHandle;
#endif

// ---------------------------------------------------------------------------
// Vulkan function pointers (loaded dynamically, no link-time dependency)
// ---------------------------------------------------------------------------
static LibHandle vk_lib = NULL;
static PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr_ = NULL;

#define VK_FUNC(name) static PFN_##name name##_ = NULL;
VK_FUNC(vkCreateInstance)
VK_FUNC(vkDestroyInstance)
VK_FUNC(vkEnumeratePhysicalDevices)
VK_FUNC(vkGetPhysicalDeviceProperties)
VK_FUNC(vkGetPhysicalDeviceMemoryProperties)
VK_FUNC(vkGetPhysicalDeviceQueueFamilyProperties)
VK_FUNC(vkCreateDevice)
VK_FUNC(vkDestroyDevice)
VK_FUNC(vkGetDeviceQueue)
VK_FUNC(vkCreateCommandPool)
VK_FUNC(vkDestroyCommandPool)
VK_FUNC(vkAllocateCommandBuffers)
VK_FUNC(vkFreeCommandBuffers)
VK_FUNC(vkBeginCommandBuffer)
VK_FUNC(vkEndCommandBuffer)
VK_FUNC(vkQueueSubmit)
VK_FUNC(vkQueueWaitIdle)
VK_FUNC(vkCreateFence)
VK_FUNC(vkDestroyFence)
VK_FUNC(vkWaitForFences)
VK_FUNC(vkResetFences)
VK_FUNC(vkResetCommandBuffer)
VK_FUNC(vkAllocateMemory)
VK_FUNC(vkFreeMemory)
VK_FUNC(vkCreateBuffer)
VK_FUNC(vkDestroyBuffer)
VK_FUNC(vkGetBufferMemoryRequirements)
VK_FUNC(vkBindBufferMemory)
VK_FUNC(vkMapMemory)
VK_FUNC(vkUnmapMemory)
VK_FUNC(vkFlushMappedMemoryRanges)
VK_FUNC(vkInvalidateMappedMemoryRanges)
VK_FUNC(vkCreateShaderModule)
VK_FUNC(vkDestroyShaderModule)
VK_FUNC(vkCreateComputePipelines)
VK_FUNC(vkDestroyPipeline)
VK_FUNC(vkCreatePipelineLayout)
VK_FUNC(vkDestroyPipelineLayout)
VK_FUNC(vkCreateDescriptorSetLayout)
VK_FUNC(vkDestroyDescriptorSetLayout)
VK_FUNC(vkCreateDescriptorPool)
VK_FUNC(vkDestroyDescriptorPool)
VK_FUNC(vkAllocateDescriptorSets)
VK_FUNC(vkUpdateDescriptorSets)
VK_FUNC(vkResetDescriptorPool)
VK_FUNC(vkCmdBindPipeline)
VK_FUNC(vkCmdBindDescriptorSets)
VK_FUNC(vkCmdDispatch)
VK_FUNC(vkCmdPushConstants)
VK_FUNC(vkCmdPipelineBarrier)
VK_FUNC(vkCmdCopyBuffer)
VK_FUNC(vkDeviceWaitIdle)
static PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR_ = NULL;
#undef VK_FUNC

// ---------------------------------------------------------------------------
// GPU state
// ---------------------------------------------------------------------------
typedef struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    uint64_t size;
    int host_visible;
} BufferAlloc;

#define MAX_BUFFERS 8192
#define MAX_DESCRIPTORS_PER_POOL 4096
#define STAGING_SIZE (128 * 1024 * 1024) // 128MB staging buffer

static struct {
    int initialized;
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family;
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd_buf;
    VkFence fence;

    VkPhysicalDeviceProperties dev_props;
    VkPhysicalDeviceMemoryProperties mem_props;

    // Staging buffer for uploads/downloads
    VkBuffer staging_buf;
    VkDeviceMemory staging_mem;
    void* staging_mapped;
    uint64_t staging_size;

    // Buffer allocator
    BufferAlloc buffers[MAX_BUFFERS];
    int buf_count;

    // Pipelines
    VkPipeline pipelines[PIPE_COUNT];
    VkPipelineLayout pipe_layouts[PIPE_COUNT];
    VkDescriptorSetLayout desc_layouts[PIPE_COUNT];
    VkDescriptorPool desc_pool;
    int pipelines_ready;

    // Subgroup size (for shader workgroup optimization)
    uint32_t subgroup_size;

    // Command buffer batching mode
    int recording; // 1 = batching dispatches, 0 = immediate submit
    int dispatch_count; // number of dispatches in current batch
    int need_barrier; // 1 = insert barrier before next dispatch
} g = {0};

// ---------------------------------------------------------------------------
// Dynamic Vulkan loading
// ---------------------------------------------------------------------------
static int load_vulkan(void) {
    vk_lib = LOAD_VULKAN();
    if (!vk_lib) return GPU_ERR_NO_VULKAN;

    vkGetInstanceProcAddr_ = (PFN_vkGetInstanceProcAddr)GET_PROC(vk_lib, "vkGetInstanceProcAddr");
    if (!vkGetInstanceProcAddr_) { CLOSE_LIB(vk_lib); vk_lib = NULL; return GPU_ERR_NO_VULKAN; }

    vkCreateInstance_ = (PFN_vkCreateInstance)vkGetInstanceProcAddr_(NULL, "vkCreateInstance");
    vkEnumeratePhysicalDevices_ = (PFN_vkEnumeratePhysicalDevices)vkGetInstanceProcAddr_(NULL, "vkEnumeratePhysicalDevices");
    return GPU_OK;
}

static void load_instance_funcs(VkInstance inst) {
    #define LOAD(name) name##_ = (PFN_##name)vkGetInstanceProcAddr_(inst, #name);
    LOAD(vkDestroyInstance)
    LOAD(vkEnumeratePhysicalDevices)
    LOAD(vkGetPhysicalDeviceProperties)
    LOAD(vkGetPhysicalDeviceMemoryProperties)
    LOAD(vkGetPhysicalDeviceQueueFamilyProperties)
    LOAD(vkCreateDevice)
    LOAD(vkDestroyDevice)
    LOAD(vkGetDeviceQueue)
    LOAD(vkCreateCommandPool)
    LOAD(vkDestroyCommandPool)
    LOAD(vkAllocateCommandBuffers)
    LOAD(vkFreeCommandBuffers)
    LOAD(vkBeginCommandBuffer)
    LOAD(vkEndCommandBuffer)
    LOAD(vkQueueSubmit)
    LOAD(vkQueueWaitIdle)
    LOAD(vkCreateFence)
    LOAD(vkDestroyFence)
    LOAD(vkWaitForFences)
    LOAD(vkResetFences)
    LOAD(vkResetCommandBuffer)
    LOAD(vkAllocateMemory)
    LOAD(vkFreeMemory)
    LOAD(vkCreateBuffer)
    LOAD(vkDestroyBuffer)
    LOAD(vkGetBufferMemoryRequirements)
    LOAD(vkBindBufferMemory)
    LOAD(vkMapMemory)
    LOAD(vkUnmapMemory)
    LOAD(vkFlushMappedMemoryRanges)
    LOAD(vkInvalidateMappedMemoryRanges)
    LOAD(vkCreateShaderModule)
    LOAD(vkDestroyShaderModule)
    LOAD(vkCreateComputePipelines)
    LOAD(vkDestroyPipeline)
    LOAD(vkCreatePipelineLayout)
    LOAD(vkDestroyPipelineLayout)
    LOAD(vkCreateDescriptorSetLayout)
    LOAD(vkDestroyDescriptorSetLayout)
    LOAD(vkCreateDescriptorPool)
    LOAD(vkDestroyDescriptorPool)
    LOAD(vkAllocateDescriptorSets)
    LOAD(vkUpdateDescriptorSets)
    LOAD(vkResetDescriptorPool)
    LOAD(vkCmdBindPipeline)
    LOAD(vkCmdBindDescriptorSets)
    LOAD(vkCmdDispatch)
    LOAD(vkCmdPushConstants)
    LOAD(vkCmdPipelineBarrier)
    LOAD(vkCmdCopyBuffer)
    LOAD(vkDeviceWaitIdle)
    #undef LOAD

    vkCmdPushDescriptorSetKHR_ = (PFN_vkCmdPushDescriptorSetKHR)
        vkGetInstanceProcAddr_(g.instance, "vkCmdPushDescriptorSetKHR");
}

// ---------------------------------------------------------------------------
// Memory helpers
// ---------------------------------------------------------------------------
static uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags flags) {
    for (uint32_t i = 0; i < g.mem_props.memoryTypeCount; i++) {
        if ((type_bits & (1 << i)) && (g.mem_props.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    return UINT32_MAX;
}

static int create_buffer(VkBuffer* buf, VkDeviceMemory* mem, uint64_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_flags) {
    VkBufferCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer_(g.device, &ci, NULL, buf) != VK_SUCCESS) return GPU_ERR_OOM;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements_(g.device, *buf, &req);

    uint32_t mt = find_memory_type(req.memoryTypeBits, mem_flags);
    if (mt == UINT32_MAX) {
        vkDestroyBuffer_(g.device, *buf, NULL);
        return GPU_ERR_OOM;
    }

    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = mt;

    if (vkAllocateMemory_(g.device, &ai, NULL, mem) != VK_SUCCESS) {
        vkDestroyBuffer_(g.device, *buf, NULL);
        return GPU_ERR_OOM;
    }

    vkBindBufferMemory_(g.device, *buf, *mem, 0);
    return GPU_OK;
}

// ---------------------------------------------------------------------------
// Command buffer helpers
// ---------------------------------------------------------------------------
static void begin_cmd(void) {
    vkResetCommandBuffer_(g.cmd_buf, 0);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer_(g.cmd_buf, &bi);
}

static void submit_and_wait(void) {
    vkEndCommandBuffer_(g.cmd_buf);

    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g.cmd_buf;

    vkResetFences_(g.device, 1, &g.fence);
    vkQueueSubmit_(g.queue, 1, &si, g.fence);
    vkWaitForFences_(g.device, 1, &g.fence, VK_TRUE, UINT64_MAX);
}

static void buffer_barrier(VkBuffer buf, uint64_t size) {
    VkBufferMemoryBarrier b = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    b.buffer = buf;
    b.offset = 0;
    b.size = size;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vkCmdPipelineBarrier_(g.cmd_buf,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, NULL, 1, &b, 0, NULL);
}

// ---------------------------------------------------------------------------
// Buffer ID <-> Vulkan buffer mapping
// ---------------------------------------------------------------------------
static GpuBuf register_buffer(VkBuffer buf, VkDeviceMemory mem, uint64_t size, int host_vis) {
    if (g.buf_count >= MAX_BUFFERS) return 0;
    int idx = g.buf_count++;
    g.buffers[idx].buffer = buf;
    g.buffers[idx].memory = mem;
    g.buffers[idx].size = size;
    g.buffers[idx].host_visible = host_vis;
    return (GpuBuf)(idx + 1);
}

static BufferAlloc* get_buf(GpuBuf id) {
    if (id == 0 || id > (GpuBuf)g.buf_count) return NULL;
    return &g.buffers[id - 1];
}

// ---------------------------------------------------------------------------
// Public API: init / shutdown
// ---------------------------------------------------------------------------
int gpu_init(void) {
    if (g.initialized) return GPU_OK;

    int rc = load_vulkan();
    if (rc != GPU_OK) return rc;

    // Create instance
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "dlgo";
    app.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;

    if (vkCreateInstance_(&ici, NULL, &g.instance) != VK_SUCCESS) return GPU_ERR_INIT_FAIL;
    load_instance_funcs(g.instance);

    // Pick physical device (prefer discrete GPU)
    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices_(g.instance, &dev_count, NULL);
    if (dev_count == 0) return GPU_ERR_NO_DEVICE;

    VkPhysicalDevice* devs = (VkPhysicalDevice*)calloc(dev_count, sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices_(g.instance, &dev_count, devs);

    g.physical_device = devs[0];
    for (uint32_t i = 0; i < dev_count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties_(devs[i], &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            g.physical_device = devs[i];
            break;
        }
    }
    free(devs);

    vkGetPhysicalDeviceProperties_(g.physical_device, &g.dev_props);
    vkGetPhysicalDeviceMemoryProperties_(g.physical_device, &g.mem_props);

    // Find compute queue family
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties_(g.physical_device, &qf_count, NULL);
    VkQueueFamilyProperties* qf = (VkQueueFamilyProperties*)calloc(qf_count, sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties_(g.physical_device, &qf_count, qf);

    g.queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < qf_count; i++) {
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            g.queue_family = i;
            break;
        }
    }
    free(qf);
    if (g.queue_family == UINT32_MAX) return GPU_ERR_NO_DEVICE;

    // Create logical device
    float priority = 1.0f;
    VkDeviceQueueCreateInfo dqci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    dqci.queueFamilyIndex = g.queue_family;
    dqci.queueCount = 1;
    dqci.pQueuePriorities = &priority;

    VkPhysicalDeviceVulkan12Features vk12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    vk12.storageBuffer8BitAccess = VK_TRUE;
    vk12.uniformAndStorageBuffer8BitAccess = VK_TRUE;
    vk12.shaderInt8 = VK_TRUE;
    vk12.scalarBlockLayout = VK_TRUE;

    VkPhysicalDeviceVulkan11Features vk11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    vk11.storageBuffer16BitAccess = VK_TRUE;
    vk11.pNext = &vk12;

    VkPhysicalDeviceFeatures2 features2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features2.pNext = &vk11;

    const char* device_extensions[] = { "VK_KHR_push_descriptor" };

    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &features2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &dqci;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = device_extensions;

    if (vkCreateDevice_(g.physical_device, &dci, NULL, &g.device) != VK_SUCCESS) return GPU_ERR_INIT_FAIL;

    vkGetDeviceQueue_(g.device, g.queue_family, 0, &g.queue);

    // Command pool + buffer
    VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = g.queue_family;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool_(g.device, &cpci, NULL, &g.cmd_pool) != VK_SUCCESS) return GPU_ERR_INIT_FAIL;

    VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = g.cmd_pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    vkAllocateCommandBuffers_(g.device, &cbai, &g.cmd_buf);

    // Fence
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkCreateFence_(g.device, &fci, NULL, &g.fence);

    // Staging buffer — prefer HOST_CACHED for fast CPU reads (download path).
    // HOST_CACHED + HOST_COHERENT uses system RAM which the CPU can read at
    // memory bandwidth instead of slow uncached PCIe BAR reads.
    g.staging_size = STAGING_SIZE;
    rc = create_buffer(&g.staging_buf, &g.staging_mem, g.staging_size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    if (rc != GPU_OK) {
        rc = create_buffer(&g.staging_buf, &g.staging_mem, g.staging_size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    if (rc != GPU_OK) return rc;
    vkMapMemory_(g.device, g.staging_mem, 0, g.staging_size, 0, &g.staging_mapped);

    // Descriptor pool
    VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MAX_DESCRIPTORS_PER_POOL * 8},
    };
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = MAX_DESCRIPTORS_PER_POOL;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = pool_sizes;
    dpci.flags = 0;
    vkCreateDescriptorPool_(g.device, &dpci, NULL, &g.desc_pool);

    g.subgroup_size = 32; // NVIDIA default
    g.recording = 0;
    g.initialized = 1;

    fprintf(stderr, "[dlgo/gpu] Initialized Vulkan on %s (%.0f MB VRAM)\n",
        g.dev_props.deviceName,
        (double)gpu_vram_bytes() / (1024.0 * 1024.0));

    return GPU_OK;
}

void gpu_shutdown(void) {
    if (!g.initialized) return;
    vkDeviceWaitIdle_(g.device);

    for (int i = 0; i < g.buf_count; i++) {
        if (g.buffers[i].buffer) vkDestroyBuffer_(g.device, g.buffers[i].buffer, NULL);
        if (g.buffers[i].memory) vkFreeMemory_(g.device, g.buffers[i].memory, NULL);
    }

    for (int i = 0; i < PIPE_COUNT; i++) {
        if (g.pipelines[i]) vkDestroyPipeline_(g.device, g.pipelines[i], NULL);
        if (g.pipe_layouts[i]) vkDestroyPipelineLayout_(g.device, g.pipe_layouts[i], NULL);
        if (g.desc_layouts[i]) vkDestroyDescriptorSetLayout_(g.device, g.desc_layouts[i], NULL);
    }

    if (g.desc_pool) vkDestroyDescriptorPool_(g.device, g.desc_pool, NULL);

    if (g.staging_mapped) vkUnmapMemory_(g.device, g.staging_mem);
    if (g.staging_buf) vkDestroyBuffer_(g.device, g.staging_buf, NULL);
    if (g.staging_mem) vkFreeMemory_(g.device, g.staging_mem, NULL);

    if (g.fence) vkDestroyFence_(g.device, g.fence, NULL);
    if (g.cmd_pool) vkDestroyCommandPool_(g.device, g.cmd_pool, NULL);
    if (g.device) vkDestroyDevice_(g.device, NULL);
    if (g.instance) vkDestroyInstance_(g.instance, NULL);
    if (vk_lib) CLOSE_LIB(vk_lib);

    memset(&g, 0, sizeof(g));
}

const char* gpu_device_name(void) {
    return g.initialized ? g.dev_props.deviceName : "none";
}

uint64_t gpu_vram_bytes(void) {
    if (!g.initialized) return 0;
    uint64_t total = 0;
    for (uint32_t i = 0; i < g.mem_props.memoryHeapCount; i++) {
        if (g.mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            total += g.mem_props.memoryHeaps[i].size;
    }
    return total;
}

int gpu_is_initialized(void) { return g.initialized; }

// ---------------------------------------------------------------------------
// Buffer management
// ---------------------------------------------------------------------------
GpuBuf gpu_alloc(uint64_t size_bytes, int usage) {
    if (!g.initialized || size_bytes == 0) return 0;

    VkBuffer buf;
    VkDeviceMemory mem;
    VkBufferUsageFlags vk_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    int rc = create_buffer(&buf, &mem, size_bytes, vk_usage,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (rc != GPU_OK) return 0;

    return register_buffer(buf, mem, size_bytes, 0);
}

void gpu_free(GpuBuf id) {
    BufferAlloc* ba = get_buf(id);
    if (!ba) return;
    if (ba->buffer) vkDestroyBuffer_(g.device, ba->buffer, NULL);
    if (ba->memory) vkFreeMemory_(g.device, ba->memory, NULL);
    ba->buffer = VK_NULL_HANDLE;
    ba->memory = VK_NULL_HANDLE;
    ba->size = 0;
}

int gpu_upload(GpuBuf dst, const void* src, uint64_t size_bytes, uint64_t offset) {
    BufferAlloc* ba = get_buf(dst);
    if (!ba || !src) return GPU_ERR_DISPATCH;

    const uint8_t* p = (const uint8_t*)src;
    uint64_t remaining = size_bytes;
    uint64_t dst_off = offset;

    while (remaining > 0) {
        uint64_t chunk = remaining < g.staging_size ? remaining : g.staging_size;
        memcpy(g.staging_mapped, p, chunk);

        if (g.recording) {
            VkBufferCopy region = {0, dst_off, chunk};
            vkCmdCopyBuffer_(g.cmd_buf, g.staging_buf, ba->buffer, 1, &region);
            VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier_(g.cmd_buf,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &mb, 0, NULL, 0, NULL);
            g.dispatch_count++;
        } else {
            begin_cmd();
            VkBufferCopy region = {0, dst_off, chunk};
            vkCmdCopyBuffer_(g.cmd_buf, g.staging_buf, ba->buffer, 1, &region);
            submit_and_wait();
        }

        p += chunk;
        dst_off += chunk;
        remaining -= chunk;
    }
    return GPU_OK;
}

int gpu_download(void* dst, GpuBuf src, uint64_t size_bytes, uint64_t offset) {
    BufferAlloc* ba = get_buf(src);
    if (!ba || !dst) return GPU_ERR_DISPATCH;

    uint8_t* p = (uint8_t*)dst;
    uint64_t remaining = size_bytes;
    uint64_t src_off = offset;

    while (remaining > 0) {
        uint64_t chunk = remaining < g.staging_size ? remaining : g.staging_size;

        if (g.recording) {
            VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier_(g.cmd_buf,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 1, &mb, 0, NULL, 0, NULL);
            VkBufferCopy region = {src_off, 0, chunk};
            vkCmdCopyBuffer_(g.cmd_buf, ba->buffer, g.staging_buf, 1, &region);
            g.dispatch_count++;
            // Must end batch to get the data, then read staging
            gpu_end_batch();
            memcpy(p, g.staging_mapped, chunk);
        } else {
            begin_cmd();
            VkBufferCopy region = {src_off, 0, chunk};
            vkCmdCopyBuffer_(g.cmd_buf, ba->buffer, g.staging_buf, 1, &region);
            submit_and_wait();
            memcpy(p, g.staging_mapped, chunk);
        }

        p += chunk;
        src_off += chunk;
        remaining -= chunk;
    }
    return GPU_OK;
}

// ---------------------------------------------------------------------------
// Shader / pipeline creation
// ---------------------------------------------------------------------------
static VkShaderModule create_shader(const uint32_t* code, size_t code_size) {
    VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = code_size;
    ci.pCode = code;

    VkShaderModule mod;
    if (vkCreateShaderModule_(g.device, &ci, NULL, &mod) != VK_SUCCESS) return VK_NULL_HANDLE;
    return mod;
}

static int create_compute_pipeline(PipelineID id, const uint32_t* spirv, size_t spirv_size,
                                   int num_buffers, int push_const_size) {
    VkShaderModule mod = create_shader(spirv, spirv_size);
    if (mod == VK_NULL_HANDLE) return GPU_ERR_SHADER;

    // Descriptor set layout: N storage buffers
    VkDescriptorSetLayoutBinding* bindings = (VkDescriptorSetLayoutBinding*)calloc(
        num_buffers, sizeof(VkDescriptorSetLayoutBinding));
    for (int i = 0; i < num_buffers; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dslci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = num_buffers;
    dslci.pBindings = bindings;
    if (vkCmdPushDescriptorSetKHR_) {
        dslci.flags = 0x00000001; /* VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR */
    }
    vkCreateDescriptorSetLayout_(g.device, &dslci, NULL, &g.desc_layouts[id]);
    free(bindings);

    // Pipeline layout with push constants
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, push_const_size > 0 ? push_const_size : 4};

    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &g.desc_layouts[id];
    if (push_const_size > 0) {
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
    }
    vkCreatePipelineLayout_(g.device, &plci, NULL, &g.pipe_layouts[id]);

    // Compute pipeline
    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod;
    cpci.stage.pName = "main";
    cpci.layout = g.pipe_layouts[id];

    VkResult r = vkCreateComputePipelines_(g.device, VK_NULL_HANDLE, 1, &cpci, NULL, &g.pipelines[id]);
    vkDestroyShaderModule_(g.device, mod, NULL);

    return r == VK_SUCCESS ? GPU_OK : GPU_ERR_SHADER;
}

// ---------------------------------------------------------------------------
// Dispatch helper: bind pipeline, descriptors, push constants, dispatch
// ---------------------------------------------------------------------------
typedef struct {
    PipelineID pipe;
    GpuBuf bufs[8];
    int num_bufs;
    void* push_data;
    int push_size;
    uint32_t groups_x, groups_y, groups_z;
} DispatchParams;

static int dispatch_compute(DispatchParams* p) {
    if (!g.pipelines[p->pipe]) return GPU_ERR_SHADER;

    // Prepare descriptor writes
    VkDescriptorBufferInfo buf_infos[8];
    VkWriteDescriptorSet writes[8];
    for (int i = 0; i < p->num_bufs; i++) {
        BufferAlloc* ba = get_buf(p->bufs[i]);
        if (!ba) return GPU_ERR_DISPATCH;
        buf_infos[i].buffer = ba->buffer;
        buf_infos[i].offset = 0;
        buf_infos[i].range = VK_WHOLE_SIZE;

        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].pNext = NULL;
        writes[i].dstSet = VK_NULL_HANDLE;
        writes[i].dstBinding = i;
        writes[i].dstArrayElement = 0;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buf_infos[i];
        writes[i].pImageInfo = NULL;
        writes[i].pTexelBufferView = NULL;
    }

    if (g.recording) {
        if (g.need_barrier) {
            VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(g.cmd_buf,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &mb, 0, NULL, 0, NULL);
            g.need_barrier = 0;
        }
        vkCmdBindPipeline_(g.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipelines[p->pipe]);
        if (vkCmdPushDescriptorSetKHR_) {
            vkCmdPushDescriptorSetKHR_(g.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                g.pipe_layouts[p->pipe], 0, p->num_bufs, writes);
        } else {
            VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            dsai.descriptorPool = g.desc_pool;
            dsai.descriptorSetCount = 1;
            dsai.pSetLayouts = &g.desc_layouts[p->pipe];
            VkDescriptorSet ds;
            if (vkAllocateDescriptorSets_(g.device, &dsai, &ds) != VK_SUCCESS) return GPU_ERR_DISPATCH;
            for (int i = 0; i < p->num_bufs; i++) writes[i].dstSet = ds;
            vkUpdateDescriptorSets_(g.device, p->num_bufs, writes, 0, NULL);
            vkCmdBindDescriptorSets_(g.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                g.pipe_layouts[p->pipe], 0, 1, &ds, 0, NULL);
        }
        if (p->push_data && p->push_size > 0) {
            vkCmdPushConstants_(g.cmd_buf, g.pipe_layouts[p->pipe],
                VK_SHADER_STAGE_COMPUTE_BIT, 0, p->push_size, p->push_data);
        }
        vkCmdDispatch_(g.cmd_buf, p->groups_x, p->groups_y, p->groups_z);
        g.dispatch_count++;
    } else {
        begin_cmd();
        vkCmdBindPipeline_(g.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipelines[p->pipe]);
        if (vkCmdPushDescriptorSetKHR_) {
            vkCmdPushDescriptorSetKHR_(g.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                g.pipe_layouts[p->pipe], 0, p->num_bufs, writes);
        } else {
            VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            dsai.descriptorPool = g.desc_pool;
            dsai.descriptorSetCount = 1;
            dsai.pSetLayouts = &g.desc_layouts[p->pipe];
            VkDescriptorSet ds;
            if (vkAllocateDescriptorSets_(g.device, &dsai, &ds) != VK_SUCCESS) return GPU_ERR_DISPATCH;
            for (int i = 0; i < p->num_bufs; i++) writes[i].dstSet = ds;
            vkUpdateDescriptorSets_(g.device, p->num_bufs, writes, 0, NULL);
            vkCmdBindDescriptorSets_(g.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                g.pipe_layouts[p->pipe], 0, 1, &ds, 0, NULL);
        }
        if (p->push_data && p->push_size > 0) {
            vkCmdPushConstants_(g.cmd_buf, g.pipe_layouts[p->pipe],
                VK_SHADER_STAGE_COMPUTE_BIT, 0, p->push_size, p->push_data);
        }
        vkCmdDispatch_(g.cmd_buf, p->groups_x, p->groups_y, p->groups_z);
        submit_and_wait();
        vkResetDescriptorPool_(g.device, g.desc_pool, 0);
    }

    return GPU_OK;
}

// ---------------------------------------------------------------------------
// Compute shader SPIR-V will be loaded from embedded data
// We use a separate file for the compiled shaders
// ---------------------------------------------------------------------------
#include "shaders_spirv.h"

int gpu_load_pipelines(void) {
    int total = sizeof(shader_registry) / sizeof(shader_registry[0]);
    if (total > PIPE_COUNT) total = PIPE_COUNT;

    for (int i = 0; i < total; i++) {
        const ShaderInfo* si = &shader_registry[i];
        int rc = create_compute_pipeline(i, si->code, si->code_size,
                                         si->num_buffers, si->push_const_size);
        if (rc != GPU_OK) {
            fprintf(stderr, "[dlgo/gpu] Failed to create pipeline %s (id=%d): %d\n",
                    si->name, i, rc);
        }
    }
    g.pipelines_ready = 1;
    fprintf(stderr, "[dlgo/gpu] Loaded %d compute pipelines\n", total);
    return GPU_OK;
}

void gpu_sync(void) {
    if (g.initialized) vkDeviceWaitIdle_(g.device);
}

void gpu_begin_batch(void) {
    if (!g.initialized || g.recording) return;
    begin_cmd();
    g.recording = 1;
    g.dispatch_count = 0;
}

void gpu_end_batch(void) {
    if (!g.initialized || !g.recording) return;
    g.recording = 0;
    if (g.dispatch_count > 0) {
        submit_and_wait();
    }
    vkResetDescriptorPool_(g.device, g.desc_pool, 0);
    g.dispatch_count = 0;
    g.need_barrier = 0;
}

void gpu_barrier(void) {
    if (g.recording && g.dispatch_count > 0) {
        g.need_barrier = 1;
    }
}

// ---------------------------------------------------------------------------
// Operations (implemented after shaders are compiled)
// ---------------------------------------------------------------------------
int gpu_matvec(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
               int rows, int cols, int qtype) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    PipelineID pipe;
    switch (qtype) {
        case QTYPE_F32:  pipe = PIPE_MATVEC_F32; break;
        case QTYPE_F16:  pipe = PIPE_MATVEC_F16; break;
        case QTYPE_Q4_0: pipe = PIPE_MATVEC_Q4_0; break;
        case QTYPE_Q8_0: pipe = PIPE_MATVEC_Q8_0; break;
        case QTYPE_Q4_K: pipe = PIPE_MATVEC_Q4_K; break;
        case QTYPE_Q5_0: pipe = PIPE_MATVEC_Q5_0; break;
        case QTYPE_Q6_K: pipe = PIPE_MATVEC_Q6_K; break;
        default: return GPU_ERR_DISPATCH;
    }

    struct { int rows; int cols; } pc = {rows, cols};
    DispatchParams dp = {0};
    dp.pipe = pipe;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = weights_buf;
    dp.bufs[2] = x_buf;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (rows + 3) / 4;
    dp.groups_y = 1;
    dp.groups_z = 1;

    return dispatch_compute(&dp);
}

int gpu_rmsnorm(GpuBuf out_buf, GpuBuf x_buf, GpuBuf weight_buf, int n, float eps) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n; float eps; } pc = {n, eps};
    DispatchParams dp = {0};
    dp.pipe = PIPE_RMSNORM;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = x_buf;
    dp.bufs[2] = weight_buf;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = 1;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_rmsnorm_heads(GpuBuf data_buf, GpuBuf weight_buf, int num_heads, int head_dim, float eps) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int head_dim; float eps; } pc = {head_dim, eps};
    DispatchParams dp = {0};
    dp.pipe = PIPE_RMSNORM_HEADS;
    dp.bufs[0] = data_buf;
    dp.bufs[1] = weight_buf;
    dp.num_bufs = 2;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = num_heads;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_softmax(GpuBuf buf, int n) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n; } pc = {n};
    DispatchParams dp = {0};
    dp.pipe = PIPE_SOFTMAX;
    dp.bufs[0] = buf;
    dp.num_bufs = 1;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = 1;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_rope(GpuBuf q_buf, GpuBuf k_buf, int num_heads, int num_kv_heads,
             int head_dim, int pos, float freq_base, int neox) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int num_heads; int num_kv_heads; int head_dim; int pos; float freq_base; int neox; } pc =
        {num_heads, num_kv_heads, head_dim, pos, freq_base, neox};
    DispatchParams dp = {0};
    dp.pipe = PIPE_ROPE;
    dp.bufs[0] = q_buf;
    dp.bufs[1] = k_buf;
    dp.num_bufs = 2;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (num_heads > num_kv_heads ? num_heads : num_kv_heads);
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_swiglu(GpuBuf out_buf, GpuBuf gate_buf, GpuBuf up_buf, int n) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n; } pc = {n};
    DispatchParams dp = {0};
    dp.pipe = PIPE_SWIGLU;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = gate_buf;
    dp.bufs[2] = up_buf;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (n + 255) / 256;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_geglu(GpuBuf out_buf, GpuBuf gate_buf, GpuBuf up_buf, int n) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n; } pc = {n};
    DispatchParams dp = {0};
    dp.pipe = PIPE_GEGLU;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = gate_buf;
    dp.bufs[2] = up_buf;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (n + 255) / 256;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_gelu(GpuBuf buf, int n) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n; } pc = {n};
    DispatchParams dp = {0};
    dp.pipe = PIPE_GELU;
    dp.bufs[0] = buf;
    dp.num_bufs = 1;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (n + 255) / 256;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_add(GpuBuf out_buf, GpuBuf a_buf, GpuBuf b_buf, int n) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n; } pc = {n};
    DispatchParams dp = {0};
    dp.pipe = PIPE_ADD;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = a_buf;
    dp.bufs[2] = b_buf;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (n + 255) / 256;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_add_bias(GpuBuf buf, GpuBuf bias_buf, int n) {
    return gpu_add(buf, buf, bias_buf, n);
}

int gpu_scale(GpuBuf buf, float s, int n) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n; float s; } pc = {n, s};
    DispatchParams dp = {0};
    dp.pipe = PIPE_SCALE;
    dp.bufs[0] = buf;
    dp.num_bufs = 1;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (n + 255) / 256;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_add_rmsnorm(GpuBuf norm_out, GpuBuf sum_out,
                    GpuBuf a_buf, GpuBuf b_buf, GpuBuf weight_buf,
                    int n, float eps) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n; float eps; } pc = {n, eps};
    DispatchParams dp = {0};
    dp.pipe = PIPE_ADD_RMSNORM;
    dp.bufs[0] = norm_out;
    dp.bufs[1] = sum_out;
    dp.bufs[2] = a_buf;
    dp.bufs[3] = b_buf;
    dp.bufs[4] = weight_buf;
    dp.num_bufs = 5;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = 1;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_copy_f32(GpuBuf dst, GpuBuf src, int n) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    BufferAlloc* s = get_buf(src);
    BufferAlloc* d = get_buf(dst);
    if (!s || !d) return GPU_ERR_DISPATCH;

    begin_cmd();
    VkBufferCopy region = {0, 0, (uint64_t)n * 4};
    vkCmdCopyBuffer_(g.cmd_buf, s->buffer, d->buffer, 1, &region);
    submit_and_wait();
    return GPU_OK;
}

int gpu_batch_matvec(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                     int rows, int cols, int npos, int qtype) {
    // For batch, dispatch one matvec per position (simple initial impl)
    // TODO: fused batch kernel
    for (int p = 0; p < npos; p++) {
        // Each position's input/output is offset by cols/rows floats
        // This requires the caller to handle offsets or we need sub-buffer views
        // For now, use single matvec dispatch
    }
    return gpu_matvec(out_buf, weights_buf, x_buf, rows, cols, qtype);
}

int gpu_dequantize(GpuBuf out_f32_buf, GpuBuf quant_buf, int n, int qtype) {
    // TODO: implement dequantize shaders
    return GPU_ERR_SHADER;
}

int gpu_attention(GpuBuf out_buf, GpuBuf q_buf, GpuBuf k_cache_buf, GpuBuf v_cache_buf,
                  int num_heads, int num_kv_heads, int head_dim, int kv_dim,
                  int seq_len, float scale) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int num_heads; int num_kv_heads; int head_dim; int kv_dim; int seq_len; float scale; } pc =
        {num_heads, num_kv_heads, head_dim, kv_dim, seq_len, scale};
    DispatchParams dp = {0};
    dp.pipe = PIPE_ATTENTION;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = q_buf;
    dp.bufs[2] = k_cache_buf;
    dp.bufs[3] = v_cache_buf;
    dp.num_bufs = 4;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = num_heads;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_kv_store(GpuBuf k_cache_buf, GpuBuf v_cache_buf,
                 GpuBuf k_buf, GpuBuf v_buf,
                 int pos, int kv_dim) {
    BufferAlloc* kc = get_buf(k_cache_buf);
    BufferAlloc* vc = get_buf(v_cache_buf);
    BufferAlloc* kb = get_buf(k_buf);
    BufferAlloc* vb = get_buf(v_buf);
    if (!kc || !vc || !kb || !vb) return GPU_ERR_DISPATCH;

    if (g.recording) {
        // Insert barrier before copy
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier_(g.cmd_buf,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 1, &mb, 0, NULL, 0, NULL);

        VkBufferCopy kr = {0, (uint64_t)pos * kv_dim * 4, (uint64_t)kv_dim * 4};
        vkCmdCopyBuffer_(g.cmd_buf, kb->buffer, kc->buffer, 1, &kr);
        VkBufferCopy vr = {0, (uint64_t)pos * kv_dim * 4, (uint64_t)kv_dim * 4};
        vkCmdCopyBuffer_(g.cmd_buf, vb->buffer, vc->buffer, 1, &vr);

        // Barrier after copy for subsequent shader reads
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier_(g.cmd_buf,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &mb, 0, NULL, 0, NULL);
        g.dispatch_count++;
    } else {
        begin_cmd();
        VkBufferCopy kr = {0, (uint64_t)pos * kv_dim * 4, (uint64_t)kv_dim * 4};
        vkCmdCopyBuffer_(g.cmd_buf, kb->buffer, kc->buffer, 1, &kr);
        VkBufferCopy vr = {0, (uint64_t)pos * kv_dim * 4, (uint64_t)kv_dim * 4};
        vkCmdCopyBuffer_(g.cmd_buf, vb->buffer, vc->buffer, 1, &vr);
        submit_and_wait();
    }
    return GPU_OK;
}
