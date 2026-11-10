// Every module the CUDA driver loads, written as the driver holds it: BOLTBEAM_MODULES/<pid>-<n>.cubin and a row in
// modules.csv. The driver loads this library itself (CUDA_INJECTION64_PATH) at cuInit and calls InitializeInjection.
// Only the CUPTI callback API is used: no counter, event or profiler session is opened. Any failure ends the process.
#include <cupti.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

static unsigned count;

static void stop(const char* what, const char* why) { fprintf(stderr, "BOLTBEAM_MODULES failed %s: %s\n", what, why); _exit(91); }

static void CUPTIAPI loaded(void*, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void* data) {
  if (domain != CUPTI_CB_DOMAIN_RESOURCE || id != CUPTI_CBID_RESOURCE_MODULE_LOADED) return;
  auto* resource = static_cast<const CUpti_ResourceData*>(data);
  auto* module = static_cast<const CUpti_ModuleResourceData*>(resource->resourceDescriptor);
  char path[4096], row[256];
  unsigned n = __sync_fetch_and_add(&count, 1);
  snprintf(path, sizeof path, "%s/%d-%u.cubin", getenv("BOLTBEAM_MODULES"), getpid(), n);
  FILE* f = fopen(path, "wbx");
  if (!f || fwrite(module->pCubin, 1, module->cubinSize, f) != module->cubinSize || fclose(f)) stop("writing", path);
  int length = snprintf(row, sizeof row, "%d-%u.cubin,%zu,%p,%u,%d\n", getpid(), n, module->cubinSize,
                        static_cast<void*>(resource->context), module->moduleId, getpid());
  snprintf(path, sizeof path, "%s/modules.csv", getenv("BOLTBEAM_MODULES"));
  f = fopen(path, "a");  // one write per row: rows of several threads or processes do not interleave
  if (!f || fwrite(row, 1, length, f) != static_cast<size_t>(length) || fclose(f)) stop("writing", path);
}

extern "C" int InitializeInjection(void) {
  CUpti_SubscriberHandle subscriber;
  const char* why = "?";
  if (!getenv("BOLTBEAM_MODULES")) stop("starting", "BOLTBEAM_MODULES is not set");
  CUptiResult r = cuptiSubscribe(&subscriber, loaded, nullptr);
  if (r != CUPTI_SUCCESS) { cuptiGetResultString(r, &why); stop("cuptiSubscribe", why); }
  r = cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_LOADED);
  if (r != CUPTI_SUCCESS) { cuptiGetResultString(r, &why); stop("cuptiEnableCallback", why); }
  fprintf(stderr, "BOLTBEAM_MODULES subscribed pid=%d\n", getpid());
  return 1;
}
