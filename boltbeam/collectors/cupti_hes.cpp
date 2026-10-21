#include <cuda.h>
#include <cupti.h>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <mutex>
#include <time.h>
static FILE* output;
static std::atomic<size_t> count{0},dropped{0};
static std::mutex lock;
static void check(CUptiResult r) {if(r!=CUPTI_SUCCESS){const char* s;cuptiGetResultString(r,&s);fprintf(stderr,"CUPTI: %s\n",s);_Exit(91);}}
static void CUPTIAPI request(uint8_t** p,size_t* n,size_t* max){*n=64*1024*1024;*max=0;if(posix_memalign((void**)p,8,*n))_Exit(92);}
static void CUPTIAPI complete(CUcontext ctx,uint32_t stream,uint8_t* p,size_t,size_t valid){
 std::lock_guard<std::mutex> guard(lock);CUpti_Activity* a=nullptr;
 for(;;){auto r=cuptiActivityGetNextRecord(p,valid,&a);if(r==CUPTI_ERROR_MAX_LIMIT_REACHED)break;check(r);
  if(a->kind!=CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)continue;
  auto k=(CUpti_ActivityKernel10*)a;
  if(!k->start||k->end<k->start)_Exit(93);
  fprintf(output,"%llu,%llu,%u,%d,%d,%d,%d,%d,%d,\"%s\"\n",(unsigned long long)k->start,(unsigned long long)(k->end-k->start),k->streamId,k->gridX,k->gridY,k->gridZ,k->blockX,k->blockY,k->blockZ,k->name);
  ++count;
 }
 size_t n;check(cuptiActivityGetNumDroppedRecords(ctx,stream,&n));dropped+=n;free(p);
}
static void finish(){check(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));if(fclose(output)||dropped||!count)_Exit(94);fprintf(stderr,"BOLTBEAM_HES kernels=%zu dropped=%zu\n",count.load(),dropped.load());}
extern "C" int InitializeInjection(){
 const char* path=getenv("BOLTBEAM_HES_OUTPUT");if(!path)_Exit(95);output=fopen(path,"wx");if(!output)_Exit(96);
 fputs("Start (ns),Duration (ns),Strm,GrdX,GrdY,GrdZ,BlkX,BlkY,BlkZ,Name\n",output);
 const char* clock_path=getenv("BOLTBEAM_HES_CLOCK");if(!clock_path)_Exit(95);
 timespec before,after;uint64_t timestamp;uint32_t version;check(cuptiGetVersion(&version));
 if(clock_gettime(CLOCK_MONOTONIC,&before))_Exit(98);
 check(cuptiGetTimestamp(&timestamp));
 if(clock_gettime(CLOCK_MONOTONIC,&after))_Exit(98);
 uint64_t begin=uint64_t(before.tv_sec)*1000000000+before.tv_nsec,end=uint64_t(after.tv_sec)*1000000000+after.tv_nsec;
 FILE* anchor=fopen(clock_path,"wx");if(!anchor)_Exit(96);
 fprintf(anchor,"{\"cupti_ns\":%llu,\"monotonic_ns\":%llu,\"uncertainty_ns\":%llu,\"cupti_version\":%u}\n",
         (unsigned long long)timestamp,(unsigned long long)(begin+(end-begin)/2),(unsigned long long)(end-begin),version);
 if(fclose(anchor))_Exit(94);
 check(cuptiActivityEnableHWTrace(1));check(cuptiActivityRegisterCallbacks(request,complete));check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));atexit(finish);return 1;
}

struct Startup {Startup(){if(cuInit(0)!=CUDA_SUCCESS)_Exit(97);InitializeInjection();}};
static Startup startup;
