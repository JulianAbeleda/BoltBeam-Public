# Direct KFD bridge boundary

`boltbeam.direct_kfd_bridge` is the CPU/static seam for the future producer-side
KFD launch path. It validates a `boltbeam.kfd_launch_request.v1` request,
requires exact code-object and digest identities, injects allocations through a
backend, submits once, waits with a bounded timeout, emits a launch observation,
and releases allocations in reverse order.

The only backend currently implemented is `FakeKFDBackend`. It simulates
success, timeout, ABI rejection, digest rejection, allocation failure, and
submit failure. No default backend exists, and the module never opens
`/dev/kfd`, calls an ioctl, maps a GPU address, or executes a code object.

The GPU gate is a separate positive-control review. It must supply a reviewed
backend and prove that the tinygrad child-process observer's code-object digest,
geometry, and completion record join exactly to the independently captured
resource/counter artifact. A fake-backend success is not a GPU observation.
