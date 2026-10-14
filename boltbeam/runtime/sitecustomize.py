"""Install BoltBeam's build bound in engine Python processes and their workers."""
import os

if os.environ.get("BOLTBEAM_BUILD_LIMIT") == "1":
  import build_jobs
  build_jobs.install_ninja_limit()
