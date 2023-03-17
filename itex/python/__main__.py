'''Main file for ITEX'''
if __name__ == "__main__":
  import sys
  import runpy
  import intel_extension_for_tensorflow as itex
  itex.experimental_ops_override()
  sys.argv = sys.argv[1:]
  runpy.run_path(sys.argv[0], init_globals=globals())
