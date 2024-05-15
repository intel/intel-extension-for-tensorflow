#Macros for building SYCL code.
def if_sycl(if_true, if_false = []):
    """Shorthand for selects()'ing on whether we're building with SYCL.

    Returns a select statement which evaluates to if_true if we're building
    with SYCL enabled.  Otherwise, the select statement evaluates to if_false.

    """
    return select({
        "@intel_extension_for_tensorflow//itex:if_using_sycl": if_true,
        "//conditions:default": if_false,
    })
    
def if_xetla(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with xetla.

    Returns a select statement which evaluates to if_true if we're building
    with xetla header.  Otherwise, the select statement evaluates to if_false.

    """
    return select({
        "@local_config_sycl//sycl:using_xetla": if_true,
        "//conditions:default": if_false,
    })     
    

def sycl_is_configured():
    """Returns true if SYCL was enabled during the configure process."""
    return %{sycl_is_configured}

def sycl_build_is_configured():
    """Returns true if SYCL compiler was enabled during the configure process."""
    return %{sycl_build_is_configured}

def if_sycl_is_configured(x):
    """Tests if the SYCL was enabled during the configure process.

    Unlike if_sycl(), this does not require that we are building with
    --config=sycl. Used to allow non-SYCL code to depend on SYCL libraries.
    """
    if sycl_is_configured():
      return select({"//conditions:default": x})
    return select({"//conditions:default": []})

def if_sycl_build_is_configured(x, y):
    if sycl_build_is_configured():
      return x
    return y
