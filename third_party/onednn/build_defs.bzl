def if_cpu_onednn(if_true, if_false = []):
    """Returns `if_true` if oneDNN on CPU is used.

    Shorthand for select()'ing on whether we're building with 'config=mkl'.

    Returns a select statement which evaluates to if_true if we're building
    with oneDNN on CPU. Otherwise, the select statement evaluates to if_false.

    """
    return select({
        "//third_party/onednn:build_with_onednn": if_true,
        "//conditions:default": if_false,
    })

def onednn_deps():
    """Shorthand for select() to pull in the correct set of oneDNN library deps.

    Returns:
      a select evaluating to a list of library dependencies, suitable for
      inclusion in the deps attribute of rules.
    """
    return select({
        str(Label("//third_party/build_option/dpcpp:build_with_dpcpp")): ["@onednn_gpu"],
        "//conditions:default": ["@onednn_cpu"],
    })
