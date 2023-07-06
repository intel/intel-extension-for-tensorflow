# Documentation Build Tips



1. Prepare the Python virtual environment using conda:
    ```
    conda create -n build_itex_doc python=3.6 -y
    conda activate build_itex_doc
    ```

2. Install documentation build dependencies:
    ```
    python -m pip install -r docs/docs_build/sphinx-requirements.txt
    ```

3. Build documentation using these commands:
    ```
    make html
    ```
    The above steps compiles an HTML websites from the markdown content in the repo into the build/html directory. You need to open the html file to check it after compilation.

4. There is also an HTTP service that can be viewed in the browser via the ip address (127.0.0.1):
    ```
    sphinx-autobuild source build/html
    ```
    or run by http.server for remote access:
    ```
    cd build/html/
    python3 -m http.server 9000
    ```
