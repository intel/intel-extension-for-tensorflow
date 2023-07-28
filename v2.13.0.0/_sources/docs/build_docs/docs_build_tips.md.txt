# Online Documentation Build Guide


## Introduction

This document shows how scripts are used to build the project documentation.

The scripts and related files are saved in the `docs/build_docs` directory.


## Update `latest` Version

Generating and publishing the `latest` documentation is triggered by merging a PR to the public GitHub repo's main branch. A GitHub `Action` executes the document-building scripts using content from the main branch and updates the `latest` [published version](https://intel.github.io/intel-extension-for-tensorflow/latest/get_started.html).

If the PR doesn't impact documentation (for example, it only contains code changes), the online documents won't be updated.

## Create Release Version

When releasing a new product version, a git tag must be added to main branch. The release version will be the same as the `tag` name.

Generating the publishing the documentation for a release must be triggered manually, and the resulting HTML content committed to the `gh-pages` branch manually. (The github.io site will be updated automatically when this `gh-pages` branch is updated.)

Note: Only GitHub users with write permission to the repo are able to update the `gh-pages` branch content.

```
git clone https://github.com/intel/intel-extension-for-tensorflow.git
git checkout <release tag name>
cd intel-extension-for-tensorflow/docs/build_docs
./build.sh version

cd ../../build_tmp/gh-pages
git add .
git commit -m "add version <release tag name>"
git push
```

## Build to Local Test

Before submitting a PR that modifies documentation, you should test
your changes by building the documentation locally.

```
git clone https://github.com/intel/intel-extension-for-tensorflow.git
# make your documentation changes
cd intel-extension-for-tensorflow/docs/build_docs

./build.sh local
cd ../../build_tmp/draft/latest
python3 -m http.server 9000
```

Use your web browser (e.g., Chrome) to open `127.0.0.1:9000` or `localhost:9000` to check your locally generated HTML content.

Note, the `version.html` version picker is not functional for locally generated documentation.

If you want to test version switching functionality using `version.html`, build for `version`:

```
git clone https://github.com/intel/intel-extension-for-tensorflow.git
# make your documentation changes
cd intel-extension-for-tensorflow/docs/build_docs

./build.sh version
cd ../../build_tmp/gh-pages/tag??
python3 -m http.server 9000

```
