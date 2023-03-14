This folder contains major src files for XPUAutoShard.

# File Structure
Every `*.td` will corresponds to `*.h` and `*.cpp`, where contains its implementation as well as additional functions.

- dialect.td : Defines HS dialect, which holds `hs.ops` and `hs.attributes`.
- ops.td : Defines `hs.ops`.
- attributes.td : Defines `hs.attribute`.