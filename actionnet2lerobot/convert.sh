export HDF5_USE_FILE_LOCKING=FALSE
export RAY_DEDUP_LOGS=0
uv run python actionnet_h5.py \
    --src-path /path/to/action_net \
    --output-path /path/to/local \
    --executor local \
    --episodes-per-task 100
