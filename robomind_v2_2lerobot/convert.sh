export HDF5_USE_FILE_LOCKING=FALSE
export RAY_DEDUP_LOGS=0

# RoboMIND ships Franka as five repos that all start at data/franka, so several
# roots can be given at once and they land in one embodiment.
python robomind_v2_h5.py \
    --src-paths /path/to/robomind_2_0/franka-part-1 \
                /path/to/robomind_2_0/franka-part-2 \
                /path/to/robomind_2_0/franka-part-3 \
                /path/to/robomind_2_0/franka-part-4 \
                /path/to/robomind_2_0/franka-part-5 \
    --output-path /path/to/local \
    --cpus-per-task 2
