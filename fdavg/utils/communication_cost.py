

def format_bytes(size):
    # List of suffixes for byte units
    power = 1024
    n = 0
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']

    # Loop to divide the size until it's less than 1024
    while size >= power and n < len(units) - 1:
        size /= power
        n += 1

    # Format the size with 5 decimal places
    return f"{size:.3f} {units[n]}"

def step_comm_cost_str(num_clients, num_nn_weights, sync_needed, method, sketch_width=250, sketch_depth=5):

    sync_bytes = 2 * num_clients * 4 * num_nn_weights
    monitor_bytes = 0

    if method == 'gm':
        monitor_bytes = 0.125  # one bit
    elif method == 'sketch':
        monitor_bytes = 4 * sketch_width * sketch_depth
    elif method == 'linear':
        monitor_bytes = 4 * 2
    elif method == 'naive':
        monitor_bytes = 4 * 1
    elif method == 'synchronous':
        monitor_bytes = 0

    monitor_bytes = monitor_bytes * num_clients

    if sync_needed:
        total_bytes_transmitted = sync_bytes + monitor_bytes
    else:
        total_bytes_transmitted = monitor_bytes

    return format_bytes(total_bytes_transmitted)

