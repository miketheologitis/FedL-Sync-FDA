
def print_current_test_info(fda_name, num_clients, batch_size, num_epochs, num_steps_until_rtc_check, theta=0.):
    print()
    print(f"------------ Current Test : ------------")
    print(f"FDA name : {fda_name}")
    print(f"Num Clients : {num_clients}")
    print(f"Batch size : {batch_size}")
    print(f"Num Epochs : {num_epochs}")
    print(f"Number of steps until we check RTC : {num_steps_until_rtc_check}")
    print(f"Theta : {theta}")
    print("-----------------------------------------")
    print()
    
    
def print_finish_testing_info(naive_test, linear_test, sketch_test, synchronous_test, num_clients, batch_size, num_epochs, 
                              num_steps_until_rtc_check, theta, start_time, end_time):
    conditions = [("naive", naive_test), ("linear", linear_test), ("sketch", sketch_test), ("synchronous", synchronous_test)]
    tests_performed = [name for name, condition in conditions if condition]
    print()
    print(f"------------ Finished Testing : ------------")
    print(f"FDA methods : {tests_performed}")
    print(f"Num Clients : {num_clients}")
    print(f"Batch size : {batch_size}")
    print(f"Num Epochs : {num_epochs}")
    print(f"Number of steps until we check RTC : {num_steps_until_rtc_check}")
    print(f"Theta : {theta}")
    print(f"Total simulation time: {end_time-start_time} sec")
    print("-----------------------------------------")
    print()