
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
    
    
def print_finish_testing_info(fda_name, num_clients, batch_size, num_epochs,
                              num_steps_until_rtc_check, theta, start_time, end_time):
    print()
    print(f"------------ Finished Testing : ------------")
    print(f"FDA name : {fda_name}")
    print(f"Num Clients : {num_clients}")
    print(f"Batch size : {batch_size}")
    print(f"Num Epochs : {num_epochs}")
    print(f"Number of steps until we check RTC : {num_steps_until_rtc_check}")
    print(f"Theta : {theta}")
    print(f"Total simulation time: {end_time-start_time} sec")
    print("-----------------------------------------")
    print()