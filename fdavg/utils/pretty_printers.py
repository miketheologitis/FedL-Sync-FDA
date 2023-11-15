
def print_current_test_info(ds_name, bias, fda_name, num_clients, batch_size, num_epochs,
                            num_steps_until_rtc_check, nn_name, bench_test, theta, aggr_scheme, **kwargs):
    print()
    print(f"------------ Current Test : ------------")
    print(f"Dataset name : {ds_name}")
    print(f"Bias : {bias}")
    print(f"Bench test : {bench_test}")
    print(f"NN name : {nn_name}")
    print(f"FDA name : {fda_name}")
    print(f"Num Clients : {num_clients}")
    print(f"Batch size : {batch_size}")
    print(f"Num Epochs : {num_epochs}")
    print(f"Number of steps until we check RTC : {num_steps_until_rtc_check}")
    print(f"Theta : {theta}")
    print(f"Aggr. Scheme : {aggr_scheme}")
    print("-----------------------------------------")
    print()
    
    
def print_finish_testing_info(start_time, end_time, ds_name, bias, fda_name, num_clients, batch_size, num_epochs,
                              num_steps_until_rtc_check, nn_name, bench_test, theta, aggr_scheme, **kwargs):
    print()
    print(f"------------ Finished Testing : ------------")
    print(f"Dataset name : {ds_name}")
    print(f"Bias : {bias}")
    print(f"Bench test : {bench_test}")
    print(f"NN name : {nn_name}")
    print(f"FDA name : {fda_name}")
    print(f"Num Clients : {num_clients}")
    print(f"Batch size : {batch_size}")
    print(f"Num Epochs : {num_epochs}")
    print(f"Number of steps until we check RTC : {num_steps_until_rtc_check}")
    print(f"Theta : {theta}")
    print(f"Aggr. Scheme : {aggr_scheme}")
    print(f"Total simulation time: {end_time-start_time} sec")
    print("-----------------------------------------")
    print()
