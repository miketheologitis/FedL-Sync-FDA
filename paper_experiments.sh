python -m fdavg.utils.create_combinations --fda linear sketch --nn LeNet-5 --ds_name MNIST --b 32 --e 300 --th 0.5 1 1.5 2 3 5 7 --num_clients 5 10 15 20 25 30 35 40 45 50 55 60 --comb_file_id 0
python -m fdavg.utils.create_combinations --fda synchronous --nn LeNet-5 --ds_name MNIST --b 32 --e 300 --th 0 --num_clients 5 10 15 20 25 30 35 40 45 50 55 60 --comb_file_id 0 --append_to
python -m fdavg.utils.create_combinations --fda synchronous FedAdam --nn LeNet-5 --ds_name MNIST --b 32 --e 1000 --th 0 --num_clients 5 10 15 20 25 30 35 40 45 50 55 60 --comb_file_id 0 --append_to
python -m fdavg.utils.create_combinations --fda linear sketch --nn LeNet-5 --ds_name MNIST --b 32 --e 300 --th 0.5 1 1.5 2 3 5 7 --num_clients 5 10 15 20 25 30 35 40 45 50 55 60 --bias 0.6 -1 --comb_file_id 0 --append_to
python -m fdavg.utils.create_combinations --fda synchronous --nn LeNet-5 --ds_name MNIST --b 32 --e 300 --th 0 --num_clients 5 10 15 20 25 30 35 40 45 50 55 60 --bias 0.6 -1 --comb_file_id 0 --append_to
python -m fdavg.utils.create_combinations --fda synchronous FedAdam --nn LeNet-5 --ds_name MNIST --b 32 --e 1000 --th 0 --num_clients 5 10 15 20 25 30 35 40 45 50 55 60 --bias 0.6 -1 --comb_file_id 0 --append_to

python -m fdavg.utils.create_combinations --fda linear sketch --nn AdvancedCNN --ds_name MNIST --b 32 --e 300 --th 20 25 30 50 75 90 100 --num_clients 5 10 15 20 25 30 35 40 45 50 55 60 --comb_file_id 0 --append_to
python -m fdavg.utils.create_combinations --fda synchronous --nn AdvancedCNN --ds_name MNIST --b 32 --e 300 --th 0 --num_clients 5 10 15 20 25 30 35 40 45 50 55 60 --comb_file_id 0 --append_to
python -m fdavg.utils.create_combinations --fda synchronous FedAdam --nn AdvancedCNN --ds_name MNIST --b 32 --e 1000 --th 0 --num_clients 5 10 15 20 25 30 35 40 45 50 55 60 --comb_file_id 0 --append_to
python -m fdavg.utils.create_combinations --fda linear sketch --nn AdvancedCNN --ds_name MNIST --b 32 --e 300 --th 20 25 30 50 75 90 100 --bias -1 -2 --num_clients 5 10 15 20 25 30 35 40 45 50 55 60 --comb_file_id 0 --append_to
python -m fdavg.utils.create_combinations --fda synchronous --nn AdvancedCNN --ds_name MNIST --b 32 --e 300 --th 0 --bias -1 -2 --num_clients 5 10 15 20 25 30 35 40 45 50 55 60 --comb_file_id 0 --append_to
python -m fdavg.utils.create_combinations --fda synchronous FedAdam --nn AdvancedCNN --ds_name MNIST --b 32 --e 1000 --th 0 --bias -1 -2 --num_clients 5 10 15 20 25 30 35 40 45 50 55 60 --comb_file_id 0 --append_to

python -m fdavg.utils.create_combinations --fda linear sketch --nn DenseNet121 --ds_name CIFAR-10 --b 32 --e 300 --th 200 250 275 300 325 350 400 --num_clients 5 10 15 20 25 30 --comb_file_id 0 --append_to
python -m fdavg.utils.create_combinations --fda synchronous --nn DenseNet121 --ds_name CIFAR-10 --b 32 --e 300 --th 0 --num_clients 5 10 15 20 25 30 --comb_file_id 0 --append_to
python -m fdavg.utils.create_combinations --fda FedAvgM --nn DenseNet121 --ds_name CIFAR-10 --b 32 --e 1000 --th 0 --num_clients 5 10 15 20 25 30 --comb_file_id 0 --append_to

python -m fdavg.utils.create_combinations --fda linear sketch --nn DenseNet201 --ds_name CIFAR-10 --b 32 --e 300 --th 200 250 275 300 325 350 400 --num_clients 5 10 15 20 25 30 --comb_file_id 0 --append_to
python -m fdavg.utils.create_combinations --fda synchronous --nn DenseNet201 --ds_name CIFAR-10 --b 32 --e 300 --th 0 --num_clients 5 10 15 20 25 30 --comb_file_id 0 --append_to
python -m fdavg.utils.create_combinations --fda FedAvgM --nn DenseNet201 --ds_name CIFAR-10 --b 32 --e 1000 --th 0 --num_clients 5 10 15 20 25 30 --comb_file_id 0 --append_to

python -m fdavg.utils.create_combinations --fda linear sketch --nn ConvNeXtLarge --ds_name CIFAR-100 --b 32 --e 30 --th 25 50 100 150 --num_clients 3 5 --comb_file_id 0 --append_to
python -m fdavg.utils.create_combinations --fda synchronous --nn ConvNeXtLarge --ds_name CIFAR-100 --b 32 --e 30 --th 0 --num_clients 3 5 --comb_file_id 0 --append_to
