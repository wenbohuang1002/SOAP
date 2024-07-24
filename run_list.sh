echo "task1"
/home/wenbo/.pyenv/versions/anaconda3-2023.03/envs/hwb/bin/python /home/wenbo/Project/Triple-master/runfull.py --way 5 --shot 1 --dataset kinetics
/home/wenbo/.pyenv/versions/anaconda3-2023.03/envs/hwb/bin/python /home/wenbo/Project/Triple-master/runfull.py --way 5 --shot 5 --dataset kinetics
/home/wenbo/.pyenv/versions/anaconda3-2023.03/envs/hwb/bin/python /home/wenbo/Project/Triple-master/runfull.py --way 5 --shot 1 --dataset ssv2 --training_iterations 75000 --learning_rate 0.01 --sch 20000 40000 50000 60000
/home/wenbo/.pyenv/versions/anaconda3-2023.03/envs/hwb/bin/python /home/wenbo/Project/Triple-master/runfull.py --way 5 --shot 5 --dataset ssv2 --training_iterations 75000 --learning_rate 0.01 --sch 20000 40000 50000 60000
/home/wenbo/.pyenv/versions/anaconda3-2023.03/envs/hwb/bin/python /home/wenbo/Project/Triple-master/runfull.py --way 5 --shot 1 --dataset hmdb
/home/wenbo/.pyenv/versions/anaconda3-2023.03/envs/hwb/bin/python /home/wenbo/Project/Triple-master/runfull.py --way 5 --shot 5 --dataset hmdb
/home/wenbo/.pyenv/versions/anaconda3-2023.03/envs/hwb/bin/python /home/wenbo/Project/Triple-master/runfull.py --way 5 --shot 1 --dataset ucf
/home/wenbo/.pyenv/versions/anaconda3-2023.03/envs/hwb/bin/python /home/wenbo/Project/Triple-master/runfull.py --way 5 --shot 5 --dataset ucf