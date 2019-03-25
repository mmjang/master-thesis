#import gprMax.gprMax
import os
import random
import math
MARGIN = 1
SPACE_STEP = 0.02

cmd_tmpl = 'C:\\Users\\liaob\\Miniconda3\\envs\\gprMax\\python.exe -m gprMax {0} -n 230 --geometry-fixed --geometry-only'
DIR = ''
TPML_IN1 = 'gpr_tmpl.in'

tmpl1 = open(DIR + TPML_IN1)
tmpl_str1 = tmpl1.read()
tmpl1.close()

# tmpl2 = open(DIR + TPML_IN2)
# tmpl_str2 = tmpl2.read()
# tmpl2.close()

def flip_coin():

    return random.random() < 0.5

def create_if_no_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

mat_list = ['metal', 'rock', 'water']

def run(name):

    
    run_dir = DIR + name
    create_if_no_exists(run_dir)
    config_path = run_dir + '\\' + name + '.in'
    obj_pos_path = run_dir + '\\' + 'pos.txt'
    ff = open(config_path,'w')

    pos_off = random.uniform(-1, 1)
    obj_size = random.uniform(0.1,0.3)
    mat = random.choice(mat_list)
    depth = random.uniform(0.1,0.4)

    ff.write(tmpl_str1.format(pos_off, obj_size, mat, depth))
    ffp = open(obj_pos_path, 'w')
    ffp.write(str(pos_off) + '\n')
    ffp.write(str(obj_size) + '\n')
    ffp.write(str(mat) + '\n')
    ffp.write(str(depth)+'\n')
    ffp.close()
    ff.close()

    os.system(cmd_tmpl.format(config_path))




for i in range(5):
        run('run' + str(i) + '_')

