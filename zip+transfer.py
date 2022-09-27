import os
from glob import glob
import shutil
from ftplib import FTP

dir = '/rds/projects/w/windowcr-granutools-engd/DigitalTwinStudy/Simulations/GranuDrum_old'
zip_dir = 'Zips'
string = 'gd_1_0_0_0_0'
sub_dirs = glob(os.path.join(dir, string, ""))
# print(sub_dirs)

for i, sub in enumerate(sub_dirs):
    try:
        print(f'Attempting to zip {sub}')
        shutil.make_archive(f'{sub_dirs[i]}', 'zip', f'{sub}')
    except:
        print(f"Couldn't zip {sub}")

    ftp = FTP('81.247.167.19', user='Ben', passwd="_cp_$;#%7##b+EL'=h|a")
    ftp.cwd('DataSimu/DigitalTwinsProject/GranuDrum')
