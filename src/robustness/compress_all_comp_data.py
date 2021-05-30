import tarfile
import pathlib
import logging
from datetime import datetime

from robustness import NETWORKS_DIR
logger = logging.getLogger(__name__)
logger.setLevel('INFO')

files_compressed = 0

for p in NETWORKS_DIR.glob('*/*/*/*/*'):
    if not p.is_dir():
        continue
    for base_name in ['comp_data', 'comp_data_fast']:
        full_file_name = p / f'{base_name}.txt'
        if not full_file_name.is_file():
            continue

        ## Remove files created under incorrect algorithm, which has been
        ## corrected on May 17, 2021
        if base_name == 'comp_data_fast':
            reference_date = datetime(2021, 5, 17) 
            modification_time = datetime.fromtimestamp(
                full_file_name.lstat().st_ctime
            )
            if modification_time < reference_date:
                full_file_name.unlink()
                continue


        ## Compress network file
        full_tar_input_name = p / f'{base_name}.tar.gz'
        tar = tarfile.open(full_tar_input_name, 'w:gz')
        tar.add(full_file_name, arcname= f'{base_name}.txt')
        tar.close()

        ## Remove network file
        full_file_name.unlink()
        files_compressed += 1
        logger.info(full_file_name)
        #input()

logger.info(f'Files compressed: {files_compressed}')